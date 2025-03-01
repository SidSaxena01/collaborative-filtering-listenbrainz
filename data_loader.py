#!/usr/bin/env python3
"""
Data loading and external API interactions for ListenBrainz Collaborative Filtering
"""

import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import polars as pl
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger

from utils import get_cache_dir, load_cache, save_cache, time_function

# Load environment variables
load_dotenv()

# Get Last.fm API key from environment
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")


@time_function
def load_data_matrix(filename: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load the user-artist-count matrix and associated metadata."""
    logger.info(f"Loading data from {filename}")
    
    # Use DuckDB to efficiently load and analyze the data
    con = duckdb.connect(":memory:")
    
    # Load the data
    con.execute(f"CREATE TABLE counts AS SELECT * FROM read_csv_auto('{filename}')")
    
    # Get unique artists and users
    df_artists = con.execute("SELECT DISTINCT artist_id FROM counts ORDER BY artist_id").pl()
    df_users = con.execute("SELECT DISTINCT user_id FROM counts ORDER BY user_id").pl()
    
    matrix_artists = df_artists['artist_id'].to_numpy()
    matrix_users = df_users['user_id'].to_numpy()
    
    # Calculate basic statistics
    stats = {
        "total_users": len(matrix_users),
        "total_artists": len(matrix_artists),
        "total_entries": con.execute("SELECT COUNT(*) FROM counts").fetchone()[0],
        "total_listens": con.execute("SELECT SUM(listen_count) FROM counts").fetchone()[0],
        "avg_listens_per_user": con.execute("SELECT AVG(listen_count) FROM (SELECT user_id, SUM(listen_count) as listen_count FROM counts GROUP BY user_id)").fetchone()[0],
        "avg_artists_per_user": con.execute("SELECT AVG(artist_count) FROM (SELECT user_id, COUNT(DISTINCT artist_id) as artist_count FROM counts GROUP BY user_id)").fetchone()[0],
        "avg_users_per_artist": con.execute("SELECT AVG(user_count) FROM (SELECT artist_id, COUNT(DISTINCT user_id) as user_count FROM counts GROUP BY artist_id)").fetchone()[0]
    }
    
    # Get top artists by listen count
    top_artists_df = con.execute("""
        SELECT artist_id, SUM(listen_count) as total_listens, COUNT(DISTINCT user_id) as unique_listeners
        FROM counts
        GROUP BY artist_id
        ORDER BY total_listens DESC
        LIMIT 50
    """).pl()
    
    # Get top users by listen count
    top_users_df = con.execute("""
        SELECT user_id, SUM(listen_count) as total_listens, COUNT(DISTINCT artist_id) as unique_artists
        FROM counts
        GROUP BY user_id
        ORDER BY total_listens DESC
        LIMIT 50
    """).pl()
    
    stats["top_artists"] = top_artists_df.to_dicts()
    stats["top_users"] = top_users_df.to_dicts()
    
    return matrix_artists, matrix_users, stats


@time_function
def load_artist_info(artist_file: str, artist_ids: Optional[List[str]] = None) -> Dict[str, str]:
    """Load artist information from the artist mapping file."""
    logger.info(f"Loading artist information from {artist_file}")
    
    # Check if file is a Parquet file
    if artist_file.endswith('.parquet'):
        df = pl.read_parquet(artist_file)
    else:
        df = pl.read_csv(artist_file)
    
    # Filter to specific artist IDs if provided
    if artist_ids:
        df = df.filter(pl.col('artist_mbid').is_in(artist_ids))
    
    # Convert to dictionary
    artist_info = {row['artist_mbid']: row['name'] for row in df.to_dicts()}
    
    logger.info(f"Loaded information for {len(artist_info)} artists")
    return artist_info


@time_function
def fetch_artist_metadata(artist_id: str, artist_name: str, working_dir: Optional[str] = None) -> Dict:
    """Fetch additional metadata for an artist from MusicBrainz API with caching."""
    if working_dir is None:
        working_dir = os.getcwd()
    
    cache_dir = get_cache_dir(working_dir)
    cache_file = cache_dir / "musicbrainz_cache.json"
    
    # Load cache
    cache = load_cache(cache_file)
    
    # Check cache
    if artist_id in cache:
        logger.debug(f"Cache hit for artist {artist_name} ({artist_id})")
        return cache[artist_id]
    
    logger.info(f"Cache miss - fetching metadata for artist: {artist_name} ({artist_id})")
    
    # MusicBrainz API endpoint
    url = f"https://musicbrainz.org/ws/2/artist/{artist_id}?inc=aliases+genres+tags&fmt=json"
    
    headers = {
        "User-Agent": "ListenBrainzReport/1.0 (your-email@example.com)"
    }
    
    try:
        # Respect MusicBrainz rate limiting
        time.sleep(1.1)
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant information
        metadata = {
            "name": data.get("name", artist_name),
            "country": data.get("country", "Unknown"),
            "genres": [g["name"] for g in data.get("genres", [])],
            "tags": [t["name"] for t in data.get("tags", [])],
            "start_date": data.get("life-span", {}).get("begin", "Unknown"),
            "end_date": data.get("life-span", {}).get("end", "Unknown"),
            "type": data.get("type", "Unknown")
        }
        
        # Update cache
        cache[artist_id] = metadata
        save_cache(cache_file, cache)
        
        return metadata
    
    except Exception as e:
        logger.warning(f"Error fetching metadata for {artist_name}: {str(e)}")
        default_metadata = {
            "name": artist_name,
            "country": "Unknown",
            "genres": [],
            "tags": [],
            "start_date": "Unknown",
            "end_date": "Unknown",
            "type": "Unknown"
        }
        # Cache the default response to avoid repeated failed requests
        cache[artist_id] = default_metadata
        save_cache(cache_file, cache)
        return default_metadata


def fetch_lastfm_similar(artist_name: str, api_key: Optional[str] = None, working_dir: Optional[str] = None) -> List[str]:
    """Fetch similar artists from Last.fm API with caching."""
    if working_dir is None:
        working_dir = os.getcwd()
    
    cache_dir = get_cache_dir(working_dir)
    cache_file = cache_dir / "lastfm_cache.json"
    
    # Load cache
    cache = load_cache(cache_file)
    
    # Check cache
    cache_key = f"similar_{artist_name}"
    if cache_key in cache:
        logger.debug(f"Cache hit for Last.fm similar artists: {artist_name}")
        return cache[cache_key]
    
    logger.info(f"Cache miss - fetching similar artists for {artist_name} from Last.fm")
    
    similar = []
    
    # If you have an API key
    if api_key:
        url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getsimilar&artist={urllib.parse.quote(artist_name)}&api_key={api_key}&format=json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            similar = [a['name'] for a in data.get('similarartists', {}).get('artist', [])[:10]]
            logger.info(f"Successfully fetched similar artists for {artist_name} using Last.fm API")
        except Exception as e:
            logger.warning(f"Error using Last.fm API: {str(e)}, falling back to scraping")
    
    # Fallback to web scraping if API fails or no key provided
    if not similar:
        try:
            url = f"https://www.last.fm/music/{urllib.parse.quote(artist_name)}/+similar"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            artist_elements = soup.select(".similar-artists-item .link-block-target")
            similar = [el.text.strip() for el in artist_elements[:10]]
            
            logger.info(f"Successfully scraped similar artists for {artist_name} from Last.fm website")
        except Exception as e:
            logger.warning(f"Error scraping Last.fm: {str(e)}")
            similar = []
    
    # Update cache
    cache[cache_key] = similar
    save_cache(cache_file, cache)
    
    # Rate limiting
    time.sleep(1.1)
    
    return similar


@time_function
def save_last_search(artist_ids_or_files: List[Any], working_dir: str) -> None:
    """Save the list of last searched artists for future reuse.
    
    Args:
        artist_ids_or_files: Either a list of artist IDs or recommendation file names
        working_dir: Directory to save the last search file
    """
    logger.info("Saving last search information")
    last_search_path = Path(working_dir) / "last_search.json"
    
    # Check if we received file names or artist IDs
    if isinstance(artist_ids_or_files[0], str) and artist_ids_or_files[0].startswith("similar_to_"):
        # Extract artist IDs from recommendation filenames
        artist_ids = [f.replace("similar_to_", "").replace(".csv", "") 
                     for f in artist_ids_or_files 
                     if f.startswith("similar_to_") and f.endswith(".csv")]
    else:
        # Already have artist IDs
        artist_ids = artist_ids_or_files
    
    with open(last_search_path, "w") as f:
        json.dump({"artist_ids": artist_ids, "timestamp": time.time()}, f, indent=2)
    
    logger.info(f"Saved {len(artist_ids)} artist IDs to {last_search_path}")


@time_function
def load_last_search(working_dir: str) -> List[str]:
    """Load the list of previously searched artists."""
    last_search_path = Path(working_dir) / "last_search.json"
    
    if not last_search_path.exists():
        logger.warning("No previous search found.")
        return []
    
    try:
        with open(last_search_path, "r") as f:
            search_data = json.load(f)
        
        artist_ids = search_data.get("artist_ids", [])
        timestamp = search_data.get("timestamp", 0)
        search_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        
        logger.info(f"Loaded previous search from {search_date} with {len(artist_ids)} artists")
        return artist_ids
    except Exception as e:
        logger.error(f"Error loading previous search: {str(e)}")
        return []
