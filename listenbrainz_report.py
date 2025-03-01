#!/usr/bin/env python3
"""
Report generator for ListenBrainz Collaborative Filtering results
"""
import argparse
import decimal
import json
import os
import random
import time
import urllib
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import requests
from bs4 import BeautifulSoup
from community import community_louvain
from dotenv import load_dotenv
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import the HTML report generation functions
from html_report import generate_html_report, open_html_report

load_dotenv()

LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")


class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle Decimal objects."""
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)
    

def convert_decimal_to_float(obj):
    """Recursively converts Decimal objects to float for JSON serialization."""
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal_to_float(item) for item in obj]
    else:
        return obj
    
    
def time_function(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        logger.info(f"Finished {func.__name__}: Took {end_time - start_time:.2f}s")
        
        return result
    return wrapper


def get_cache_dir(working_dir: str) -> Path:
    """Create and return the cache directory path."""
    cache_dir = Path(working_dir) / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def load_cache(cache_file: Path) -> dict:
    """Load cached data from a file."""
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache_file: Path, data: dict) -> None:
    """Save data to cache file."""
    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2)


def create_directories(output_dir: str) -> Dict[str, str]:
    """Create report subdirectories and return paths.
    
    Args:
        output_dir: Base directory for reports
        
    Returns:
        Dictionary of directory paths
    """
    # Create base directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    images_dir = os.path.join(output_dir, "images")
    data_dir = os.path.join(output_dir, "data")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    return {
        "base": output_dir,
        "images": images_dir,
        "data": data_dir
    }

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
def load_artist_info(artist_file: str, artist_ids: List[str] = None) -> Dict[str, str]:
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
def fetch_artist_metadata(artist_id: str, artist_name: str, working_dir: str = None) -> Dict:
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

@time_function
def generate_dataset_summary(stats: Dict, artist_info: Dict, dirs: Dict[str, str]) -> None:
    """Generate a summary of the dataset."""
    logger.info("Generating dataset summary")
    
    # Create a report dictionary
    report = {
        "dataset_statistics": {
            "total_users": stats["total_users"],
            "total_artists": stats["total_artists"],
            "total_entries": stats["total_entries"],
            "total_listens": stats["total_listens"],
            "avg_listens_per_user": stats["avg_listens_per_user"],
            "avg_artists_per_user": stats["avg_artists_per_user"],
            "avg_users_per_artist": stats["avg_users_per_artist"]
        },
        "top_artists": []
    }
    
    # Add top artists with names
    for artist in stats["top_artists"][:20]:
        artist_id = artist["artist_id"]
        artist["name"] = artist_info.get(artist_id, "Unknown Artist")
        report["top_artists"].append(artist)
    
    # Write JSON report to data directory
    with open(os.path.join(dirs["data"], "dataset_summary.json"), "w") as f:
        json.dump(report, f, indent=2, cls=DecimalEncoder)
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    
    # Plot top 20 artists by listen count
    top_artist_names = [artist_info.get(a["artist_id"], "Unknown")[:20] for a in stats["top_artists"][:20]]
    top_artist_listens = [a["total_listens"] for a in stats["top_artists"][:20]]
    
    plt.bar(range(len(top_artist_names)), top_artist_listens)
    plt.xticks(range(len(top_artist_names)), top_artist_names, rotation=45, ha="right")
    plt.title("Top 20 Artists by Total Listens")
    plt.xlabel("Artist")
    plt.ylabel("Total Listens")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["images"], "top_artists.png"), dpi=300)
    plt.close()
    
    
    # 1. Listen count distribution (histogram)
    user_listen_counts = [float(u["total_listens"]) for u in stats["top_users"]]
    
    plt.figure(figsize=(12, 6))
    plt.hist(user_listen_counts, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Listen Counts Among Users")
    plt.xlabel("Number of Listens")
    plt.ylabel("Number of Users")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["images"], "user_listen_distribution.png"), dpi=300)
    plt.close()
    
    # 2. User engagement scatter plot - Listen count vs. Unique artists
    unique_artists = [float(u["unique_artists"]) for u in stats["top_users"]]
    user_listens = [float(u["total_listens"]) for u in stats["top_users"]]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(unique_artists, user_listens, alpha=0.6, c='teal', edgecolors='darkblue')
    plt.title("User Listening Diversity")
    plt.xlabel("Number of Unique Artists")
    plt.ylabel("Total Listen Count")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    if unique_artists and user_listens:
        z = np.polyfit(unique_artists, user_listens, 1)
        p = np.poly1d(z)
        plt.plot(sorted(unique_artists), p(sorted(unique_artists)), "r--", alpha=0.8, 
                 label=f"Trend: y = {z[0]:.2f}x + {z[1]:.2f}")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["images"], "user_listening_diversity.png"), dpi=300)
    plt.close()
    
    # 3. User engagement segments - categorize users by activity
    if len(stats["top_users"]) > 100:
        # Calculate percentiles if we have enough users
        listen_counts = sorted([float(u["total_listens"]) for u in stats["top_users"]], reverse=True)
        total_listens = sum(listen_counts)
        
        # Calculate what percentage of listens come from different user segments
        segments = {
            "Top 1% Users": sum(listen_counts[:max(1, int(len(listen_counts) * 0.01))]) / total_listens * 100,
            "Next 9% Users": sum(listen_counts[max(1, int(len(listen_counts) * 0.01)):int(len(listen_counts) * 0.1)]) / total_listens * 100,
            "Next 40% Users": sum(listen_counts[int(len(listen_counts) * 0.1):int(len(listen_counts) * 0.5)]) / total_listens * 100,
            "Bottom 50% Users": sum(listen_counts[int(len(listen_counts) * 0.5):]) / total_listens * 100
        }
        
        # Pie chart of user segments
        plt.figure(figsize=(10, 7))
        plt.pie(segments.values(), labels=segments.keys(), autopct='%1.1f%%', 
                startangle=90, shadow=False, colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        plt.axis('equal')
        plt.title("Distribution of Listens by User Segments")
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["images"], "user_segments.png"), dpi=300)
        plt.close()
        
    # 4. Artist-to-Listen ratio comparison for top users
    # This shows which users discover more artists vs those who listen to the same artists repeatedly
    if len(stats["top_users"]) >= 20:
        top_20_users = stats["top_users"][:20]
        indices = np.arange(len(top_20_users))
        
        # Convert to float to avoid Decimal type issues
        artist_to_listen_ratios = [(float(u["unique_artists"]) / float(u["total_listens"])) * 1000 for u in top_20_users]
        user_listens = [float(u["total_listens"]) for u in top_20_users]
        
        # Create a two-panel chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot total listens on the top panel
        bars1 = ax1.bar(indices, user_listens, color='#3498db')
        ax1.set_title("Listen Count for Top 20 Users")
        ax1.set_ylabel("Total Listen Count")
        ax1.grid(True, linestyle='--', axis='y', alpha=0.7)
        
        # Annotate the bars with listen counts
        for i, v in enumerate(user_listens):
            ax1.text(i, v + max(user_listens) * 0.01, f"{v:,}", 
                     ha='center', va='bottom', rotation=90, fontsize=8)
        
        # Plot artist-to-listen ratios on the bottom panel
        bars2 = ax2.bar(indices, artist_to_listen_ratios, color='#e74c3c')
        ax2.set_title("Artist-to-Listen Ratio for Top 20 Users (higher = more diverse)")
        ax2.set_xlabel("User Rank")
        ax2.set_ylabel("Artists per 1000 Listens")
        ax2.grid(True, linestyle='--', axis='y', alpha=0.7)
        ax2.set_xticks(indices)
        ax2.set_xticklabels([f"User {i+1}" for i in indices], rotation=45, ha='right')
        
        # Annotate the bars with the ratios
        for i, v in enumerate(artist_to_listen_ratios):
            ax2.text(i, v + max(artist_to_listen_ratios) * 0.01, f"{v:.1f}", 
                    ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["images"], "top_users_comparison.png"), dpi=300)
        plt.close()
        
    logger.info(f"Dataset summary saved to {dirs['base']}")

@time_function
def analyze_artist_recommendations(working_dir: str, artist_info: Dict, dirs: Dict[str, str]) -> None:
    """Analyze artist recommendations from the model."""
    logger.info("Analyzing artist recommendations")
    
    # Look for recommendation files in the recommendations directory
    recommendations_dir = os.path.join(working_dir, "recommendations")
    if not os.path.exists(recommendations_dir):
        logger.warning("No recommendations directory found")
        return
        
    recommendation_files = [f for f in os.listdir(recommendations_dir) 
                          if f.startswith("similar_to_") and f.endswith(".csv")]
    
    logger.info(f"Found {len(recommendation_files)} recommendation files")
    
    if not recommendation_files:
        logger.warning("No recommendation files found in recommendations directory")
        return
    
    all_recommendations = []
    
    for rec_file in recommendation_files:
        try:
            # Extract source artist from filename
            source_artist = rec_file.replace("similar_to_", "").replace(".csv", "")
            logger.info(f"Processing recommendations for {source_artist}")
            
            # Read recommendations and skip the first row (self-recommendation)
            df = pl.read_csv(os.path.join(recommendations_dir, rec_file))
            df = df.filter(pl.col("rank") > 1)  # Skip the self-recommendation
            
            # Add artist names
            recommendations = []
            for row in df.to_dicts():
                artist_id = row["artist_id"]
                row["artist_name"] = artist_info.get(artist_id, "Unknown Artist")
                recommendations.append(row)
            
            all_recommendations.append({
                "source_artist": source_artist,
                "recommendations": recommendations
            })
        except Exception as e:
            logger.warning(f"Error processing recommendation file {rec_file}: {str(e)}")
    
    # Write recommendation analysis to file
    with open(os.path.join(dirs["data"], "recommendations_analysis.json"), "w") as f:
        json.dump(all_recommendations, f, indent=2, cls=DecimalEncoder)
    
    logger.info(f"Recommendation analysis saved to {dirs['data']} for {len(all_recommendations)} artists")

@time_function
def analyze_artist_diversity(stats: Dict, artist_info: Dict, dirs: Dict[str, str]) -> None:
    """Analyze the diversity of artists in the dataset."""
    logger.info("Analyzing artist diversity")
    
    # Fetch additional metadata for top artists
    top_artists = stats["top_artists"][:20]
    
    # Create a cache for artist metadata to avoid duplicate API calls
    artist_metadata_cache = {}
    
    # Collect artist metadata for analysis
    artist_metadata = []
    
    for artist in top_artists:
        artist_id = artist["artist_id"]
        artist_name = artist_info.get(artist_id, "Unknown Artist")
        
        # Use cached metadata if available
        if artist_id in artist_metadata_cache:
            metadata = artist_metadata_cache[artist_id]
        else:
            metadata = fetch_artist_metadata(artist_id, artist_name, dirs["base"])
            artist_metadata_cache[artist_id] = metadata
            
        metadata["total_listens"] = artist["total_listens"]
        metadata["unique_listeners"] = artist["unique_listeners"]
        metadata["artist_id"] = artist_id
        
        artist_metadata.append(metadata)
    
    # --- COUNTRY DISTRIBUTION ---
    country_counts = {}
    for artist in artist_metadata:
        country = artist["country"]
        if country != "Unknown" and country is not None:
            country_counts[country] = country_counts.get(country, 0) + 1
    
    # --- GENRE DISTRIBUTION ---
    genre_counts = {}
    for artist in artist_metadata:
        for genre in artist["genres"]:
            if genre:  # Ensure genre is not empty
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # --- TIME PERIOD DISTRIBUTION ---
    decades = {}
    for artist in artist_metadata:
        start_date = artist.get("start_date", "Unknown")
        if start_date != "Unknown":
            try:
                year = int(start_date.split("-")[0])
                decade = (year // 10) * 10
                decades[decade] = decades.get(decade, 0) + 1
            except (ValueError, AttributeError):
                pass
    
    # Save diversity analysis
    diversity_analysis = {
        "country_distribution": country_counts,
        "genre_distribution": genre_counts,
        "decade_distribution": decades,
        "artist_metadata": convert_decimal_to_float(artist_metadata)
    }
    
    with open(os.path.join(dirs["data"], "artist_diversity.json"), "w") as f:
        json.dump(diversity_analysis, f, indent=2, cls=DecimalEncoder)
    
    # Create visualizations
    # 1. Country Distribution - Now using a donut chart instead of bar chart
    if country_counts:
        plt.figure(figsize=(12, 8))
        
        # Sort countries by count for better visualization
        sorted_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)
        
        # For a cleaner visualization, group small countries as "Other"
        # if there are too many countries
        if len(sorted_countries) > 8:
            main_countries = sorted_countries[:7]  # Keep top 7 countries
            other_count = sum(country[1] for country in sorted_countries[7:])
            
            # Only add "Other" if it's not empty
            if other_count > 0:
                chart_data = main_countries + [("Other", other_count)]
            else:
                chart_data = main_countries
        else:
            chart_data = sorted_countries
        
        # Extract data for the pie chart
        labels = [f"{country} ({count})" for country, count in chart_data]
        counts = [count for _, count in chart_data]
        
        # Create a colorful palette for the chart
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(chart_data)))
        
        # Create a donut chart
        wedges, texts, autotexts = plt.pie(
            counts, 
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor='w'),  # The width creates the donut hole
            textprops=dict(color="k", fontsize=12)
        )
        
        # Equal aspect ratio ensures the pie chart is circular
        plt.axis('equal')  
        
        plt.title("Country Distribution of Top Artists", fontsize=16, pad=20)
        
        # Add a center circle for better donut appearance
        center_circle = plt.Circle((0, 0), 0.25, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(center_circle)
        
        # Adjust styling
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["images"], "country_distribution.png"), dpi=300)
        plt.close()
    
    # 2. Genre Distribution
    if genre_counts:
        plt.figure(figsize=(12, 6))
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        
        if top_genres:
            genres = [g[0] for g in top_genres]
            counts = [g[1] for g in top_genres]
            plt.bar(genres, counts, color='lightgreen')
            plt.title("Genre Distribution of Top Artists")
            plt.xlabel("Genre")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(dirs["images"], "genre_distribution.png"), dpi=300)
        plt.close()
    
    # 3. Time Period Distribution (improved histogram)
    if decades:
        plt.figure(figsize=(12, 6))
        sorted_decades = sorted(decades.items())
        labels = [f"{d}s" for d, _ in sorted_decades]
        values = [c for _, c in sorted_decades]
        
        plt.bar(labels, values, color='salmon')
        plt.title("Time Period Distribution of Top Artists")
        plt.xlabel("Decade")
        plt.ylabel("Number of Artists")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["images"], "time_period_distribution.png"), dpi=300)
        plt.savefig(os.path.join(dirs["images"], "artist_decades.png"), dpi=300)
        plt.close()
    
    logger.info(f"Artist diversity analysis saved to {dirs['data']}")

@time_function
def generate_full_report(user_artist_file: str, artist_file: str, working_dir: str, 
                        output_dir: str = None, open_report: bool = True,
                        use_last_search: bool = False, mapping_stats: Dict = None) -> None:
    """Generate a full report on the collaborative filtering results."""
    logger.info("Generating full report")
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(working_dir, "report")
    
    # Create directories
    dirs = create_directories(output_dir)
    
    # Load data
    matrix_artists, matrix_users, stats = load_data_matrix(user_artist_file)
    
    # Convert artist IDs to a list for lookup
    artist_ids = list(matrix_artists)
    
    # Add top artists from stats
    for artist in stats["top_artists"]:
        artist_ids.append(artist["artist_id"])
    
    # Load artist info
    artist_info = load_artist_info(artist_file, artist_ids)
    
    # If mapping statistics are provided, save them
    if mapping_stats:
        save_data_quality_metrics(mapping_stats, dirs)
        logger.info("Using provided mapping statistics for data quality metrics")
    else:
        logger.info("No mapping statistics provided; data quality section will be minimal")
    
    # Find recommendation files - check recommendations directory
    recommendations_dir = os.path.join(working_dir, "recommendations")
    if os.path.exists(recommendations_dir):
        recommendation_files = [f for f in os.listdir(recommendations_dir) 
                              if f.startswith("similar_to_") and f.endswith(".csv")]
    else:
        # Create recommendations directory if it doesn't exist
        os.makedirs(recommendations_dir, exist_ok=True)
        recommendation_files = []
    
    # Handle last search option
    if use_last_search and not recommendation_files:
        last_artists = load_last_search(working_dir)
        if last_artists:
            logger.info(f"Using {len(last_artists)} artists from previous search")
            # Here you would typically re-run the searches for these artists
            # But since we're just using existing files, we'll check if they exist
            for artist_id in last_artists:
                expected_file = f"similar_to_{artist_id}.csv"
                if os.path.exists(os.path.join(recommendations_dir, expected_file)):
                    recommendation_files.append(expected_file)
                else:
                    logger.warning(f"Previous search file not found: {expected_file}")
    
    # Save current search for future use
    if recommendation_files:
        save_last_search(recommendation_files, working_dir)
    
    # Generate dataset summary
    generate_dataset_summary(stats, artist_info, dirs)
    
    # Analyze artist recommendations
    analyze_artist_recommendations(working_dir, artist_info, dirs)
    
    # Generate artist network visualization
    generate_artist_network(working_dir, artist_info, dirs, max_artists=30)
    
    # Analyze user segments
    analyze_user_segments(user_artist_file, dirs)
    
    # Compare with Last.fm - use the API key from environment variables
    compare_with_lastfm(working_dir, artist_info, dirs, LASTFM_API_KEY)
    
    # Analyze artist diversity
    analyze_artist_diversity(stats, artist_info, dirs)
    
    # Add the new artist distribution analysis
    analyze_artist_distribution(user_artist_file, dirs)
        
    # Generate an HTML report
    report_path = os.path.join(dirs["base"], 'report.html')
    generate_html_report(stats, artist_info, dirs)
    
    logger.info(f"Full report generated in {dirs['base']}")
    logger.info(f"Main HTML report: {report_path}")
    
    # Open the report if requested
    if open_report:
        open_html_report(report_path)

@time_function
def compare_with_lastfm(working_dir: str, artist_info: Dict, dirs: Dict[str, str], api_key: str = None) -> None:
    """Compare our model's recommendations with Last.fm recommendations."""
    logger.info("Comparing model recommendations with Last.fm")
    
    # Find recommendation files in the recommendations directory
    recommendations_dir = os.path.join(working_dir, "recommendations")
    if not os.path.exists(recommendations_dir):
        logger.warning("No recommendations directory found")
        return
        
    recommendation_files = [f for f in os.listdir(recommendations_dir) 
                          if f.startswith("similar_to_") and f.endswith(".csv")]
    
    if not recommendation_files:
        logger.warning("No recommendation files found in recommendations directory")
        return
    
    # Limit to a reasonable number of comparisons
    if len(recommendation_files) > 10:
        files_to_compare = random.sample(recommendation_files, 10)
    else:
        files_to_compare = recommendation_files
    
    comparison_results = []
    
    for rec_file in files_to_compare:
        try:
            # Extract artist name from filename
            source_artist = rec_file.replace("similar_to_", "").replace(".csv", "")
            artist_name = source_artist.replace("_", " ")
            
            logger.info(f"Comparing recommendations for {artist_name}")
            
            # Read our model's recommendations and skip self-recommendation
            rec_df = pl.read_csv(os.path.join(recommendations_dir, rec_file))
            rec_df = rec_df.filter(pl.col("rank") > 1)  # Skip the self-recommendation
            
            # Extract artist names if they exist in the dataframe, otherwise use IDs
            if 'artist_name' in rec_df.columns:
                model_recommendations = rec_df.select('artist_name').head(10).to_series().to_list()
            else:
                # Get the artist IDs and map to names
                artist_ids = rec_df.select('artist_id').head(10).to_series().to_list()
                model_recommendations = [artist_info.get(artist_id, f"Unknown Artist ({artist_id})") 
                                         for artist_id in artist_ids]
            
            # Get Last.fm recommendations
            lastfm_recommendations = fetch_lastfm_similar(artist_name, api_key, working_dir)
            
            # Calculate overlap
            overlap = set(model_recommendations) & set(lastfm_recommendations)
            overlap_percentage = len(overlap) / min(len(model_recommendations), len(lastfm_recommendations)) * 100 if lastfm_recommendations else 0
            
            comparison_results.append({
                'artist_name': artist_name,
                'model_recommendations': model_recommendations,
                'lastfm_recommendations': lastfm_recommendations,
                'overlap': list(overlap),
                'overlap_percentage': overlap_percentage
            })
            
            # Rate limiting for Last.fm
            time.sleep(1)
            
        except Exception as e:
            logger.warning(f"Error comparing recommendations for {source_artist}: {str(e)}")
    
    # Save comparison results
    with open(os.path.join(dirs["data"], "lastfm_comparison.json"), "w") as f:
        json.dump(comparison_results, f, indent=2, cls=DecimalEncoder)
    
    # Create visualization
    if comparison_results:
        artist_names = [r['artist_name'] for r in comparison_results]
        overlap_percentages = [r['overlap_percentage'] for r in comparison_results]
        
        plt.figure(figsize=(12, 6))
        plt.bar(artist_names, overlap_percentages, color='purple')
        plt.title('Overlap Between Model and Last.fm Recommendations')
        plt.xlabel('Artist')
        plt.ylabel('Overlap Percentage (%)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(dirs["images"], "lastfm_overlap.png"), dpi=300)
        plt.close()
    
    # Calculate and log average overlap
    if comparison_results:
        avg_overlap = sum(r['overlap_percentage'] for r in comparison_results) / len(comparison_results)
        logger.info(f"Average overlap with Last.fm recommendations: {avg_overlap:.2f}%")
    
    logger.info(f"Last.fm comparison saved to {dirs['data']}")

def fetch_lastfm_similar(artist_name: str, api_key: str = None, working_dir: str = None) -> List[str]:
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
def generate_artist_network(working_dir: str, artist_info: Dict, dirs: Dict[str, str], max_artists: int = 50) -> None:
    """Generate a network visualization of artist relationships based on similarity scores."""
    logger.info("Generating artist network visualization")
    
    # Find recommendation files in the recommendations directory
    recommendations_dir = os.path.join(working_dir, "recommendations")
    if not os.path.exists(recommendations_dir):
        logger.warning("No recommendations directory found for network visualization")
        create_placeholder_image("No recommendations directory found", 
                               os.path.join(dirs["images"], "artist_network.png"))
        return
        
    recommendation_files = [f for f in os.listdir(recommendations_dir) 
                          if f.startswith("similar_to_") and f.endswith(".csv")]
    
    if not recommendation_files:
        logger.warning("No recommendation files found in recommendations directory")
        create_placeholder_image("No recommendation files found", 
                               os.path.join(dirs["images"], "artist_network.png"))
        return
    
    # Create a graph
    G = nx.Graph()
    
    # Process each artist's recommendations
    edges = []
    artist_names = {}  # Store artist names for better visualization
    
    # Limit to a reasonable number to prevent overcrowding
    if len(recommendation_files) > max_artists:
        files_to_process = random.sample(recommendation_files, max_artists)
    else:
        files_to_process = recommendation_files
        
    logger.info(f"Processing {len(files_to_process)} files for network visualization")
    
    for rec_file in files_to_process:
        try:
            # Extract source artist from filename
            source_artist_id = rec_file.replace("similar_to_", "").replace(".csv", "")
            source_name = artist_info.get(source_artist_id, source_artist_id)
            source_name = source_name.replace("_", " ")
            artist_names[source_artist_id] = source_name
            
            # Read recommendations and skip self-recommendation
            df = pl.read_csv(os.path.join(recommendations_dir, rec_file))
            df = df.filter(pl.col("rank") > 1)  # Skip self-recommendation
            
            # Add edges for top recommendations
            for row in df.head(5).to_dicts():  # Only use top 5 recommendations
                target_artist_id = row["artist_id"]
                target_name = artist_info.get(target_artist_id, f"Unknown ({target_artist_id})")
                artist_names[target_artist_id] = target_name
                
                # Use similarity score as edge weight
                weight = float(row["similarity_score"])
                edges.append((source_artist_id, target_artist_id, weight))
        except Exception as e:
            logger.warning(f"Error processing {rec_file} for network graph: {e}")
    
    # Add edges to graph
    for source, target, weight in edges:
        G.add_edge(source, target, weight=weight)
    
    # Check if graph has edges
    if not G.edges():
        logger.warning("No valid edges found for network visualization")
        create_placeholder_image("No artist relationships found", 
                               os.path.join(dirs["images"], "artist_network.png"))
        return
    
    logger.info(f"Drawing network with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Draw the graph
    plt.figure(figsize=(15, 15))
    
    # Use community detection for coloring nodes if possible
    try:
        partition = community_louvain.best_partition(G)
        colors = [partition.get(node, 0) for node in G.nodes()]
    except ImportError:
        # Fallback to degree-based coloring
        colors = [min(G.degree(node) / 5, 1.0) for node in G.nodes()]
    
    # Layout for the graph - use spring layout but be robust with parameters
    try:
        pos = nx.spring_layout(G, k=0.15, iterations=50)
    except Exception as e:
        logger.warning(f"Error with spring_layout: {e}, falling back to circular layout")
        pos = nx.circular_layout(G)
    
    # Draw edges with varying thickness based on weight
    edge_weights = [G[u][v]['weight'] * 1.5 for u, v in G.edges()]
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights)
    
    # Draw nodes
    node_size = [G.degree(node) * 20 + 50 for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=colors, alpha=0.8, 
                          cmap=plt.cm.viridis)
    
    # Draw labels for higher degree nodes with artist names
    large_degree_nodes = {node: artist_names.get(node, node) 
                          for node in G.nodes() if G.degree(node) > 2}
    nx.draw_networkx_labels(G, pos, labels=large_degree_nodes, font_size=10)
    
    plt.title("Artist Similarity Network", fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    
    try:
        plt.savefig(os.path.join(dirs["images"], "artist_network.png"), dpi=300, bbox_inches='tight')
        logger.info(f"Artist network visualization saved to {dirs["images"]}")
    except Exception as e:
        logger.error(f"Error saving network visualization: {e}")
        create_placeholder_image("Error generating artist network", 
                               os.path.join(dirs["images"], "artist_network.png"))
    
    plt.close()

def create_placeholder_image(message: str, output_path: str):
    """Create a placeholder image with a message."""
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, message, horizontalalignment='center',
             verticalalignment='center', fontsize=14)
    plt.axis('off')
    try:
        plt.savefig(output_path, dpi=100)
        plt.close()
        logger.info(f"Created placeholder image at {output_path}")
    except Exception as e:
        logger.error(f"Failed to create placeholder image: {e}")

@time_function
def analyze_user_segments(user_artist_file: str, dirs: Dict[str, str]) -> Dict:
    """Perform clustering analysis to identify user segments based on listening patterns."""
    logger.info("Analyzing user segments")
        
    # Use DuckDB to efficiently process the data
    con = duckdb.connect(":memory:")
    
    try:
        # Load the data
        con.execute(f"CREATE TABLE counts AS SELECT * FROM read_csv_auto('{user_artist_file}')")
        
        # Extract features for clustering
        user_features_df = con.execute("""
            SELECT 
                user_id,
                COUNT(DISTINCT artist_id) as artist_count,
                SUM(listen_count) as total_listens,
                AVG(listen_count) as avg_listens_per_artist,
                STDDEV(listen_count) as stddev_listens,
                MAX(listen_count) as max_listens
            FROM counts
            GROUP BY user_id
            HAVING COUNT(DISTINCT artist_id) > 5
        """).pl()
        
        if len(user_features_df) < 10:
            logger.warning("Not enough user data for segmentation analysis")
            create_placeholder_image("Not enough user data for segmentation analysis",
                                    os.path.join(dirs["images"], "user_segments_scatter.png"))
            return {}
            
        logger.info(f"Processing data for {len(user_features_df)} users")
        
        # Prepare features for clustering
        features = user_features_df.select(
            ['artist_count', 'total_listens', 'avg_listens_per_artist', 
             'stddev_listens', 'max_listens']
        ).to_numpy()
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Hard-code to 3 clusters for simplicity and reliability
        optimal_clusters = 3
        
        # Perform clustering with fixed number of clusters
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to the original dataframe
        user_features_df = user_features_df.with_columns(pl.Series(name="cluster", values=clusters))
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in range(optimal_clusters):
            cluster_df = user_features_df.filter(pl.col("cluster") == cluster_id)
            stats = {
                "cluster_id": cluster_id,
                "user_count": len(cluster_df),
                "avg_artists": float(cluster_df["artist_count"].mean()),
                "avg_listens": float(cluster_df["total_listens"].mean()),
                "avg_listens_per_artist": float(cluster_df["avg_listens_per_artist"].mean()),
                "example_users": cluster_df["user_id"].head(5).to_list()
            }
            cluster_stats.append(stats)
        
        # Create cluster visualization - scatter plot
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(
            features[:, 0],  # artist_count
            features[:, 1],  # total_listens
            c=clusters,
            cmap='viridis',
            alpha=0.6,
            s=50,
            edgecolors='w'
        )
        
        plt.xlabel('Number of Unique Artists', fontsize=12)
        plt.ylabel('Total Listens', fontsize=12)
        plt.title('User Segments Based on Listening Patterns', fontsize=16)
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["images"], "user_segments_scatter.png"), dpi=300)
        plt.close()
        
        # Create bar chart of cluster characteristics
        plt.figure(figsize=(14, 8))
        
        # Prepare data for the plot
        cluster_ids = [stats["cluster_id"] for stats in cluster_stats]
        user_counts = [stats["user_count"] for stats in cluster_stats]
        avg_artists = [stats["avg_artists"] for stats in cluster_stats]
        avg_listens = [stats["avg_listens"] / 100 for stats in cluster_stats]  # Scale down for visibility
        
        # Create bar chart with multiple metrics
        x = np.arange(len(cluster_ids))
        width = 0.3
        
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot user count
        bars1 = ax1.bar(x - width, user_counts, width, label='User Count', color='skyblue')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Users', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        
        # Create a second y-axis for artist count
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x, avg_artists, width, label='Avg. Artists', color='orange')
        ax2.set_ylabel('Average Number of Artists', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Create a third y-axis for listen count
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))  # Offset the right spine
        bars3 = ax3.bar(x + width, avg_listens, width, label='Avg. Listens (÷100)', color='green')
        ax3.set_ylabel('Average Listens (÷100)', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        
        # Add cluster size as text
        for i, count in enumerate(user_counts):
            ax1.text(i - width, count + max(user_counts) * 0.03, f"{count}", 
                    ha='center', va='bottom', fontsize=10)
        
        # Set x-ticks
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Cluster {i}" for i in cluster_ids])
        
        plt.title('Characteristics of User Segments', fontsize=16)
        
        fig.tight_layout()
        
        # Create custom legend
        legend_elements = [
            plt.Line2D([0], [0], color='skyblue', lw=4, label='User Count'),
            plt.Line2D([0], [0], color='orange', lw=4, label='Avg. Artists'),
            plt.Line2D([0], [0], color='green', lw=4, label='Avg. Listens (÷100)')
        ]
        ax1.legend(handles=legend_elements, loc='upper left')
        
        plt.savefig(os.path.join(dirs["images"], "user_segments_characteristics.png"), dpi=300)
        plt.close()
        
        # Save cluster statistics
        segmentation_results = {
            "optimal_clusters": optimal_clusters,
            "cluster_stats": cluster_stats
        }
        
        with open(os.path.join(dirs["data"], "user_segments.json"), "w") as f:
            json.dump(segmentation_results, f, indent=2, cls=DecimalEncoder)
        
        logger.info(f"User segmentation analysis saved to {dirs['data']}")
        return segmentation_results
        
    except Exception as e:
        logger.error(f"Error during user segmentation: {e}")
        create_placeholder_image("Error during user segmentation analysis",
                                os.path.join(dirs["images"], "user_segments_scatter.png"))
        create_placeholder_image("Error during user segmentation analysis",
                                os.path.join(dirs["images"], "user_segments_characteristics.png"))
        return {}

@time_function
def save_last_search(artist_ids_or_files: List, working_dir: str) -> None:
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

@time_function
def analyze_artist_distribution(user_artist_file: str, dirs: Dict[str, str]) -> None:
    """Analyze and visualize the distribution of listeners per artist.
    
    Args:
        user_artist_file: Path to the user-artist-count CSV file
        output_dir: Directory to save visualizations
    """
    logger.info("Analyzing artist listener distribution for report")
    
    try:
        # Use DuckDB for efficient analysis
        con = duckdb.connect(":memory:")
        
        # Load the data
        con.execute(f"CREATE TABLE counts AS SELECT * FROM read_csv_auto('{user_artist_file}')")
        
        # Get total artists and users
        total_artists = con.execute("SELECT COUNT(DISTINCT artist_id) FROM counts").fetchone()[0]
        total_users = con.execute("SELECT COUNT(DISTINCT user_id) FROM counts").fetchone()[0]
        
        # Calculate MSID mapping statistics for data quality section
        mapping_stats = {
            "total_artists": total_artists,
            "total_users": total_users,
        }
        
        # Get distribution of listeners per artist
        listener_distribution = con.execute("""
            WITH listener_counts AS (
                SELECT artist_id, COUNT(DISTINCT user_id) as listener_count
                FROM counts
                GROUP BY artist_id
            )
            SELECT listener_count, COUNT(*) as artist_count
            FROM listener_counts
            GROUP BY listener_count
            ORDER BY listener_count
        """).fetchall()
        
        # Calculate percentiles for report
        percentiles = con.execute("""
            WITH listener_counts AS (
                SELECT artist_id, COUNT(DISTINCT user_id) as listener_count
                FROM counts
                GROUP BY artist_id
            )
            SELECT 
                APPROX_QUANTILE(listener_count, 0.5) as median,
                APPROX_QUANTILE(listener_count, 0.9) as p90,
                APPROX_QUANTILE(listener_count, 0.95) as p95,
                APPROX_QUANTILE(listener_count, 0.99) as p99,
                MAX(listener_count) as max
            FROM listener_counts
        """).fetchone()
        
        mapping_stats["percentiles"] = {
            "median": percentiles[0],
            "p90": percentiles[1],
            "p95": percentiles[2], 
            "p99": percentiles[3],
            "max": percentiles[4]
        }
        
        # Calculate stats for bucketed segments
        buckets = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
        bucket_stats = []
        
        for low, high in buckets:
            if high == float('inf'):
                query = f"""
                    WITH listener_counts AS (
                        SELECT artist_id, COUNT(DISTINCT user_id) as listener_count
                        FROM counts
                        GROUP BY artist_id
                    )
                    SELECT COUNT(*) 
                    FROM listener_counts
                    WHERE listener_count >= {low}
                """
                label = f"{low}+"
            else:
                query = f"""
                    WITH listener_counts AS (
                        SELECT artist_id, COUNT(DISTINCT user_id) as listener_count
                        FROM counts
                        GROUP BY artist_id
                    )
                    SELECT COUNT(*) 
                    FROM listener_counts
                    WHERE listener_count BETWEEN {low} AND {high}
                """
                label = f"{low}-{high}"
            
            artist_count = con.execute(query).fetchone()[0]
            percent = (artist_count / total_artists) * 100
            
            bucket_stats.append({
                "label": label,
                "range": [low, high if high != float('inf') else mapping_stats["percentiles"]["max"]],
                "count": int(artist_count),
                "percentage": percent
            })
        
        mapping_stats["buckets"] = bucket_stats
        
        # Calculate mean and stddev for artist listeners
        stats = con.execute("""
            WITH listener_counts AS (
                SELECT artist_id, COUNT(DISTINCT user_id) as listener_count
                FROM counts
                GROUP BY artist_id
            )
            SELECT 
                AVG(listener_count) as mean,
                STDDEV(listener_count) as stddev
            FROM listener_counts
        """).fetchone()
        
        mapping_stats["mean"] = float(stats[0])
        mapping_stats["stddev"] = float(stats[1])
        
        # Save statistics for use in the report
        with open(os.path.join(dirs["data"], "artist_distribution.json"), "w") as f:
            json.dump(mapping_stats, f, indent=2, cls=DecimalEncoder)
        
        # Create visualizations
        
        # 1. Histogram of artist listener counts (log scale)
        plt.figure(figsize=(12, 6))
        
        # Extract x (listener counts) and y (number of artists) from distribution
        x_values = [row[0] for row in listener_distribution]
        y_values = [row[1] for row in listener_distribution]
        
        # Plot histogram style bars but with actual data rather than binning
        plt.bar(x_values[:100], y_values[:100], width=0.8, color='skyblue', alpha=0.7)
        plt.yscale('log')  # Log scale for y-axis to show the long tail
        plt.title('Distribution of Listeners per Artist (first 100 values)', fontsize=14)
        plt.xlabel('Number of Listeners', fontsize=12)
        plt.ylabel('Number of Artists (log scale)', fontsize=12)
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["images"], "artist_listener_histogram.png"), dpi=300)
        plt.close()
        
        # 2. Pie chart of listener segments
        plt.figure(figsize=(10, 8))
        
        # Prepare data for the pie chart
        labels = [f"{b['label']} listeners ({b['percentage']:.1f}%)" for b in bucket_stats]
        sizes = [b['count'] for b in bucket_stats]
        
        # Use a colorful palette
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(bucket_stats)))
        
        # Create pie chart with percentages
        plt.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
        
        # Add a legend with custom ordering (most to least frequent)
        plt.legend(labels, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.title('Artist Distribution by Listener Count', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["images"], "artist_listener_segments.png"), dpi=300)
        plt.close()
        
        # 3. Cumulative distribution chart
        plt.figure(figsize=(12, 6))
        
        # Convert to cumulative data
        x_cum = range(1, 101)  # First 100 points
        y_cum = []
        total = sum(y_values)
        cumulative = 0
        
        for i in range(min(100, len(y_values))):
            cumulative += y_values[i]
            y_cum.append((cumulative / total) * 100)
        
        plt.plot(x_cum, y_cum, marker='o', linewidth=2, markersize=5)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title('Cumulative Distribution of Artists by Listener Count', fontsize=14)
        plt.xlabel('Number of Listeners', fontsize=12)
        plt.ylabel('Cumulative Percentage of Artists', fontsize=12)
        
        # Highlight key percentiles
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.5)
        plt.text(x_cum[-1] * 0.8, 52, '50% of artists', color='r')
        plt.axhline(y=90, color='g', linestyle='--', alpha=0.5)
        plt.text(x_cum[-1] * 0.8, 92, '90% of artists', color='g')
        
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["images"], "artist_listener_cumulative.png"), dpi=300)
        plt.close()
        
        # 4. Bar chart of the buckets
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        bucket_labels = [b['label'] for b in bucket_stats]
        bucket_counts = [b['count'] for b in bucket_stats]
        
        # Create bar chart
        plt.bar(bucket_labels, bucket_counts, color='skyblue')
        plt.yscale('log')  # Log scale for better visualization
        plt.title('Artist Counts by Listener Range (Log Scale)', fontsize=14)
        plt.xlabel('Number of Listeners', fontsize=12)
        plt.ylabel('Number of Artists (log scale)', fontsize=12)
        plt.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on top of bars
        for i, count in enumerate(bucket_counts):
            plt.text(i, count * 1.1, f"{count:,}", ha='center', va='bottom', rotation=0)
            
        plt.tight_layout()
        plt.savefig(os.path.join(dirs["images"], "artist_listener_buckets.png"), dpi=300)
        plt.close()

        logger.info(f"Artist distribution visualizations saved to {dirs["images"]}")
        
        return mapping_stats
    
    except Exception as e:
        logger.error(f"Error analyzing artist distribution: {e}")
        create_placeholder_image("Error analyzing artist distribution", 
                                os.path.join(dirs["images"], "artist_listener_histogram.png"))
        return None

@time_function
def save_data_quality_metrics(mapping_stats: Dict, dirs: Dict[str, str]) -> None:
    """Save data quality metrics to a JSON file.
    
    Args:
        mapping_stats: Dictionary containing mapped and unmapped counts
        dirs: Dictionary of directory paths
    """
    logger.info("Saving data quality metrics")
    
    # Check for required fields in mapping stats
    if not mapping_stats:
        logger.warning("No mapping statistics provided")
        return
    
    # Support two possible formats of mapping_stats
    if "mapped_count" in mapping_stats and "unmapped_count" in mapping_stats:
        # Format with explicit counts
        mapped_count = mapping_stats["mapped_count"]
        unmapped_count = mapping_stats["unmapped_count"]
        total = mapped_count + unmapped_count
        
        if total == 0:
            logger.warning("No mapping data available (total is 0)")
            return
        
        mapped_pct = (mapped_count / total) * 100
        unmapped_pct = (unmapped_count / total) * 100
        
        quality_metrics = {
            "mapped": mapped_pct,
            "unmapped": unmapped_pct,
            "total": total,
            "mapped_count": mapped_count,
            "unmapped_count": unmapped_count,
            "quality_score": mapped_pct  # Simple quality score is just the mapping percentage
        }
    elif "mapped" in mapping_stats and "unmapped" in mapping_stats:
        # Format with percentages
        mapped_pct = mapping_stats["mapped"]
        unmapped_pct = mapping_stats["unmapped"]
        
        # If total is provided, use it; otherwise calculate from count fields if available
        if "total" in mapping_stats:
            total = mapping_stats["total"]
        elif "mapped_count" in mapping_stats and "unmapped_count" in mapping_stats:
            total = mapping_stats["mapped_count"] + mapping_stats["unmapped_count"]
        else:
            # If no counts are available, use a placeholder total
            total = 100
            logger.warning("No count information in mapping stats; using placeholder total")
        
        # Calculate counts from percentages if not provided
        mapped_count = mapping_stats.get("mapped_count", int(total * mapped_pct / 100))
        unmapped_count = mapping_stats.get("unmapped_count", int(total * unmapped_pct / 100))
        
        quality_metrics = {
            "mapped": mapped_pct,
            "unmapped": unmapped_pct,
            "total": total,
            "mapped_count": mapped_count,
            "unmapped_count": unmapped_count,
            "quality_score": mapped_pct
        }
    else:
        logger.warning("Invalid mapping statistics format")
        return
    
    # Save metrics to JSON for use in the report
    with open(os.path.join(dirs["data"], "data_quality.json"), 'w') as f:
        json.dump(quality_metrics, f, indent=2, cls=DecimalEncoder)
    
    # Generate data quality visualization
    plt.figure(figsize=(8, 6))
    labels = ['Mapped MSIDs', 'Unmapped MSIDs']
    sizes = [mapped_pct, unmapped_pct]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # explode the 1st slice for emphasis
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
    plt.title('MSID to MBID Mapping Coverage', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["images"], "msid_mbid_mapping.png"), dpi=300)
    plt.close()
    
    logger.info(f"Data quality metrics saved: {quality_metrics}")

def main():
    """Main function to run the report generator."""
    parser = argparse.ArgumentParser(description="ListenBrainz Collaborative Filtering Report Generator")
    parser.add_argument("--user-artist-file", required=True, help="Path to the user-artist-count CSV file")
    parser.add_argument("--artist-file", required=True, help="Path to the artist mapping file (CSV or Parquet)")
    parser.add_argument("--working-dir", required=True, help="Path to the working directory with model outputs")
    parser.add_argument("--output-dir", help="Path to save the report (defaults to working_dir/report)")
    parser.add_argument("--no-open-report", action="store_true", help="Don't automatically open the HTML report when finished")
    parser.add_argument("--use-last-search", action="store_true", 
                      help="Reuse the list of artists from the previous search")
    
    args = parser.parse_args()
        
    generate_full_report(
        args.user_artist_file,
        args.artist_file,
        args.working_dir,
        args.output_dir,
        not args.no_open_report,  # Invert the flag to maintain the default behavior
        args.use_last_search
    )

if __name__ == "__main__":
    main()
