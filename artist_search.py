#!/usr/bin/env python3
"""
Artist search utility for ListenBrainz Collaborative Filtering
"""

import logging
import time
import urllib.parse
from typing import Dict, List

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("artist_search")

def search_artist(name: str) -> List[Dict]:
    """Search for an artist by name using the MusicBrainz API.
    
    Args:
        name: Artist name to search for
        
    Returns:
        List of matching artists with their MBIDs and metadata
    """
    # MusicBrainz API endpoint for artist search
    url = f"https://musicbrainz.org/ws/2/artist?query={urllib.parse.quote(name)}&fmt=json"
    
    # Add proper user agent to avoid being blocked
    headers = {
        "User-Agent": "ListenBrainzSearch/1.0 (your-email@example.com)"
    }
    
    try:
        # Respect MusicBrainz rate limiting (1 request per second)
        time.sleep(1.1)
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Extract artists from the response
        artists = []
        for artist in data.get("artists", [])[:10]:  # Limit to top 10 matches
            artists.append({
                "mbid": artist.get("id", ""),
                "name": artist.get("name", ""),
                "country": artist.get("country", "Unknown"),
                "type": artist.get("type", "Unknown"),
                "score": artist.get("score", 0),
                "disambiguation": artist.get("disambiguation", "")
            })
        
        return artists
    
    except Exception as e:
        logger.error(f"Error searching for artist '{name}': {str(e)}")
        return []

def interactive_artist_search() -> List[str]:
    """Interactively search for artists and let the user select them.
    
    Returns:
        List of selected artist MBIDs
    """
    selected_artists = []
    
    print("\n=== Artist Search Tool ===")
    print("Search for artists to analyze with the collaborative filtering model.")
    print("Type 'done' when you've finished selecting artists.\n")
    
    while True:
        # Ask for a search term
        search_term = input("\nEnter artist name to search (or 'done' to finish): ")
        
        if search_term.lower() == 'done':
            break
        
        # Search for artists
        print(f"Searching for '{search_term}'...")
        artists = search_artist(search_term)
        
        if not artists:
            print("No artists found matching that name.")
            continue
        
        # Display results
        print("\nSearch results:")
        for i, artist in enumerate(artists):
            disambiguation = f" ({artist['disambiguation']})" if artist['disambiguation'] else ""
            print(f"{i+1}. {artist['name']}{disambiguation} - {artist['country']} [{artist['type']}]")
        
        # Let user select
        selection = input("\nEnter the number of the artist to select (or 'skip' to try another search): ")
        
        if selection.lower() == 'skip':
            continue
        
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(artists):
                selected_artist = artists[idx]
                print(f"Selected: {selected_artist['name']} ({selected_artist['mbid']})")
                selected_artists.append(selected_artist['mbid'])
            else:
                print("Invalid selection number.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nYou selected {len(selected_artists)} artists.")
    return selected_artists

def check_artists_in_dataset(artist_ids: List[str], matrix_artists, artist_map: Dict[str, str]):
    """Check if the selected artists exist in the dataset.
    
    Args:
        artist_ids: List of artist MBIDs to check
        matrix_artists: Array of artist MBIDs in the dataset
        artist_map: Dictionary mapping artist MBIDs to names
        
    Returns:
        List of found artist MBIDs
    """
    found_artists = []
    not_found = []
    
    print("\nChecking if selected artists exist in the dataset...")
    
    for artist_id in artist_ids:
        try:
            # Try to find the artist index
            indices = [i for i, a in enumerate(matrix_artists) if a == artist_id]
            if indices:
                idx = indices[0]
                artist_name = artist_map.get(artist_id, "Unknown Artist")
                print(f"✓ Found: {artist_name}")
                found_artists.append(artist_id)
            else:
                raise ValueError(f"Artist ID not found: {artist_id}")
        except Exception:
            # Get artist name if possible
            artist_name = None
            response = requests.get(
                f"https://musicbrainz.org/ws/2/artist/{artist_id}?fmt=json",
                headers={"User-Agent": "ListenBrainzSearch/1.0"}
            )
            if response.status_code == 200:
                artist_data = response.json()
                artist_name = artist_data.get("name", None)
            
            if artist_name:
                print(f"✗ Not found: {artist_name}")
                not_found.append(artist_name)
            else:
                print(f"✗ Not found: {artist_id}")
                not_found.append(artist_id)
    
    if not found_artists:
        print("\nNone of the selected artists were found in the dataset.")
        if not_found:
            print("Try some of these popular artists instead:")
            try:
                # Get top 5 artists from the dataset
                artist_counts = {}
                for user_idx in range(matrix_artists.shape[0]):
                    for artist_idx, count in enumerate(matrix_artists[user_idx]):
                        if count > 0:
                            artist_id = matrix_artists[artist_idx]
                            artist_counts[artist_id] = artist_counts.get(artist_id, 0) + count
                
                top_artists = sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for artist_id, count in top_artists:
                    artist_name = artist_map.get(artist_id, "Unknown Artist")
                    print(f"- {artist_name} ({artist_id})")
            except Exception:
                # Fallback suggestions if we can't get top artists
                print("- The Beatles (b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d)")
                print("- The Clash (8f92558c-2baa-4758-8c38-615519e9deda)")
                print("- Taylor Swift (20244d07-534f-4eff-b4d4-930878889970)")
    else:
        print(f"\nFound {len(found_artists)} out of {len(artist_ids)} selected artists in the dataset.")
    
    return found_artists

if __name__ == "__main__":
    # Test the search functionality
    selected_artists = interactive_artist_search()
    if selected_artists:
        print("Selected artist MBIDs:")
        for mbid in selected_artists:
            print(mbid)
    else:
        print("No artists selected.")