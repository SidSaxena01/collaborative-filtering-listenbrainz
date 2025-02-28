#!/usr/bin/env python3
"""
ListenBrainz Collaborative Filtering Processor using Parquet and DuckDB
======================================================================

This implementation uses Parquet for efficient storage and DuckDB for
analytics processing, significantly improving memory usage and performance
over the original implementation.
"""
import argparse
import hashlib
import importlib
import json
import logging
import os
import pickle
import shutil
import subprocess

# import tempfile
import time

# from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb

# import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd
import polars as pl

# import pyarrow as pa
# import pyarrow.parquet as pq
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

# import the artist search module
import artist_search

# Import the report generator module
import listenbrainz_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("listenbrainz_processing.log")
    ]
)
logger = logging.getLogger("listenbrainz")

# Set of temporary files to clean up at exit
temp_files = set()
temp_dirs = set()

def register_temp_file(file_path: str) -> None:
    """Register a temporary file for cleanup."""
    temp_files.add(file_path)

def register_temp_dir(dir_path: str) -> None:
    """Register a temporary directory for cleanup."""
    temp_dirs.add(dir_path)

def cleanup_temp_files() -> None:
    """Clean up all registered temporary files and directories."""
    # Clean up files
    for file_path in temp_files:
        if os.path.exists(file_path):
            try:
                logger.info(f"Cleaning up temporary file: {file_path}")
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {str(e)}")
    
    # Clean up directories
    for dir_path in temp_dirs:
        if os.path.exists(dir_path):
            try:
                logger.info(f"Cleaning up temporary directory: {dir_path}")
                shutil.rmtree(dir_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary directory {dir_path}: {str(e)}")

def cleanup_on_exception(func):
    """Decorator to ensure cleanup occurs even if an exception is raised."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            logger.info("Cleaning up temporary files due to exception")
            cleanup_temp_files()
            raise
    return wrapper

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

def decompress_zst_file(input_file: str, output_file: str = None) -> str:
    """Decompress a zstandard file using either Python's zstandard library or system command.
    
    Args:
        input_file: Path to the zst file to decompress
        output_file: Path for the decompressed output (if None, removes the .zst extension)
        
    Returns:
        Path to the decompressed file
    """
    if output_file is None:
        output_file = input_file.rstrip('.zst')
    
    logger.info(f"Decompressing {input_file} to {output_file}")
    
    # Register the output file for cleanup
    register_temp_file(output_file)
    
    # Try to use Python's zstandard library first
    try:
        zstandard_spec = importlib.util.find_spec("zstandard")
        if zstandard_spec is not None:
            logger.info("Using Python's zstandard library for decompression")
            import zstandard as zstd
            
            with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
                dctx = zstd.ZstdDecompressor()
                dctx.copy_stream(f_in, f_out)
            
            logger.info(f"Successfully decompressed to {output_file}")
            return output_file
    except ImportError:
        logger.info("Python zstandard library not available, will try system command")
    except Exception as e:
        logger.warning(f"Error using Python's zstandard library: {str(e)}, will try system command")
    
    # Fall back to system command
    try:
        # Check if unzstd is available
        which_result = subprocess.run(['which', 'unzstd'], 
                                    capture_output=True, 
                                    text=True)
        
        if which_result.returncode != 0:
            # Try zstd -d as an alternative
            which_result = subprocess.run(['which', 'zstd'], 
                                        capture_output=True, 
                                        text=True)
            
            if which_result.returncode != 0:
                raise RuntimeError("Neither unzstd nor zstd command found. Please install zstandard.")
            
            # Use zstd -d with the full path
            result = subprocess.run(['zstd', '-d', input_file, '-o', output_file], 
                                  capture_output=True, 
                                  text=True)
        else:
            # Use unzstd with the full path
            if output_file == input_file.rstrip('.zst'):
                # Default behavior of unzstd is to output to the same path without .zst
                result = subprocess.run(['unzstd', input_file], 
                                      capture_output=True, 
                                      text=True)
            else:
                # Specify output file
                result = subprocess.run(['unzstd', '-o', output_file, input_file], 
                                      capture_output=True, 
                                      text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Decompression failed: {result.stderr}")
        
        logger.info(f"Successfully decompressed to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to decompress file: {str(e)}")
        raise

@time_function
def convert_listens_to_parquet(input_file: str, output_file: str, batch_size: int = 100000) -> Tuple[str, Set[str]]:
    """Convert a ListenBrainz jsonlines file to Parquet format, extracting only needed fields."""
    logger.info(f"Converting {input_file} to Parquet at {output_file}")
    
    # Check if output already exists
    if os.path.exists(output_file):
        logger.info(f"Parquet file {output_file} already exists, skipping conversion")
        # Try to get unique MSIDs from existing file
        try:
            df = pl.read_parquet(output_file)
            unique_msids = set(df["recording_msid"].unique().to_list())
            return output_file, unique_msids
        except Exception as e:
            logger.warning(f"Could not extract MSIDs from existing file: {str(e)}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process in batches to manage memory
    all_batches = []
    total_rows = 0
    unique_msids = set()
    
    try:
        with open(input_file, 'r') as f:
            batch = []
            for line_num, line in enumerate(f):
                if line_num % 50000 == 0 and line_num > 0:
                    logger.info(f"Processed {line_num} lines from {input_file}")
                
                try:
                    data = json.loads(line)
                    user_id = data.get('user_id')
                    
                    # Extract recording_msid from various possible locations
                    recording_msid = None
                    
                    # Try to get from top level
                    if 'recording_msid' in data:
                        recording_msid = data['recording_msid']
                    
                    # Try to get from track_metadata
                    if recording_msid is None and 'track_metadata' in data:
                        track_metadata = data['track_metadata']
                        if isinstance(track_metadata, dict):
                            if 'recording_msid' in track_metadata:
                                recording_msid = track_metadata['recording_msid']
                            elif 'additional_info' in track_metadata:
                                additional_info = track_metadata['additional_info']
                                if isinstance(additional_info, dict) and 'recording_msid' in additional_info:
                                    recording_msid = additional_info['recording_msid']
                    
                    # Only add valid entries
                    if user_id is not None and recording_msid is not None:
                        batch.append({
                            'user_id': user_id,
                            'recording_msid': recording_msid
                        })
                        unique_msids.add(recording_msid)
                    
                    # Process batch when it reaches the batch size
                    if len(batch) >= batch_size:
                        all_batches.append(pl.DataFrame(batch))
                        total_rows += len(batch)
                        batch = []
                
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {str(e)}, skipping")
                    continue
            
            # Process any remaining batch
            if batch:
                all_batches.append(pl.DataFrame(batch))
                total_rows += len(batch)
        
        # Combine all batches and write to Parquet
        if all_batches:
            combined_df = pl.concat(all_batches)
            combined_df.write_parquet(output_file)
            logger.info(f"Successfully converted {input_file} to Parquet with {total_rows} rows and {len(unique_msids)} unique MSIDs")
        else:
            logger.warning(f"No valid data extracted from {input_file}")
    
    except Exception as e:
        logger.error(f"Error converting {input_file} to Parquet: {str(e)}")
        if os.path.exists(output_file):
            os.remove(output_file)
        raise
    
    return output_file, unique_msids

@time_function
def convert_csv_to_parquet(input_file: str, output_file: str, selected_columns: Optional[List[str]] = None) -> str:
    """Convert a CSV file to Parquet format.
    
    Args:
        input_file: Path to the CSV file
        output_file: Path for the Parquet output
        selected_columns: Optional list of columns to keep (None for all)
        
    Returns:
        Path to the Parquet file
    """
    logger.info(f"Converting {input_file} to Parquet at {output_file}")
    
    # Check if output already exists
    if os.path.exists(output_file):
        logger.info(f"Parquet file {output_file} already exists, skipping conversion")
        return output_file
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Read the CSV into a Polars DataFrame
        df = pl.scan_csv(input_file)
        
        # Select only specific columns if requested
        if selected_columns:
            df = df.select(selected_columns)
        
        # Write to Parquet
        df.collect().write_parquet(output_file)
        
        logger.info(f"Successfully converted {input_file} to Parquet")
        return output_file
    
    except Exception as e:
        logger.error(f"Error converting {input_file} to Parquet: {str(e)}")
        # Remove incomplete output file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
        raise

@time_function
def convert_all_listens(data_root: str, parquet_root: str, months: List[int]) -> Dict[int, Tuple[str, Set[str]]]:
    """Convert all the specified ListenBrainz listen files to Parquet.
    
    Args:
        data_root: Path to the original data files
        parquet_root: Path to store the Parquet files
        months: List of months to process
        
    Returns:
        Dictionary mapping month to (parquet_file_path, set of unique MSIDs)
    """
    results = {}
    all_unique_msids = set()
    
    for month in months:
        listen_file = os.path.join(data_root, f"{month}.listens.zst")
        parquet_file = os.path.join(parquet_root, f"listens_{month}.parquet")
        
        # Decompress if needed
        if not os.path.exists(listen_file.replace('.zst', '')):
            decompressed_file = decompress_zst_file(listen_file)
        else:
            decompressed_file = listen_file.replace('.zst', '')
        
        # Convert to Parquet
        parquet_file, unique_msids = convert_listens_to_parquet(decompressed_file, parquet_file)
        
        # Add to results
        results[month] = (parquet_file, unique_msids)
        all_unique_msids.update(unique_msids)
        
        # Clean up decompressed file
        if decompressed_file != listen_file.replace('.zst', ''):
            try:
                os.remove(decompressed_file)
            except Exception as e:
                logger.warning(f"Error removing temporary file {decompressed_file}: {str(e)}")
    
    logger.info(f"Converted {len(months)} listen files to Parquet with {len(all_unique_msids)} total unique MSIDs")
    return results, all_unique_msids

@time_function
def convert_mapping_file(data_root: str, parquet_root: str, unique_msids: Set[str], quality_levels: List[str]) -> str:
    """Convert the MSID mapping file to Parquet, filtering to only include relevant MSIDs and match qualities.
    
    Args:
        data_root: Path to the original data files
        parquet_root: Path to store the Parquet files
        unique_msids: Set of MSIDs to include
        quality_levels: List of match quality levels to include
        
    Returns:
        Path to the Parquet file
    """
    mapping_file = os.path.join(data_root, "listenbrainz_msid_mapping.csv.zst")
    parquet_file = os.path.join(parquet_root, "msid_mapping.parquet")
    
    # Check if output already exists
    if os.path.exists(parquet_file):
        logger.info(f"Mapping Parquet file {parquet_file} already exists, skipping conversion")
        return parquet_file
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(parquet_file), exist_ok=True)
    
    # Decompress if needed
    if not os.path.exists(mapping_file.replace('.zst', '')):
        decompressed_file = decompress_zst_file(mapping_file)
    else:
        decompressed_file = mapping_file.replace('.zst', '')
    
    # Convert to Parquet using DuckDB for efficient filtering
    logger.info(f"Converting and filtering mapping file to Parquet")
    
    try:
        # Create a temporary file for the unique MSIDs
        msids_file = os.path.join(parquet_root, "temp_msids.csv")
        with open(msids_file, 'w') as f:
            f.write("msid\n")  # Header
            for msid in unique_msids:
                f.write(f"{msid}\n")
        
        register_temp_file(msids_file)
        
        # Create a temporary file for the quality levels
        quality_file = os.path.join(parquet_root, "temp_quality.csv")
        with open(quality_file, 'w') as f:
            f.write("quality\n")  # Header
            for quality in quality_levels:
                f.write(f"{quality}\n")
        
        register_temp_file(quality_file)
        
        # Use DuckDB to efficiently filter and convert
        con = duckdb.connect(":memory:")
        
        # Register the lists as tables
        con.execute(f"CREATE TABLE msids AS SELECT * FROM read_csv_auto('{msids_file}');")
        con.execute(f"CREATE TABLE qualities AS SELECT * FROM read_csv_auto('{quality_file}');")
        
        # Filter and write to Parquet
        query = f"""
        COPY (
            SELECT m.recording_msid, m.recording_mbid, m.match_type
            FROM read_csv_auto('{decompressed_file}') m
            JOIN msids ON m.recording_msid = msids.msid
            JOIN qualities ON m.match_type = qualities.quality
        ) TO '{parquet_file}' (FORMAT 'PARQUET');
        """
        
        con.execute(query)
        
        # Clean up
        if decompressed_file != mapping_file.replace('.zst', ''):
            try:
                os.remove(decompressed_file)
            except Exception as e:
                logger.warning(f"Error removing temporary file {decompressed_file}: {str(e)}")
        
        # Clean up temporary files
        os.remove(msids_file)
        os.remove(quality_file)
        
        logger.info(f"Successfully converted and filtered mapping file to Parquet")
        return parquet_file
    
    except Exception as e:
        logger.error(f"Error converting mapping file to Parquet: {str(e)}")
        # Remove incomplete output file if it exists
        if os.path.exists(parquet_file):
            os.remove(parquet_file)
        raise

@time_function
def convert_redirect_file(data_root: str, parquet_root: str) -> str:
    """Convert the canonical redirect file to Parquet.
    
    Args:
        data_root: Path to the original data files
        parquet_root: Path to store the Parquet files
        
    Returns:
        Path to the Parquet file
    """
    redirect_file = os.path.join(data_root, "canonical_recording_redirect.csv.zst")
    parquet_file = os.path.join(parquet_root, "canonical_redirects.parquet")
    
    # Check if output already exists
    if os.path.exists(parquet_file):
        logger.info(f"Redirect Parquet file {parquet_file} already exists, skipping conversion")
        return parquet_file
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(parquet_file), exist_ok=True)
    
    # Decompress if needed
    if not os.path.exists(redirect_file.replace('.zst', '')):
        decompressed_file = decompress_zst_file(redirect_file)
    else:
        decompressed_file = redirect_file.replace('.zst', '')
    
    # Convert to Parquet
    logger.info(f"Converting redirect file to Parquet")
    
    try:
        # Use DuckDB for efficient conversion
        con = duckdb.connect(":memory:")
        
        # Convert to Parquet
        query = f"""
        COPY (
            SELECT recording_mbid, canonical_recording_mbid
            FROM read_csv_auto('{decompressed_file}')
        ) TO '{parquet_file}' (FORMAT 'PARQUET');
        """
        
        con.execute(query)
        
        # Clean up
        if decompressed_file != redirect_file.replace('.zst', ''):
            try:
                os.remove(decompressed_file)
            except Exception as e:
                logger.warning(f"Error removing temporary file {decompressed_file}: {str(e)}")
        
        logger.info(f"Successfully converted redirect file to Parquet")
        return parquet_file
    
    except Exception as e:
        logger.error(f"Error converting redirect file to Parquet: {str(e)}")
        # Remove incomplete output file if it exists
        if os.path.exists(parquet_file):
            os.remove(parquet_file)
        raise

@time_function
def convert_canonical_data_file(data_root: str, parquet_root: str) -> str:
    """Convert the canonical MusicBrainz data file to Parquet.
    
    Args:
        data_root: Path to the original data files
        parquet_root: Path to store the Parquet files
        
    Returns:
        Path to the Parquet file
    """
    canonical_file = os.path.join(data_root, "canonical_musicbrainz_data.csv.zst")
    parquet_file = os.path.join(parquet_root, "canonical_data.parquet")
    
    # Check if output already exists
    if os.path.exists(parquet_file):
        logger.info(f"Canonical data Parquet file {parquet_file} already exists, skipping conversion")
        return parquet_file
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(parquet_file), exist_ok=True)
    
    # Decompress if needed
    if not os.path.exists(canonical_file.replace('.zst', '')):
        decompressed_file = decompress_zst_file(canonical_file)
    else:
        decompressed_file = canonical_file.replace('.zst', '')
    
    # Convert to Parquet
    logger.info(f"Converting canonical data file to Parquet")
    
    try:
        # Use DuckDB for efficient conversion
        con = duckdb.connect(":memory:")
        
        # Convert to Parquet
        query = f"""
        COPY (
            SELECT recording_mbid, artist_mbids, artist_credit_name
            FROM read_csv_auto('{decompressed_file}')
        ) TO '{parquet_file}' (FORMAT 'PARQUET');
        """
        
        con.execute(query)
        
        # Clean up
        if decompressed_file != canonical_file.replace('.zst', ''):
            try:
                os.remove(decompressed_file)
            except Exception as e:
                logger.warning(f"Error removing temporary file {decompressed_file}: {str(e)}")
        
        logger.info(f"Successfully converted canonical data file to Parquet")
        return parquet_file
    
    except Exception as e:
        logger.error(f"Error converting canonical data file to Parquet: {str(e)}")
        # Remove incomplete output file if it exists
        if os.path.exists(parquet_file):
            os.remove(parquet_file)
        raise

@time_function
def convert_artist_file(data_root: str, parquet_root: str) -> str:
    """Convert the MusicBrainz artist file to Parquet.
    
    Args:
        data_root: Path to the original data files
        parquet_root: Path to store the Parquet files
        
    Returns:
        Path to the Parquet file
    """
    artist_file = os.path.join(data_root, "musicbrainz_artist.csv")
    parquet_file = os.path.join(parquet_root, "artists.parquet")
    
    # Check if output already exists
    if os.path.exists(parquet_file):
        logger.info(f"Artist Parquet file {parquet_file} already exists, skipping conversion")
        return parquet_file
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(parquet_file), exist_ok=True)
    
    # Convert to Parquet
    logger.info(f"Converting artist file to Parquet")
    
    try:
        # Use DuckDB for efficient conversion
        con = duckdb.connect(":memory:")
        
        # Convert to Parquet
        query = f"""
        COPY (
            SELECT artist_mbid, name
            FROM read_csv_auto('{artist_file}')
        ) TO '{parquet_file}' (FORMAT 'PARQUET');
        """
        
        con.execute(query)
        
        logger.info(f"Successfully converted artist file to Parquet")
        return parquet_file
    
    except Exception as e:
        logger.error(f"Error converting artist file to Parquet: {str(e)}")
        # Remove incomplete output file if it exists
        if os.path.exists(parquet_file):
            os.remove(parquet_file)
        raise

@time_function
def save_mapping_stats(mapped_count: int, unmapped_count: int, output_dir: str) -> None:
    """Save the MSID to MBID mapping statistics for reporting purposes.
    
    Args:
        mapped_count: Number of MSIDs successfully mapped to MBIDs
        unmapped_count: Number of MSIDs that couldn't be mapped
        output_dir: Directory to save the statistics
    """
    logger.info("Saving mapping statistics")
    
    total = mapped_count + unmapped_count
    
    if total == 0:
        logger.warning("No mapping data to save (total is 0)")
        return
    
    mapped_pct = (mapped_count / total) * 100
    unmapped_pct = (unmapped_count / total) * 100
    
    stats = {
        "mapped": mapped_pct,
        "unmapped": unmapped_pct,
        "total": total,
        "mapped_count": mapped_count,
        "unmapped_count": unmapped_count,
        "quality_score": mapped_pct
    }
    
    # Ensure directory exists
    os.makedirs(os.path.join(output_dir, "report/data"), exist_ok=True)
    
    # Save to data quality file for use in reporting
    with open(os.path.join(output_dir, "report/data/data_quality.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Mapping statistics saved: {mapped_pct:.1f}% mapped ({mapped_count:,} out of {total:,})")

@time_function
def generate_user_artist_counts(parquet_root: str, output_file: str) -> str:
    """Generate the user-artist-count matrix using DuckDB for efficient processing.
    
    Args:
        parquet_root: Path to the Parquet files
        output_file: Path to the output CSV file
        
    Returns:
        Path to the output file
    """
    logger.info(f"Generating user-artist counts")
    
    # Check if output already exists
    if os.path.exists(output_file):
        logger.info(f"User-artist counts file {output_file} already exists, skipping generation")
        return output_file
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create a temporary DuckDB database file instead of in-memory for larger datasets
    db_file = os.path.join(parquet_root, "processing.duckdb")
    if os.path.exists(db_file):
        os.remove(db_file)
    
    register_temp_file(db_file)
    
    con = duckdb.connect(db_file)
    
    try:
        # Get listen files (listens_1.parquet, listens_2.parquet, etc.)
        listen_files = [os.path.join(parquet_root, f) for f in os.listdir(parquet_root) 
                      if f.startswith("listens_") and f.endswith(".parquet")]
        
        if not listen_files:
            raise ValueError("No listen files found in parquet_root")
        
        # Create a view for all listen files
        con.execute("""
        CREATE OR REPLACE VIEW all_listens AS
        SELECT user_id, recording_msid
        FROM (
        """ + "\nUNION ALL\n".join([
            f"SELECT user_id, recording_msid FROM parquet_scan('{f}')" for f in listen_files
        ]) + ");")
        
        # Create indices for faster joins
        con.execute("CREATE TABLE listens AS SELECT * FROM all_listens;")
        
        # Path to other Parquet files
        mapping_file = os.path.join(parquet_root, "msid_mapping.parquet")
        redirects_file = os.path.join(parquet_root, "canonical_redirects.parquet")
        canonical_file = os.path.join(parquet_root, "canonical_data.parquet")
        
        # Add diagnostic information - track data flow through each step
        rows_in_listens = con.execute("SELECT COUNT(*) FROM listens").fetchone()[0]
        logger.info(f"Initial listens count: {rows_in_listens}")
        
        # Check raw MSIDs to MBID mapping coverage
        con.execute(f"""
            CREATE TABLE msid_mbid_mapping AS 
            SELECT l.user_id, l.recording_msid, m.recording_mbid
            FROM listens l
            LEFT JOIN parquet_scan('{mapping_file}') m ON l.recording_msid = m.recording_msid
        """)
        
        mapped_rows = con.execute("SELECT COUNT(*) FROM msid_mbid_mapping WHERE recording_mbid IS NOT NULL").fetchone()[0]
        unmapped_rows = con.execute("SELECT COUNT(*) FROM msid_mbid_mapping WHERE recording_mbid IS NULL").fetchone()[0]
        logger.info(f"MSID to MBID mapping: {mapped_rows} mapped ({mapped_rows/rows_in_listens*100:.1f}%), {unmapped_rows} unmapped")
        
        # Save mapping stats to a file for reporting
        output_dir = os.path.dirname(os.path.dirname(output_file))
        save_mapping_stats(mapped_rows, unmapped_rows, output_dir)
        
        # Generate user-artist-counts through a series of transformations
        logger.info("Running SQL query to generate user-artist counts")
        
        # This query does the entire pipeline in one SQL statement with improved tracking:
        # 1. Join listens with MSID mapping
        # 2. Apply canonical redirects
        # 3. Look up artists for recordings
        # 4. Group by user and artist to get listen counts
        sql = f"""
        COPY (
            WITH listens_with_mbid AS (
                -- Join listens with MSID mapping, filtering out missing MBIDs
                SELECT l.user_id, m.recording_mbid
                FROM listens l
                JOIN parquet_scan('{mapping_file}') m ON l.recording_msid = m.recording_msid
                WHERE m.recording_mbid IS NOT NULL
            ),
            canonical_mbids AS (
                -- Apply canonical redirects, preserving originals when no canonical exists
                SELECT 
                    l.user_id,
                    COALESCE(r.canonical_recording_mbid, l.recording_mbid) AS canonical_mbid
                FROM listens_with_mbid l
                LEFT JOIN parquet_scan('{redirects_file}') r ON l.recording_mbid = r.recording_mbid
            ),
            recording_artists AS (
                -- Get artists for each recording - use LEFT JOIN to keep all listens
                -- even those without canonical data
                SELECT 
                    c.user_id,
                    CASE 
                        WHEN m.artist_mbids IS NULL OR m.artist_mbids = '' THEN array[c.canonical_mbid]
                        ELSE string_split(m.artist_mbids, ';')
                    END as artist_ids
                FROM canonical_mbids c
                LEFT JOIN parquet_scan('{canonical_file}') m ON c.canonical_mbid = m.recording_mbid
            ),
            unnested_artists AS (
                -- Unnest the artist arrays
                SELECT
                    user_id,
                    unnest(artist_ids) as artist_id
                FROM recording_artists
                WHERE array_length(artist_ids) > 0
            )
            -- Generate the final counts
            SELECT 
                user_id, 
                artist_id, 
                COUNT(*) AS listen_count
            FROM unnested_artists
            WHERE artist_id IS NOT NULL
            GROUP BY user_id, artist_id
            ORDER BY user_id, artist_id
        ) TO '{output_file}' (FORMAT 'CSV', HEADER);
        """
        
        con.execute(sql)
        
        # Get stats about the generated file
        row_count = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{output_file}')").fetchone()[0]
        user_count = con.execute(f"SELECT COUNT(DISTINCT user_id) FROM read_csv_auto('{output_file}')").fetchone()[0]
        artist_count = con.execute(f"SELECT COUNT(DISTINCT artist_id) FROM read_csv_auto('{output_file}')").fetchone()[0]
        
        logger.info(f"Generated user-artist counts with {row_count} entries, {user_count} users, and {artist_count} artists")
        
        # Get distribution of listeners per artist for better understanding
        listener_distribution = con.execute(f"""
            WITH listener_counts AS (
                SELECT artist_id, COUNT(DISTINCT user_id) as listener_count
                FROM read_csv_auto('{output_file}')
                GROUP BY artist_id
            )
            SELECT listener_count, COUNT(*) as artist_count
            FROM listener_counts
            GROUP BY listener_count
            ORDER BY listener_count
        """).fetchall()
        
        # Calculate cumulative statistics
        total_artists = sum(row[1] for row in listener_distribution)
        single_listener_artists = listener_distribution[0][1] if listener_distribution and listener_distribution[0][0] == 1 else 0
        percent_single = (single_listener_artists / total_artists) * 100 if total_artists > 0 else 0
        
        logger.info(f"Artists with only one listener: {single_listener_artists:,} ({percent_single:.1f}% of all artists)")
        logger.info(f"Artist listener distribution: {', '.join([f'{count} artists with {listeners} listener(s)' for listeners, count in listener_distribution[:5]])}, ...")
        
        # Close and cleanup database
        con.close()
        os.remove(db_file)
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error generating user-artist counts: {str(e)}")
        # Close connection
        con.close()
        # Remove incomplete output file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
        raise

@time_function
def load_data_matrix(filename: str) -> Tuple[np.ndarray, np.ndarray, Any]:
    """Load the user-artist-count matrix using DuckDB for efficiency.
    
    Args:
        filename: Path to the user-artist-count CSV file
        
    Returns:
        Tuple of (artist_ids, user_ids, plays matrix)
    """
    logger.info(f"Loading data matrix from {filename}")
    
    # Use DuckDB to efficiently load and process the data
    con = duckdb.connect(":memory:")
    
    # Load the data
    con.execute(f"CREATE TABLE counts AS SELECT * FROM read_csv_auto('{filename}')")
    
    # Get unique artists and users
    df_artists = con.execute("SELECT DISTINCT artist_id FROM counts ORDER BY artist_id").pl()
    df_users = con.execute("SELECT DISTINCT user_id FROM counts ORDER BY user_id").pl()
    
    matrix_artists = df_artists['artist_id'].to_numpy()
    matrix_users = df_users['user_id'].to_numpy()
    
    # Create mappings from IDs to indices
    artist_to_idx = {artist: i for i, artist in enumerate(matrix_artists)}
    user_to_idx = {user: i for i, user in enumerate(matrix_users)}
    
    # Get the data as (user_idx, artist_idx, count) for efficient sparse matrix creation
    query = """
    SELECT u.user_idx, a.artist_idx, c.listen_count
    FROM counts c
    JOIN (SELECT artist_id, row_number() OVER (ORDER BY artist_id) - 1 AS artist_idx FROM (SELECT DISTINCT artist_id FROM counts) t) a 
        ON c.artist_id = a.artist_id
    JOIN (SELECT user_id, row_number() OVER (ORDER BY user_id) - 1 AS user_idx FROM (SELECT DISTINCT user_id FROM counts) t) u 
        ON c.user_id = u.user_id
    """
    
    df_counts = con.execute(query).pl()
    
    # Create the sparse matrix
    from scipy.sparse import csr_matrix
    
    plays = csr_matrix((df_counts['listen_count'],
                       (df_counts['user_idx'], df_counts['artist_idx'])),
                      shape=(len(matrix_users), len(matrix_artists)))
    
    logger.info(f"Created matrix of shape {plays.shape} with {plays.nnz} non-zero entries")
    logger.info(f"Matrix has {len(matrix_users)} users and {len(matrix_artists)} artists")
    
    return matrix_artists, matrix_users, plays

@time_function
def build_model(plays, factors=64, regularization=0.05, alpha=2.0, use_native=True):
    """Build the collaborative filtering model.
    
    Args:
        plays: User-artist play count matrix
        factors: Number of latent factors
        regularization: Regularization factor
        alpha: Weighting factor
        use_native: Whether to use native optimizations
        
    Returns:
        Trained ALS model
    """
    logger.info(f"Building ALS model with factors={factors}, regularization={regularization}, alpha={alpha}")
    
    # Initialize the model
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        calculate_training_loss=True,
        use_native=use_native,
        num_threads=0  # Use all available threads
    )
    
    # Weight the matrix using BM25 weighting
    logger.info("Applying BM25 weighting with K1=100, B=0.8")
    artist_user_plays = bm25_weight(plays, K1=100, B=0.8)
    
    # Fit the model
    logger.info("Fitting model (this may take a while)...")
    model.fit(artist_user_plays.tocsr())
    
    logger.info("Model training complete")
    return model

@time_function
def get_artist_map(artist_file: str) -> Dict[str, str]:
    """Load the artist name mapping.
    
    Args:
        artist_file: Path to the artist Parquet file
        
    Returns:
        Dictionary mapping artist MBIDs to names
    """
    logger.info(f"Loading artist mapping from {artist_file}")
    
    # Use Polars to efficiently load the mapping
    df = pl.read_parquet(artist_file)
    
    # Convert to dictionary
    artist_map = dict(zip(df['artist_mbid'].to_list(), df['name'].to_list()))
    
    logger.info(f"Loaded {len(artist_map)} artist mappings")
    return artist_map

def artist_index(artists: np.ndarray, artist_id: str) -> int:
    """Find the index of an artist in the artists array.
    
    Args:
        artists: Array of artist IDs
        artist_id: Artist ID to find
        
    Returns:
        Index of the artist in the array
        
    Raises:
        ValueError: If the artist ID is not found in the array
    """
    indices = np.where(artists == artist_id)[0]
    if len(indices) == 0:
        raise ValueError(f"Artist ID not found: {artist_id}")
    return indices[0]

@cleanup_on_exception
@time_function
def process_all_files(data_root: str, working_root: str, scratch_root: str, months: List[int], quality_levels: List[str], force_reprocess: bool = False) -> str:
    """Process all files to generate the user-artist-count matrix.
    
    Args:
        data_root: Path to the original data files
        working_root: Path to store final results
        scratch_root: Path for temporary files
        months: List of months to process
        quality_levels: List of match quality levels to include
        force_reprocess: Whether to force reprocessing of files
        
    Returns:
        Path to the generated user-artist-count matrix
    """
    # Create directories
    os.makedirs(working_root, exist_ok=True)
    os.makedirs(scratch_root, exist_ok=True)
    
    # Set up paths for Parquet files
    parquet_root = os.path.join(scratch_root, "parquet")
    os.makedirs(parquet_root, exist_ok=True)
    
    # Output file for user-artist counts
    output_file = os.path.join(working_root, "userid-artist-counts.csv")
    
    # If output file exists and we're not forcing reprocessing, return it
    if os.path.exists(output_file) and not force_reprocess:
        logger.info(f"Output file {output_file} already exists, skipping processing")
        return output_file
    
    # Step 1: Convert all listen files to Parquet
    logger.info(f"Step 1: Converting listen files to Parquet")
    listen_files, all_unique_msids = convert_all_listens(data_root, parquet_root, months)
    
    # Step 2: Convert and filter MSID mapping
    logger.info(f"Step 2: Converting and filtering MSID mapping")
    mapping_file = convert_mapping_file(data_root, parquet_root, all_unique_msids, quality_levels)
    
    # Step 3: Convert canonical redirect mapping
    logger.info(f"Step 3: Converting canonical redirect mapping")
    redirect_file = convert_redirect_file(data_root, parquet_root)
    
    # Step 4: Convert canonical metadata
    logger.info(f"Step 4: Converting canonical metadata")
    canonical_file = convert_canonical_data_file(data_root, parquet_root)
    
    # Step 5: Convert artist mapping
    logger.info(f"Step 5: Converting artist mapping")
    artist_file = convert_artist_file(data_root, parquet_root)
    
    # Step 6: Generate user-artist counts
    logger.info(f"Step 6: Generating user-artist counts")
    user_artist_file = generate_user_artist_counts(parquet_root, output_file)
    
    # New Step 7: Analyze the listener distribution in detail
    logger.info(f"Step 7: Analyzing artist listener distribution")
    analyze_artist_listener_distribution(user_artist_file)
    
    logger.info(f"Processing complete, user-artist counts saved to {user_artist_file}")
    return user_artist_file

@time_function
def save_artist_recommendations(model, matrix_artists, artist_map, artist_ids, working_dir):
    """Save artist recommendations to CSV files."""
    logger.info(f"Saving artist recommendations for {len(artist_ids)} artists")
    
    # Create recommendations directory
    recommendations_dir = os.path.join(working_dir, "recommendations")
    os.makedirs(recommendations_dir, exist_ok=True)
    
    saved_count = 0
    saved_artist_names = []
    
    for artist_id in artist_ids:
        try:
            # Find artist index
            idx = artist_index(matrix_artists, artist_id)
            artist_name = artist_map.get(artist_id, "Unknown Artist")
            
            # Create a consistent, safe filename
            # THIS IS CRITICAL: file naming must match what report generator expects
            safe_name = artist_name.replace(' ', '_').replace('&', 'and').replace("'", "")
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
            
            # Get similar artists
            similar_ids, scores = model.similar_items(idx, N=20)
            
            # Create recommendations data
            recommendations = []
            for i, (similar_id, score) in enumerate(zip(similar_ids, scores)):
                if similar_id < len(matrix_artists):  # Ensure index is valid
                    similar_artist_id = matrix_artists[similar_id]
                    similar_artist_name = artist_map.get(similar_artist_id, "Unknown Artist")
                    
                    recommendations.append({
                        "rank": i+1,
                        "artist_id": similar_artist_id,
                        "artist_name": similar_artist_name,
                        "similarity_score": float(score)
                    })
            
            # Save to CSV
            import pandas as pd
            output_file = os.path.join(recommendations_dir, f"similar_to_{safe_name}.csv")
            pd.DataFrame(recommendations).to_csv(output_file, index=False)
            
            logger.info(f"Saved recommendations for {artist_name} to {output_file}")
            saved_count += 1
            saved_artist_names.append(artist_name)
            
        except ValueError as e:
            logger.warning(f"Could not find artist {artist_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Error saving recommendations for {artist_id}: {str(e)}")
    
    logger.info(f"Saved recommendations for {saved_count} artists: {', '.join(saved_artist_names)}")
    return saved_count

@time_function
def save_model_cache(model, matrix_artists, matrix_users, plays, working_dir: str, data_hash: str = None) -> str:
    """Save the trained model and associated data to disk for faster reuse."""
    cache_dir = os.path.join(working_dir, "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a versioned filename
    if data_hash is None:
        factors = model.factors
        matrix_shape = f"{len(matrix_users)}x{len(matrix_artists)}"
        data_hash = hashlib.md5(f"factors{factors}-shape{matrix_shape}".encode()).hexdigest()[:10]
    
    cache_file = os.path.join(cache_dir, f"model-{data_hash}.pkl")
    
    # Save model and data (now including plays)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'matrix_artists': matrix_artists,
            'matrix_users': matrix_users,
            'plays': plays,  # Add plays matrix to cache
            'timestamp': time.time(),
            'data_hash': data_hash
        }, f)
    
    logger.info(f"Model saved to cache: {cache_file}")
    return cache_file

@time_function
def load_model_cache(working_dir: str, data_hash: str = None) -> Optional[Tuple]:
    """Try to load a cached model if available.
    
    Args:
        working_dir: Directory where the cache is stored
        data_hash: Optional hash to find a specific cached model
        
    Returns:
        Tuple of (model, matrix_artists, matrix_users) if cache exists, None otherwise
    """
    cache_dir = os.path.join(working_dir, "model_cache")
    
    if not os.path.exists(cache_dir):
        logger.info("No model cache directory found")
        return None
    
    # Find the latest cached model or a specific one by hash
    if data_hash:
        cache_file = os.path.join(cache_dir, f"model-{data_hash}.pkl")
        if not os.path.exists(cache_file):
            logger.info(f"No cached model found with hash: {data_hash}")
            return None
        cache_files = [cache_file]
    else:
        cache_files = sorted(
            [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.startswith("model-") and f.endswith(".pkl")],
            key=os.path.getmtime,
            reverse=True
        )
    
    if not cache_files:
        logger.info("No cached models found")
        return None
    
    # Try to load the cached model
    try:
        with open(cache_files[0], 'rb') as f:
            cache_data = pickle.load(f)
        
        model = cache_data['model']
        matrix_artists = cache_data['matrix_artists']
        matrix_users = cache_data['matrix_users']
        plays = cache_data.get('plays')  # Get plays from cache
        timestamp = cache_data.get('timestamp', 0)
        
        age_hours = (time.time() - timestamp) / 3600 if timestamp else 0
        
        logger.info(f"Loaded cached model from {cache_files[0]} (age: {age_hours:.1f} hours)")
        return model, matrix_artists, matrix_users, plays  # Return plays as well
    
    except Exception as e:
        logger.warning(f"Error loading cached model: {str(e)}")
        return None

@time_function
def analyze_artist_listener_distribution(user_artist_file: str) -> None:
    """Analyze the distribution of listeners per artist to better understand the data.
    
    Args:
        user_artist_file: Path to the user-artist-count CSV file
    """
    logger.info("Analyzing artist listener distribution")
    
    try:
        # Use DuckDB for efficient analysis
        con = duckdb.connect(":memory:")
        
        # Load the data
        con.execute(f"CREATE TABLE counts AS SELECT * FROM read_csv_auto('{user_artist_file}')")
        
        # Get total artists and users
        total_artists = con.execute("SELECT COUNT(DISTINCT artist_id) FROM counts").fetchone()[0]
        total_users = con.execute("SELECT COUNT(DISTINCT user_id) FROM counts").fetchone()[0]
        
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
        
        # Calculate percentiles
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
        
        # Log detailed statistics
        single_listener = next((count for listeners, count in listener_distribution if listeners == 1), 0)
        percent_single = (single_listener / total_artists) * 100 if total_artists > 0 else 0
        
        logger.info(f"Artist Listener Distribution Analysis:")
        logger.info(f"Total artists: {total_artists:,}")
        logger.info(f"Total users: {total_users:,}")
        logger.info(f"Artists with only one listener: {single_listener:,} ({percent_single:.1f}%)")
        logger.info(f"Listener percentiles - Median: {percentiles[0]}, 90th: {percentiles[1]}, 95th: {percentiles[2]}, 99th: {percentiles[3]}, Max: {percentiles[4]}")
        
        # Log the distribution in more detail
        buckets = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, float('inf'))]
        for low, high in buckets:
            if high == float('inf'):
                artist_count = con.execute(f"""
                    WITH listener_counts AS (
                        SELECT artist_id, COUNT(DISTINCT user_id) as listener_count
                        FROM counts
                        GROUP BY artist_id
                    )
                    SELECT COUNT(*) 
                    FROM listener_counts
                    WHERE listener_count >= {low}
                """).fetchone()[0]
                percent = (artist_count / total_artists) * 100
                logger.info(f"Artists with {low}+ listeners: {artist_count:,} ({percent:.1f}%)")
            else:
                artist_count = con.execute(f"""
                    WITH listener_counts AS (
                        SELECT artist_id, COUNT(DISTINCT user_id) as listener_count
                        FROM counts
                        GROUP BY artist_id
                    )
                    SELECT COUNT(*) 
                    FROM listener_counts
                    WHERE listener_count BETWEEN {low} AND {high}
                """).fetchone()[0]
                percent = (artist_count / total_artists) * 100
                logger.info(f"Artists with {low}-{high} listeners: {artist_count:,} ({percent:.1f}%)")
                
        # Calculate mean and std dev listeners per artist
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
        
        logger.info(f"Mean listeners per artist: {stats[0]:.2f}")
        logger.info(f"StdDev listeners per artist: {stats[1]:.2f}")
        
        return listener_distribution
    
    except Exception as e:
        logger.error(f"Error analyzing artist listener distribution: {e}")
        return None

def generate_data_hash(user_artist_file: str) -> str:
    """Generate a hash of the input data file.
    
    This can be used to invalidate the cache when the data changes.
    
    Args:
        user_artist_file: Path to the user-artist-count CSV file
        
    Returns:
        MD5 hash string
    """
    # Use file modification time and size for a quick hash
    stat = os.stat(user_artist_file)
    data_str = f"{user_artist_file}-{stat.st_mtime}-{stat.st_size}"
    return hashlib.md5(data_str.encode()).hexdigest()[:10]

def main():
    """Main function to run the ListenBrainz collaborative filtering pipeline."""
    parser = argparse.ArgumentParser(description="ListenBrainz Collaborative Filtering with Parquet and DuckDB")
    parser.add_argument("--data-root", required=True, help="Path to the original data files")
    parser.add_argument("--working-root", required=True, help="Path to store intermediate and final results")
    parser.add_argument("--scratch-root", required=True, help="Path for temporary files")
    parser.add_argument("--months", type=str, default="1-12", help="Range of months to process (e.g., '1-3' or '1,3,5')")
    parser.add_argument("--quality-levels", type=str, default="exact_match,high_quality", 
                        help="Comma-separated list of match quality levels to include")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing of all files")
    parser.add_argument("--analyze-artists", nargs="*", help="List of artist MBIDs to analyze")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--search-artists", action="store_true", 
                        help="Interactively search for artists to analyze")
    parser.add_argument("--force-retrain", action="store_true", 
                   help="Force retraining the model even if a cached version exists")
    parser.add_argument("--no-open-report", action="store_true",
                   help="Don't automatically open the HTML report when finished")
    parser.add_argument("--use-last-search", action="store_true",
                   help="Reuse the list of artists from the previous search")
    
    args = parser.parse_args()
    
    # Configure logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Determine which months to process
    months_to_process = []
    if '-' in args.months:
        start, end = map(int, args.months.split('-'))
        months_to_process = list(range(start, end + 1))
    else:
        months_to_process = list(map(int, args.months.split(',')))
    
    # Parse quality levels
    quality_levels = args.quality_levels.split(',')
    
    # Process files
    start_time = time.time()
    
    user_artist_file = process_all_files(
        args.data_root,
        args.working_root,
        args.scratch_root,
        months=months_to_process,
        quality_levels=quality_levels,
        force_reprocess=args.force_reprocess
    )
    
    # Generate a hash of the input data
    data_hash = generate_data_hash(user_artist_file)

    # Try to load cached model first
    cached_data = load_model_cache(args.working_root, data_hash)

    if cached_data and not args.force_retrain:
        # Use cached model
        model, matrix_artists, matrix_users, plays = cached_data
        logger.info("Using cached model (skipping model building step)")
    else:
        # Build the model from scratch
        logger.info("Building collaborative filtering model...")
        matrix_artists, matrix_users, plays = load_data_matrix(user_artist_file)
        model = build_model(plays)
        
        # Cache the model for future use
        save_model_cache(model, matrix_artists, matrix_users, plays, args.working_root, data_hash) 
               
    # Load artist mapping
    artist_file = os.path.join(args.scratch_root, "parquet", "artists.parquet")
    artist_map = get_artist_map(artist_file)
    
    # ARTIST SELECTION AND RECOMMENDATION GENERATION
    # =============================================

    # Step 1: Determine which artists to analyze based on command-line arguments or search
    artists_to_analyze = []
    
    # Check for last search first if that option is enabled
    if args.use_last_search:
        logger.info("Attempting to load artists from previous search")
        last_artists = listenbrainz_report.load_last_search(args.working_root)
        if last_artists:
            # Verify these artists exist in our current dataset
            for artist_id in last_artists:
                try:
                    idx = artist_index(matrix_artists, artist_id)
                    artists_to_analyze.append(artist_id)
                    artist_name = artist_map.get(artist_id, "Unknown Artist")
                    logger.info(f"Reusing artist from previous search: {artist_name} ({artist_id})")
                except ValueError:
                    logger.warning(f"Artist from previous search not found in current dataset: {artist_id}")
            
            # If we found valid artists from last search, don't proceed with other selection methods
            if artists_to_analyze:
                logger.info(f"Successfully loaded {len(artists_to_analyze)} artists from previous search")
                
                # Skip the rest of the artist selection logic
                # This is the key change - we're skipping all other selection methods
                # when we have successfully loaded artists from the last search
                
    # Only proceed with other selection methods if we don't have artists yet
    if not artists_to_analyze:
        # Option 1: Interactive search mode
        if args.search_artists:
            logger.info("Starting interactive artist search...")
            selected_artists = artist_search.interactive_artist_search()
            
            if selected_artists:
                # Check if selected artists exist in our dataset
                artists_to_analyze = artist_search.check_artists_in_dataset(
                    selected_artists, matrix_artists, artist_map)
                
                if not artists_to_analyze:
                    logger.warning("None of the searched artists were found in the dataset.")

        # Option 2: Explicit artist list from command line
        elif args.analyze_artists:
            logger.info(f"Using {len(args.analyze_artists)} artists specified via command line")
            for artist_id in args.analyze_artists:
                try:
                    # Verify artist exists in dataset
                    idx = artist_index(matrix_artists, artist_id)
                    artists_to_analyze.append(artist_id)
                    artist_name = artist_map.get(artist_id, "Unknown Artist")
                    logger.info(f"Found artist in dataset: {artist_name} ({artist_id})")
                except ValueError as e:
                    logger.warning(f"Artist ID not found in dataset: {artist_id}")

        # Option 3: Find popular artists if none of the above provided any artists
        if not artists_to_analyze:
            logger.info("No artists specified, finding popular artists in dataset")
            
            # Find top artists based on listen count
            artist_listen_counts = plays.sum(axis=0).A1
            top_artist_indices = artist_listen_counts.argsort()[-10:][::-1]  # Top 10, highest first
            
            for idx in top_artist_indices:
                if idx < len(matrix_artists):
                    artist_id = matrix_artists[idx]
                    artist_name = artist_map.get(artist_id, "Unknown Artist")
                    logger.info(f"Selected popular artist: {artist_name} ({artist_id})")
                    artists_to_analyze.append(artist_id)
            
            # Also try some well-known artists if they exist in our data
            default_artists = [
                "169c4c28-858e-497b-81a4-8bc15e0026ea",  # Porcupine Tree
                "a74b1b7f-71a5-4011-9441-d0b5e4122711",  # Radiohead
                "46eb0fb7-9725-43af-97d7-6c717682a799"   # The Midnight
            ]
            
            for artist_id in default_artists:
                try:
                    idx = artist_index(matrix_artists, artist_id)
                    if artist_id not in artists_to_analyze:  # Only add if not already in list
                        artist_name = artist_map.get(artist_id, "Unknown Artist")
                        logger.info(f"Added default artist: {artist_name} ({artist_id})")
                        artists_to_analyze.append(artist_id)
                except ValueError:
                    pass

    # Step 2: Generate and save artist recommendations
    if artists_to_analyze:
        # Additionally save the currently used artists for future use
        listenbrainz_report.save_last_search(artists_to_analyze, args.working_root)
        
        logger.info(f"Generating recommendations for {len(artists_to_analyze)} artists")
        saved_count = save_artist_recommendations(model, matrix_artists, artist_map, artists_to_analyze, args.working_root)
        
        if saved_count == 0:
            logger.warning("Failed to save recommendations for any artists")
    else:
        logger.warning("No artists to analyze. No recommendations will be generated.")    
    end_time = time.time()
    logger.info(f"Total processing time: {end_time - start_time:.2f}s")

    logger.info("Generating report...")

    # Generate the report
    listenbrainz_report.generate_full_report(
        user_artist_file,
        os.path.join(args.scratch_root, "parquet", "artists.parquet"),
        args.working_root,
        open_report=not args.no_open_report,  # Open report by default, unless specifically disabled
        use_last_search=args.use_last_search  # Pass the use_last_search flag
    )

    logger.info(f"Report generated in {os.path.join(args.working_root, 'report')}")


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_temp_files()