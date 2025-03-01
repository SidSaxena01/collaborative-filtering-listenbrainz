#!/usr/bin/env python3
"""
Utility module for directory management in ListenBrainz project.
"""
import os
from typing import Dict

from loguru import logger


def ensure_directories(working_root: str) -> Dict[str, str]:
    """
    Ensure all necessary directories exist and return their paths.
    
    Args:
        working_root: Base working directory
        
    Returns:
        Dictionary of directory paths
    """
    # Create main directories
    dirs = {
        "recommendations": os.path.join(working_root, "recommendations"),
        "report": os.path.join(working_root, "report"),
        "model_cache": os.path.join(working_root, "model_cache"),
        "cache": os.path.join(working_root, ".cache"),
    }
    
    # Create report subdirectories
    dirs["report_images"] = os.path.join(dirs["report"], "images")
    dirs["report_data"] = os.path.join(dirs["report"], "data")
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")
    
    return dirs

def get_recommendations_dir(working_root: str) -> str:
    """
    Get path to recommendations directory and ensure it exists.
    
    Args:
        working_root: Base working directory
        
    Returns:
        Path to recommendations directory
    """
    recommendations_dir = os.path.join(working_root, "recommendations")
    os.makedirs(recommendations_dir, exist_ok=True)
    return recommendations_dir

def find_recommendation_files(working_root: str) -> list:
    """
    Find recommendation files in the proper directory.
    
    Args:
        working_root: Base working directory
        
    Returns:
        List of recommendation filenames
    """
    recommendations_dir = get_recommendations_dir(working_root)
    if not os.path.exists(recommendations_dir):
        return []
    
    return [f for f in os.listdir(recommendations_dir) 
            if f.startswith("similar_to_") and f.endswith(".csv")]
