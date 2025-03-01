#!/usr/bin/env python3
"""
Utility functions for ListenBrainz Collaborative Filtering Report Generator
"""
import decimal
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, TypeVar

from loguru import logger

# Set up type variables for decorator
F = TypeVar('F', bound=Callable[..., Any])



class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle Decimal objects."""
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)
    

def convert_decimal_to_float(obj: Any) -> Any:
    """Recursively converts Decimal objects to float for JSON serialization."""
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimal_to_float(item) for item in obj]
    else:
        return obj
    
    
def time_function(func: F) -> F:
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        logger.info(f"Finished {func.__name__}: Took {end_time - start_time:.2f}s")
        
        return result
    return wrapper  # type: ignore


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


def create_placeholder_image(message: str, output_path: str) -> None:
    """Create a placeholder image with a message."""
    import matplotlib.pyplot as plt
    
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
