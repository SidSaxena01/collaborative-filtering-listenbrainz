#!/usr/bin/env python3
"""
Data quality analysis module for ListenBrainz processing.
"""
import json
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
from loguru import logger


def analyze_mapping_quality(mapping_stats: Dict) -> Dict:
    """Analyze the quality of MSID to MBID mapping.
    
    Args:
        mapping_stats: Dictionary with mapping statistics
        
    Returns:
        Dictionary with quality metrics
    """
    if not mapping_stats or "mapped" not in mapping_stats or "unmapped" not in mapping_stats:
        logger.warning("Invalid mapping statistics provided")
        return {}
    
    total = mapping_stats["mapped"] + mapping_stats["unmapped"]
    
    if total == 0:
        logger.warning("No mapping data available (total is 0)")
        return {}
    
    mapped_pct = (mapping_stats["mapped"] / total) * 100
    unmapped_pct = (mapping_stats["unmapped"] / total) * 100
    
    return {
        "mapped": mapped_pct,
        "unmapped": unmapped_pct,
        "total": total,
        "quality_score": mapped_pct  # Simple quality score is just the mapping percentage
    }

def visualize_mapping_quality(quality_metrics: Dict, output_dir: str) -> None:
    """Create visualizations for mapping quality.
    
    Args:
        quality_metrics: Dictionary with quality metrics
        output_dir: Directory to save visualizations
    """
    if not quality_metrics or "mapped" not in quality_metrics:
        logger.warning("Invalid quality metrics provided")
        return
    
    plt.figure(figsize=(8, 6))
    labels = ['Mapped MSIDs', 'Unmapped MSIDs']
    sizes = [quality_metrics["mapped"], quality_metrics["unmapped"]]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # explode the 1st slice for emphasis
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
    plt.title('MSID to MBID Mapping Coverage', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "msid_mbid_mapping.png"), dpi=300)
    plt.close()
    
    # Save metrics to JSON for use in the report
    with open(os.path.join(output_dir, "data_quality.json"), 'w') as f:
        json.dump(quality_metrics, f, indent=2)
    
    logger.info("Mapping quality visualizations saved")

def extract_mapping_stats(logs_file: str) -> Optional[Dict]:
    """Extract MSID to MBID mapping statistics from log file.
    
    Args:
        logs_file: Path to log file
        
    Returns:
        Dictionary with mapped and unmapped counts, or None if not found
    """
    try:
        with open(logs_file, 'r') as f:
            for line in f:
                if "MSID to MBID mapping:" in line:
                    # Parse the line: "MSID to MBID mapping: X mapped (Y%), Z unmapped"
                    parts = line.strip().split("MSID to MBID mapping:")[1].strip()
                    
                    # Extract mapped count
                    mapped_count = int(parts.split(" mapped")[0].strip().replace(',', ''))
                    
                    # Extract unmapped count
                    unmapped_count = int(parts.split("unmapped")[0].split(',')[-1].strip().replace(',', ''))
                    
                    return {
                        "mapped": mapped_count,
                        "unmapped": unmapped_count
                    }
        
        logger.warning(f"Mapping statistics not found in log file: {logs_file}")
        return None
    
    except Exception as e:
        logger.error(f"Error extracting mapping stats from log: {str(e)}")
        return None

def analyze_and_visualize_from_logs(logs_file: str, output_dir: str) -> Dict:
    """Extract mapping stats from logs and generate visualizations.
    
    Args:
        logs_file: Path to log file
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary with quality metrics
    """
    mapping_stats = extract_mapping_stats(logs_file)
    
    if mapping_stats:
        quality_metrics = analyze_mapping_quality(mapping_stats)
        visualize_mapping_quality(quality_metrics, output_dir)
        return quality_metrics
    
    return {}
