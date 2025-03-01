# ListenBrainz Collaborative Filtering

A collaborative filtering system for music recommendations using ListenBrainz listening history data.

---

Information on Assignment Deliverables can be found in [Assignment.md](./Assignment%20Report.md)

---

## Project Overview

This project implements a scalable collaborative filtering system that analyzes ListenBrainz user listening data to generate artist recommendations. It uses efficient data processing techniques with Parquet and DuckDB to handle large datasets while keeping memory usage manageable.

The system takes raw ListenBrainz data, processes it through a series of transformations, builds a collaborative filtering model using Alternating Least Squares (ALS), and generates artist recommendations. It also creates comprehensive visual reports to analyze the dataset and evaluate the recommendation quality.

### Key Features

- Efficient processing of large ListenBrainz datasets using DuckDB and Parquet
- Artist recommendation generation using implicit ALS collaborative filtering
- Interactive artist search via MusicBrainz API

      The interactive artist search feature allows users to find and analyze specific artists without knowing their MusicBrainz IDs:

      API Integration: Uses the MusicBrainz API to search for artists by name
      Disambiguation: Shows additional context when multiple artists share the same name
      Rate Limiting: Implements appropriate delays to respect the MusicBrainz API usage policy
      Dataset Verification: Checks if selected artists exist in the processed dataset
      Alternative Suggestions: Recommends popular artists from the dataset if searched artists aren't found
      This feature makes the system more accessible for exploratory analysis, as users can discover artists interactively rather than providing MusicBrainz IDs directly.

- Comprehensive HTML report generation with visualizations
- Artist diversity analysis by country, genre, and time period

      The system analyzes artist diversity across multiple dimensions:

      Geographic Distribution: Shows the country of origin for top artists
      Genre Analysis: Identifies the most common genres in the dataset
      Time Period Distribution: Visualizes when artists began their careers by decade
      Metadata Enrichment: Fetches additional artist information from MusicBrainz API
      Visualization: Generates charts showing the distribution across these dimensions
      This analysis helps identify potential biases in the dataset and recommendation system, such as over-representation of certain countries or time periods.

- Comparison with Last.fm recommendations
      The system compares its recommendations with Last.fm's similar artists:

      Dual Approach: Uses Last.fm API when possible, with a fallback to web scraping
      Overlap Analysis: Calculates the percentage of artists that appear in both recommendation sets
      Visualization: Shows the overlap percentage for different source artists
      Caching: Stores Last.fm results to avoid repeated API calls
      Detailed Comparison: Lists the specific artists that both systems recommend
      This comparison provides an external benchmark for the recommendation quality, showing how the collaborative filtering approach compares to Last.fm's established recommendation system.

## System Requirements

- Python 3.8+
- Sufficient RAM to handle your dataset size (8GB+ recommended)
- Storage space for the ListenBrainz dataset and intermediate files

## Installation

1. Clone this repository
2. Install dependencies using `uv` or your preferred package manager:

```bash
uv sync
```

## Modules

### `listenbrainz.py`

The main module that:
1. Processes ListenBrainz data files (decompression, conversion to Parquet)
2. Maps recording MSIDs to MusicBrainz IDs
3. Applies canonical redirects
4. Extracts artist information
5. Generates user-artist listen count matrix
6. Builds the collaborative filtering model
7. Generates artist recommendations

### `listenbrainz_report.py`

Generates detailed reports and visualizations to analyze:
1. Dataset characteristics (users, artists, listening patterns)
2. Artist diversity (country, genre, time period)
3. Recommendation quality
4. Comparison with Last.fm recommendations

### `artist_search.py`

Provides utilities for:
1. Searching for artists by name using the MusicBrainz API
2. Interactive artist selection
3. Verification that selected artists exist in the dataset

## Data Requirements

This system requires the ListenBrainz dataset, which includes:

1. Listen files (monthly `.listens.zst` files)
2. MSID mapping file (`listenbrainz_msid_mapping.csv.zst`)
3. Canonical redirect file (`canonical_recording_redirect.csv.zst`)
4. Canonical MusicBrainz data (`canonical_musicbrainz_data.csv.zst`)
5. MusicBrainz artist data (`musicbrainz_artist.csv`)

You can obtain this dataset from the ListenBrainz data dumps: [https://data.metabrainz.org/](https://data.metabrainz.org/)

## Usage

### Directory Structure

You'll need to set up three main directories:

1. `data-root`: Directory containing the raw ListenBrainz dataset files
2. `working-root`: Directory for storing results and the generated report
3. `scratch-root`: Directory for temporary files and converted Parquet files

## Usage

### Basic Usage

```bash
uv run listenbrainz.py \
  --data-root "/path/to/data" \
  --working-root ./results/ \
  --scratch-root "/path/to/scratch"
```

### Command Options

- `--quality-levels`: Specify match quality levels to include (e.g., `exact_match,high_quality,medium_quality`)
- `--search-artists`: Enable interactive artist search
- `--months`: Process specific months of data (e.g., `1-3` for a range or `1,3,5` for individual months)
- `--analyze-artists`: Analyze specific artists by their MusicBrainz IDs
- `--force-reprocess`: Force reprocessing of all files
- `--force-retrain`: Force retraining the model even if cached version exists
- `--no-open-report`: Don't automatically open the HTML report in browser

## Report Generation

If you want to generate a report separately after processing the data:

```bash
uv run listenbrainz_report.py \
  --user-artist-file /path/to/working_root/userid-artist-counts.csv \
  --artist-file /path/to/scratch_root/parquet/artists.parquet \
  --working-dir /path/to/working_root
```

### Report Options

- `--output-dir`: Custom location to save the report (defaults to working_dir/report)
- `--no-open-report`: Don't automatically open the report in a browser

## Limitations and Future Work

- The current implementation focuses on artist recommendations; track-level recommendations could be added
- The model uses fixed hyperparameters; future versions could implement hyperparameter tuning
- More sophisticated evaluation metrics could be implemented
- Support for incremental updates as new ListenBrainz data becomes available
