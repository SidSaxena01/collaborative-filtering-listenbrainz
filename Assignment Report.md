# Report

You must include a discussion in your report discussing the following items.

> Following the implicit tutorial section “Recommending similar items“ and using the code in the notebook, use the MusicBrainz site to search for some artists that you are familiar with and determine their MusicBrainz IDs. If these artists are in the model that you just developed, use the model to identify similar artists. Comment on if in your opinion these artists are similar to each other. Use another site, such as last.fm or spotify to find which artists that site says are similar to the artists that you selected. Comment on any differences that you see between the similarities from your model and from the site.

- This is being handled programmatically with the use of the `-search-artists` flag where users can search for artists with the Musicbrainz and Last.fm APIs.
- The recommendations are quite decent for the most part. We plot a network graph to get a more intuitive understanding into how the collaborative filtering works.

> Investigate the data model that you have built. Determine how many users’ data were used to build this model.

- ~ 15-16K users with complete end-to-end mapping.

> Look at the documentation for the Lastfm 360k dataset to determine how many users are in it and comment on which model you think would be more representative. (If you want, you could also generate a model using the lastfm 360k dataset and compare results, although won’t consider this subtask when grading the assignment) 

- The Lastfm dataset would definitely be more representative considering the much larger number of users (Unique Users: 359,347)
The Listenbrainz platform is still in its infancy compared to a much older platform that is Last.fm

> Perform an analysis of the most common artists in the dataset. Use an external resource (MusicBrainz, Wikipedia, Last.fm, spotify etc) to look up where they were from, their general styles/genres, and when they were most active (You can do this manually, don’t do more than 10 or so). Discuss if you think that you think that there will be a bias in this recommender system based on the type of data that you found. 

- The analysis of the top 20 most common artists is available in the [report](./results/report/report.html).

Potential Biases - `"Artist Diversity Analysis" section`

- Geographic Bias: The "Country Distribution of Top Artists" visualization clearly shows that certain countries such as the US and the UK are overrepresented in the dataset.

Genre Bias: The "Genre Distribution of Top Artists" visualization shows some genres are more prevalent than others, specifically Pop and its subgenres, some Rock and a bit of Hip Hop & Rap. This could lead to bias in recommendations.

Temporal Bias: The "Artist Time Period Analysis" section shows the distribution of top artists by decade they started their careers, showing that 1980s and 1990s make up the majority of it.

Popularity Bias: We can safely conclude from the top 20 artists that recommendations may be biased towards popular artists with many listeners.

## Deliverables

For this task, provide us with:
- The code used to convert the ListenBrainz dataset into a matrix of user/artist counts - 
    [listenbrainz.py](listenbrainz.py)
- The code used to generate some artist similarities - 
    [listenbrainz_report.py](listenbrainz_report.py)
- A short report (~1 page) describing the process that you took to convert the data and generate the model. Include your observations on the accuracy of the model with artists that you are familiar with, and compare it to another online music service’s recommendations.

## Implementation Details
> **🔍 More information can be found in [report.html](./results/report/report.html) and [README.md](README.md)**


### Data Processing Pipeline

1. **Data Conversion**: Raw ListenBrainz data is converted to Parquet format for efficient processing
2. **MSID Mapping**: Recording MSIDs are mapped to MusicBrainz recording MBIDs
3. **Canonical Redirects**: Recording MBIDs are redirected to their canonical versions
4. **Artist Extraction**: Artist MBIDs are extracted from the recordings
5. **Matrix Creation**: A sparse user-artist listen count matrix is created
6. **Model Building**: An ALS model is trained on the weighted matrix
7. **Recommendation Generation**: Artist recommendations are generated based on the model

### Technical Decisions

1. **DuckDB and Parquet**: Used for efficient data processing that can handle much larger datasets than in-memory processing

    Parquet Format

    The implementation converts large CSV and JSON files to Parquet format, which offers several advantages:

    Column-oriented storage: Allows faster reading of specific columns without loading the entire dataset
    Efficient compression: Reduces file sizes by up to 75% compared to CSV
    Schema enforcement: Maintains data types across operations
    Predicate pushdown: Enables filtering at the storage level before loading data
    This conversion happens at the beginning of the pipeline and significantly reduces the memory footprint and processing time for subsequent operations.

    DuckDB for Analytics
    DuckDB is used as an embedded analytical database engine because:

    SQL-based processing: Leverages optimized query execution for data transformations
    Zero-copy integration: Works directly with Parquet files without intermediate conversions
    Join optimization: Efficiently handles complex joins between large tables
    Low memory overhead: Processes data without loading everything into memory
    Vectorized execution: Uses CPU SIMD instructions for faster processing
    The system uses DuckDB to perform complex transformations like mapping recording IDs to artists and generating the user-artist matrix, which would be memory-intensive with traditional approaches.

    Polars for Data Manipulation
    Polars is used for dataframe operations because:

    Rust-based performance: Much faster than pandas for many operations
    Lazy evaluation: Optimizes operation chains before execution
    Memory efficiency: Uses Arrow memory model for reduced memory usage
    Multi-threaded execution: Automatically parallelizes operations when possible
    The combination of Parquet, DuckDB, and Polars allows the system to process datasets that would otherwise require much more RAM and processing time.

2. **Caching**: Model and API calls are cached to avoid expensive retraining and search operations.

    Caching System
    To avoid redundant processing and API calls, the system implements multiple caching mechanisms:

    Model Caching: Saves trained models with data hashes to avoid retraining
    API Response Caching: Stores MusicBrainz and Last.fm API responses
    Intermediate Results: Caches processed dataframes between runs
    Automatic Invalidation: Uses content hashing to determine when caches should be refreshed
    The caching system significantly speeds up repeated runs and analysis of the same dataset.

3. **Batch Processing**: Data is processed in batches to manage memory usage

### Optimization Techniques

1. **Memory Management**: Efficient use of memory by processing data in chunks and using Parquet format

    Memory Management
    The system implements several strategies to manage memory usage with large datasets:

    Batch Processing: Processes data in manageable chunks rather than all at once
    File-Based Intermediate Storage: Uses disk storage for intermediate results
    Query Optimization: Uses DuckDB's query optimizer to minimize memory requirements
    Garbage Collection: Explicitly cleans up unused objects during processing
    Temporary File Management: Tracks and removes temporary files when no longer needed
    These strategies allow processing multi-gigabyte datasets on machines with limited RAM.

2. **Parallel Processing**: Leveraging multi-core processors to speed up data processing tasks
3. **Lazy Evaluation**: Using lazy evaluation techniques to delay computation until necessary, reducing memory overhead
4. **Indexing**: Creating indexes on frequently accessed columns to speed up query performance


### Output Files

The system generates the following key outputs in the working directory:

1. `userid-artist-counts.csv`: The user-artist listen count matrix
2. `similar_to_*.csv`: Files containing similar artists for each analyzed artist
3. `report/`: Directory containing the generated HTML report and visualizations
4. `model_cache/`: Directory containing cached models for future use

### Report Content

The generated HTML report includes:

1. **Dataset Summary**: Statistics on users, artists, and listening patterns
2. **Top Artists**: Ranking of the most popular artists in the dataset
3. **Artist Diversity**: Analysis of countries, genres, and time periods
4. **Recommendations**: Examples of generated artist recommendations
5. **Last.fm Comparison**: Comparison with Last.fm's similar artist recommendations
6. **Visualizations**: Charts and graphs illustrating the data and results

### Last.fm API Integration

For Last.fm comparisons, you can set your Last.fm API key in a `.env` file:

```
LASTFM_API_KEY=your_api_key_here
```

If no API key is provided, the system will fall back to web scraping (when BeautifulSoup is installed).


## Use of LLMs or coding AI tools

If you use an LLM service such as Chatgpt or Claude, then add a section to your report
indicating that you did this. Additionally, include a summary of the following items
- If the code didn’t work for any reason, explain what went wrong, why you think it went wrong, and how you adjusted your prompts or the code to make it work
- For each block of generated code that you run (i.e. for each preprocessing step), explain how you verified that you believe that this code is accurate and that it computes what you expect it to.

Claude was used liberally for these tasks
- Generation of the HTML Report
- Data Visualization
- Generation of docstrings
- Utility functions for caching etc.
- Refactoring of the code

Potential issues that were encountered were:
- A lot of the generated code did not work in one go, there was a lot of back and forth with the LLM trying to make it understand exactly what the requirements were and what it kept missing.
- A lot of the code for the data visualization is unoptimized and redundant.
- The HTML report could be done much better, there are a lot of structural issues that require a lot of manual work to fix. The LLM was incapable of understanding and fixing the exact issues.
- Some of the API caching could be done much better through native API caching methods, the current implementation of storing and loading json is very naive.