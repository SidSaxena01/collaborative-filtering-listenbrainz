#!/usr/bin/env python3
"""
HTML report generation module for ListenBrainz Collaborative Filtering results
"""
import decimal  # Added import for DecimalEncoder
import json
import os
import time
import webbrowser
from typing import Dict

from loguru import logger


class DecimalEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle Decimal objects."""
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def generate_html_report(stats: Dict, artist_info: Dict, dirs: Dict[str, str]) -> None:
    """Generate an HTML report that combines all analyses."""
    logger.info("Generating HTML report")
    
    # Create HTML content with executive summary and navigation
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>ListenBrainz Collaborative Filtering Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .section {{
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
                clear: both;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
                font-size: 0.9em;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 6px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            /* Two-column layout for visualizations */
            .visualization-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                gap: 15px;
                margin: 15px 0;
            }}
            /* Make visualizations smaller */
            .visualization {{
                text-align: center;
                width: calc(50% - 15px);
                margin-bottom: 15px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .visualization img {{
                max-width: 100%;
                height: auto;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                object-fit: contain;
            }}
            .visualization p {{
                font-size: 0.85em;
                margin-top: 5px;
            }}
            /* Full-width visualization when needed */
            .visualization.full-width {{
                width: 100%;
            }}
            .stats-box {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
                width: calc(33.33% - 15px);
                box-sizing: border-box;
                float: left;
                margin-right: 15px;
            }}
            .stats-box h3 {{
                margin-top: 0;
                border-bottom: 1px solid #ddd;
                padding-bottom: 10px;
                font-size: 1.1em;
            }}
            .stats-box p {{
                font-size: 0.9em;
                margin: 5px 0;
            }}
            .executive-summary {{
                background-color: #f0f7ff;
                border-left: 5px solid #3498db;
                padding: 15px;
                margin-bottom: 30px;
                border-radius: 0 5px 5px 0;
            }}
            /* New navigation bar style */
            .navigation {{
                background-color: #2c3e50;
                position: sticky;
                top: 0;
                padding: 10px 20px;
                margin-bottom: 30px;
                border-radius: 5px;
                z-index: 100;
            }}
            .navigation ul {{
                list-style-type: none;
                padding: 0;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 0;
            }}
            .navigation li {{
                margin: 3px;
                font-size: 0.9em;
            }}
            .navigation a {{
                display: inline-block;
                padding: 4px 8px;
                background-color: #34495e;
                border-radius: 3px;
                text-decoration: none;
                color: #fff;
                transition: background-color 0.2s;
            }}
            .navigation a:hover {{
                background-color: #3498db;
            }}
            .stats-summary {{
                background-color: #f5f5f5;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                border-left: 4px solid #3498db;
            }}
            .highlighted-stat {{
                font-size: 1.2em;
                color: #e74c3c;
                font-weight: bold;
            }}
            .two-column {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            .two-column > div {{
                width: calc(50% - 10px);
            }}
            .clearfix::after {{
                content: "";
                clear: both;
                display: table;
            }}
            footer {{
                margin-top: 30px;
                padding-top: 10px;
                border-top: 1px solid #eee;
                text-align: center;
                font-size: 0.8em;
                color: #777;
            }}
            /* Responsive design */
            @media (max-width: 768px) {{
                .stats-box, .visualization, .two-column > div {{
                    width: 100%;
                    margin-right: 0;
                    float: none;
                }}
                table {{
                    font-size: 0.8em;
                }}
            }}
        </style>
    </head>
    <body>
        <h1>ListenBrainz Collaborative Filtering Report</h1>
        <p>Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <!-- Executive Summary -->
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <p>This report analyzes the ListenBrainz collaborative filtering model applied to a dataset of <strong>{stats["total_users"]:,}</strong> 
            users and <strong>{stats["total_artists"]:,}</strong> artists, with approximately <strong>{stats["total_listens"]:,}</strong> total listens.</p>
            
            <p>Key findings:</p>
            <ul>
                <li>The average user listens to <strong>{stats["avg_artists_per_user"]:.0f}</strong> different artists with <strong>{stats["avg_listens_per_user"]:.0f}</strong> total listens</li>
                <li>The most popular artist ({artist_info.get(stats["top_artists"][0]["artist_id"], "Unknown")}) has <strong>{stats["top_artists"][0]["total_listens"]:,}</strong> listens from <strong>{stats["top_artists"][0]["unique_listeners"]:,}</strong> unique users</li>
                <li>Artist recommendations show strong genre clustering and musical similarity patterns</li>
            </ul>
        </div>
        
        <!-- Navigation -->
        <div class="navigation">
            <ul>
                <li><a href="#dataset-summary">Dataset Overview</a></li>
                <li><a href="#data-quality">Data Quality</a></li>
                <li><a href="#artist-distribution">Artist Distribution</a></li>
                <li><a href="#user-listening">User Patterns</a></li>
                <li><a href="#user-segments">User Segments</a></li>
                <li><a href="#artist-diversity">Artist Diversity</a></li>
                <li><a href="#time-analysis">Time Analysis</a></li>
                <li><a href="#artist-recommendations">Recommendations</a></li>
                <li><a href="#artist-network">Artist Network</a></li>
                <li><a href="#lastfm-comparison">Last.fm Comparison</a></li>
                <li><a href="#conclusion">Conclusion</a></li>
            </ul>
        </div>
        
        <!-- 1. DATASET OVERVIEW -->
        <div class="section" id="dataset-summary">
            <h2>Dataset Overview</h2>
            <div class="clearfix">
                <div class="stats-box">
                    <h3>User Statistics</h3>
                    <p><strong>Total Users:</strong> {stats["total_users"]:,}</p>
                    <p><strong>Avg. Listens per User:</strong> {stats["avg_listens_per_user"]:.2f}</p>
                    <p><strong>Avg. Artists per User:</strong> {stats["avg_artists_per_user"]:.2f}</p>
                </div>
                
                <div class="stats-box">
                    <h3>Artist Statistics</h3>
                    <p><strong>Total Artists:</strong> {stats["total_artists"]:,}</p>
                    <p><strong>Avg. Users per Artist:</strong> {stats["avg_users_per_artist"]:.2f}</p>
                </div>
                
                <div class="stats-box">
                    <h3>Overall Statistics</h3>
                    <p><strong>Total Entries:</strong> {stats["total_entries"]:,}</p>
                    <p><strong>Total Listens:</strong> {stats["total_listens"]:,}</p>
                </div>
            </div>
            
            <div class="visualization full-width">
                <img src="images/top_artists.png" alt="Top 20 Artists by Total Listens">
                <p><em>Top 20 Artists by Total Listens</em></p>
            </div>

            <h3>Top Artists by Listen Count</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Artist</th>
                    <th>Total Listens</th>
                    <th>Unique Listeners</th>
                </tr>
    """
    
    # Add top artists to the table
    for i, artist in enumerate(stats["top_artists"][:20]):
        artist_id = artist["artist_id"]
        artist_name = artist_info.get(artist_id, "Unknown Artist")
        html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{artist_name}</td>
                    <td>{artist["total_listens"]:,}</td>
                    <td>{artist["unique_listeners"]:,}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <!-- 2. DATA QUALITY -->
        <div class="section" id="data-quality">
            <h2>Data Quality Analysis</h2>
            <p>This section analyzes the quality of the dataset and mapping processes.</p>
    """
    
    # If we have the MSID to MBID mapping visualization, add it to the data quality section
    if os.path.exists(os.path.join(dirs["images"], "msid_mbid_mapping.png")):
        html_content += """
            <div class="visualization">
                <img src="images/msid_mbid_mapping.png" alt="MSID to MBID Mapping">
                <p><em>Percentage of MSIDs that were successfully mapped to canonical MBIDs.</em></p>
            </div>
        """
    
    # Load data quality metrics if available - with more debugging
    data_quality_file = os.path.join(dirs["data"], "data_quality.json")
    logger.info(f"Looking for data quality metrics at: {data_quality_file}")
    
    if os.path.exists(data_quality_file):
        try:
            with open(data_quality_file, 'r') as f:
                quality_metrics = json.load(f)
                logger.info(f"Loaded quality metrics: {quality_metrics}")
            
            # Display data quality metrics without any conditional checks
            html_content += f"""
                <div class="stats-summary">
                    <p>Data mapping quality metrics:</p>
                    <ul>
                        <li><strong>{quality_metrics.get('mapped', 0):.1f}%</strong> of MSIDs were successfully mapped to MBIDs ({quality_metrics.get('mapped_count', 0):,} artists)</li>
                        <li><strong>{quality_metrics.get('unmapped', 0):.1f}%</strong> of MSIDs could not be mapped ({quality_metrics.get('unmapped_count', 0):,} artists)</li>
                        <li>Total artists processed: <strong>{quality_metrics.get('total', 0):,}</strong></li>
                        <li>Overall mapping quality score: <strong>{quality_metrics.get('quality_score', 0):.1f}</strong></li>
                    </ul>
                </div>
            """
        except Exception as e:
            logger.error(f"Error processing data quality metrics: {e}")
            html_content += f"<p>Error loading data quality metrics: {e}</p>"
    else:
        logger.warning(f"No data quality metrics found at {data_quality_file}")
        html_content += "<p>No data quality metrics are available for this report.</p>"
    
    html_content += """
        </div>
        
        <!-- 3. ARTIST DISTRIBUTION -->
        <div class="section" id="artist-distribution">
            <h2>Artist Listener Distribution</h2>
            <p>This section analyzes how listeners are distributed across artists, revealing the "long tail" pattern
            characteristic of music consumption.</p>
    """
    
    # Check if we have the artist distribution JSON data and visualizations
    artist_dist_file = os.path.join(dirs["data"], "artist_distribution.json")
    if os.path.exists(artist_dist_file):
        with open(artist_dist_file, 'r') as f:
            artist_dist = json.load(f)
        
        # Add key statistics
        html_content += f"""
            <div class="stats-summary">
                <p>Of the <strong>{artist_dist["total_artists"]:,}</strong> artists in the dataset:</p>
                <ul>
                    <li><span class="highlighted-stat">{artist_dist["buckets"][0]["percentage"]:.1f}%</span> have only one listener</li>
                    <li>The median artist has <strong>{artist_dist["percentiles"]["median"]}</strong> listeners</li>
                    <li>Only <strong>{artist_dist["buckets"][-1]["percentage"]:.1f}%</strong> of artists have more than 100 listeners</li>
                </ul>
            </div>
        """
    
    # Add the visualizations in a container for 2-column layout
    html_content += """
            <div class="visualization-container">
    """
    
    if os.path.exists(os.path.join(dirs["images"], "artist_listener_segments.png")):
        html_content += """
                <div class="visualization">
                    <img src="images/artist_listener_segments.png" alt="Artist Listener Segments">
                    <p><em>Distribution of artists by listener count ranges</em></p>
                </div>
        """
    
    if os.path.exists(os.path.join(dirs["images"], "artist_listener_buckets.png")):
        html_content += """
                <div class="visualization">
                    <img src="images/artist_listener_buckets.png" alt="Artist Listener Buckets">
                    <p><em>Number of artists in each listener count range (log scale)</em></p>
                </div>
        """
    
    if os.path.exists(os.path.join(dirs["images"], "artist_listener_histogram.png")):
        html_content += """
                <div class="visualization">
                    <img src="images/artist_listener_histogram.png" alt="Artist Listener Distribution">
                    <p><em>Detailed distribution of artists by listener count</em></p>
                </div>
        """
    
    if os.path.exists(os.path.join(dirs["images"], "artist_listener_cumulative.png")):
        html_content += """
                <div class="visualization">
                    <img src="images/artist_listener_cumulative.png" alt="Artist Listener Cumulative Distribution">
                    <p><em>Cumulative percentage of artists by listener count</em></p>
                </div>
        """
    
    html_content += """
            </div>
            
            <h3>Implications for Recommendations</h3>
            <p>This long-tail distribution has significant implications for the recommendation system:</p>
            <ul>
                <li><strong>Cold start problem:</strong> Many artists have very few listeners, making it challenging to generate reliable recommendations for them.</li>
                <li><strong>Data sparsity:</strong> The user-artist matrix is extremely sparse, with most artists having very few listeners.</li>
                <li><strong>Popularity bias:</strong> Recommendations may be biased towards popular artists with many listeners.</li>
            </ul>
        </div>
        
        <!-- 4. USER LISTENING PATTERNS -->
        <div class="section" id="user-listening">
            <h2>User Listening Patterns</h2>
            <p>The visualizations below show different aspects of user engagement and listening patterns.</p>
            
            <div class="visualization-container">
    """
    
    # Add user listening visualizations in 2-column layout
    
    if os.path.exists(os.path.join(dirs["images"], "user_listen_distribution.png")):
        html_content += """
                <div class="visualization">
                    <img src="images/user_listen_distribution.png" alt="Distribution of Listen Counts">
                    <p><em>Distribution of total listen counts across users</em></p>
                </div>
        """
    
    if os.path.exists(os.path.join(dirs["images"], "user_segments.png")):
        html_content += """
                <div class="visualization">
                    <img src="images/user_segments.png" alt="User Activity Segments">
                    <p><em>Distribution of total listens across different user segments</em></p>
                </div>
        """
    
    # The Top Users Comparison should be in a new row for better layout
    html_content += """
            </div>
            
            <div class="visualization-container">
    """
    
    if os.path.exists(os.path.join(dirs["images"], "top_users_comparison.png")):
        html_content += """
                <div class="visualization full-width">
                    <img src="images/top_users_comparison.png" alt="Top Users Comparison">
                    <p><em>Listen counts and artist-to-listen ratios for top users</em></p>
                </div>
        """
    
    html_content += """
            </div>
        </div>
        
        <!-- 5. USER SEGMENTS -->
    """
    
    if os.path.exists(os.path.join(dirs["images"], "user_segments_scatter.png")):
        html_content += """
        <div class="section" id="user-segments">
            <h2>User Segmentation Analysis</h2>
            <p>Users have been clustered into distinct segments based on their listening behavior. 
            These segments represent different types of music consumers with unique characteristics.</p>
            
            <div class="visualization-container">
        """
        
        if os.path.exists(os.path.join(dirs["images"], "user_segments_scatter.png")):
            html_content += """
                <div class="visualization">
                    <img src="images/user_segments_scatter.png" alt="User Segments Scatter Plot">
                    <p><em>User clusters based on listening diversity and volume</em></p>
                </div>
            """
        
        if os.path.exists(os.path.join(dirs["images"], "user_segments_characteristics.png")):
            html_content += """
                <div class="visualization">
                    <img src="images/user_segments_characteristics.png" alt="User Segment Characteristics">
                    <p><em>Characteristics of each identified user segment</em></p>
                </div>
            """
        
        html_content += """
            </div>
        """
        
        # Read segment info from JSON if available
        segments_file = os.path.join(dirs["data"], "user_segments.json")
        if os.path.exists(segments_file):
            try:
                with open(segments_file, 'r') as f:
                    segments_data = json.load(f)
                cluster_stats = segments_data.get("cluster_stats", [])
                
                if cluster_stats:
                    # Create a container for segment descriptions
                    html_content += """
                    <h3>Segment Descriptions</h3>
                    <div class="clearfix">
                    """
                    
                    for i, cluster in enumerate(cluster_stats):
                        avg_artists = cluster.get("avg_artists", 0)
                        avg_listens = cluster.get("avg_listens", 0)
                        user_count = cluster.get("user_count", 0)
                        
                        # Create a descriptive name based on cluster characteristics from user_segments.json
                        cluster_name = "Unknown Segment"
                        description = "A group of users with similar listening patterns."
                        
                        if i == 0:  # Cluster 0: Largest group with moderate artists, lower listens
                            cluster_name = "Casual Listeners"
                            description = "The majority group with moderate artist diversity but lower overall listening time."
                        elif i == 1:  # Cluster 1: Smallest group with very high listens per artist
                            cluster_name = "Power Users"
                            description = "A small group of highly engaged users with extremely high listen counts and strong artist loyalty."
                        elif i == 2:  # Cluster 2: Medium group with highest artist diversity
                            cluster_name = "Explorers"
                            description = "Users who discover and listen to many different artists with good overall engagement."
                        
                        html_content += f"""
                        <div class="stats-box">
                            <h3>{cluster_name}</h3>
                            <p>{description}</p>
                            <p><strong>Users:</strong> {user_count:,}</p>
                            <p><strong>Avg. Artists:</strong> {avg_artists:.1f}</p>
                            <p><strong>Avg. Listens:</strong> {avg_listens:.1f}</p>
                        </div>
                        """
                    
                    html_content += """
                    </div>
                    """
            except Exception as e:
                logger.warning(f"Error processing user segments JSON: {e}")
        
        html_content += """
        </div>
        """
    
    # 6. ARTIST DIVERSITY
    html_content += """
        <div class="section" id="artist-diversity">
            <h2>Artist Diversity Analysis</h2>
            <div class="visualization-container">
    """
    
    if os.path.exists(os.path.join(dirs["images"], "country_distribution.png")):
        html_content += """
                <div class="visualization">
                    <img src="images/country_distribution.png" alt="Country Distribution of Top Artists">
                    <p><em>Country distribution of top artists</em></p>
                </div>
    """
    
    if os.path.exists(os.path.join(dirs["images"], "genre_distribution.png")):
        html_content += """
                <div class="visualization">
                    <img src="images/genre_distribution.png" alt="Genre Distribution of Top Artists">
                    <p><em>Genre distribution of top artists</em></p>
                </div>
    """
    
    html_content += """
            </div>
        </div>
        
        <!-- 7. TIME PERIOD ANALYSIS -->
        <div class="section" id="time-analysis">
            <h2>Artist Time Period Analysis</h2>
            <p>This analysis shows the time period distribution of top artists in the dataset.</p>
            
            <div class="visualization">
                <img src="images/artist_decades.png" alt="Time Period Distribution of Top Artists">
                <p><em>Distribution of top artists by decade they started their career</em></p>
            </div>
            
            <p>The distribution shows when the most popular artists in the dataset began their careers.
            This can indicate potential biases in the dataset towards certain eras of music.</p>
        </div>
        
        <!-- 8. ARTIST RECOMMENDATIONS -->
        <div class="section" id="artist-recommendations">
            <h2>Artist Recommendations</h2>
            <p>The collaborative filtering model generates artist recommendations based on listening patterns.</p>
    """
    
    # Check if recommendations file exists
    recommendations_file = os.path.join(dirs["data"], "recommendations_analysis.json")
    if os.path.exists(recommendations_file):
        with open(recommendations_file, "r") as f:
            recommendations = json.load(f)
        
        # Create a 2-column layout for recommendations
        html_content += """
            <div class="two-column">
        """
        
        # Add recommendation examples (limited to keep the page manageable)
        for i, rec_set in enumerate(recommendations[:10]): 
            source_artist = rec_set["source_artist"]
            recs = rec_set["recommendations"][:10]  # 5 recommendations per artist
            
            html_content += f"""
                <div>
                <h3>Artists similar to {source_artist}</h3>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Artist</th>
                        <th>Score</th>
                    </tr>
            """
            
            for rec in recs:
                html_content += f"""
                    <tr>
                        <td>{rec["rank"]}</td>
                        <td>{rec["artist_name"]}</td>
                        <td>{rec["similarity_score"]:.4f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                </div>
            """
        
        html_content += """
            </div>
        """
    else:
        html_content += "<p>No recommendation data available.</p>"
    
    html_content += """
        </div>
        
        <!-- 9. ARTIST NETWORK -->
    """
    
    if os.path.exists(os.path.join(dirs["images"], "artist_network.png")):
        html_content += """
        <div class="section" id="artist-network">
            <h2>Artist Relationship Network</h2>
            <p>The network visualization below shows relationships between artists based on similarity scores.
            Connected artists are frequently listened to by the same users.</p>
            
            <div class="visualization full-width">
                <img src="images/artist_network.png" alt="Artist Relationship Network">
                <p><em>Network graph showing artist relationships based on listener overlap. 
                Clusters indicate genres or musical styles.</em></p>
            </div>
            
            <div class="two-column">
                <div>
                    <h3>Network Insights</h3>
                    <ul>
                        <li><strong>Central Connectors:</strong> Artists that bridge multiple musical styles</li>
                        <li><strong>Tight Clusters:</strong> Genre-specific communities with high internal similarity</li>
                        <li><strong>Isolated Nodes:</strong> Unique artists with distinctive audiences</li>
                    </ul>
                </div>
                <div>
                    <h3>Network Applications</h3>
                    <ul>
                        <li><strong>Discovery:</strong> Finding bridge artists that connect different genres</li>
                        <li><strong>Recommendations:</strong> Identifying similar artists within clusters</li>
                        <li><strong>Genre Mapping:</strong> Automatically detecting music genres from listening patterns</li>
                    </ul>
                </div>
            </div>
        </div>
        """
    
    # 10. LASTFM COMPARISON
    html_content += """
        <div class="section" id="lastfm-comparison">
            <h2>Comparison with Last.fm</h2>
            <p>This section compares the collaborative filtering model's recommendations 
            with those from Last.fm, a popular music service.</p>
            
            <div class="visualization">
                <img src="images/lastfm_overlap.png" alt="Overlap Between Model and Last.fm Recommendations">
                <p><em>Overlap percentage between our model and Last.fm recommendations</em></p>
            </div>
    """
    
    # Add Last.fm comparison details if available
    lastfm_file = os.path.join(dirs["data"], "lastfm_comparison.json")
    if os.path.exists(lastfm_file):
        with open(lastfm_file, "r") as f:
            lastfm_data = json.load(f)
        
        if lastfm_data:
            # Calculate average overlap
            avg_overlap = sum(r['overlap_percentage'] for r in lastfm_data) / len(lastfm_data)
            
            html_content += f"""
            <p>The model's recommendations have an average overlap of <strong>{avg_overlap:.2f}%</strong> 
            with Last.fm's recommendations for the same artists.</p>
            
            <h3>Detailed Comparison</h3>
            <table>
                <tr>
                    <th>Artist</th>
                    <th>Overlap %</th>
                    <th>Common Recommendations</th>
                </tr>
            """
            
            for comp in lastfm_data[:10]:  # Limit to 10 comparisons for brevity
                html_content += f"""
                <tr>
                    <td>{comp['artist_name']}</td>
                    <td>{comp['overlap_percentage']:.2f}%</td>
                    <td>{', '.join(comp['overlap'][:5]) if comp['overlap'] else 'None'}</td>
                </tr>
                """
            
            html_content += """
            </table>
            """
    
    html_content += """
        </div>
        
        <!-- 11. CONCLUSION -->
        <div class="section" id="conclusion">
            <h2>Conclusion</h2>
            <p>The collaborative filtering model successfully identifies relationships between artists based on user listening patterns.
            The recommendations appear to match general musical similarity and could be used for artist discovery.</p>
            
            <h3>Key Findings</h3>
            <ul>
                <li>There is a diverse range of artists in the dataset, though certain countries and genres may be overrepresented.</li>
                <li>The collaborative filtering approach appears to find meaningful similarities between artists.</li>
                <li>User segmentation reveals distinct patterns of music consumption behavior.</li>
                <li>Artist network analysis shows clear genre clusters with interesting bridges between musical styles.</li>
            </ul>
            
            <h3>Limitations and Future Work</h3>
            <ul>
                <li>The dataset may contain biases toward certain genres or time periods</li>
                <li>The collaborative filtering approach considers only listening counts, not contextual information</li>
                <li>Future work could incorporate audio features or lyrical content analysis</li>
                <li>A hybrid recommendation system combining collaborative filtering with content-based methods could improve results</li>
            </ul>
        </div>
        
        <footer>
            <p>Generated using ListenBrainz Collaborative Filtering.</p>
        </footer>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(os.path.join(dirs["base"], "report.html"), "w") as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated: {os.path.join(dirs['base'], 'report.html')}")


def open_html_report(report_path: str) -> bool:
    """Open the HTML report in the default web browser.
    
    Args:
        report_path: Path to the HTML report file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if os.path.exists(report_path):
            logger.info(f"Opening HTML report in web browser: {report_path}")
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
            return True
        else:
            logger.warning(f"HTML report not found at {report_path}")
            return False
    except Exception as e:
        logger.warning(f"Failed to open HTML report: {str(e)}")
        return False


def create_report_template():
    """Return the base HTML template for the report."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>ListenBrainz Collaborative Filtering Report</title>
        <link rel="stylesheet" href="styles.css">
    </head>
    <body>
        <h1>ListenBrainz Collaborative Filtering Report</h1>
        <p class="generated-date">Generated on {date}</p>
        
        <!-- Content will be inserted here -->
        {content}
        
        <footer>
            <p>Generated using ListenBrainz Collaborative Filtering.</p>
        </footer>
    </body>
    </html>
    """


# Potential future enhancement: 
# Split the report into separate components that can be imported individually
# This would make the HTML generation more modular and easier to maintain
class ReportSection:
    """Base class for report sections."""
    
    def __init__(self, title, id_attr):
        self.title = title
        self.id = id_attr
        
    def generate(self, data, dirs):
        """Generate the HTML for this section."""
        raise NotImplementedError("Subclasses must implement this method")
