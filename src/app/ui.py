"""
Streamlit web application for the Spotify Mood Recommender.

This module provides an interactive web interface for exploring mood-based
song recommendations with visualization and analysis capabilities.
"""

import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Any

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from io_spotify import SpotifyAPI, load_dataset
from features import AudioFeatureProcessor
from cluster import MoodClusterer
from nn import MoodRecommender
from viz import MoodVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🎧 Spotify Mood Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1DB954;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load pre-trained models and data."""
    try:
        # Load dataset
        df = load_dataset('data/dataset.csv')
        
        # Initialize processor
        processor = AudioFeatureProcessor()
        df_processed = processor.fit_transform(df)
        
        # Get feature columns
        feature_cols = processor.get_audio_feature_columns()
        available_cols = [col for col in feature_cols if col in df_processed.columns]
        X = df_processed[available_cols].values
        
        # Load or create clustering model
        clusterer_path = 'models/clusterer.pkl'
        if Path(clusterer_path).exists():
            clusterer = MoodClusterer()
            clusterer.load_model(clusterer_path)
        else:
            clusterer = MoodClusterer(algorithm='kmeans')
            clusterer.fit(X, auto_tune=True)
            clusterer.save_model(clusterer_path)
        
        # Load or create recommendation model
        recommender_path = 'models/recommender.pkl'
        if Path(recommender_path).exists():
            recommender = MoodRecommender()
            recommender.load_model(recommender_path)
        else:
            recommender = MoodRecommender(algorithm='ball_tree', metric='cosine')
            recommender.fit(X, df_processed, clusterer.cluster_labels)
            recommender.save_model(recommender_path)
        
        return df, df_processed, processor, clusterer, recommender, X, available_cols
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None, None, None

def display_header():
    """Display the application header."""
    st.markdown('<h1 class="main-header">🎧 Spotify Mood Recommender</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Discover songs with similar vibes using AI-powered mood clustering
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar_stats(df: pd.DataFrame, clusterer: MoodClusterer):
    """Display statistics in the sidebar."""
    st.sidebar.markdown("## 📊 Dataset Statistics")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Songs", f"{len(df):,}")
    with col2:
        st.metric("Mood Clusters", len(clusterer.cluster_stats))
    
    st.sidebar.markdown("## 🎭 Mood Clusters")
    for i, stat in enumerate(clusterer.cluster_stats):
        mood_label = clusterer.mood_labels.get(stat['cluster_id'], f"Cluster {stat['cluster_id']}")
        st.sidebar.markdown(f"**{mood_label}**")
        st.sidebar.markdown(f"• {stat['size']} songs ({stat['percentage']:.1f}%)")

def search_tracks(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Search for tracks in the dataset."""
    if not query:
        return df.head(10)
    
    # Search in track name, artists, and album
    mask = (
        df['track_name'].str.contains(query, case=False, na=False) |
        df['artists'].str.contains(query, case=False, na=False) |
        df['album_name'].str.contains(query, case=False, na=False)
    )
    
    return df[mask].head(20)

def display_track_search(df: pd.DataFrame):
    """Display track search interface."""
    st.markdown("## 🔍 Search for a Track")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Enter track name, artist, or album:", placeholder="e.g., Bohemian Rhapsody")
    with col2:
        search_button = st.button("Search", type="primary")
    
    if search_query or search_button:
        search_results = search_tracks(df, search_query)
        
        if len(search_results) > 0:
            st.markdown(f"**Found {len(search_results)} tracks:**")
            
            for idx, row in search_results.iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        st.write(f"**{row['track_name']}**")
                        st.write(f"by {row['artists']}")
                        st.write(f"from {row['album_name']}")
                    
                    with col2:
                        st.write(f"Popularity: {row['popularity']}")
                        st.write(f"Duration: {row['duration_ms'] // 1000}s")
                    
                    with col3:
                        if st.button(f"Select", key=f"select_{idx}"):
                            st.session_state.selected_track = row.to_dict()
                            st.rerun()
        else:
            st.warning("No tracks found. Try a different search term.")

def display_recommendations(recommender: MoodRecommender, track_data: Dict[str, Any]):
    """Display recommendations for a selected track."""
    st.markdown("## 🎵 Recommendations")
    
    track_id = track_data['track_id']
    track_name = track_data['track_name']
    artists = track_data['artists']
    
    st.markdown(f"**Recommendations for:** {track_name} by {artists}")
    
    # Get recommendations
    with st.spinner("Finding similar songs..."):
        recommendations = recommender.recommend_similar_songs(
            track_id, n_recommendations=10, same_mood_only=True
        )
    
    if recommendations:
        st.markdown(f"**Found {len(recommendations)} similar songs:**")
        
        for i, rec in enumerate(recommendations, 1):
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    st.write(f"**{i}.**")
                
                with col2:
                    st.write(f"**{rec['track_name']}**")
                    st.write(f"by {rec['artists']}")
                    st.write(f"from {rec['album_name']}")
                
                with col3:
                    similarity = rec.get('similarity_score', 0)
                    st.metric("Similarity", f"{similarity:.2f}")
    else:
        st.warning("No recommendations found for this track.")

def display_track_analysis(track_data: Dict[str, Any], visualizer: MoodVisualizer):
    """Display detailed analysis of a track."""
    st.markdown("## 📈 Track Analysis")
    
    # Create radar chart
    feature_cols = [
        'danceability', 'energy', 'valence', 'tempo', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness'
    ]
    
    track_features = {col: track_data.get(col, 0) for col in feature_cols if col in track_data}
    
    if track_features:
        fig = visualizer.create_radar_chart(track_features, f"Audio Features - {track_data['track_name']}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Display feature values
    st.markdown("### Audio Feature Values")
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (feature, value) in enumerate(track_features.items()):
            if i % 2 == 0:
                st.metric(feature.replace('_', ' ').title(), f"{value:.3f}")
    
    with col2:
        for i, (feature, value) in enumerate(track_features.items()):
            if i % 2 == 1:
                st.metric(feature.replace('_', ' ').title(), f"{value:.3f}")

def display_cluster_analysis(clusterer: MoodClusterer, visualizer: MoodVisualizer, X: np.ndarray, df: pd.DataFrame):
    """Display cluster analysis and visualizations."""
    st.markdown("## 🎭 Mood Cluster Analysis")
    
    # Cluster overview
    st.markdown("### Cluster Overview")
    cluster_data = []
    for stat in clusterer.cluster_stats:
        mood_label = clusterer.mood_labels.get(stat['cluster_id'], f"Cluster {stat['cluster_id']}")
        cluster_data.append({
            'Mood': mood_label,
            'Songs': stat['size'],
            'Percentage': f"{stat['percentage']:.1f}%"
        })
    
    cluster_df = pd.DataFrame(cluster_data)
    st.dataframe(cluster_df, use_container_width=True)
    
    # Visualizations
    st.markdown("### Cluster Visualizations")
    
    viz_method = st.selectbox("Choose visualization method:", ["PCA", "t-SNE", "UMAP"])
    
    if st.button("Generate Visualization"):
        with st.spinner("Creating visualization..."):
            fig = visualizer.create_cluster_scatter_plot(
                X, clusterer.cluster_labels, clusterer.mood_labels,
                method=viz_method.lower()
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.markdown("### Feature Distributions by Mood")
    if st.button("Show Feature Distributions"):
        with st.spinner("Creating distribution plots..."):
            fig = visualizer.create_feature_distribution_plot(
                df, clusterer.cluster_labels, clusterer.mood_labels
            )
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function."""
    # Load models and data
    with st.spinner("Loading models and data..."):
        df, df_processed, processor, clusterer, recommender, X, available_cols = load_models()
    
    if df is None:
        st.error("Failed to load models. Please check the data files.")
        return
    
    # Initialize visualizer
    visualizer = MoodVisualizer()
    
    # Display header
    display_header()
    
    # Display sidebar stats
    display_sidebar_stats(df, clusterer)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Find Recommendations", "📊 Cluster Analysis", "ℹ️ About"])
    
    with tab1:
        # Track search
        display_track_search(df)
        
        # Display selected track and recommendations
        if 'selected_track' in st.session_state:
            selected_track = st.session_state.selected_track
            
            st.markdown("---")
            display_track_analysis(selected_track, visualizer)
            
            st.markdown("---")
            display_recommendations(recommender, selected_track)
    
    with tab2:
        display_cluster_analysis(clusterer, visualizer, X, df_processed)
    
    with tab3:
        st.markdown("""
        ## About Spotify Mood Recommender
        
        This application uses machine learning to group songs by mood based on their audio features.
        
        ### How it works:
        1. **Audio Features**: We analyze 11 audio features from Spotify including danceability, energy, valence, tempo, and more.
        2. **Clustering**: Songs are grouped into mood clusters using KMeans clustering.
        3. **Recommendations**: For any song, we find similar songs within the same mood cluster.
        
        ### Features:
        - **Mood-based clustering** instead of genre-based
        - **Interactive visualizations** with radar charts and scatter plots
        - **Real-time recommendations** based on audio similarity
        - **Comprehensive analysis** of track characteristics
        
        ### Technical Stack:
        - **Python 3.11+** with modern ML libraries
        - **scikit-learn** for clustering and recommendations
        - **Streamlit** for the web interface
        - **Plotly** for interactive visualizations
        - **UMAP/t-SNE** for dimensionality reduction
        
        ### Dataset:
        The model is trained on a dataset of songs with their Spotify audio features,
        allowing for accurate mood-based recommendations.
        """)

if __name__ == "__main__":
    main()
