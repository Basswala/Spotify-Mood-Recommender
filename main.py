"""
Main script to run the complete Spotify Mood Recommender pipeline.

This script demonstrates the full workflow from data loading to model training
and provides an example of how to use the recommendation system.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from io_spotify import load_dataset
from features import AudioFeatureProcessor
from cluster import MoodClusterer
from nn import MoodRecommender
from viz import MoodVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline function."""
    logger.info("🎧 Starting Spotify Mood Recommender Pipeline")
    
    try:
        # Step 1: Load dataset
        logger.info("📊 Loading dataset...")
        df = load_dataset('data/dataset.csv')
        logger.info(f"Loaded {len(df)} songs")
        
        # Step 2: Feature processing
        logger.info("🔧 Processing audio features...")
        processor = AudioFeatureProcessor(scaler_type='standard')
        df_processed = processor.fit_transform(df, remove_outliers=True)
        logger.info(f"Processed {len(df_processed)} songs after outlier removal")
        
        # Save processor
        processor.save_processor('models/feature_processor.pkl')
        logger.info("✅ Feature processor saved")
        
        # Step 3: Clustering
        logger.info("🎭 Performing mood clustering...")
        feature_cols = processor.get_audio_feature_columns()
        available_cols = [col for col in feature_cols if col in df_processed.columns]
        X = df_processed[available_cols].values
        
        clusterer = MoodClusterer(algorithm='kmeans', random_state=42)
        clusterer.fit(X, auto_tune=True)
        logger.info(f"✅ Clustering completed with {len(clusterer.cluster_stats)} clusters")
        
        # Save clusterer
        clusterer.save_model('models/clusterer.pkl')
        logger.info("✅ Clustering model saved")
        
        # Step 4: Recommendation system
        logger.info("🎵 Building recommendation system...")
        recommender = MoodRecommender(algorithm='ball_tree', metric='euclidean', n_neighbors=20)
        recommender.fit(X, df_processed, clusterer.cluster_labels)
        logger.info("✅ Recommendation system built")
        
        # Save recommender
        recommender.save_model('models/recommender.pkl')
        logger.info("✅ Recommendation model saved")
        
        # Step 5: Visualizations
        logger.info("📈 Creating visualizations...")
        visualizer = MoodVisualizer()
        visualizer.save_all_visualizations(
            X, df_processed, clusterer.cluster_labels, clusterer.mood_labels,
            output_dir='visualizations'
        )
        logger.info("✅ Visualizations saved")
        
        # Step 6: Demo recommendations
        logger.info("🎯 Testing recommendation system...")
        demo_recommendations(df_processed, clusterer, recommender)
        
        # Step 7: Summary
        print_summary(df, clusterer, recommender)
        
        logger.info("🎉 Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        raise


def demo_recommendations(df: pd.DataFrame, clusterer: MoodClusterer, recommender: MoodRecommender):
    """Demonstrate the recommendation system with sample tracks."""
    logger.info("🎵 Demo Recommendations:")
    
    # Get a few sample tracks
    sample_tracks = df.sample(min(3, len(df)), random_state=42)
    
    for i, (_, track) in enumerate(sample_tracks.iterrows(), 1):
        track_id = track['track_id']
        track_name = track['track_name']
        artists = track['artists']
        
        logger.info(f"\n{i}. {track_name} by {artists}")
        
        # Get recommendations
        recommendations = recommender.recommend_similar_songs(track_id, n_recommendations=3)
        
        if recommendations:
            logger.info(f"   Similar songs:")
            for j, rec in enumerate(recommendations, 1):
                logger.info(f"   {j}. {rec['track_name']} by {rec['artists']} "
                           f"(similarity: {rec['similarity_score']:.3f})")
        else:
            logger.info("   No recommendations found")


def print_summary(df: pd.DataFrame, clusterer: MoodClusterer, recommender: MoodRecommender):
    """Print a summary of the pipeline results."""
    print("\n" + "="*60)
    print("🎧 SPOTIFY MOOD RECOMMENDER - PIPELINE SUMMARY")
    print("="*60)
    
    print(f"📊 Dataset: {len(df):,} songs")
    print(f"🎭 Mood Clusters: {len(clusterer.cluster_stats)}")
    print(f"🎵 Recommendation System: Ready")
    
    print(f"\n🎭 Mood Clusters:")
    for stat in clusterer.cluster_stats:
        mood_label = clusterer.mood_labels.get(stat['cluster_id'], f"Cluster {stat['cluster_id']}")
        print(f"   • {mood_label}: {stat['size']} songs ({stat['percentage']:.1f}%)")
    
    print(f"\n📁 Models saved:")
    print(f"   • models/feature_processor.pkl")
    print(f"   • models/clusterer.pkl")
    print(f"   • models/recommender.pkl")
    
    print(f"\n📈 Visualizations saved in: visualizations/")
    print(f"🌐 Run Streamlit app: streamlit run src/app/ui.py")
    
    print("="*60)


if __name__ == "__main__":
    main()
