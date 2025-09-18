#!/usr/bin/env python3
"""
Demo script for the Spotify Mood Recommender.

This script demonstrates the key functionality of the recommendation system.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

def demo_basic_functionality():
    """Demonstrate basic functionality without running the full pipeline."""
    print("🎧 Spotify Mood Recommender - Demo")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from io_spotify import load_dataset, get_audio_feature_columns
        from features import AudioFeatureProcessor
        from cluster import MoodClusterer
        from nn import MoodRecommender
        from viz import MoodVisualizer
        print("✅ All modules imported successfully!")
        
        # Test dataset loading
        print("\n📊 Testing dataset loading...")
        df = load_dataset('data/dataset.csv')
        print(f"✅ Dataset loaded: {len(df)} songs, {len(df.columns)} columns")
        
        # Test feature processing
        print("\n🔧 Testing feature processing...")
        processor = AudioFeatureProcessor(scaler_type='standard')
        df_processed = processor.fit_transform(df.head(100))  # Use small sample for demo
        print(f"✅ Features processed: {df_processed.shape}")
        
        # Test clustering
        print("\n🎭 Testing clustering...")
        feature_cols = processor.get_audio_feature_columns()
        available_cols = [col for col in feature_cols if col in df_processed.columns]
        X = df_processed[available_cols].values
        
        clusterer = MoodClusterer(algorithm='kmeans', random_state=42)
        clusterer.fit(X, n_clusters=3)  # Use 3 clusters for demo
        print(f"✅ Clustering completed: {len(clusterer.cluster_stats)} clusters")
        
        # Test recommendation system
        print("\n🎵 Testing recommendation system...")
        recommender = MoodRecommender(algorithm='ball_tree', metric='cosine')
        recommender.fit(X, df_processed, clusterer.cluster_labels)
        print("✅ Recommendation system ready!")
        
        # Test visualization
        print("\n📈 Testing visualization...")
        visualizer = MoodVisualizer()
        print("✅ Visualization tools ready!")
        
        # Demo recommendations
        print("\n🎯 Demo recommendations:")
        if len(df_processed) > 0:
            test_track_id = df_processed.iloc[0]['track_id']
            recommendations = recommender.recommend_similar_songs(test_track_id, n_recommendations=3)
            
            if recommendations:
                print(f"  Recommendations for track {test_track_id}:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"    {i}. {rec['track_name']} by {rec['artists']} "
                          f"(similarity: {rec['similarity_score']:.3f})")
            else:
                print("    No recommendations found")
        
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("  • Run 'python main.py' for full pipeline")
        print("  • Run 'streamlit run src/app/ui.py' for web app")
        print("  • Run 'python run_app.py check' to verify setup")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("\nTroubleshooting:")
        print("  • Check if data/dataset.csv exists")
        print("  • Install requirements: pip install -r requirements.txt")
        print("  • Check Python version (3.11+ recommended)")
        return False
    
    return True

if __name__ == "__main__":
    success = demo_basic_functionality()
    sys.exit(0 if success else 1)

