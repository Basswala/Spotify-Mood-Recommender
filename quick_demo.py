#!/usr/bin/env python3
"""
Quick demo of the Spotify Mood Recommender clustering analysis.
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from features import AudioFeatureProcessor
from cluster import MoodClusterer
from viz import MoodVisualizer

def main():
    print("🎧 Spotify Mood Recommender - Quick Demo\n")

    # Load and process data
    print("📊 Loading dataset...")
    df = pd.read_csv('data/dataset.csv')
    print(f"   Loaded {len(df):,} songs")

    # Process features
    print("\n🔧 Processing audio features...")
    processor = AudioFeatureProcessor(scaler_type='standard')
    df_processed = processor.fit_transform(df, remove_outliers=True)
    print(f"   Processed {len(df_processed):,} songs (outliers removed)")

    # Get feature matrix
    feature_cols = ['danceability', 'energy', 'valence', 'tempo', 'loudness',
                   'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                   'key_sin', 'key_cos', 'mode_sin', 'mode_cos']
    available_cols = [col for col in feature_cols if col in df_processed.columns]
    X = df_processed[available_cols].values
    print(f"   Feature matrix shape: {X.shape}")

    # Quick clustering with fixed number of clusters
    print("\n🎭 Performing mood clustering...")
    clusterer = MoodClusterer(algorithm='kmeans', random_state=42)

    # Use a fixed number of clusters for quick demo
    n_clusters = 6
    print(f"   Using {n_clusters} mood clusters (for quick demo)")
    clusterer.fit(X, n_clusters=n_clusters)

    # Display cluster information
    print("\n📈 Cluster Analysis:")
    print("=" * 50)

    for i, stat in enumerate(clusterer.cluster_stats):
        mood_label = clusterer.mood_labels.get(stat['cluster_id'], f"Cluster {stat['cluster_id']}")
        print(f"\n🎵 {mood_label}")
        print(f"   Size: {stat['size']:,} songs ({stat['percentage']:.1f}%)")

        # Show top features for this cluster
        feature_means = stat['mean_features'][:5]  # First 5 features
        feature_names = available_cols[:5]

        print("   Key characteristics:")
        for fname, fmean in zip(feature_names, feature_means):
            if fmean > 0.5:
                level = "High"
            elif fmean < -0.5:
                level = "Low"
            else:
                level = "Medium"
            print(f"     • {fname}: {level} ({fmean:.2f})")

    # Quick evaluation
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(X, clusterer.cluster_labels)
    print(f"\n📊 Clustering Quality:")
    print(f"   Silhouette Score: {silhouette:.3f}")

    # Create a simple visualization
    print("\n📊 Creating visualization...")
    visualizer = MoodVisualizer()

    # PCA for 2D visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=clusterer.cluster_labels,
                         cmap='viridis',
                         alpha=0.6,
                         s=1)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Mood Clusters Visualization (PCA)')
    plt.tight_layout()

    # Save the plot
    plt.savefig('mood_clusters_pca.png', dpi=150, bbox_inches='tight')
    print("   Saved visualization to mood_clusters_pca.png")

    # Sample recommendations
    print("\n🎵 Sample Recommendations:")
    print("=" * 50)

    # Pick a random song from each cluster and show similar songs
    for cluster_id in range(n_clusters):
        cluster_mask = clusterer.cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) > 0:
            # Pick a random song from this cluster
            sample_idx = np.random.choice(cluster_indices)
            sample_song = df_processed.iloc[sample_idx]

            mood_label = clusterer.mood_labels.get(cluster_id, f"Cluster {cluster_id}")

            # Find similar songs (same cluster, closest by distance)
            cluster_features = X[cluster_mask]
            sample_features = X[sample_idx].reshape(1, -1)

            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(sample_features, cluster_features)[0]
            similar_indices = np.argsort(distances)[1:4]  # Top 3 similar (excluding itself)

            print(f"\n🎭 {mood_label} Example:")
            if 'track_name' in sample_song.index:
                print(f"   Seed: {sample_song['track_name']} by {sample_song.get('artists', 'Unknown')}")
            print(f"   Similar tracks in same mood cluster:")

            for i, sim_idx in enumerate(similar_indices, 1):
                actual_idx = cluster_indices[sim_idx]
                sim_song = df_processed.iloc[actual_idx]
                if 'track_name' in sim_song.index:
                    print(f"     {i}. {sim_song['track_name']} by {sim_song.get('artists', 'Unknown')}")

    print("\n✅ Demo completed successfully!")
    print("=" * 50)
    print("\nRun 'python main.py' for full analysis with optimized clustering")
    print("Run 'streamlit run src/app/ui.py' for interactive web app")

if __name__ == "__main__":
    main()