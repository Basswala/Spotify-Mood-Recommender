"""
Unit tests for the cluster module.

Tests the MoodClusterer class and related functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import os

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from cluster import MoodClusterer, evaluate_clustering_quality


class TestMoodClusterer:
    """Test cases for MoodClusterer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Create data with clear clusters
        X = np.random.randn(n_samples, n_features)
        
        # Create 3 distinct clusters
        X[:30] += [2, 0, 0, 0, 0]  # Cluster 0
        X[30:60] += [0, 2, 0, 0, 0]  # Cluster 1
        X[60:] += [0, 0, 2, 0, 0]  # Cluster 2
        
        return X
    
    def test_initialization(self):
        """Test clusterer initialization."""
        clusterer = MoodClusterer(algorithm='kmeans', random_state=42)
        assert clusterer.algorithm == 'kmeans'
        assert clusterer.random_state == 42
        assert not clusterer.is_fitted
    
    def test_fit_kmeans(self, sample_data):
        """Test fitting KMeans clustering."""
        clusterer = MoodClusterer(algorithm='kmeans', random_state=42)
        clusterer.fit(sample_data, n_clusters=3)
        
        assert clusterer.is_fitted
        assert clusterer.model is not None
        assert clusterer.cluster_labels is not None
        assert len(clusterer.cluster_labels) == len(sample_data)
        assert len(np.unique(clusterer.cluster_labels)) == 3
    
    def test_fit_with_auto_tune(self, sample_data):
        """Test fitting with automatic parameter tuning."""
        clusterer = MoodClusterer(algorithm='kmeans', random_state=42)
        clusterer.fit(sample_data, auto_tune=True)
        
        assert clusterer.is_fitted
        assert clusterer.cluster_labels is not None
        assert len(clusterer.cluster_labels) == len(sample_data)
    
    def test_predict_without_fit(self, sample_data):
        """Test that predict raises error if not fitted."""
        clusterer = MoodClusterer()
        
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            clusterer.predict(sample_data)
    
    def test_predict(self, sample_data):
        """Test prediction on new data."""
        clusterer = MoodClusterer(algorithm='kmeans', random_state=42)
        clusterer.fit(sample_data, n_clusters=3)
        
        # Predict on same data
        predictions = clusterer.predict(sample_data)
        assert len(predictions) == len(sample_data)
        assert len(np.unique(predictions)) <= 3
    
    def test_get_cluster_info(self, sample_data):
        """Test getting cluster information."""
        clusterer = MoodClusterer(algorithm='kmeans', random_state=42)
        clusterer.fit(sample_data, n_clusters=3)
        
        # Get info for each cluster
        for cluster_id in range(3):
            info = clusterer.get_cluster_info(cluster_id)
            
            assert 'cluster_id' in info
            assert 'mood_label' in info
            assert 'size' in info
            assert 'percentage' in info
            assert 'mean_features' in info
            assert 'std_features' in info
            
            assert info['cluster_id'] == cluster_id
            assert info['size'] > 0
            assert 0 <= info['percentage'] <= 100
    
    def test_get_cluster_info_invalid(self, sample_data):
        """Test getting info for invalid cluster."""
        clusterer = MoodClusterer(algorithm='kmeans', random_state=42)
        clusterer.fit(sample_data, n_clusters=3)
        
        with pytest.raises(ValueError, match="Cluster 999 not found"):
            clusterer.get_cluster_info(999)
    
    def test_save_and_load_model(self, sample_data):
        """Test saving and loading model."""
        clusterer = MoodClusterer(algorithm='kmeans', random_state=42)
        clusterer.fit(sample_data, n_clusters=3)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save model
            clusterer.save_model(tmp_path)
            assert Path(tmp_path).exists()
            
            # Load model
            new_clusterer = MoodClusterer()
            new_clusterer.load_model(tmp_path)
            
            assert new_clusterer.is_fitted
            assert new_clusterer.algorithm == clusterer.algorithm
            assert new_clusterer.random_state == clusterer.random_state
            assert np.array_equal(new_clusterer.cluster_labels, clusterer.cluster_labels)
            
        finally:
            # Clean up
            if Path(tmp_path).exists():
                os.unlink(tmp_path)
    
    def test_find_optimal_clusters(self, sample_data):
        """Test finding optimal number of clusters."""
        clusterer = MoodClusterer(algorithm='kmeans', random_state=42)
        optimal_params = clusterer.find_optimal_clusters(sample_data, max_clusters=10, min_clusters=2)
        
        assert 'optimal_clusters' in optimal_params
        assert 'silhouette_score' in optimal_params
        assert 'calinski_harabasz_score' in optimal_params
        assert 'davies_bouldin_score' in optimal_params
        assert 'all_results' in optimal_params
        
        assert 2 <= optimal_params['optimal_clusters'] <= 10
        assert optimal_params['silhouette_score'] > 0
    
    def test_mood_label_generation(self, sample_data):
        """Test mood label generation."""
        clusterer = MoodClusterer(algorithm='kmeans', random_state=42)
        clusterer.fit(sample_data, n_clusters=3)
        
        assert clusterer.mood_labels is not None
        assert len(clusterer.mood_labels) == 3
        
        # Check that all clusters have mood labels
        for cluster_id in range(3):
            assert cluster_id in clusterer.mood_labels
            assert isinstance(clusterer.mood_labels[cluster_id], str)
            assert len(clusterer.mood_labels[cluster_id]) > 0
    
    def test_cluster_statistics(self, sample_data):
        """Test cluster statistics calculation."""
        clusterer = MoodClusterer(algorithm='kmeans', random_state=42)
        clusterer.fit(sample_data, n_clusters=3)
        
        assert clusterer.cluster_stats is not None
        assert len(clusterer.cluster_stats) == 3
        
        for stat in clusterer.cluster_stats:
            assert 'cluster_id' in stat
            assert 'size' in stat
            assert 'percentage' in stat
            assert 'mean_features' in stat
            assert 'std_features' in stat
            
            assert stat['size'] > 0
            assert 0 <= stat['percentage'] <= 100
            assert len(stat['mean_features']) == sample_data.shape[1]
            assert len(stat['std_features']) == sample_data.shape[1]


class TestClusteringQuality:
    """Test cases for clustering quality evaluation."""
    
    def test_evaluate_clustering_quality(self):
        """Test clustering quality evaluation."""
        # Create data with clear clusters
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X[:50] += [2, 0, 0, 0, 0]
        X[50:] += [0, 2, 0, 0, 0]
        
        # Create labels
        labels = np.array([0] * 50 + [1] * 50)
        
        metrics = evaluate_clustering_quality(X, labels)
        
        assert 'silhouette_score' in metrics
        assert 'calinski_harabasz_score' in metrics
        assert 'davies_bouldin_score' in metrics
        
        assert metrics['silhouette_score'] > 0
        assert metrics['calinski_harabasz_score'] > 0
        assert metrics['davies_bouldin_score'] > 0
    
    def test_evaluate_clustering_quality_single_cluster(self):
        """Test evaluation with single cluster."""
        X = np.random.randn(100, 5)
        labels = np.zeros(100)  # All same cluster
        
        metrics = evaluate_clustering_quality(X, labels)
        
        assert metrics['silhouette_score'] == -1.0
        assert metrics['calinski_harabasz_score'] == 0.0
        assert metrics['davies_bouldin_score'] == float('inf')


if __name__ == "__main__":
    pytest.main([__file__])
