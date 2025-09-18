"""
Unit tests for the nn module.

Tests the MoodRecommender class and related functionality.
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

from nn import MoodRecommender, create_recommendation_summary


class TestMoodRecommender:
    """Test cases for MoodRecommender class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Create feature matrix
        X = np.random.randn(n_samples, n_features)
        
        # Create track data
        track_data = pd.DataFrame({
            'track_id': [f'track_{i}' for i in range(n_samples)],
            'track_name': [f'Song {i}' for i in range(n_samples)],
            'artists': [f'Artist {i}' for i in range(n_samples)],
            'album_name': [f'Album {i}' for i in range(n_samples)],
            'popularity': np.random.randint(0, 100, n_samples)
        })
        
        # Create cluster labels
        cluster_labels = np.random.randint(0, 5, n_samples)
        
        return X, track_data, cluster_labels
    
    def test_initialization(self):
        """Test recommender initialization."""
        recommender = MoodRecommender(algorithm='ball_tree', metric='euclidean', n_neighbors=20)
        assert recommender.algorithm == 'ball_tree'
        assert recommender.metric == 'euclidean'
        assert recommender.n_neighbors == 20
        assert not recommender.is_fitted
    
    def test_fit(self, sample_data):
        """Test fitting the recommender."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender(algorithm='ball_tree', metric='euclidean')
        recommender.fit(X, track_data, cluster_labels)
        
        assert recommender.is_fitted
        assert recommender.nn_model is not None
        assert recommender.feature_data is not None
        assert recommender.track_data is not None
        assert recommender.cluster_labels is not None
    
    def test_fit_mismatched_data(self, sample_data):
        """Test fitting with mismatched data lengths."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender()
        
        # Create mismatched data
        wrong_track_data = track_data.iloc[:-10]  # Remove 10 rows
        
        with pytest.raises(ValueError, match="Feature matrix, track data, and cluster labels must have same length"):
            recommender.fit(X, wrong_track_data, cluster_labels)
    
    def test_recommend_similar_songs(self, sample_data):
        """Test recommending similar songs."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender(algorithm='ball_tree', metric='euclidean')
        recommender.fit(X, track_data, cluster_labels)
        
        # Test with first track
        track_id = track_data.iloc[0]['track_id']
        recommendations = recommender.recommend_similar_songs(track_id, n_recommendations=5)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        
        # Check recommendation structure
        if recommendations:
            rec = recommendations[0]
            assert 'track_id' in rec
            assert 'track_name' in rec
            assert 'artists' in rec
            assert 'similarity_score' in rec
            assert 'distance' in rec
    
    def test_recommend_similar_songs_not_found(self, sample_data):
        """Test recommending for non-existent track."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender()
        recommender.fit(X, track_data, cluster_labels)
        
        recommendations = recommender.recommend_similar_songs('non_existent_track', n_recommendations=5)
        assert recommendations == []
    
    def test_recommend_similar_songs_not_fitted(self, sample_data):
        """Test recommending without fitting."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender()
        
        with pytest.raises(ValueError, match="Model must be fitted before making recommendations"):
            recommender.recommend_similar_songs('track_0', n_recommendations=5)
    
    def test_recommend_by_mood(self, sample_data):
        """Test recommending by mood cluster."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender()
        recommender.fit(X, track_data, cluster_labels)
        
        # Test with cluster 0
        recommendations = recommender.recommend_by_mood(0, n_recommendations=10)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 10
        
        # Check that all recommendations are from the same cluster
        if recommendations:
            for rec in recommendations:
                assert 'track_id' in rec
                assert 'similarity_score' in rec
                assert rec['similarity_score'] == 1.0  # All same mood have equal similarity
    
    def test_recommend_by_mood_empty_cluster(self, sample_data):
        """Test recommending from empty cluster."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender()
        recommender.fit(X, track_data, cluster_labels)
        
        # Test with non-existent cluster
        recommendations = recommender.recommend_by_mood(999, n_recommendations=10)
        assert recommendations == []
    
    def test_find_similar_tracks(self, sample_data):
        """Test finding similar tracks by features."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender()
        recommender.fit(X, track_data, cluster_labels)
        
        # Test with sample features
        test_features = X[0]  # Use first track's features
        recommendations = recommender.find_similar_tracks(test_features, n_recommendations=5)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        
        if recommendations:
            rec = recommendations[0]
            assert 'track_id' in rec
            assert 'similarity_score' in rec
            assert 'distance' in rec
    
    def test_find_similar_tracks_not_fitted(self, sample_data):
        """Test finding similar tracks without fitting."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender()
        
        with pytest.raises(ValueError, match="Model must be fitted before finding similar tracks"):
            recommender.find_similar_tracks(X[0], n_recommendations=5)
    
    def test_get_track_cluster(self, sample_data):
        """Test getting track cluster."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender()
        recommender.fit(X, track_data, cluster_labels)
        
        # Test with first track
        track_id = track_data.iloc[0]['track_id']
        cluster = recommender.get_track_cluster(track_id)
        
        assert cluster is not None
        assert isinstance(cluster, int)
        assert 0 <= cluster < len(np.unique(cluster_labels))
    
    def test_get_track_cluster_not_found(self, sample_data):
        """Test getting cluster for non-existent track."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender()
        recommender.fit(X, track_data, cluster_labels)
        
        cluster = recommender.get_track_cluster('non_existent_track')
        assert cluster is None
    
    def test_get_cluster_tracks(self, sample_data):
        """Test getting tracks in a cluster."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender()
        recommender.fit(X, track_data, cluster_labels)
        
        # Test with cluster 0
        tracks = recommender.get_cluster_tracks(0)
        
        assert isinstance(tracks, list)
        assert len(tracks) > 0
        
        # Check track structure
        if tracks:
            track = tracks[0]
            assert 'track_id' in track
            assert 'track_name' in track
            assert 'artists' in track
    
    def test_save_and_load_model(self, sample_data):
        """Test saving and loading model."""
        X, track_data, cluster_labels = sample_data
        recommender = MoodRecommender(algorithm='ball_tree', metric='euclidean')
        recommender.fit(X, track_data, cluster_labels)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save model
            recommender.save_model(tmp_path)
            assert Path(tmp_path).exists()
            
            # Load model
            new_recommender = MoodRecommender()
            new_recommender.load_model(tmp_path)
            
            assert new_recommender.is_fitted
            assert new_recommender.algorithm == recommender.algorithm
            assert new_recommender.metric == recommender.metric
            assert np.array_equal(new_recommender.cluster_labels, recommender.cluster_labels)
            
        finally:
            # Clean up
            if Path(tmp_path).exists():
                os.unlink(tmp_path)
    
    def test_different_algorithms(self, sample_data):
        """Test different nearest neighbors algorithms."""
        X, track_data, cluster_labels = sample_data
        algorithms = ['ball_tree', 'kd_tree', 'brute']
        
        for algorithm in algorithms:
            recommender = MoodRecommender(algorithm=algorithm)
            recommender.fit(X, track_data, cluster_labels)
            
            assert recommender.is_fitted
            assert recommender.algorithm == algorithm
    
    def test_different_metrics(self, sample_data):
        """Test different distance metrics."""
        X, track_data, cluster_labels = sample_data
        metrics = ['euclidean', 'manhattan', 'minkowski']
        
        for metric in metrics:
            recommender = MoodRecommender(metric=metric)
            recommender.fit(X, track_data, cluster_labels)
            
            assert recommender.is_fitted
            assert recommender.metric == metric


class TestRecommendationSummary:
    """Test cases for recommendation summary functionality."""
    
    def test_create_recommendation_summary(self):
        """Test creating recommendation summary."""
        recommendations = [
            {'similarity_score': 0.8, 'distance': 0.2},
            {'similarity_score': 0.6, 'distance': 0.4},
            {'similarity_score': 0.9, 'distance': 0.1}
        ]
        
        summary = create_recommendation_summary(recommendations)
        
        assert 'count' in summary
        assert 'avg_similarity' in summary
        assert 'min_similarity' in summary
        assert 'max_similarity' in summary
        assert 'avg_distance' in summary
        
        assert summary['count'] == 3
        assert summary['avg_similarity'] == 0.7666666666666666  # (0.8 + 0.6 + 0.9) / 3
        assert summary['min_similarity'] == 0.6
        assert summary['max_similarity'] == 0.9
    
    def test_create_recommendation_summary_empty(self):
        """Test creating summary for empty recommendations."""
        recommendations = []
        summary = create_recommendation_summary(recommendations)
        
        assert summary['count'] == 0
        assert summary['avg_similarity'] == 0.0
        assert summary['min_similarity'] == 0.0
        assert summary['max_similarity'] == 0.0
        assert summary['avg_distance'] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
