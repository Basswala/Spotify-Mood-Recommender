"""
Nearest neighbors recommendation engine for mood-based song recommendations.

This module implements a recommendation system that finds similar songs
within the same mood cluster using various distance metrics and algorithms.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, BallTree, KDTree
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import joblib
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MoodRecommender:
    """
    Mood-based song recommendation system using nearest neighbors.
    
    Recommends songs within the same mood cluster based on audio feature similarity,
    with support for multiple distance metrics and algorithms.
    """
    
    def __init__(self, algorithm: str = 'ball_tree', metric: str = 'euclidean', 
                 n_neighbors: int = 20, random_state: int = 42):
        """
        Initialize the mood recommender.
        
        Args:
            algorithm: Nearest neighbors algorithm ('ball_tree', 'kd_tree', 'brute')
            metric: Distance metric ('euclidean', 'cosine', 'manhattan', 'minkowski')
            n_neighbors: Number of neighbors to consider for recommendations
            random_state: Random state for reproducibility
            
        Example:
            >>> recommender = MoodRecommender(algorithm='ball_tree', metric='cosine')
        """
        self.algorithm = algorithm
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        
        self.nn_model = None
        self.feature_data = None
        self.track_data = None
        self.cluster_labels = None
        self.is_fitted = False
        
        logger.info(f"Initialized MoodRecommender with {algorithm} algorithm and {metric} metric")
    
    def _create_nn_model(self) -> NearestNeighbors:
        """Create nearest neighbors model with specified parameters."""
        return NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 to exclude the query point itself
            algorithm=self.algorithm,
            metric=self.metric
        )
    
    def fit(self, X: np.ndarray, track_data: pd.DataFrame, 
            cluster_labels: np.ndarray) -> 'MoodRecommender':
        """
        Fit the recommendation model to the data.
        
        Args:
            X: Feature matrix (scaled audio features)
            track_data: DataFrame with track information (track_id, track_name, artists, etc.)
            cluster_labels: Cluster labels for each track
            
        Returns:
            Self for method chaining
            
        Example:
            >>> recommender = MoodRecommender()
            >>> recommender.fit(X, track_df, cluster_labels)
        """
        try:
            logger.info("Fitting mood recommendation model")
            
            # Validate inputs
            if len(X) != len(track_data) or len(X) != len(cluster_labels):
                raise ValueError("Feature matrix, track data, and cluster labels must have same length")
            
            # Store data
            self.feature_data = X.copy()
            self.track_data = track_data.copy()
            self.cluster_labels = cluster_labels.copy()
            
            # Create and fit nearest neighbors model
            self.nn_model = self._create_nn_model()
            self.nn_model.fit(X)
            
            self.is_fitted = True
            logger.info(f"Recommendation model fitted with {len(X)} tracks")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting recommendation model: {str(e)}")
            raise
    
    def recommend_similar_songs(self, track_id: str, n_recommendations: int = 10,
                              same_mood_only: bool = True) -> List[Dict[str, Any]]:
        """
        Recommend similar songs for a given track.
        
        Args:
            track_id: Spotify track ID
            n_recommendations: Number of recommendations to return
            same_mood_only: Whether to only recommend songs from the same mood cluster
            
        Returns:
            List of recommendation dictionaries with track info and similarity scores
            
        Example:
            >>> recommender = MoodRecommender()
            >>> recommender.fit(X, track_df, cluster_labels)
            >>> recommendations = recommender.recommend_similar_songs('4iV5W9uYEdYUVa79Axb7Rh')
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        try:
            # Find the track in the dataset
            track_idx = self._find_track_index(track_id)
            if track_idx is None:
                logger.warning(f"Track {track_id} not found in dataset")
                return []
            
            # Get track features
            track_features = self.feature_data[track_idx].reshape(1, -1)
            track_cluster = self.cluster_labels[track_idx]
            
            # Find nearest neighbors
            distances, indices = self.nn_model.kneighbors(track_features)
            
            # Filter by mood if requested
            if same_mood_only:
                same_mood_mask = self.cluster_labels[indices[0]] == track_cluster
                filtered_indices = indices[0][same_mood_mask]
                filtered_distances = distances[0][same_mood_mask]
            else:
                filtered_indices = indices[0]
                filtered_distances = distances[0]
            
            # Remove the query track itself
            query_mask = filtered_indices != track_idx
            filtered_indices = filtered_indices[query_mask]
            filtered_distances = filtered_distances[query_mask]
            
            # Limit to requested number of recommendations
            n_recommendations = min(n_recommendations, len(filtered_indices))
            top_indices = filtered_indices[:n_recommendations]
            top_distances = filtered_distances[:n_recommendations]
            
            # Create recommendations
            recommendations = []
            for idx, distance in zip(top_indices, top_distances):
                track_info = self.track_data.iloc[idx].to_dict()
                track_info['similarity_score'] = 1.0 / (1.0 + distance)  # Convert distance to similarity
                track_info['distance'] = distance
                recommendations.append(track_info)
            
            logger.info(f"Generated {len(recommendations)} recommendations for track {track_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for track {track_id}: {str(e)}")
            return []
    
    def recommend_by_mood(self, mood_cluster: int, n_recommendations: int = 20,
                         exclude_tracks: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Recommend songs from a specific mood cluster.
        
        Args:
            mood_cluster: Cluster ID for the desired mood
            n_recommendations: Number of recommendations to return
            exclude_tracks: List of track IDs to exclude from recommendations
            
        Returns:
            List of recommendation dictionaries
            
        Example:
            >>> recommender = MoodRecommender()
            >>> recommendations = recommender.recommend_by_mood(mood_cluster=2, n_recommendations=15)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        try:
            # Filter tracks by mood cluster
            mood_mask = self.cluster_labels == mood_cluster
            mood_indices = np.where(mood_mask)[0]
            
            if len(mood_indices) == 0:
                logger.warning(f"No tracks found for mood cluster {mood_cluster}")
                return []
            
            # Exclude specified tracks
            if exclude_tracks:
                exclude_mask = ~self.track_data['track_id'].isin(exclude_tracks)
                mood_indices = mood_indices[exclude_mask[mood_indices]]
            
            # Limit to requested number of recommendations
            n_recommendations = min(n_recommendations, len(mood_indices))
            selected_indices = np.random.choice(mood_indices, size=n_recommendations, replace=False)
            
            # Create recommendations
            recommendations = []
            for idx in selected_indices:
                track_info = self.track_data.iloc[idx].to_dict()
                track_info['similarity_score'] = 1.0  # All tracks in same mood have equal similarity
                track_info['distance'] = 0.0
                recommendations.append(track_info)
            
            logger.info(f"Generated {len(recommendations)} recommendations for mood cluster {mood_cluster}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating mood recommendations: {str(e)}")
            return []
    
    def find_similar_tracks(self, track_features: np.ndarray, n_recommendations: int = 10,
                          mood_cluster: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Find similar tracks based on audio features.
        
        Args:
            track_features: Audio features array
            n_recommendations: Number of recommendations to return
            mood_cluster: Optional mood cluster to filter by
            
        Returns:
            List of recommendation dictionaries
            
        Example:
            >>> recommender = MoodRecommender()
            >>> features = np.array([0.8, 0.7, 0.6, 120, -5, 0.1, 0.2, 0.0, 0.1])
            >>> recommendations = recommender.find_similar_tracks(features)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar tracks")
        
        try:
            # Ensure features are in correct shape
            if track_features.ndim == 1:
                track_features = track_features.reshape(1, -1)
            
            # Find nearest neighbors
            distances, indices = self.nn_model.kneighbors(track_features)
            
            # Filter by mood cluster if specified
            if mood_cluster is not None:
                same_mood_mask = self.cluster_labels[indices[0]] == mood_cluster
                filtered_indices = indices[0][same_mood_mask]
                filtered_distances = distances[0][same_mood_mask]
            else:
                filtered_indices = indices[0]
                filtered_distances = distances[0]
            
            # Limit to requested number of recommendations
            n_recommendations = min(n_recommendations, len(filtered_indices))
            top_indices = filtered_indices[:n_recommendations]
            top_distances = filtered_distances[:n_recommendations]
            
            # Create recommendations
            recommendations = []
            for idx, distance in zip(top_indices, top_distances):
                track_info = self.track_data.iloc[idx].to_dict()
                track_info['similarity_score'] = 1.0 / (1.0 + distance)
                track_info['distance'] = distance
                recommendations.append(track_info)
            
            logger.info(f"Found {len(recommendations)} similar tracks")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error finding similar tracks: {str(e)}")
            return []
    
    def _find_track_index(self, track_id: str) -> Optional[int]:
        """Find the index of a track in the dataset."""
        try:
            if 'track_id' in self.track_data.columns:
                matches = self.track_data['track_id'] == track_id
                if matches.any():
                    return matches.idxmax()
            return None
        except Exception as e:
            logger.error(f"Error finding track index for {track_id}: {str(e)}")
            return None
    
    def get_track_cluster(self, track_id: str) -> Optional[int]:
        """
        Get the mood cluster for a specific track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Cluster ID or None if track not found
            
        Example:
            >>> cluster = recommender.get_track_cluster('4iV5W9uYEdYUVa79Axb7Rh')
        """
        try:
            track_idx = self._find_track_index(track_id)
            if track_idx is not None:
                return int(self.cluster_labels[track_idx])
            return None
        except Exception as e:
            logger.error(f"Error getting track cluster: {str(e)}")
            return None
    
    def get_cluster_tracks(self, cluster_id: int) -> List[Dict[str, Any]]:
        """
        Get all tracks in a specific mood cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            List of track dictionaries
            
        Example:
            >>> tracks = recommender.get_cluster_tracks(2)
        """
        try:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            tracks = []
            for idx in cluster_indices:
                track_info = self.track_data.iloc[idx].to_dict()
                tracks.append(track_info)
            
            logger.info(f"Found {len(tracks)} tracks in cluster {cluster_id}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error getting cluster tracks: {str(e)}")
            return []
    
    def save_model(self, file_path: str) -> None:
        """
        Save the fitted recommendation model to disk.
        
        Args:
            file_path: Path to save the model
            
        Example:
            >>> recommender = MoodRecommender()
            >>> recommender.fit(X, track_df, cluster_labels)
            >>> recommender.save_model('models/recommender.pkl')
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'nn_model': self.nn_model,
                'feature_data': self.feature_data,
                'track_data': self.track_data,
                'cluster_labels': self.cluster_labels,
                'algorithm': self.algorithm,
                'metric': self.metric,
                'n_neighbors': self.n_neighbors,
                'random_state': self.random_state,
                'is_fitted': self.is_fitted
            }
            
            joblib.dump(model_data, file_path)
            logger.info(f"Recommendation model saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving recommendation model: {str(e)}")
            raise
    
    def load_model(self, file_path: str) -> None:
        """
        Load a fitted recommendation model from disk.
        
        Args:
            file_path: Path to the model file
            
        Example:
            >>> recommender = MoodRecommender()
            >>> recommender.load_model('models/recommender.pkl')
        """
        try:
            model_data = joblib.load(file_path)
            
            self.nn_model = model_data['nn_model']
            self.feature_data = model_data['feature_data']
            self.track_data = model_data['track_data']
            self.cluster_labels = model_data['cluster_labels']
            self.algorithm = model_data['algorithm']
            self.metric = model_data['metric']
            self.n_neighbors = model_data['n_neighbors']
            self.random_state = model_data['random_state']
            self.is_fitted = model_data['is_fitted']
            
            logger.info(f"Recommendation model loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading recommendation model: {str(e)}")
            raise


def create_recommendation_summary(recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create a summary of recommendation results.
    
    Args:
        recommendations: List of recommendation dictionaries
        
    Returns:
        Dictionary with recommendation summary statistics
        
    Example:
        >>> summary = create_recommendation_summary(recommendations)
        >>> print(summary['avg_similarity'])
    """
    try:
        if not recommendations:
            return {
                'count': 0,
                'avg_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0,
                'avg_distance': 0.0
            }
        
        similarities = [rec.get('similarity_score', 0.0) for rec in recommendations]
        distances = [rec.get('distance', 0.0) for rec in recommendations]
        
        summary = {
            'count': len(recommendations),
            'avg_similarity': np.mean(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'avg_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
        
        logger.info(f"Created recommendation summary: {summary}")
        return summary
        
    except Exception as e:
        logger.error(f"Error creating recommendation summary: {str(e)}")
        return {}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load and process data
    import pandas as pd
    from features import AudioFeatureProcessor
    from cluster import MoodClusterer
    
    df = pd.read_csv('data/dataset.csv')
    processor = AudioFeatureProcessor()
    df_processed = processor.fit_transform(df)
    
    # Get feature columns for clustering
    feature_cols = processor.get_audio_feature_columns()
    available_cols = [col for col in feature_cols if col in df_processed.columns]
    X = df_processed[available_cols].values
    
    # Fit clustering model
    clusterer = MoodClusterer(algorithm='kmeans')
    clusterer.fit(X, auto_tune=True)
    cluster_labels = clusterer.cluster_labels
    
    # Initialize and fit recommender
    recommender = MoodRecommender(algorithm='ball_tree', metric='cosine')
    recommender.fit(X, df_processed, cluster_labels)
    
    # Test recommendations
    if len(df_processed) > 0:
        test_track_id = df_processed.iloc[0]['track_id']
        recommendations = recommender.recommend_similar_songs(test_track_id, n_recommendations=5)
        
        print(f"Recommendations for track {test_track_id}:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['track_name']} by {rec['artists']} (similarity: {rec['similarity_score']:.3f})")
    
    # Save model
    recommender.save_model('models/recommender.pkl')
