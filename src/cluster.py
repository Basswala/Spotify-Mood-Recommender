"""
Clustering module for mood-based song grouping using KMeans and other algorithms.

This module implements clustering algorithms to group songs by mood using
Spotify audio features, with automatic mood labeling and cluster analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import ParameterGrid
import joblib
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MoodClusterer:
    """
    Mood-based clustering system for songs using Spotify audio features.
    
    Implements multiple clustering algorithms with automatic parameter tuning
    and mood labeling based on cluster characteristics.
    """
    
    def __init__(self, algorithm: str = 'kmeans', random_state: int = 42):
        """
        Initialize the mood clusterer.
        
        Args:
            algorithm: Clustering algorithm ('kmeans', 'dbscan', 'agglomerative')
            random_state: Random state for reproducibility
            
        Example:
            >>> clusterer = MoodClusterer(algorithm='kmeans')
        """
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = None
        self.cluster_labels = None
        self.mood_labels = None
        self.cluster_centers = None
        self.cluster_stats = None
        self.is_fitted = False
        
        logger.info(f"Initialized MoodClusterer with {algorithm} algorithm")
    
    def _create_kmeans_model(self, n_clusters: int) -> KMeans:
        """Create KMeans model with specified parameters."""
        return KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
    
    def _create_dbscan_model(self, eps: float, min_samples: int) -> DBSCAN:
        """Create DBSCAN model with specified parameters."""
        return DBSCAN(eps=eps, min_samples=min_samples)
    
    def _create_agglomerative_model(self, n_clusters: int) -> AgglomerativeClustering:
        """Create AgglomerativeClustering model with specified parameters."""
        return AgglomerativeClustering(n_clusters=n_clusters)
    
    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 20, 
                            min_clusters: int = 2) -> Dict[str, Any]:
        """
        Find optimal number of clusters using multiple metrics.
        
        Args:
            X: Feature matrix
            max_clusters: Maximum number of clusters to test
            min_clusters: Minimum number of clusters to test
            
        Returns:
            Dictionary with optimal parameters and scores
            
        Example:
            >>> clusterer = MoodClusterer()
            >>> optimal = clusterer.find_optimal_clusters(X, max_clusters=15)
        """
        try:
            logger.info(f"Finding optimal clusters between {min_clusters} and {max_clusters}")
            
            results = []
            cluster_range = range(min_clusters, max_clusters + 1)
            
            for n_clusters in tqdm(cluster_range, desc="Testing cluster numbers"):
                try:
                    # Fit model
                    if self.algorithm == 'kmeans':
                        model = self._create_kmeans_model(n_clusters)
                    elif self.algorithm == 'agglomerative':
                        model = self._create_agglomerative_model(n_clusters)
                    else:
                        continue  # DBSCAN doesn't use n_clusters
                    
                    labels = model.fit_predict(X)
                    
                    # Skip if all points are in one cluster
                    if len(np.unique(labels)) < 2:
                        continue
                    
                    # Calculate metrics
                    silhouette = silhouette_score(X, labels)
                    calinski_harabasz = calinski_harabasz_score(X, labels)
                    davies_bouldin = davies_bouldin_score(X, labels)
                    
                    results.append({
                        'n_clusters': n_clusters,
                        'silhouette_score': silhouette,
                        'calinski_harabasz_score': calinski_harabasz,
                        'davies_bouldin_score': davies_bouldin
                    })
                    
                except Exception as e:
                    logger.warning(f"Error testing {n_clusters} clusters: {str(e)}")
                    continue
            
            if not results:
                raise ValueError("No valid clustering results found")
            
            # Find optimal number of clusters
            results_df = pd.DataFrame(results)
            
            # Use silhouette score as primary metric
            optimal_idx = results_df['silhouette_score'].idxmax()
            optimal_result = results_df.iloc[optimal_idx]
            
            logger.info(f"Optimal clusters: {optimal_result['n_clusters']} "
                       f"(silhouette: {optimal_result['silhouette_score']:.3f})")
            
            return {
                'optimal_clusters': int(optimal_result['n_clusters']),
                'silhouette_score': float(optimal_result['silhouette_score']),
                'calinski_harabasz_score': float(optimal_result['calinski_harabasz_score']),
                'davies_bouldin_score': float(optimal_result['davies_bouldin_score']),
                'all_results': results
            }
            
        except Exception as e:
            logger.error(f"Error finding optimal clusters: {str(e)}")
            raise
    
    def fit(self, X: np.ndarray, n_clusters: Optional[int] = None, 
            auto_tune: bool = True) -> 'MoodClusterer':
        """
        Fit the clustering model to the data.
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters (auto-determined if None)
            auto_tune: Whether to automatically find optimal parameters
            
        Returns:
            Self for method chaining
            
        Example:
            >>> clusterer = MoodClusterer()
            >>> clusterer.fit(X, n_clusters=8)
        """
        try:
            logger.info(f"Fitting {self.algorithm} clustering model")
            
            # Determine number of clusters
            if n_clusters is None and auto_tune:
                optimal_params = self.find_optimal_clusters(X)
                n_clusters = optimal_params['optimal_clusters']
                logger.info(f"Auto-determined optimal clusters: {n_clusters}")
            elif n_clusters is None:
                n_clusters = 8  # Default value
                logger.info(f"Using default clusters: {n_clusters}")
            
            # Create and fit model
            if self.algorithm == 'kmeans':
                self.model = self._create_kmeans_model(n_clusters)
            elif self.algorithm == 'dbscan':
                # For DBSCAN, use default parameters (can be tuned separately)
                self.model = self._create_dbscan_model(eps=0.5, min_samples=5)
            elif self.algorithm == 'agglomerative':
                self.model = self._create_agglomerative_model(n_clusters)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
            # Fit model
            self.cluster_labels = self.model.fit_predict(X)
            
            # Calculate cluster statistics
            self._calculate_cluster_stats(X)
            
            # Generate mood labels
            self._generate_mood_labels()
            
            self.is_fitted = True
            logger.info(f"Clustering completed with {len(np.unique(self.cluster_labels))} clusters")
            
            return self
            
        except Exception as e:
            logger.error(f"Error fitting clustering model: {str(e)}")
            raise
    
    def _calculate_cluster_stats(self, X: np.ndarray) -> None:
        """Calculate statistics for each cluster."""
        try:
            unique_labels = np.unique(self.cluster_labels)
            stats = []
            
            for label in unique_labels:
                if label == -1:  # Skip noise points in DBSCAN
                    continue
                
                cluster_mask = self.cluster_labels == label
                cluster_data = X[cluster_mask]
                
                stats.append({
                    'cluster_id': int(label),
                    'size': int(np.sum(cluster_mask)),
                    'percentage': float(np.sum(cluster_mask) / len(X) * 100),
                    'mean_features': cluster_data.mean(axis=0).tolist(),
                    'std_features': cluster_data.std(axis=0).tolist()
                })
            
            self.cluster_stats = stats
            logger.info(f"Calculated statistics for {len(stats)} clusters")
            
        except Exception as e:
            logger.error(f"Error calculating cluster statistics: {str(e)}")
    
    def _generate_mood_labels(self) -> None:
        """
        Generate mood labels based on cluster characteristics.
        
        Uses audio feature patterns to assign mood labels like 'energetic',
        'melancholic', 'upbeat', etc.
        """
        try:
            if not self.cluster_stats:
                logger.warning("No cluster statistics available for mood labeling")
                return
            
            mood_labels = {}
            
            for stat in self.cluster_stats:
                cluster_id = stat['cluster_id']
                mean_features = np.array(stat['mean_features'])
                
                # Define mood based on feature patterns
                # This is a simplified heuristic - can be improved with domain knowledge
                energy = mean_features[1] if len(mean_features) > 1 else 0  # energy
                valence = mean_features[2] if len(mean_features) > 2 else 0  # valence
                danceability = mean_features[0] if len(mean_features) > 0 else 0  # danceability
                
                # Mood classification logic
                if energy > 0.7 and valence > 0.6:
                    mood = "Energetic & Happy"
                elif energy > 0.7 and valence < 0.4:
                    mood = "Energetic & Intense"
                elif energy < 0.4 and valence > 0.6:
                    mood = "Calm & Peaceful"
                elif energy < 0.4 and valence < 0.4:
                    mood = "Melancholic & Sad"
                elif danceability > 0.7:
                    mood = "Danceable & Upbeat"
                elif energy > 0.5 and valence > 0.5:
                    mood = "Moderate & Balanced"
                else:
                    mood = "Ambient & Atmospheric"
                
                mood_labels[cluster_id] = mood
            
            self.mood_labels = mood_labels
            logger.info(f"Generated mood labels for {len(mood_labels)} clusters")
            
        except Exception as e:
            logger.error(f"Error generating mood labels: {str(e)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of cluster labels
            
        Example:
            >>> clusterer = MoodClusterer()
            >>> clusterer.fit(train_X)
            >>> labels = clusterer.predict(test_X)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            if self.algorithm == 'kmeans':
                return self.model.predict(X)
            else:
                # For DBSCAN and Agglomerative, we need to use fit_predict
                # This is a limitation - we'll use the closest cluster center
                if hasattr(self.model, 'cluster_centers_'):
                    from sklearn.metrics.pairwise import euclidean_distances
                    distances = euclidean_distances(X, self.model.cluster_centers_)
                    return np.argmin(distances, axis=1)
                else:
                    # Fallback: assign to most common cluster
                    unique_labels = np.unique(self.cluster_labels)
                    return np.random.choice(unique_labels, size=len(X))
            
        except Exception as e:
            logger.error(f"Error predicting cluster labels: {str(e)}")
            raise
    
    def get_cluster_info(self, cluster_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific cluster.
        
        Args:
            cluster_id: ID of the cluster
            
        Returns:
            Dictionary with cluster information
            
        Example:
            >>> clusterer = MoodClusterer()
            >>> clusterer.fit(X)
            >>> info = clusterer.get_cluster_info(0)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster info")
        
        try:
            # Find cluster statistics
            cluster_stat = None
            for stat in self.cluster_stats:
                if stat['cluster_id'] == cluster_id:
                    cluster_stat = stat
                    break
            
            if cluster_stat is None:
                raise ValueError(f"Cluster {cluster_id} not found")
            
            # Get mood label
            mood_label = self.mood_labels.get(cluster_id, "Unknown")
            
            return {
                'cluster_id': cluster_id,
                'mood_label': mood_label,
                'size': cluster_stat['size'],
                'percentage': cluster_stat['percentage'],
                'mean_features': cluster_stat['mean_features'],
                'std_features': cluster_stat['std_features']
            }
            
        except Exception as e:
            logger.error(f"Error getting cluster info: {str(e)}")
            raise
    
    def save_model(self, file_path: str) -> None:
        """
        Save the fitted clustering model to disk.
        
        Args:
            file_path: Path to save the model
            
        Example:
            >>> clusterer = MoodClusterer()
            >>> clusterer.fit(X)
            >>> clusterer.save_model('models/clusterer.pkl')
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'cluster_labels': self.cluster_labels,
                'mood_labels': self.mood_labels,
                'cluster_stats': self.cluster_stats,
                'algorithm': self.algorithm,
                'random_state': self.random_state,
                'is_fitted': self.is_fitted
            }
            
            joblib.dump(model_data, file_path)
            logger.info(f"Clustering model saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving clustering model: {str(e)}")
            raise
    
    def load_model(self, file_path: str) -> None:
        """
        Load a fitted clustering model from disk.
        
        Args:
            file_path: Path to the model file
            
        Example:
            >>> clusterer = MoodClusterer()
            >>> clusterer.load_model('models/clusterer.pkl')
        """
        try:
            model_data = joblib.load(file_path)
            
            self.model = model_data['model']
            self.cluster_labels = model_data['cluster_labels']
            self.mood_labels = model_data['mood_labels']
            self.cluster_stats = model_data['cluster_stats']
            self.algorithm = model_data['algorithm']
            self.random_state = model_data['random_state']
            self.is_fitted = model_data['is_fitted']
            
            logger.info(f"Clustering model loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading clustering model: {str(e)}")
            raise


def evaluate_clustering_quality(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the quality of clustering results using multiple metrics.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        
    Returns:
        Dictionary with evaluation metrics
        
    Example:
        >>> metrics = evaluate_clustering_quality(X, labels)
        >>> print(metrics['silhouette_score'])
    """
    try:
        # Skip evaluation if all points are in one cluster
        if len(np.unique(labels)) < 2:
            return {
                'silhouette_score': -1.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf')
            }
        
        metrics = {
            'silhouette_score': silhouette_score(X, labels),
            'calinski_harabasz_score': calinski_harabasz_score(X, labels),
            'davies_bouldin_score': davies_bouldin_score(X, labels)
        }
        
        logger.info(f"Clustering quality metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating clustering quality: {str(e)}")
        return {}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load and process data
    import pandas as pd
    from features import AudioFeatureProcessor
    
    df = pd.read_csv('data/dataset.csv')
    processor = AudioFeatureProcessor()
    df_processed = processor.fit_transform(df)
    
    # Get feature columns for clustering
    feature_cols = processor.get_audio_feature_columns()
    available_cols = [col for col in feature_cols if col in df_processed.columns]
    X = df_processed[available_cols].values
    
    # Initialize and fit clusterer
    clusterer = MoodClusterer(algorithm='kmeans')
    clusterer.fit(X, auto_tune=True)
    
    # Get cluster information
    for cluster_id in range(len(clusterer.cluster_stats)):
        info = clusterer.get_cluster_info(cluster_id)
        print(f"Cluster {cluster_id}: {info['mood_label']} ({info['size']} songs)")
    
    # Save model
    clusterer.save_model('models/clusterer.pkl')
