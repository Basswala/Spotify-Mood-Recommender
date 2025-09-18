"""
Visualization module for mood-based song clustering and recommendations.

This module provides various visualization tools including radar charts,
PCA/UMAP scatter plots, and cluster analysis visualizations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MoodVisualizer:
    """
    Visualization toolkit for mood-based song clustering analysis.
    
    Provides methods for creating radar charts, scatter plots, and
    cluster analysis visualizations.
    """
    
    def __init__(self, style: str = 'seaborn', color_palette: str = 'viridis'):
        """
        Initialize the mood visualizer.
        
        Args:
            style: Matplotlib style ('seaborn', 'ggplot', 'default')
            color_palette: Color palette for plots ('viridis', 'plasma', 'inferno')
            
        Example:
            >>> visualizer = MoodVisualizer(style='seaborn', color_palette='viridis')
        """
        self.style = style
        self.color_palette = color_palette
        
        # Set matplotlib style
        try:
            plt.style.use(style)
        except OSError:
            # If style is not available, use default
            pass

        # Set color palette
        sns.set_palette(color_palette)
        
        logger.info(f"Initialized MoodVisualizer with {style} style and {color_palette} palette")
    
    def create_radar_chart(self, features: Dict[str, float], title: str = "Audio Features",
                          save_path: Optional[str] = None) -> go.Figure:
        """
        Create a radar chart for audio features.
        
        Args:
            features: Dictionary of feature names and values
            title: Chart title
            save_path: Optional path to save the chart
            
        Returns:
            Plotly figure object
            
        Example:
            >>> features = {'danceability': 0.8, 'energy': 0.7, 'valence': 0.6}
            >>> fig = visualizer.create_radar_chart(features, "Song Features")
        """
        try:
            # Define feature categories and their ranges
            feature_categories = {
                'danceability': (0, 1),
                'energy': (0, 1),
                'valence': (0, 1),
                'speechiness': (0, 1),
                'acousticness': (0, 1),
                'instrumentalness': (0, 1),
                'liveness': (0, 1),
                'tempo': (0, 200),  # BPM
                'loudness': (-60, 0)  # dB
            }
            
            # Filter features that are in our categories
            available_features = {k: v for k, v in features.items() if k in feature_categories}
            
            if not available_features:
                logger.warning("No valid features found for radar chart")
                return go.Figure()
            
            # Normalize features to 0-1 range
            normalized_features = {}
            for feature, value in available_features.items():
                min_val, max_val = feature_categories[feature]
                normalized_value = (value - min_val) / (max_val - min_val)
                normalized_features[feature] = normalized_value
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=list(normalized_features.values()),
                theta=list(normalized_features.keys()),
                fill='toself',
                name='Audio Features',
                line_color='blue',
                fillcolor='rgba(0, 100, 200, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title=title,
                showlegend=True,
                font=dict(size=12)
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Radar chart saved to {save_path}")
            
            logger.info("Radar chart created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {str(e)}")
            return go.Figure()
    
    def create_cluster_scatter_plot(self, X: np.ndarray, cluster_labels: np.ndarray,
                                   mood_labels: Dict[int, str], method: str = 'pca',
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create a scatter plot of clusters using dimensionality reduction.
        
        Args:
            X: Feature matrix
            cluster_labels: Cluster labels
            mood_labels: Dictionary mapping cluster IDs to mood names
            method: Dimensionality reduction method ('pca', 'tsne', 'umap')
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
            
        Example:
            >>> fig = visualizer.create_cluster_scatter_plot(X, labels, mood_labels, method='umap')
        """
        try:
            logger.info(f"Creating {method.upper()} scatter plot for {len(np.unique(cluster_labels))} clusters")
            
            # Apply dimensionality reduction
            if method.lower() == 'pca':
                reducer = PCA(n_components=2, random_state=42)
                embedding = reducer.fit_transform(X)
                explained_variance = reducer.explained_variance_ratio_
                logger.info(f"PCA explained variance: {explained_variance}")
                
            elif method.lower() == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                embedding = reducer.fit_transform(X)
                
            elif method.lower() == 'umap':
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
                embedding = reducer.fit_transform(X)
                
            else:
                raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
            # Create DataFrame for plotting
            plot_df = pd.DataFrame({
                'x': embedding[:, 0],
                'y': embedding[:, 1],
                'cluster': cluster_labels,
                'mood': [mood_labels.get(label, f'Cluster {label}') for label in cluster_labels]
            })
            
            # Create scatter plot
            fig = px.scatter(
                plot_df, x='x', y='y', color='mood',
                title=f'Mood Clusters - {method.upper()} Visualization',
                labels={'x': f'{method.upper()} Component 1', 'y': f'{method.upper()} Component 2'},
                hover_data=['cluster']
            )
            
            fig.update_layout(
                width=800,
                height=600,
                showlegend=True,
                font=dict(size=12)
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Scatter plot saved to {save_path}")
            
            logger.info(f"{method.upper()} scatter plot created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cluster scatter plot: {str(e)}")
            return go.Figure()
    
    def create_feature_distribution_plot(self, df: pd.DataFrame, cluster_labels: np.ndarray,
                                       mood_labels: Dict[int, str],
                                       save_path: Optional[str] = None) -> go.Figure:
        """
        Create distribution plots for audio features across clusters.
        
        Args:
            df: DataFrame with audio features
            cluster_labels: Cluster labels
            mood_labels: Dictionary mapping cluster IDs to mood names
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
            
        Example:
            >>> fig = visualizer.create_feature_distribution_plot(df, labels, mood_labels)
        """
        try:
            # Get audio feature columns
            feature_cols = [
                'danceability', 'energy', 'valence', 'tempo', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness'
            ]
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if not available_cols:
                logger.warning("No audio feature columns found")
                return go.Figure()
            
            # Add cluster and mood information
            plot_df = df[available_cols].copy()
            plot_df['cluster'] = cluster_labels
            plot_df['mood'] = [mood_labels.get(label, f'Cluster {label}') for label in cluster_labels]
            
            # Create subplots
            n_features = len(available_cols)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=available_cols,
                specs=[[{'type': 'box'} for _ in range(n_cols)] for _ in range(n_rows)]
            )
            
            # Create box plots for each feature
            for i, feature in enumerate(available_cols):
                row = i // n_cols + 1
                col = i % n_cols + 1
                
                for mood in plot_df['mood'].unique():
                    mood_data = plot_df[plot_df['mood'] == mood][feature]
                    
                    fig.add_trace(
                        go.Box(
                            y=mood_data,
                            name=mood,
                            showlegend=(i == 0),  # Only show legend for first subplot
                            boxpoints='outliers'
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(
                title="Audio Feature Distributions by Mood Cluster",
                height=200 * n_rows,
                showlegend=True,
                font=dict(size=10)
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Distribution plot saved to {save_path}")
            
            logger.info("Feature distribution plot created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature distribution plot: {str(e)}")
            return go.Figure()
    
    def create_cluster_heatmap(self, df: pd.DataFrame, cluster_labels: np.ndarray,
                              mood_labels: Dict[int, str],
                              save_path: Optional[str] = None) -> go.Figure:
        """
        Create a heatmap showing average feature values for each cluster.
        
        Args:
            df: DataFrame with audio features
            cluster_labels: Cluster labels
            mood_labels: Dictionary mapping cluster IDs to mood names
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
            
        Example:
            >>> fig = visualizer.create_cluster_heatmap(df, labels, mood_labels)
        """
        try:
            # Get audio feature columns
            feature_cols = [
                'danceability', 'energy', 'valence', 'tempo', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness'
            ]
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if not available_cols:
                logger.warning("No audio feature columns found")
                return go.Figure()
            
            # Calculate mean features for each cluster
            cluster_means = []
            cluster_names = []
            
            for cluster_id in np.unique(cluster_labels):
                if cluster_id == -1:  # Skip noise points
                    continue
                
                cluster_mask = cluster_labels == cluster_id
                cluster_data = df[available_cols][cluster_mask]
                cluster_mean = cluster_data.mean()
                
                cluster_means.append(cluster_mean.values)
                cluster_names.append(mood_labels.get(cluster_id, f'Cluster {cluster_id}'))
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cluster_means,
                x=available_cols,
                y=cluster_names,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Feature Value")
            ))
            
            fig.update_layout(
                title="Average Audio Features by Mood Cluster",
                xaxis_title="Audio Features",
                yaxis_title="Mood Clusters",
                width=800,
                height=400,
                font=dict(size=12)
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Heatmap saved to {save_path}")
            
            logger.info("Cluster heatmap created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cluster heatmap: {str(e)}")
            return go.Figure()
    
    def create_recommendation_visualization(self, track_features: Dict[str, float],
                                           recommendations: List[Dict[str, Any]],
                                           save_path: Optional[str] = None) -> go.Figure:
        """
        Create a visualization comparing a track with its recommendations.
        
        Args:
            track_features: Features of the query track
            recommendations: List of recommendation dictionaries
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
            
        Example:
            >>> fig = visualizer.create_recommendation_visualization(track_features, recommendations)
        """
        try:
            # Prepare data for comparison
            feature_cols = [
                'danceability', 'energy', 'valence', 'tempo', 'loudness',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness'
            ]
            
            # Get query track features
            query_features = [track_features.get(col, 0) for col in feature_cols]
            
            # Get recommendation features
            rec_features = []
            rec_names = []
            
            for rec in recommendations[:5]:  # Limit to top 5 recommendations
                rec_feature_values = [rec.get(col, 0) for col in feature_cols]
                rec_features.append(rec_feature_values)
                rec_names.append(f"{rec.get('track_name', 'Unknown')} - {rec.get('artists', 'Unknown')}")
            
            # Create radar chart comparison
            fig = go.Figure()
            
            # Add query track
            fig.add_trace(go.Scatterpolar(
                r=query_features,
                theta=feature_cols,
                fill='toself',
                name='Query Track',
                line_color='red',
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
            
            # Add recommendations
            colors = ['blue', 'green', 'orange', 'purple', 'brown']
            for i, (rec_feature, rec_name) in enumerate(zip(rec_features, rec_names)):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatterpolar(
                    r=rec_feature,
                    theta=feature_cols,
                    fill='toself',
                    name=rec_name,
                    line_color=color,
                    fillcolor=f'rgba({hash(color) % 255}, {hash(color) % 255}, {hash(color) % 255}, 0.2)'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Track vs Recommendations - Audio Feature Comparison",
                showlegend=True,
                font=dict(size=10)
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Recommendation visualization saved to {save_path}")
            
            logger.info("Recommendation visualization created successfully")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating recommendation visualization: {str(e)}")
            return go.Figure()
    
    def save_all_visualizations(self, X: np.ndarray, df: pd.DataFrame,
                               cluster_labels: np.ndarray, mood_labels: Dict[int, str],
                               output_dir: str = 'visualizations') -> None:
        """
        Save all visualization types to files.
        
        Args:
            X: Feature matrix
            df: DataFrame with audio features
            cluster_labels: Cluster labels
            mood_labels: Dictionary mapping cluster IDs to mood names
            output_dir: Directory to save visualizations
            
        Example:
            >>> visualizer.save_all_visualizations(X, df, labels, mood_labels)
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Create scatter plots
            for method in ['pca', 'tsne', 'umap']:
                try:
                    fig = self.create_cluster_scatter_plot(
                        X, cluster_labels, mood_labels, method=method,
                        save_path=f"{output_dir}/cluster_scatter_{method}.html"
                    )
                except Exception as e:
                    logger.warning(f"Failed to create {method} scatter plot: {str(e)}")
            
            # Create distribution plot
            try:
                fig = self.create_feature_distribution_plot(
                    df, cluster_labels, mood_labels,
                    save_path=f"{output_dir}/feature_distributions.html"
                )
            except Exception as e:
                logger.warning(f"Failed to create distribution plot: {str(e)}")
            
            # Create heatmap
            try:
                fig = self.create_cluster_heatmap(
                    df, cluster_labels, mood_labels,
                    save_path=f"{output_dir}/cluster_heatmap.html"
                )
            except Exception as e:
                logger.warning(f"Failed to create heatmap: {str(e)}")
            
            logger.info(f"All visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving visualizations: {str(e)}")


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
    
    # Create visualizations
    visualizer = MoodVisualizer()
    
    # Create scatter plot
    fig = visualizer.create_cluster_scatter_plot(
        X, clusterer.cluster_labels, clusterer.mood_labels, method='umap'
    )
    
    # Save all visualizations
    visualizer.save_all_visualizations(
        X, df_processed, clusterer.cluster_labels, clusterer.mood_labels
    )
