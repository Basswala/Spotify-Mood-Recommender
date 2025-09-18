"""
Spotify API integration for fetching audio features and track information.

This module handles authentication with Spotify API and provides functions
to fetch audio features for tracks, artists, and playlists.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm.auto import tqdm
import numpy as np

logger = logging.getLogger(__name__)

class SpotifyAPI:
    """
    Spotify API client for fetching audio features and track information.
    
    Handles authentication and provides methods to fetch audio features
    for tracks, artists, and playlists with rate limiting and error handling.
    """
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        """
        Initialize Spotify API client.
        
        Args:
            client_id: Spotify client ID. If None, will try to get from environment.
            client_secret: Spotify client secret. If None, will try to get from environment.
            
        Raises:
            ValueError: If credentials are not provided or invalid.
        """
        try:
            # Get credentials from environment if not provided
            client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                raise ValueError(
                    "Spotify credentials not found. Please set SPOTIFY_CLIENT_ID "
                    "and SPOTIFY_CLIENT_SECRET environment variables or pass them directly."
                )
            
            # Initialize Spotify client
            client_credentials_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
            
            # Test connection
            self.sp.search('test', limit=1)
            logger.info("Spotify API client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Spotify API client: {str(e)}")
            raise
    
    def get_track_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get audio features for a single track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            Dictionary containing audio features and track info, or None if failed
            
        Example:
            >>> api = SpotifyAPI()
            >>> features = api.get_track_features('4iV5W9uYEdYUVa79Axb7Rh')
            >>> print(features['danceability'])
        """
        try:
            # Get track info
            track_info = self.sp.track(track_id)
            
            # Get audio features
            audio_features = self.sp.audio_features(track_id)[0]
            
            if not audio_features:
                logger.warning(f"No audio features found for track {track_id}")
                return None
            
            # Combine track info and audio features
            result = {
                'track_id': track_id,
                'track_name': track_info['name'],
                'artists': ', '.join([artist['name'] for artist in track_info['artists']]),
                'album_name': track_info['album']['name'],
                'popularity': track_info['popularity'],
                'duration_ms': track_info['duration_ms'],
                'explicit': track_info['explicit'],
                **audio_features
            }
            
            logger.debug(f"Successfully fetched features for track: {track_info['name']}")
            return result
            
        except Exception as e:
            logger.error(f"Error fetching features for track {track_id}: {str(e)}")
            return None
    
    def get_multiple_track_features(self, track_ids: List[str], batch_size: int = 50) -> List[Dict[str, Any]]:
        """
        Get audio features for multiple tracks with batching and rate limiting.
        
        Args:
            track_ids: List of Spotify track IDs
            batch_size: Number of tracks to process in each batch
            
        Returns:
            List of dictionaries containing audio features and track info
            
        Example:
            >>> api = SpotifyAPI()
            >>> track_ids = ['4iV5W9uYEdYUVa79Axb7Rh', '3n3Ppam7vgaVa1iaRUpq9E']
            >>> features = api.get_multiple_track_features(track_ids)
        """
        results = []
        
        try:
            # Process tracks in batches
            for i in tqdm(range(0, len(track_ids), batch_size), desc="Fetching track features"):
                batch = track_ids[i:i + batch_size]
                
                # Get track info for batch
                track_infos = self.sp.tracks(batch)
                
                # Get audio features for batch
                audio_features = self.sp.audio_features(batch)
                
                # Combine results
                for track_info, audio_feature in zip(track_infos['tracks'], audio_features):
                    if track_info and audio_feature:
                        result = {
                            'track_id': track_info['id'],
                            'track_name': track_info['name'],
                            'artists': ', '.join([artist['name'] for artist in track_info['artists']]),
                            'album_name': track_info['album']['name'],
                            'popularity': track_info['popularity'],
                            'duration_ms': track_info['duration_ms'],
                            'explicit': track_info['explicit'],
                            **audio_feature
                        }
                        results.append(result)
                    else:
                        logger.warning(f"Failed to fetch data for track in batch starting at index {i}")
                
                # Rate limiting - Spotify allows 100 requests per second
                time.sleep(0.1)
            
            logger.info(f"Successfully fetched features for {len(results)} tracks")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching multiple track features: {str(e)}")
            return results
    
    def search_tracks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for tracks by query string.
        
        Args:
            query: Search query (track name, artist, etc.)
            limit: Maximum number of results to return
            
        Returns:
            List of track dictionaries with basic info
            
        Example:
            >>> api = SpotifyAPI()
            >>> tracks = api.search_tracks('Bohemian Rhapsody', limit=5)
        """
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            tracks = []
            
            for track in results['tracks']['items']:
                track_info = {
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'artists': ', '.join([artist['name'] for artist in track['artists']]),
                    'album_name': track['album']['name'],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms'],
                    'explicit': track['explicit']
                }
                tracks.append(track_info)
            
            logger.info(f"Found {len(tracks)} tracks for query: {query}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error searching tracks for query '{query}': {str(e)}")
            return []
    
    def get_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        """
        Get all tracks from a Spotify playlist.
        
        Args:
            playlist_id: Spotify playlist ID
            
        Returns:
            List of track dictionaries with basic info
            
        Example:
            >>> api = SpotifyAPI()
            >>> tracks = api.get_playlist_tracks('37i9dQZF1DXcBWIGoYBM5M')
        """
        try:
            tracks = []
            results = self.sp.playlist_tracks(playlist_id)
            
            while results:
                for item in results['items']:
                    if item['track']:
                        track_info = {
                            'track_id': item['track']['id'],
                            'track_name': item['track']['name'],
                            'artists': ', '.join([artist['name'] for artist in item['track']['artists']]),
                            'album_name': item['track']['album']['name'],
                            'popularity': item['track']['popularity'],
                            'duration_ms': item['track']['duration_ms'],
                            'explicit': item['track']['explicit']
                        }
                        tracks.append(track_info)
                
                # Get next page if available
                results = self.sp.next(results) if results['next'] else None
            
            logger.info(f"Found {len(tracks)} tracks in playlist {playlist_id}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error fetching playlist tracks for {playlist_id}: {str(e)}")
            return []


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the existing dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the dataset
        
    Example:
        >>> df = load_dataset('data/dataset.csv')
        >>> print(df.shape)
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {str(e)}")
        raise


def save_features_to_cache(features: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save audio features to JSON cache file.
    
    Args:
        features: List of feature dictionaries
        file_path: Path to save the cache file
        
    Example:
        >>> features = [{'track_id': '123', 'danceability': 0.8}]
        >>> save_features_to_cache(features, 'data/cached/features.json')
    """
    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(features, f, indent=2)
        
        logger.info(f"Saved {len(features)} features to cache: {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving features to cache: {str(e)}")
        raise


def load_features_from_cache(file_path: str) -> List[Dict[str, Any]]:
    """
    Load audio features from JSON cache file.
    
    Args:
        file_path: Path to the cache file
        
    Returns:
        List of feature dictionaries
        
    Example:
        >>> features = load_features_from_cache('data/cached/features.json')
    """
    try:
        with open(file_path, 'r') as f:
            features = json.load(f)
        
        logger.info(f"Loaded {len(features)} features from cache: {file_path}")
        return features
        
    except Exception as e:
        logger.error(f"Error loading features from cache: {str(e)}")
        return []


def get_audio_feature_columns() -> List[str]:
    """
    Get the list of audio feature columns used for clustering.
    
    Returns:
        List of audio feature column names
        
    Example:
        >>> columns = get_audio_feature_columns()
        >>> print(columns)
    """
    return [
        'danceability', 'energy', 'valence', 'tempo', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'mode', 'key'
    ]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load existing dataset
    df = load_dataset('data/dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Example of using Spotify API (requires credentials)
    # api = SpotifyAPI()
    # track = api.search_tracks('Bohemian Rhapsody', limit=1)
    # if track:
    #     features = api.get_track_features(track[0]['track_id'])
    #     print(features)
