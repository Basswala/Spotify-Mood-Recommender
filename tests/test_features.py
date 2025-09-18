"""
Unit tests for the features module.

Tests the AudioFeatureProcessor class and related functionality.
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

from features import AudioFeatureProcessor, create_feature_summary


class TestAudioFeatureProcessor:
    """Test cases for AudioFeatureProcessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample audio feature data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'track_id': [f'track_{i}' for i in range(n_samples)],
            'track_name': [f'Song {i}' for i in range(n_samples)],
            'artists': [f'Artist {i}' for i in range(n_samples)],
            'danceability': np.random.uniform(0, 1, n_samples),
            'energy': np.random.uniform(0, 1, n_samples),
            'valence': np.random.uniform(0, 1, n_samples),
            'tempo': np.random.uniform(60, 200, n_samples),
            'loudness': np.random.uniform(-60, 0, n_samples),
            'speechiness': np.random.uniform(0, 1, n_samples),
            'acousticness': np.random.uniform(0, 1, n_samples),
            'instrumentalness': np.random.uniform(0, 1, n_samples),
            'liveness': np.random.uniform(0, 1, n_samples),
            'mode': np.random.randint(0, 2, n_samples),
            'key': np.random.randint(0, 12, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test processor initialization."""
        processor = AudioFeatureProcessor(scaler_type='standard')
        assert processor.scaler_type == 'standard'
        assert processor.scaler is not None
        assert processor.imputer is not None
        assert not processor.is_fitted
    
    def test_get_audio_feature_columns(self):
        """Test getting audio feature columns."""
        processor = AudioFeatureProcessor()
        columns = processor.get_audio_feature_columns()
        
        expected_columns = [
            'danceability', 'energy', 'valence', 'tempo', 'loudness',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'mode', 'key'
        ]
        
        assert set(columns) == set(expected_columns)
    
    def test_fit_transform(self, sample_data):
        """Test fit_transform method."""
        processor = AudioFeatureProcessor(scaler_type='standard')
        df_processed = processor.fit_transform(sample_data)
        
        # Check that processor is fitted
        assert processor.is_fitted
        
        # Check that data is processed
        assert len(df_processed) == len(sample_data)
        assert df_processed.shape[1] >= sample_data.shape[1]  # May have more columns due to encoding
        
        # Check that circular features are encoded
        if 'key' in sample_data.columns:
            assert 'key_sin' in df_processed.columns
            assert 'key_cos' in df_processed.columns
            assert 'key' not in df_processed.columns
        
        if 'mode' in sample_data.columns:
            assert 'mode_sin' in df_processed.columns
            assert 'mode_cos' in df_processed.columns
            assert 'mode' not in df_processed.columns
    
    def test_transform_without_fit(self, sample_data):
        """Test that transform raises error if not fitted."""
        processor = AudioFeatureProcessor()
        
        with pytest.raises(ValueError, match="Processor must be fitted before transform"):
            processor.transform(sample_data)
    
    def test_save_and_load_processor(self, sample_data):
        """Test saving and loading processor."""
        processor = AudioFeatureProcessor(scaler_type='standard')
        processor.fit_transform(sample_data)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Save processor
            processor.save_processor(tmp_path)
            assert Path(tmp_path).exists()
            
            # Load processor
            new_processor = AudioFeatureProcessor()
            new_processor.load_processor(tmp_path)
            
            assert new_processor.is_fitted
            assert new_processor.scaler_type == processor.scaler_type
            assert new_processor.feature_columns == processor.feature_columns
            
        finally:
            # Clean up
            if Path(tmp_path).exists():
                os.unlink(tmp_path)
    
    def test_detect_outliers(self, sample_data):
        """Test outlier detection."""
        processor = AudioFeatureProcessor()
        
        # Add some outliers
        sample_data.loc[0, 'danceability'] = 10  # Extreme outlier
        sample_data.loc[1, 'energy'] = -5  # Extreme outlier
        
        df_clean = processor.detect_outliers(sample_data, method='iqr')
        
        # Should remove outliers
        assert len(df_clean) < len(sample_data)
        assert 0 not in df_clean.index  # First outlier should be removed
        assert 1 not in df_clean.index  # Second outlier should be removed
    
    def test_encode_circular_features(self, sample_data):
        """Test circular feature encoding."""
        processor = AudioFeatureProcessor()
        
        df_encoded = processor.encode_circular_features(sample_data)
        
        # Check that circular features are encoded
        if 'key' in sample_data.columns:
            assert 'key_sin' in df_encoded.columns
            assert 'key_cos' in df_encoded.columns
            assert 'key' not in df_encoded.columns
            
            # Check that encoding preserves circular nature
            key_values = sample_data['key'].values
            expected_sin = np.sin(2 * np.pi * key_values / 12)
            expected_cos = np.cos(2 * np.pi * key_values / 12)
            
            np.testing.assert_array_almost_equal(df_encoded['key_sin'].values, expected_sin)
            np.testing.assert_array_almost_equal(df_encoded['key_cos'].values, expected_cos)
    
    def test_different_scaler_types(self, sample_data):
        """Test different scaler types."""
        scaler_types = ['standard', 'minmax', 'robust']
        
        for scaler_type in scaler_types:
            processor = AudioFeatureProcessor(scaler_type=scaler_type)
            df_processed = processor.fit_transform(sample_data)
            
            assert processor.is_fitted
            assert len(df_processed) == len(sample_data)


class TestFeatureSummary:
    """Test cases for feature summary functionality."""
    
    def test_create_feature_summary(self):
        """Test creating feature summary."""
        # Create sample data
        data = {
            'danceability': [0.8, 0.6, 0.9],
            'energy': [0.7, 0.5, 0.8],
            'valence': [0.6, 0.4, 0.7],
            'tempo': [120, 140, 100],
            'loudness': [-5, -10, -3]
        }
        df = pd.DataFrame(data)
        
        summary = create_feature_summary(df)
        
        # Check summary structure
        assert 'count' in summary
        assert 'features' in summary
        assert 'mean' in summary
        assert 'std' in summary
        assert 'min' in summary
        assert 'max' in summary
        assert 'missing_values' in summary
        
        # Check values
        assert summary['count'] == 3
        assert len(summary['features']) == 5
        assert all(val == 0 for val in summary['missing_values'].values())
    
    def test_create_feature_summary_empty(self):
        """Test creating feature summary with empty DataFrame."""
        df = pd.DataFrame()
        summary = create_feature_summary(df)
        
        assert summary['count'] == 0
        assert len(summary['features']) == 0


if __name__ == "__main__":
    pytest.main([__file__])
