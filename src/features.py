"""
Feature engineering and preprocessing for Spotify audio features.

This module handles data cleaning, scaling, and encoding of circular features
for the mood clustering pipeline.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

class AudioFeatureProcessor:
    """
    Audio feature processor for cleaning, scaling, and encoding Spotify features.
    
    Handles missing values, outlier detection, scaling, and circular feature encoding
    for optimal clustering performance.
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize the feature processor.
        
        Args:
            scaler_type: Type of scaler to use ('standard', 'minmax', or 'robust')
            
        Example:
            >>> processor = AudioFeatureProcessor(scaler_type='standard')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        self.circular_features = ['key', 'mode']
        self.is_fitted = False
        
        # Initialize scaler based on type
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.imputer = SimpleImputer(strategy='median')
        logger.info(f"Initialized AudioFeatureProcessor with {scaler_type} scaler")
    
    def get_audio_feature_columns(self) -> List[str]:
        """
        Get the list of audio feature columns used for clustering.
        
        Returns:
            List of audio feature column names
            
        Example:
            >>> processor = AudioFeatureProcessor()
            >>> columns = processor.get_audio_feature_columns()
        """
        return [
            'danceability', 'energy', 'valence', 'tempo', 'loudness',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'mode', 'key'
        ]
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and optionally remove outliers from the dataset.
        
        Args:
            df: DataFrame with audio features
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier information or cleaned data
            
        Example:
            >>> processor = AudioFeatureProcessor()
            >>> df_clean = processor.detect_outliers(df, method='iqr')
        """
        try:
            feature_cols = self.get_audio_feature_columns()
            available_cols = [col for col in feature_cols if col in df.columns]
            
            outlier_mask = pd.Series([False] * len(df), index=df.index)
            
            for col in available_cols:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers = z_scores > threshold
                
                else:
                    raise ValueError(f"Unknown outlier detection method: {method}")
                
                outlier_mask |= outliers
            
            outlier_count = outlier_mask.sum()
            logger.info(f"Detected {outlier_count} outliers ({outlier_count/len(df)*100:.2f}%)")
            
            return df[~outlier_mask]
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return df
    
    def encode_circular_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode circular features (key, mode) using sine and cosine transformation.
        
        Args:
            df: DataFrame with audio features
            
        Returns:
            DataFrame with encoded circular features
            
        Example:
            >>> processor = AudioFeatureProcessor()
            >>> df_encoded = processor.encode_circular_features(df)
        """
        try:
            df_encoded = df.copy()
            
            # Encode key (0-11, circular)
            if 'key' in df.columns:
                df_encoded['key_sin'] = np.sin(2 * np.pi * df['key'] / 12)
                df_encoded['key_cos'] = np.cos(2 * np.pi * df['key'] / 12)
                df_encoded = df_encoded.drop('key', axis=1)
                logger.info("Encoded 'key' feature using sine/cosine transformation")
            
            # Encode mode (0-1, binary but treated as circular)
            if 'mode' in df.columns:
                df_encoded['mode_sin'] = np.sin(2 * np.pi * df['mode'] / 2)
                df_encoded['mode_cos'] = np.cos(2 * np.pi * df['mode'] / 2)
                df_encoded = df_encoded.drop('mode', axis=1)
                logger.info("Encoded 'mode' feature using sine/cosine transformation")
            
            return df_encoded
            
        except Exception as e:
            logger.error(f"Error encoding circular features: {str(e)}")
            return df
    
    def fit_transform(self, df: pd.DataFrame, remove_outliers: bool = True) -> pd.DataFrame:
        """
        Fit the processor and transform the data.
        
        Args:
            df: DataFrame with audio features
            remove_outliers: Whether to remove outliers before processing
            
        Returns:
            Transformed DataFrame ready for clustering
            
        Example:
            >>> processor = AudioFeatureProcessor()
            >>> df_processed = processor.fit_transform(df)
        """
        try:
            logger.info("Starting feature processing pipeline")
            
            # Get feature columns
            self.feature_columns = self.get_audio_feature_columns()
            available_cols = [col for col in self.feature_columns if col in df.columns]
            
            if not available_cols:
                raise ValueError("No audio feature columns found in the dataset")
            
            logger.info(f"Processing {len(available_cols)} audio features: {available_cols}")
            
            # Remove outliers if requested
            if remove_outliers:
                df_processed = self.detect_outliers(df)
                logger.info(f"Dataset size after outlier removal: {len(df_processed)}")
            else:
                df_processed = df.copy()
            
            # Encode circular features
            df_processed = self.encode_circular_features(df_processed)

            # Update available columns after encoding (key and mode are replaced)
            encoded_cols = []
            for col in available_cols:
                if col == 'key':
                    encoded_cols.extend(['key_sin', 'key_cos'])
                elif col == 'mode':
                    encoded_cols.extend(['mode_sin', 'mode_cos'])
                else:
                    encoded_cols.append(col)

            # Select and prepare features for scaling
            feature_data = df_processed[encoded_cols].copy()
            
            # Handle missing values
            if feature_data.isnull().any().any():
                logger.info("Handling missing values with median imputation")
                feature_data = pd.DataFrame(
                    self.imputer.fit_transform(feature_data),
                    columns=feature_data.columns,
                    index=feature_data.index
                )
            
            # Scale features
            logger.info(f"Scaling features using {self.scaler_type} scaler")
            scaled_features = self.scaler.fit_transform(feature_data)

            # Create final DataFrame
            df_final = df_processed.copy()
            df_final[encoded_cols] = scaled_features
            
            self.is_fitted = True
            logger.info("Feature processing completed successfully")
            
            return df_final
            
        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            raise
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted processor.
        
        Args:
            df: DataFrame with audio features
            
        Returns:
            Transformed DataFrame
            
        Example:
            >>> processor = AudioFeatureProcessor()
            >>> processor.fit_transform(train_df)  # Fit first
            >>> df_transformed = processor.transform(test_df)
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transform")
        
        try:
            logger.info("Transforming new data")
            
            # Encode circular features
            df_processed = self.encode_circular_features(df.copy())
            
            # Select features
            available_cols = [col for col in self.feature_columns if col in df.columns]
            feature_data = df_processed[available_cols].copy()
            
            # Handle missing values
            if feature_data.isnull().any().any():
                feature_data = pd.DataFrame(
                    self.imputer.transform(feature_data),
                    columns=feature_data.columns,
                    index=feature_data.index
                )
            
            # Scale features
            scaled_features = self.scaler.transform(feature_data)
            
            # Create final DataFrame
            df_final = df_processed.copy()
            df_final[available_cols] = scaled_features
            
            return df_final
            
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            raise
    
    def save_processor(self, file_path: str) -> None:
        """
        Save the fitted processor to disk.
        
        Args:
            file_path: Path to save the processor
            
        Example:
            >>> processor = AudioFeatureProcessor()
            >>> processor.fit_transform(df)
            >>> processor.save_processor('models/processor.pkl')
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            processor_data = {
                'scaler': self.scaler,
                'imputer': self.imputer,
                'feature_columns': self.feature_columns,
                'circular_features': self.circular_features,
                'scaler_type': self.scaler_type,
                'is_fitted': self.is_fitted
            }
            
            joblib.dump(processor_data, file_path)
            logger.info(f"Processor saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving processor: {str(e)}")
            raise
    
    def load_processor(self, file_path: str) -> None:
        """
        Load a fitted processor from disk.
        
        Args:
            file_path: Path to the processor file
            
        Example:
            >>> processor = AudioFeatureProcessor()
            >>> processor.load_processor('models/processor.pkl')
        """
        try:
            processor_data = joblib.load(file_path)
            
            self.scaler = processor_data['scaler']
            self.imputer = processor_data['imputer']
            self.feature_columns = processor_data['feature_columns']
            self.circular_features = processor_data['circular_features']
            self.scaler_type = processor_data['scaler_type']
            self.is_fitted = processor_data['is_fitted']
            
            logger.info(f"Processor loaded from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading processor: {str(e)}")
            raise


def create_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a summary of audio features for analysis.
    
    Args:
        df: DataFrame with audio features
        
    Returns:
        Dictionary containing feature statistics
        
    Example:
        >>> summary = create_feature_summary(df)
        >>> print(summary['mean'])
    """
    try:
        feature_cols = [
            'danceability', 'energy', 'valence', 'tempo', 'loudness',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness'
        ]
        
        available_cols = [col for col in feature_cols if col in df.columns]
        
        summary = {
            'count': len(df),
            'features': available_cols,
            'mean': df[available_cols].mean().to_dict(),
            'std': df[available_cols].std().to_dict(),
            'min': df[available_cols].min().to_dict(),
            'max': df[available_cols].max().to_dict(),
            'missing_values': df[available_cols].isnull().sum().to_dict()
        }
        
        logger.info(f"Created feature summary for {len(available_cols)} features")
        return summary
        
    except Exception as e:
        logger.error(f"Error creating feature summary: {str(e)}")
        return {}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load dataset
    import pandas as pd
    df = pd.read_csv('data/dataset.csv')
    
    # Initialize processor
    processor = AudioFeatureProcessor(scaler_type='standard')
    
    # Process features
    df_processed = processor.fit_transform(df)
    
    # Create summary
    summary = create_feature_summary(df_processed)
    print(f"Processed dataset shape: {df_processed.shape}")
    print(f"Feature summary: {summary}")
    
    # Save processor
    processor.save_processor('models/feature_processor.pkl')
