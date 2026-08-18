import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.config import RANDOM_STATE, TEST_SIZE, FEATURES_CONFIG, TARGET_CLASSIFICATION, TARGET_REGRESSION
from src.logger import setup_logger

logger = setup_logger("preprocessing")

class DataPreprocessor:
    """
    Handles cleaning, missing value imputation, outlier treatment,
    categorical encoding, and feature scaling for the elevator dataset.
    Follows Scikit-Learn transformer pattern for reproducibility.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns: List[str] = []
        self.is_fit = False
        
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fits the scaler and encoders based on the training data.
        
        Args:
            df: Training DataFrame with engineered features.
        """
        logger.info("Fitting DataPreprocessor...")
        
        # 1. Determine feature columns
        numeric_cols = FEATURES_CONFIG["numeric_features"] + FEATURES_CONFIG["engineered_features"]
        self.feature_columns = numeric_cols + FEATURES_CONFIG["categorical_features"]
        
        # 2. Fit Numeric Scaler
        # Handle missing values by fitting on available non-null data (or assume data is already cleaned)
        # In synthetic data, there are no NaNs, but we'll add robust handling anyway
        numeric_data = df[numeric_cols].fillna(df[numeric_cols].median())
        self.scaler.fit(numeric_data)
        
        # 3. Fit Target Label Encoder
        if TARGET_CLASSIFICATION in df.columns:
            self.label_encoder.fit(df[TARGET_CLASSIFICATION].astype(str))
            
        self.is_fit = True
        logger.info("DataPreprocessor successfully fit.")
        return self
        
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Transforms the input DataFrame using the fitted scaler and encoders.
        
        Args:
            df: Input DataFrame to preprocess.
            
        Returns:
            Tuple containing:
                - Preprocessed feature matrix (X) as a numpy array.
                - Dict containing targets (y_class, y_reg) if they exist in the input DataFrame.
        """
        if not self.is_fit:
            raise ValueError("DataPreprocessor must be fit before calling transform.")
            
        df_clean = df.copy()
        
        # 1. Clean missing values and handle outliers (Capping at 1st and 99th percentiles for numerical columns)
        numeric_cols = FEATURES_CONFIG["numeric_features"] + FEATURES_CONFIG["engineered_features"]
        for col in numeric_cols:
            if col in df_clean.columns:
                # Impute missing values with median
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                
                # Outlier treatment: cap values at 1st and 99th percentiles
                q1 = df_clean[col].quantile(0.01)
                q99 = df_clean[col].quantile(0.99)
                df_clean[col] = np.clip(df_clean[col], q1, q99)
                
        # 2. Encode categorical columns
        # For 'brake_status', we map 'Normal' to 0 and 'Worn' to 1
        for col in FEATURES_CONFIG["categorical_features"]:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map({"Normal": 0, "Worn": 1}).fillna(0).astype(int)
                
        # 3. Scale numeric features
        scaled_numeric = self.scaler.transform(df_clean[numeric_cols])
        df_scaled = pd.DataFrame(scaled_numeric, columns=numeric_cols, index=df_clean.index)
        
        # Re-attach categorical columns
        for col in FEATURES_CONFIG["categorical_features"]:
            if col in df_clean.columns:
                df_scaled[col] = df_clean[col]
                
        # Arrange columns in a consistent order
        X = df_scaled[self.feature_columns].values
        
        # 4. Extract target columns if present
        targets = {}
        if TARGET_CLASSIFICATION in df.columns:
            targets["y_class"] = self.label_encoder.transform(df[TARGET_CLASSIFICATION].astype(str))
        if TARGET_REGRESSION in df.columns:
            targets["y_reg"] = df[TARGET_REGRESSION].values
            
        return X, targets

def split_and_stratify(
    df: pd.DataFrame, 
    test_size: float = TEST_SIZE, 
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe into train and test sets, stratifying on the classification target.
    
    Args:
        df: Input DataFrame.
        test_size: Split ratio.
        random_state: Seed.
        
    Returns:
        Tuple: Train and test DataFrames.
    """
    logger.info("Splitting data into train/test sets...")
    
    if TARGET_CLASSIFICATION not in df.columns:
        # Fallback to simple split if target not present
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    else:
        # Stratified split on health_status
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df[TARGET_CLASSIFICATION]
        )
        
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df
