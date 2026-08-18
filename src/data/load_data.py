import pandas as pd
from typing import Tuple, Dict
import numpy as np
from src.config import SYNTHETIC_DATA_DIR
from src.data.synthetic_generator import generate_synthetic_data
from src.data.feature_engineering import engineer_features
from src.data.preprocessing import DataPreprocessor, split_and_stratify
from src.logger import setup_logger

logger = setup_logger("load_data")

def get_or_generate_dataset() -> pd.DataFrame:
    """
    Checks for existing elevator data. If it doesn't exist, generates it synthetically.
    
    Returns:
        pd.DataFrame: Elevator dataset.
    """
    data_path = SYNTHETIC_DATA_DIR / "elevator_data.csv"
    if not data_path.exists():
        logger.info(f"Dataset not found at {data_path}. Triggering generation...")
        df = generate_synthetic_data()
    else:
        logger.info(f"Loading existing dataset from {data_path}...")
        df = pd.read_csv(data_path)
    return df

def run_data_pipeline() -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray], DataPreprocessor]:
    """
    Runs the entire data ingestion, feature engineering, and preprocessing pipeline.
    
    Returns:
        Tuple:
            - X_train: Preprocessed train feature matrix.
            - X_test: Preprocessed test feature matrix.
            - y_train_dict: Dict of train targets (y_class, y_reg).
            - y_test_dict: Dict of test targets (y_class, y_reg).
            - preprocessor: Fitted DataPreprocessor object.
    """
    logger.info("Starting End-to-End Data Pipeline...")
    
    # 1. Load or generate raw data
    df_raw = get_or_generate_dataset()
    
    # 2. Feature Engineering
    df_engineered = engineer_features(df_raw, is_training=True)
    
    # 3. Train-test Split (Stratified on health_status)
    df_train, df_test = split_and_stratify(df_engineered)
    
    # 4. Preprocess (Scale, encode, outlier handling)
    preprocessor = DataPreprocessor()
    preprocessor.fit(df_train)
    
    X_train, y_train_dict = preprocessor.transform(df_train)
    X_test, y_test_dict = preprocessor.transform(df_test)
    
    logger.info("Data Pipeline executed successfully.")
    logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    return X_train, X_test, y_train_dict, y_test_dict, preprocessor

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = run_data_pipeline()
