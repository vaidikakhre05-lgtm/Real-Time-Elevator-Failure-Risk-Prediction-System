import joblib
from pathlib import Path
from typing import Any, List
from src.config import (
    MODELS_DIR, CLASSIFIER_MODELS_DIR, REGRESSOR_MODELS_DIR
)
from src.data.preprocessing import DataPreprocessor
from src.logger import setup_logger

logger = setup_logger("save_models")

def save_model_artifact(model: Any, filename: str, subfolder: str = "") -> None:
    """
    Saves a trained model or preprocessing object using Joblib.
    
    Args:
        model: Trained model or preprocessing object to save.
        filename: Name of the file (e.g. 'logistic_regression.pkl').
        subfolder: Optional subfolder inside models/ ('classifier' or 'regressor').
    """
    target_dir = MODELS_DIR
    if subfolder == "classifier":
        target_dir = CLASSIFIER_MODELS_DIR
    elif subfolder == "regressor":
        target_dir = REGRESSOR_MODELS_DIR
        
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / filename
    
    logger.info(f"Saving model artifact to {file_path}...")
    joblib.dump(model, file_path)
    logger.info("Artifact saved successfully.")

def load_model_artifact(filename: str, subfolder: str = "") -> Any:
    """
    Loads a saved model or preprocessing object using Joblib.
    
    Args:
        filename: Name of the file.
        subfolder: Optional subfolder ('classifier' or 'regressor').
        
    Returns:
        The loaded Python object.
    """
    target_dir = MODELS_DIR
    if subfolder == "classifier":
        target_dir = CLASSIFIER_MODELS_DIR
    elif subfolder == "regressor":
        target_dir = REGRESSOR_MODELS_DIR
        
    file_path = target_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {file_path}")
        
    logger.info(f"Loading model artifact from {file_path}...")
    obj = joblib.load(file_path)
    return obj

def save_preprocessor_assets(preprocessor: DataPreprocessor) -> None:
    """
    Saves the internal components of DataPreprocessor to individual files.
    This fulfills the required folder structure layout.
    
    Args:
        preprocessor: Fitted DataPreprocessor object.
    """
    logger.info("Saving DataPreprocessor assets individually...")
    save_model_artifact(preprocessor.scaler, "scaler.pkl")
    save_model_artifact(preprocessor.label_encoder, "label_encoder.pkl")
    save_model_artifact(preprocessor.feature_columns, "feature_columns.pkl")
    # Also save the preprocessor object itself for easier loaded inference
    save_model_artifact(preprocessor, "preprocessor.pkl")
