import numpy as np
from sklearn.linear_model import LinearRegression
from src.logger import setup_logger
from src.models.save_models import save_model_artifact

logger = setup_logger("train_regressor")

def train_regressor(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Trains a Linear Regression model to predict the Days Until Failure (Remaining Useful Life).
    
    Args:
        X_train: Preprocessed features.
        y_train: Target values (Days Until Failure).
        
    Returns:
        LinearRegression: Trained model.
    """
    logger.info("Training Linear Regression Regressor...")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    
    save_model_artifact(lin_reg, "linear_regression.pkl", subfolder="regressor")
    logger.info("Regressor trained and saved successfully.")
    
    return lin_reg
