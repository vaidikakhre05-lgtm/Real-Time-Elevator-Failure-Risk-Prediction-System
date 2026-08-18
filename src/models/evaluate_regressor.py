import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FIGURES_DIR, METRICS_DIR
from src.logger import setup_logger

logger = setup_logger("evaluate_regressor")

def evaluate_regressor_model(
    model: Any, 
    X_test: np.ndarray, 
    y_test: np.ndarray,
    name: str = "linear_regression"
) -> Dict[str, float]:
    """
    Evaluates the regression model (predicting Days Until Failure/RUL).
    Saves validation plot and writes metrics to file.
    
    Args:
        model: Trained regressor model.
        X_test: Preprocessed features.
        y_test: Actual Days Until Failure values.
        name: Name of the model.
        
    Returns:
        Dict: Dictionary containing MAE, RMSE, and R2.
    """
    logger.info(f"Evaluating regressor model '{name}'...")
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2_Score": r2
    }
    
    logger.info(f"Regressor Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2 Score: {r2:.4f}")
    
    # Save metrics to CSV
    df_metrics = pd.DataFrame([metrics])
    metrics_path = METRICS_DIR / "regressor_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    logger.info(f"Saved regressor metrics to {metrics_path}")
    
    # 1. Generate & Save Predicted vs. Actual Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color="teal", edgecolors="w")
    
    # Perfect prediction diagonal line
    lims = [
        np.min([plt.xlim()[0], plt.ylim()[0]]),  # min of both axes
        np.max([plt.xlim()[1], plt.ylim()[1]]),  # max of both axes
    ]
    plt.plot(lims, lims, "r--", alpha=0.75, zorder=3, label="Perfect Prediction")
    
    plt.title(f"Predicted vs. Actual Days Until Failure ({name.replace('_', ' ').title()})")
    plt.xlabel("Actual Days Until Failure")
    plt.ylabel("Predicted Days Until Failure")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    plot_path = FIGURES_DIR / f"regressor_predicted_vs_actual_{name}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved Regressor Validation Plot to {plot_path}")
    
    return metrics
