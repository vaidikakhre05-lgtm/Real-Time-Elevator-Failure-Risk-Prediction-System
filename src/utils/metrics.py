from typing import Dict
import numpy as np

def calculate_predictive_maintenance_score(
    accuracy: float,
    f1_score: float,
    r2_score: float,
    rmse: float
) -> float:
    """
    Computes a composite score (0-100) representing how well the overall system
    performs on both classification (health state) and regression (RUL).
    Used as an overall system quality metric.
    """
    # Normalize RMSE (assume 60 days error is the maximum acceptable threshold for 0 score)
    rmse_score = max(0.0, 1.0 - (rmse / 60.0))
    # Normalize R2 (clip below 0)
    r2_score_norm = max(0.0, r2_score)
    
    # Weight components
    composite_score = (
        accuracy * 0.25 +
        f1_score * 0.35 +
        r2_score_norm * 0.20 +
        rmse_score * 0.20
    ) * 100.0
    
    return float(np.clip(composite_score, 0.0, 100.0))
