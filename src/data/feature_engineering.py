import pandas as pd
import numpy as np
from src.constants import MAINTENANCE_OVERDUE_DAYS
from src.logger import setup_logger

logger = setup_logger("feature_engineering")

def engineer_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Engineers predictive maintenance features for the elevator dataset.
    Includes rolling statistics (sorted by operating hours), interactions, and flags.
    
    Args:
        df: Input pandas DataFrame.
        is_training: Whether we are in training mode (helps decide fit vs transform logic if stateful).
        
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    logger.info("Engineering features...")
    df_feat = df.copy()
    
    # 1. Sort by operating hours to establish a temporal timeline for rolling calculations
    # If elevator_id is present, we should group by it. Otherwise, assume a single fleet/timeline.
    has_elevator_id = "elevator_id" in df_feat.columns
    
    # Sort for rolling computations
    if has_elevator_id:
        df_feat = df_feat.sort_values(by=["elevator_id", "operating_hours"]).reset_index(drop=True)
    else:
        df_feat = df_feat.sort_values(by="operating_hours").reset_index(drop=True)
        
    # 2. Rolling Statistics (window size = 10)
    window_size = 10
    logger.info(f"Computing rolling features with window size {window_size}...")
    
    if has_elevator_id:
        # Grouped rolling
        df_feat["vibration_magnitude_rolling_mean"] = (
            df_feat.groupby("elevator_id")["vibration_magnitude"]
            .rolling(window=window_size, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df_feat["vibration_magnitude_rolling_std"] = (
            df_feat.groupby("elevator_id")["vibration_magnitude"]
            .rolling(window=window_size, min_periods=1)
            .std()
            .fillna(0.0)
            .reset_index(level=0, drop=True)
        )
        df_feat["temperature_rolling_mean"] = (
            df_feat.groupby("elevator_id")["temperature"]
            .rolling(window=window_size, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        # Global rolling
        df_feat["vibration_magnitude_rolling_mean"] = (
            df_feat["vibration_magnitude"]
            .rolling(window=window_size, min_periods=1)
            .mean()
        )
        df_feat["vibration_magnitude_rolling_std"] = (
            df_feat["vibration_magnitude"]
            .rolling(window=window_size, min_periods=1)
            .std()
            .fillna(0.0)
        )
        df_feat["temperature_rolling_mean"] = (
            df_feat["temperature"]
            .rolling(window=window_size, min_periods=1)
            .mean()
        )
        
    # 3. Interaction Features
    logger.info("Computing interaction features...")
    # Current x Vibration
    df_feat["current_vibration_interaction"] = (
        df_feat["motor_current"] * df_feat["vibration_magnitude"]
    )
    # Temperature x Current
    df_feat["temp_current_interaction"] = (
        df_feat["temperature"] * df_feat["motor_current"]
    )
    
    # 4. Maintenance Overdue Flag (1 if days_since_last_maintenance > threshold, else 0)
    logger.info(f"Computing maintenance overdue flag (threshold: {MAINTENANCE_OVERDUE_DAYS} days)...")
    df_feat["maintenance_overdue_flag"] = (
        (df_feat["days_since_last_maintenance"] > MAINTENANCE_OVERDUE_DAYS).astype(int)
    )
    
    logger.info(f"Feature engineering completed. Shape: {df_feat.shape}")
    return df_feat
