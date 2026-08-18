import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_engineering import engineer_features
from src.data.preprocessing import DataPreprocessor, split_and_stratify

def test_feature_engineering():
    """
    Test that engineered features are correctly created and have no NaN values.
    """
    # Create a small dummy dataframe matching sensor reading columns
    df = pd.DataFrame({
        "motor_current": [10.0, 12.0, 15.0],
        "torque": [50.0, 60.0, 70.0],
        "temperature": [30.0, 35.0, 40.0],
        "humidity": [50.0, 48.0, 45.0],
        "vibration_x": [0.5, 0.6, 0.7],
        "vibration_y": [0.4, 0.5, 0.6],
        "vibration_z": [0.3, 0.4, 0.5],
        "vibration_magnitude": [0.707, 0.877, 1.048],
        "passenger_load": [100, 200, 300],
        "door_cycle_count": [1000, 1010, 1020],
        "trips_per_hour": [10, 12, 14],
        "operating_hours": [100.0, 101.0, 102.0],
        "brake_status": ["Normal", "Normal", "Worn"],
        "days_since_last_maintenance": [10.0, 15.0, 20.0]
    })
    
    df_feat = engineer_features(df)
    
    # Check that new columns are created
    expected_cols = [
        "vibration_magnitude_rolling_mean",
        "vibration_magnitude_rolling_std",
        "temperature_rolling_mean",
        "current_vibration_interaction",
        "temp_current_interaction",
        "maintenance_overdue_flag"
    ]
    for col in expected_cols:
        assert col in df_feat.columns
        assert not df_feat[col].isna().any()

def test_preprocessing():
    """
    Test that preprocessing correctly scales values and handles targets.
    """
    df = pd.DataFrame({
        "motor_current": [10.0, 12.0, 15.0, 14.0, 11.0],
        "torque": [50.0, 60.0, 70.0, 65.0, 55.0],
        "temperature": [30.0, 35.0, 40.0, 38.0, 32.0],
        "humidity": [50.0, 48.0, 45.0, 47.0, 49.0],
        "vibration_x": [0.5, 0.6, 0.7, 0.65, 0.55],
        "vibration_y": [0.4, 0.5, 0.6, 0.55, 0.45],
        "vibration_z": [0.3, 0.4, 0.5, 0.45, 0.35],
        "vibration_magnitude": [0.707, 0.877, 1.048, 0.95, 0.80],
        "passenger_load": [100, 200, 300, 250, 150],
        "door_cycle_count": [1000, 1010, 1020, 1015, 1005],
        "trips_per_hour": [10, 12, 14, 13, 11],
        "operating_hours": [100.0, 101.0, 102.0, 101.5, 100.5],
        "brake_status": ["Normal", "Normal", "Worn", "Normal", "Worn"],
        "days_since_last_maintenance": [10.0, 15.0, 20.0, 18.0, 12.0],
        "days_until_failure": [300.0, 280.0, 250.0, 260.0, 290.0],
        "health_status": ["Normal", "Normal", "Warning", "Normal", "Warning"]
    })
    
    df_feat = engineer_features(df)
    train_df, test_df = split_and_stratify(df_feat, test_size=0.4, random_state=42)
    
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_df)
    
    X_train, y_train = preprocessor.transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)
    
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 2
    assert "y_class" in y_train
    assert "y_reg" in y_train
