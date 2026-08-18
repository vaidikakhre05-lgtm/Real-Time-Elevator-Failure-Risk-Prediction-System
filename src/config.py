import os
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_MODELS_DIR = MODELS_DIR / "classifier"
REGRESSOR_MODELS_DIR = MODELS_DIR / "regressor"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE_PATH = LOGS_DIR / "app.log"

# Ensure directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SYNTHETIC_DATA_DIR,
    CLASSIFIER_MODELS_DIR,
    REGRESSOR_MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Random Seed
RANDOM_STATE = 42

# Data Generation Parameters
NUM_SAMPLES = 20000

# Training Settings
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature names and types
FEATURES_CONFIG = {
    "numeric_features": [
        "motor_current",
        "torque",
        "temperature",
        "humidity",
        "vibration_x",
        "vibration_y",
        "vibration_z",
        "vibration_magnitude",
        "passenger_load",
        "door_cycle_count",
        "trips_per_hour",
        "operating_hours",
        "days_since_last_maintenance",
    ],
    "categorical_features": [
        "brake_status",  # 'Normal' or 'Worn'
    ],
    "engineered_features": [
        "vibration_magnitude_rolling_mean",
        "vibration_magnitude_rolling_std",
        "temperature_rolling_mean",
        "current_vibration_interaction",
        "temp_current_interaction",
        "maintenance_overdue_flag",
    ]
}

TARGET_CLASSIFICATION = "health_status"
TARGET_REGRESSION = "days_until_failure"
