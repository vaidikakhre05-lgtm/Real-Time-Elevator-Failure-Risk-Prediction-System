import os
import sys
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import ElevatorPredictor

def test_prediction_flow():
    """
    Verifies that ElevatorPredictor loads models and makes valid inference.
    Note: Requires models to be already trained and saved.
    """
    # Check if models exist first, otherwise skip
    model_file = PROJECT_ROOT / "models" / "preprocessor.pkl"
    if not model_file.exists():
        pytest.skip("Models not trained yet. Run pipeline first.")
        
    predictor = ElevatorPredictor(classifier_name="random_forest")
    
    sample_input = {
        "motor_current": 10.5,
        "torque": 45.0,
        "temperature": 32.5,
        "humidity": 45.0,
        "vibration_x": 0.2,
        "vibration_y": 0.3,
        "vibration_z": 0.1,
        "vibration_magnitude": 0.374,
        "passenger_load": 150.0,
        "door_cycle_count": 5000.0,
        "trips_per_hour": 15.0,
        "operating_hours": 1000.0,
        "brake_status": "Normal",
        "days_since_last_maintenance": 15.0
    }
    
    res = predictor.predict_single(sample_input)
    
    assert "predicted_class" in res
    assert "probabilities" in res
    assert "predicted_days_until_failure" in res
    assert "health_score" in res
    
    assert res["predicted_class"] in ["Normal", "Warning", "Critical"]
    assert 0.0 <= res["health_score"] <= 100.0
    assert 0.0 <= res["predicted_days_until_failure"] <= 365.0
