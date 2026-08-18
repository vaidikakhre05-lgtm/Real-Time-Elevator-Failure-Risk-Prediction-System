from typing import Dict, Any, Tuple, List
from src.config import FEATURES_CONFIG

def validate_sensor_reading(reading: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates a raw sensor reading dictionary.
    Checks that all required keys are present and values are physically reasonable.
    
    Args:
        reading: Dictionary of sensor key-values.
        
    Returns:
        Tuple: (is_valid, list of validation error messages)
    """
    errors = []
    
    # Required keys are the base numeric features + categorical features
    required_keys = FEATURES_CONFIG["numeric_features"] + FEATURES_CONFIG["categorical_features"]
    
    # Check keys
    for key in required_keys:
        if key not in reading:
            errors.append(f"Missing required key: '{key}'")
            
    if errors:
        return False, errors
        
    # Check numeric bounds
    # Current (Amps) - shouldn't be negative
    if reading["motor_current"] < 0:
        errors.append("motor_current cannot be negative")
        
    # Torque (Nm) - shouldn't be negative
    if reading["torque"] < 0:
        errors.append("torque cannot be negative")
        
    # Temperature (Celsius) - physically realistic
    if reading["temperature"] < -50 or reading["temperature"] > 200:
        errors.append("temperature must be between -50C and 200C")
        
    # Humidity (%) - must be between 0 and 100
    if reading["humidity"] < 0 or reading["humidity"] > 100:
        errors.append("humidity must be between 0% and 100%")
        
    # Vibration magnitude - shouldn't be negative
    if reading["vibration_magnitude"] < 0:
        errors.append("vibration_magnitude cannot be negative")
        
    # Passenger load (kg) - shouldn't be negative, max limit of 2000
    if reading["passenger_load"] < 0 or reading["passenger_load"] > 2000:
        errors.append("passenger_load must be between 0 and 2000 kg")
        
    # Days since last maintenance - shouldn't be negative
    if reading["days_since_last_maintenance"] < 0:
        errors.append("days_since_last_maintenance cannot be negative")
        
    # Brake status - must be 'Normal' or 'Worn'
    if reading["brake_status"] not in ["Normal", "Worn"]:
        errors.append("brake_status must be either 'Normal' or 'Worn'")
        
    return len(errors) == 0, errors
