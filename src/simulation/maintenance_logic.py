from typing import Dict, Any, Tuple
from src.constants import BRAKE_NORMAL

def perform_maintenance_reset(
    current_reading: Dict[str, Any],
    wear_state: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Simulates performing maintenance on the elevator.
    Resets wear parameters, clears anomalies, and restores sensor values to healthy baselines.
    
    Args:
        current_reading: The current sensor readings.
        wear_state: The current simulation wear state.
        
    Returns:
        Tuple: (restored_sensor_reading, restored_wear_state)
    """
    # 1. Reset wear state parameters
    restored_wear = {
        "cumulative_wear": 0.05,
        "door_wear": 0.01,
        "brake_wear": 0.0,
        "anomaly_active": False,
        "anomaly_type": None,
        "ticks": wear_state.get("ticks", 0) # Keep simulation clock tick count
    }
    
    # 2. Reset sensor parameters
    restored_reading = current_reading.copy()
    restored_reading["days_since_last_maintenance"] = 0.0
    restored_reading["brake_status"] = BRAKE_NORMAL
    
    # Immediately drop temperature, current, and vibration to nominal baselines
    # representing fixed and lubricated parts
    passenger_load = float(current_reading.get("passenger_load", 0.0))
    load_factor = passenger_load / 1000.0
    
    restored_reading["motor_current"] = float(9.0 + (load_factor * 6.0) + 0.05 * 10.0) # Nominal wear current
    restored_reading["torque"] = float(45.0 + (load_factor * 35.0) + 0.05 * 30.0)
    restored_reading["temperature"] = 35.0  # Cool chamber
    restored_reading["vibration_magnitude"] = 0.35  # Smooth shaft
    
    # Recalculate components
    restored_reading["vibration_x"] = 0.2
    restored_reading["vibration_y"] = 0.2
    restored_reading["vibration_z"] = 0.19
    
    return restored_reading, restored_wear
