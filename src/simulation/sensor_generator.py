import numpy as np
from typing import Dict, Any, Tuple
from src.constants import (
    HEALTH_NORMAL, HEALTH_WARNING, HEALTH_CRITICAL,
    BRAKE_NORMAL, BRAKE_WORN,
    MOTOR_CURRENT_THRESHOLD_WARNING, TORQUE_THRESHOLD_WARNING,
    TEMPERATURE_THRESHOLD_WARNING, VIBRATION_THRESHOLD_WARNING
)

def step_simulation(
    current_reading: Dict[str, Any],
    wear_state: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Simulates the evolution of elevator sensor values over one time step (tick).
    Evolves parameters using physical equations, usage factors, and wear degradation.
    
    Args:
        current_reading: The last recorded sensor measurements.
        wear_state: Internal state tracking degradation, anomalies, and ticks.
        
    Returns:
        Tuple: (updated_sensor_reading, updated_wear_state)
    """
    # 1. Update internal simulation clock
    ticks = wear_state.get("ticks", 0) + 1
    
    # Evolve usage times
    operating_hours = float(current_reading["operating_hours"]) + 0.1  # 6 minutes per tick
    days_since_last_maintenance = float(current_reading["days_since_last_maintenance"]) + 0.05  # 1.2 hours per tick
    door_cycle_count = float(current_reading["door_cycle_count"]) + np.random.randint(1, 4)
    trips_per_hour = np.clip(
        float(current_reading["trips_per_hour"]) + np.random.normal(0, 1.5), 
        0.0, 35.0
    )
    
    # Evolve cumulative wear based on hours, maintenance delay, and passenger stress
    cumulative_wear = wear_state.get("cumulative_wear", 0.05)
    door_wear = wear_state.get("door_wear", 0.01)
    # Wear increases faster if maintenance is overdue (> 180 days)
    maint_penalty = 2.0 if days_since_last_maintenance > 180 else 1.0
    cumulative_wear += 0.00015 * maint_penalty
    
    # Track brake status degradation
    brake_wear = wear_state.get("brake_wear", 0.0)
    brake_wear += 0.0003 * (trips_per_hour / 15.0)
    brake_status = BRAKE_NORMAL
    if brake_wear > 0.8:
        brake_status = BRAKE_WORN
        
    # 2. Check for stochastically generated anomalies (if not already active)
    anomaly_active = wear_state.get("anomaly_active", False)
    anomaly_type = wear_state.get("anomaly_type", None)
    
    # If no anomaly, minor probability to trigger one (increases with wear)
    if not anomaly_active:
        trigger_prob = 0.005 + (cumulative_wear * 0.05)
        if np.random.rand() < trigger_prob:
            anomaly_active = True
            anomaly_type = np.random.choice(["vibration_bearing", "motor_overload", "cooling_failure"])
            
    # 3. Passenger Load Simulator (changes stochastically per trip)
    # Let's say load is high during rush hours (ticks 20-40, 80-100) or randomly
    passenger_load = np.clip(
        float(current_reading["passenger_load"]) + np.random.normal(0, 80.0), 
        0.0, 900.0
    )
    # Heavy passenger load spikes current and torque
    load_factor = passenger_load / 1000.0  # 0 to 0.9
    
    # 4. Sensor values calculation based on physics and wear
    # Motor Current
    current_noise = np.random.normal(0, 0.25)
    base_current = 9.0 + (load_factor * 6.0) + (cumulative_wear * 10.0)
    if anomaly_active and anomaly_type == "motor_overload":
        base_current += 6.0  # significant current draw
    motor_current = np.clip(base_current + current_noise, 3.0, 28.0)
    
    # Torque
    torque_noise = np.random.normal(0, 1.0)
    base_torque = 45.0 + (load_factor * 35.0) + (cumulative_wear * 30.0)
    if anomaly_active and anomaly_type == "motor_overload":
        base_torque += 25.0
    torque = np.clip(base_torque + torque_noise, 15.0, 140.0)
    
    # Vibration Magnitude
    vib_noise = np.random.normal(0, 0.08)
    base_vib = 0.4 + (cumulative_wear * 3.5)
    if anomaly_active and anomaly_type == "vibration_bearing":
        base_vib += 2.2  # high bearings wear vibration
    vibration_magnitude = np.clip(base_vib + vib_noise, 0.1, 6.5)
    
    # Vector Components
    angles_theta = np.random.uniform(0, np.pi)
    angles_phi = np.random.uniform(0, 2 * np.pi)
    vibration_x = vibration_magnitude * np.sin(angles_theta) * np.cos(angles_phi)
    vibration_y = vibration_magnitude * np.sin(angles_theta) * np.sin(angles_phi)
    vibration_z = vibration_magnitude * np.cos(angles_theta)
    
    # Chamber Temperature (heats up with load, current, and friction wear)
    temp_dissipation = 0.05  # cooling rate
    ambient_temp = 25.0
    heat_generated = (motor_current * 1.6) + (torque * 0.12) + (cumulative_wear * 12.0)
    if anomaly_active and anomaly_type == "cooling_failure":
        heat_generated += 20.0  # cooling fan broken
        temp_dissipation = 0.01  # slower heat dissipation
        
    temp_target = ambient_temp + heat_generated
    prev_temp = float(current_reading["temperature"])
    temperature = prev_temp + (temp_target - prev_temp) * temp_dissipation + np.random.normal(0, 0.4)
    temperature = np.clip(temperature, 15.0, 95.0)
    
    # Humidity
    humidity = np.clip(50.0 - (temperature - 30.0) * 0.35 + np.random.normal(0, 1.0), 10.0, 90.0)
    
    # Compile sensor updates
    updated_reading = {
        "motor_current": float(motor_current),
        "torque": float(torque),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "vibration_x": float(vibration_x),
        "vibration_y": float(vibration_y),
        "vibration_z": float(vibration_z),
        "vibration_magnitude": float(vibration_magnitude),
        "passenger_load": float(passenger_load),
        "door_cycle_count": float(door_cycle_count),
        "trips_per_hour": float(trips_per_hour),
        "operating_hours": float(operating_hours),
        "brake_status": brake_status,
        "days_since_last_maintenance": float(days_since_last_maintenance)
    }
    
    updated_wear = {
        "cumulative_wear": float(np.clip(cumulative_wear, 0.0, 1.5)),
        "door_wear": float(np.clip(door_wear + 0.0001, 0.0, 1.0)),
        "brake_wear": float(np.clip(brake_wear, 0.0, 1.2)),
        "anomaly_active": anomaly_active,
        "anomaly_type": anomaly_type,
        "ticks": ticks
    }
    
    return updated_reading, updated_wear
