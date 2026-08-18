import numpy as np
import pandas as pd
from typing import Tuple
from src.config import RANDOM_STATE, NUM_SAMPLES, SYNTHETIC_DATA_DIR, DATA_DIR
from src.constants import (
    HEALTH_NORMAL, HEALTH_WARNING, HEALTH_CRITICAL,
    BRAKE_NORMAL, BRAKE_WORN,
    MOTOR_CURRENT_THRESHOLD_WARNING, MOTOR_CURRENT_THRESHOLD_CRITICAL,
    TORQUE_THRESHOLD_WARNING, TORQUE_THRESHOLD_CRITICAL,
    TEMPERATURE_THRESHOLD_WARNING, TEMPERATURE_THRESHOLD_CRITICAL,
    VIBRATION_THRESHOLD_WARNING, VIBRATION_THRESHOLD_CRITICAL,
    MAINTENANCE_OVERDUE_DAYS, MAX_RUL_DAYS, MIN_RUL_DAYS
)
from src.logger import setup_logger

logger = setup_logger("synthetic_generator")

def generate_synthetic_data(num_samples: int = NUM_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset for elevator predictive maintenance.
    The simulation includes degradation curves and physical equations.
    
    Args:
        num_samples: Number of rows to generate.
        seed: Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Synthetic dataset.
    """
    logger.info(f"Generating {num_samples} samples of synthetic elevator data with seed {seed}...")
    np.random.seed(seed)
    
    # 1. Base Time & Usage Features
    operating_hours = np.random.uniform(500, 15000, num_samples)
    days_since_last_maintenance = np.random.uniform(1, 300, num_samples)
    door_cycle_count = operating_hours * np.random.uniform(8, 12, num_samples)
    trips_per_hour = np.random.uniform(5, 30, num_samples)
    
    # Brake status wear probability increases with days since maintenance and operating hours
    brake_wear_prob = 1 / (1 + np.exp(-((days_since_last_maintenance - 200) / 40 + (operating_hours - 10000) / 5000)))
    brake_status = np.where(np.random.rand(num_samples) < brake_wear_prob, BRAKE_WORN, BRAKE_NORMAL)
    
    # Passenger Load (max load is 1000kg)
    passenger_load = np.random.uniform(0, 950, num_samples)
    
    # 2. Base Health Degradation Score (Wear Factor)
    # Wear increases with time, usage, brake wear, and poor maintenance
    wear_factor = (
        (operating_hours / 15000.0) * 0.35 +
        (days_since_last_maintenance / 300.0) * 0.45 +
        (door_cycle_count / 150000.0) * 0.10 +
        np.where(brake_status == BRAKE_WORN, 0.25, 0.0)
    )
    # Add a stochastic degradation modifier
    wear_factor += np.random.normal(0, 0.05, num_samples)
    wear_factor = np.clip(wear_factor, 0.0, 1.2)
    
    # 3. Physics-based Sensor Generation
    # Current (Amps) - Increases with load and wear factor
    base_current = 8.0 + (passenger_load / 100.0) * 0.8 + (wear_factor * 8.0)
    motor_current = base_current + np.random.normal(0, 0.5, num_samples)
    
    # Torque (Nm) - Proportional to current and passenger load
    base_torque = 40.0 + (passenger_load / 100.0) * 5.0 + (wear_factor * 35.0)
    torque = base_torque + np.random.normal(0, 2.0, num_samples)
    
    # Vibration (g) - Increases with wear, high load, and speed
    # Vibration X, Y, Z are components, vibration_magnitude is the vector magnitude
    base_vib_mag = 0.5 + (wear_factor * 3.5)
    vibration_magnitude = base_vib_mag + np.random.normal(0, 0.15, num_samples)
    vibration_magnitude = np.clip(vibration_magnitude, 0.1, 7.0)
    
    # Distribute magnitude into X, Y, Z components with noise
    angles_theta = np.random.uniform(0, np.pi, num_samples)
    angles_phi = np.random.uniform(0, 2 * np.pi, num_samples)
    vibration_x = vibration_magnitude * np.sin(angles_theta) * np.cos(angles_phi)
    vibration_y = vibration_magnitude * np.sin(angles_theta) * np.sin(angles_phi)
    vibration_z = vibration_magnitude * np.cos(angles_theta)
    
    # Temperature (Celsius) - Increases with current, torque, and ambient factors
    # High wear causes friction, increasing temperature
    base_temp = 25.0 + (motor_current * 1.5) + (torque * 0.15) + (wear_factor * 15.0)
    temperature = base_temp + np.random.normal(0, 1.5, num_samples)
    
    # Humidity (%) - Ambient variable, slightly correlated with temperature
    humidity = 50.0 - (temperature - 30.0) * 0.3 + np.random.normal(0, 5.0, num_samples)
    humidity = np.clip(humidity, 10.0, 95.0)
    
    # 4. Target Variables
    # Regression: Days Until Failure (Remaining Useful Life)
    # Higher wear factor means fewer days until failure
    # Also, sudden sensor spikes (anomalies) can drastically reduce days until failure
    base_days = MAX_RUL_DAYS * np.exp(-1.8 * wear_factor)
    
    # Anomaly factors
    current_anomaly = np.clip((motor_current - MOTOR_CURRENT_THRESHOLD_WARNING) / 10.0, 0, 1)
    temp_anomaly = np.clip((temperature - TEMPERATURE_THRESHOLD_WARNING) / 25.0, 0, 1)
    vib_anomaly = np.clip((vibration_magnitude - VIBRATION_THRESHOLD_WARNING) / 3.0, 0, 1)
    
    max_anomaly = np.maximum(np.maximum(current_anomaly, temp_anomaly), vib_anomaly)
    
    # Reduce days until failure based on max anomaly
    days_until_failure = base_days * (1.0 - 0.85 * max_anomaly)
    days_until_failure += np.random.normal(0, 5.0, num_samples)
    days_until_failure = np.clip(days_until_failure, MIN_RUL_DAYS, MAX_RUL_DAYS)
    
    # Classification: Health Status (Normal, Warning, Critical)
    # Based on remaining useful life and current sensor violations
    health_status = []
    for i in range(num_samples):
        # Critical conditions: Very low RUL or critical sensor thresholds exceeded
        if (
            days_until_failure[i] <= 20 or
            motor_current[i] >= MOTOR_CURRENT_THRESHOLD_CRITICAL or
            temperature[i] >= TEMPERATURE_THRESHOLD_CRITICAL or
            vibration_magnitude[i] >= VIBRATION_THRESHOLD_CRITICAL or
            torque[i] >= TORQUE_THRESHOLD_CRITICAL
        ):
            health_status.append(HEALTH_CRITICAL)
        # Warning conditions: Moderate RUL or warning sensor thresholds exceeded
        elif (
            days_until_failure[i] <= 75 or
            days_since_last_maintenance[i] >= MAINTENANCE_OVERDUE_DAYS or
            motor_current[i] >= MOTOR_CURRENT_THRESHOLD_WARNING or
            temperature[i] >= TEMPERATURE_THRESHOLD_WARNING or
            vibration_magnitude[i] >= VIBRATION_THRESHOLD_WARNING or
            torque[i] >= TORQUE_THRESHOLD_WARNING or
            brake_status[i] == BRAKE_WORN
        ):
            health_status.append(HEALTH_WARNING)
        else:
            health_status.append(HEALTH_NORMAL)
            
    # Assemble DataFrame
    df = pd.DataFrame({
        "motor_current": motor_current,
        "torque": torque,
        "temperature": temperature,
        "humidity": humidity,
        "vibration_x": vibration_x,
        "vibration_y": vibration_y,
        "vibration_z": vibration_z,
        "vibration_magnitude": vibration_magnitude,
        "passenger_load": passenger_load,
        "door_cycle_count": door_cycle_count,
        "trips_per_hour": trips_per_hour,
        "operating_hours": operating_hours,
        "brake_status": brake_status,
        "days_since_last_maintenance": days_since_last_maintenance,
        "days_until_failure": days_until_failure,
        "health_status": health_status
    })
    
    # Save to disk
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = SYNTHETIC_DATA_DIR / "elevator_data.csv"
    df.to_csv(out_file, index=False)
    logger.info(f"Dataset successfully saved to {out_file}.")
    
    # Also save a small sample of input for demonstration/testing
    sample_file = DATA_DIR / "sample_input.csv"
    df.head(100).to_csv(sample_file, index=False)
    logger.info(f"Sample dataset saved to {sample_file}.")
    
    # Display class distribution
    class_counts = df["health_status"].value_counts()
    for cls, cnt in class_counts.items():
        logger.info(f"Class '{cls}': {cnt} ({cnt/num_samples*100:.2f}%)")
        
    return df

if __name__ == "__main__":
    generate_synthetic_data()
