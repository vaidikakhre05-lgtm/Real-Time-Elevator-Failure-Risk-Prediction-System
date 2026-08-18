import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.sensor_generator import step_simulation
from src.simulation.maintenance_logic import perform_maintenance_reset
from src.simulation.realtime_sensor import RealTimeSensorSimulator

def test_simulation_stepping():
    """
    Verifies that stepping the simulation advances the tick clock,
    accumulates usage hours, and updates sensor readings.
    """
    initial_sensor = {
        "motor_current": 10.0,
        "torque": 45.0,
        "temperature": 35.0,
        "humidity": 45.0,
        "vibration_x": 0.2,
        "vibration_y": 0.3,
        "vibration_z": 0.1,
        "vibration_magnitude": 0.374,
        "passenger_load": 0.0,
        "door_cycle_count": 1000.0,
        "trips_per_hour": 10.0,
        "operating_hours": 1000.0,
        "brake_status": "Normal",
        "days_since_last_maintenance": 15.0
    }
    
    initial_wear = {
        "cumulative_wear": 0.05,
        "door_wear": 0.01,
        "brake_wear": 0.0,
        "anomaly_active": False,
        "anomaly_type": None,
        "ticks": 0
    }
    
    next_sensor, next_wear = step_simulation(initial_sensor, initial_wear)
    
    assert next_wear["ticks"] == 1
    assert next_sensor["operating_hours"] > initial_sensor["operating_hours"]
    assert next_sensor["days_since_last_maintenance"] > initial_sensor["days_since_last_maintenance"]
    assert next_sensor["door_cycle_count"] > initial_sensor["door_cycle_count"]
    assert "motor_current" in next_sensor
    assert "torque" in next_sensor

def test_maintenance_reset():
    """
    Verifies that triggering maintenance resets cumulative wear indices,
    clears active anomalies, and returns sensors to healthy baselines.
    """
    degraded_sensor = {
        "motor_current": 25.0,
        "torque": 120.0,
        "temperature": 85.0,
        "humidity": 30.0,
        "vibration_x": 2.5,
        "vibration_y": 2.0,
        "vibration_z": 1.5,
        "vibration_magnitude": 3.55,
        "passenger_load": 200.0,
        "door_cycle_count": 25000.0,
        "trips_per_hour": 15.0,
        "operating_hours": 2500.0,
        "brake_status": "Worn",
        "days_since_last_maintenance": 195.0
    }
    
    degraded_wear = {
        "cumulative_wear": 0.95,
        "door_wear": 0.55,
        "brake_wear": 0.85,
        "anomaly_active": True,
        "anomaly_type": "cooling_failure",
        "ticks": 45
    }
    
    restored_sensor, restored_wear = perform_maintenance_reset(degraded_sensor, degraded_wear)
    
    # Wear factors reset
    assert restored_wear["cumulative_wear"] == 0.05
    assert restored_wear["brake_wear"] == 0.0
    assert restored_wear["anomaly_active"] is False
    assert restored_wear["anomaly_type"] is None
    
    # Ticks maintained
    assert restored_wear["ticks"] == 45
    
    # Sensor thresholds drop back
    assert restored_sensor["days_since_last_maintenance"] == 0.0
    assert restored_sensor["brake_status"] == "Normal"
    assert restored_sensor["motor_current"] < degraded_sensor["motor_current"]
    assert restored_sensor["temperature"] == 35.0
    assert restored_sensor["vibration_magnitude"] == 0.35

def test_realtime_sensor_simulator_class():
    """
    Verifies that the stateful RealTimeSensorSimulator class works correctly.
    """
    initial_sensor = {
        "motor_current": 10.0,
        "torque": 45.0,
        "temperature": 35.0,
        "humidity": 45.0,
        "vibration_x": 0.2,
        "vibration_y": 0.3,
        "vibration_z": 0.1,
        "vibration_magnitude": 0.374,
        "passenger_load": 0.0,
        "door_cycle_count": 1000.0,
        "trips_per_hour": 10.0,
        "operating_hours": 1000.0,
        "brake_status": "Normal",
        "days_since_last_maintenance": 15.0
    }
    
    initial_wear = {
        "cumulative_wear": 0.05,
        "door_wear": 0.01,
        "brake_wear": 0.0,
        "anomaly_active": False,
        "anomaly_type": None,
        "ticks": 0
    }
    
    sim = RealTimeSensorSimulator(initial_sensor, initial_wear)
    
    # Test step
    sensor, wear = sim.step()
    assert sim.wear_state["ticks"] == 1
    
    # Test manual update
    sim.update_sensor_value("passenger_load", 800.0)
    assert sim.sensor_data["passenger_load"] == 800.0
    
    # Test maintenance
    sim.trigger_maintenance()
    assert sim.wear_state["cumulative_wear"] == 0.05
    assert sim.sensor_data["days_since_last_maintenance"] == 0.0
