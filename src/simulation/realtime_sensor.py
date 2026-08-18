import pandas as pd
from typing import Dict, Any, Tuple, List
from src.simulation.sensor_generator import step_simulation
from src.simulation.maintenance_logic import perform_maintenance_reset

class RealTimeSensorSimulator:
    """
    Stateful manager for the elevator IoT sensor simulation loop.
    Controls simulation steps, wear degradation tracks, and maintenance updates.
    """
    def __init__(self, initial_reading: Dict[str, Any], initial_wear: Dict[str, Any]):
        self.sensor_data = initial_reading.copy()
        self.wear_state = initial_wear.copy()
        
    def step(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Advances the simulation by one tick.
        """
        self.sensor_data, self.wear_state = step_simulation(self.sensor_data, self.wear_state)
        return self.sensor_data, self.wear_state
        
    def trigger_maintenance(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Triggers emergency or scheduled maintenance, resetting degradation indices.
        """
        self.sensor_data, self.wear_state = perform_maintenance_reset(self.sensor_data, self.wear_state)
        return self.sensor_data, self.wear_state
        
    def update_sensor_value(self, key: str, val: Any) -> None:
        """
        Manually overrides a sensor reading parameter.
        """
        if key in self.sensor_data:
            self.sensor_data[key] = val
