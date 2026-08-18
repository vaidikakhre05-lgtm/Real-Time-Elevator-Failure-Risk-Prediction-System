"""
Constants for the elevator predictive maintenance system.
Defines target categories, sensor limits, and default configurations.
"""

# Health States
HEALTH_NORMAL = "Normal"
HEALTH_WARNING = "Warning"
HEALTH_CRITICAL = "Critical"

HEALTH_CLASSES = [HEALTH_NORMAL, HEALTH_WARNING, HEALTH_CRITICAL]

# Brake Status Values
BRAKE_NORMAL = "Normal"
BRAKE_WORN = "Worn"

# Operational Thresholds (for synthetic data & simulation)
MOTOR_CURRENT_THRESHOLD_WARNING = 15.0  # Amps
MOTOR_CURRENT_THRESHOLD_CRITICAL = 22.0  # Amps

TORQUE_THRESHOLD_WARNING = 80.0  # Nm
TORQUE_THRESHOLD_CRITICAL = 110.0  # Nm

TEMPERATURE_THRESHOLD_WARNING = 65.0  # Celsius
TEMPERATURE_THRESHOLD_CRITICAL = 80.0  # Celsius

VIBRATION_THRESHOLD_WARNING = 2.5  # g (acceleration)
VIBRATION_THRESHOLD_CRITICAL = 4.5  # g (acceleration)

MAINTENANCE_OVERDUE_DAYS = 180  # Days after which maintenance is overdue

# RUL limits
MAX_RUL_DAYS = 365
MIN_RUL_DAYS = 0
