import streamlit as st
from typing import Dict, Any, List
from src.constants import (
    HEALTH_NORMAL, HEALTH_WARNING, HEALTH_CRITICAL,
    MOTOR_CURRENT_THRESHOLD_WARNING, MOTOR_CURRENT_THRESHOLD_CRITICAL,
    TORQUE_THRESHOLD_WARNING, TORQUE_THRESHOLD_CRITICAL,
    TEMPERATURE_THRESHOLD_WARNING, TEMPERATURE_THRESHOLD_CRITICAL,
    VIBRATION_THRESHOLD_WARNING, VIBRATION_THRESHOLD_CRITICAL,
    BRAKE_WORN, MAINTENANCE_OVERDUE_DAYS
)

def analyze_anomalies(sensor_reading: Dict[str, Any]) -> List[str]:
    """
    Checks individual sensor parameters to diagnose what issues are causing Warning/Critical states.
    
    Args:
        sensor_reading: Dictionary of raw sensor readings.
        
    Returns:
        List[str]: List of failure/warning description strings.
    """
    anomalies = []
    
    # Motor Current Checks
    current = float(sensor_reading.get("motor_current", 0.0))
    if current >= MOTOR_CURRENT_THRESHOLD_CRITICAL:
        anomalies.append(f"🔴 Motor Current critical overload: {current:.1f} A (Threshold: {MOTOR_CURRENT_THRESHOLD_CRITICAL} A)")
    elif current >= MOTOR_CURRENT_THRESHOLD_WARNING:
        anomalies.append(f"🟡 Motor Current elevated: {current:.1f} A (Threshold: {MOTOR_CURRENT_THRESHOLD_WARNING} A)")
        
    # Torque Checks
    torque = float(sensor_reading.get("torque", 0.0))
    if torque >= TORQUE_THRESHOLD_CRITICAL:
        anomalies.append(f"🔴 Motor Torque critical limit: {torque:.1f} Nm (Threshold: {TORQUE_THRESHOLD_CRITICAL} Nm)")
    elif torque >= TORQUE_THRESHOLD_WARNING:
        anomalies.append(f"🟡 Motor Torque high load: {torque:.1f} Nm (Threshold: {TORQUE_THRESHOLD_WARNING} Nm)")
        
    # Temperature Checks
    temp = float(sensor_reading.get("temperature", 0.0))
    if temp >= TEMPERATURE_THRESHOLD_CRITICAL:
        anomalies.append(f"🔴 Chamber Temperature critical: {temp:.1f} °C (Threshold: {TEMPERATURE_THRESHOLD_CRITICAL} °C)")
    elif temp >= TEMPERATURE_THRESHOLD_WARNING:
        anomalies.append(f"🟡 Chamber Temperature high: {temp:.1f} °C (Threshold: {TEMPERATURE_THRESHOLD_WARNING} °C)")
        
    # Vibration Checks
    vib = float(sensor_reading.get("vibration_magnitude", 0.0))
    if vib >= VIBRATION_THRESHOLD_CRITICAL:
        anomalies.append(f"🔴 Shaft Vibration critical: {vib:.2f} g (Threshold: {VIBRATION_THRESHOLD_CRITICAL} g)")
    elif vib >= VIBRATION_THRESHOLD_WARNING:
        anomalies.append(f"🟡 Shaft Vibration high: {vib:.2f} g (Threshold: {VIBRATION_THRESHOLD_WARNING} g)")
        
    # Brake Checks
    brake = sensor_reading.get("brake_status", "Normal")
    if brake == BRAKE_WORN:
        anomalies.append("🟡 Mechanical Brake shoe wear detected (Worn status)")
        
    # Maintenance Checks
    maint_days = float(sensor_reading.get("days_since_last_maintenance", 0.0))
    if maint_days >= MAINTENANCE_OVERDUE_DAYS:
        anomalies.append(f"🟡 Maintenance Overdue: {int(maint_days)} days since last check (Threshold: {MAINTENANCE_OVERDUE_DAYS} days)")
        
    return anomalies

def render_health_alerts(predicted_class: str, sensor_reading: Dict[str, Any]) -> None:
    """
    Renders the color-coded alerts and anomaly list onto the streamlit dashboard.
    
    Args:
        predicted_class: Predicted health state ('Normal', 'Warning', 'Critical').
        sensor_reading: Raw sensor dictionary.
    """
    anomalies = analyze_anomalies(sensor_reading)
    
    if predicted_class == HEALTH_CRITICAL:
        st.markdown(
            f"""
            <div class="alert-banner alert-critical">
                <div>
                    <strong style="font-size: 1.1rem; display: block;">🚨 CRITICAL RISK STATE DETECTED</strong>
                    <span>Elevator has immediate failure risk. Automatic dispatching of emergency maintenance triggered!</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if anomalies:
            with st.expander("🔍 VIEW CRITICAL ANOMALIES LOG", expanded=True):
                for anomaly in anomalies:
                    st.markdown(f"<span style='color: #f87171; font-family: monospace;'>{anomaly}</span>", unsafe_allow_html=True)
                    
    elif predicted_class == HEALTH_WARNING:
        st.markdown(
            f"""
            <div class="alert-banner alert-warning">
                <div>
                    <strong style="font-size: 1.1rem; display: block;">⚠️ ELEVATOR WARNING ACTIVE</strong>
                    <span>Operational performance is degrading. Schedule preventative maintenance soon.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if anomalies:
            with st.expander("🔍 VIEW DEGRADATION SIGNS", expanded=True):
                for anomaly in anomalies:
                    st.markdown(f"<span style='color: #fbbf24; font-family: monospace;'>{anomaly}</span>", unsafe_allow_html=True)
                    
    else:
        st.markdown(
            f"""
            <div class="alert-banner alert-normal">
                <div>
                    <strong style="font-size: 1.1rem; display: block;">🟢 SYSTEM NOMINAL</strong>
                    <span>All parameters within standard operational limits. Regular operations.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if anomalies:
            # Sometime sensor shows warnings, but model evaluates normal. Still helpful to display!
            with st.expander("🔍 VIEW MINOR WARNINGS", expanded=False):
                for anomaly in anomalies:
                    st.markdown(f"<span style='color: #34d399; font-family: monospace;'>{anomaly}</span>", unsafe_allow_html=True)
