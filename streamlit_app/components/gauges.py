import plotly.graph_objects as go
import streamlit as st
from src.constants import (
    MOTOR_CURRENT_THRESHOLD_WARNING, MOTOR_CURRENT_THRESHOLD_CRITICAL,
    TORQUE_THRESHOLD_WARNING, TORQUE_THRESHOLD_CRITICAL,
    TEMPERATURE_THRESHOLD_WARNING, TEMPERATURE_THRESHOLD_CRITICAL,
    VIBRATION_THRESHOLD_WARNING, VIBRATION_THRESHOLD_CRITICAL
)

def render_radial_gauge(
    val: float,
    min_val: float,
    max_val: float,
    title: str,
    unit: str,
    warning_thresh: float,
    critical_thresh: float
) -> go.Figure:
    """
    Generates a Plotly radial dial gauge matching the dark theme.
    
    Args:
        val: Current metric value.
        min_val: Minimum range value.
        max_val: Maximum range value.
        title: Title of the metric.
        unit: Unit of measurement.
        warning_thresh: Warning status threshold.
        critical_thresh: Critical status threshold.
        
    Returns:
        go.Figure: Radial gauge figure.
    """
    # Dynamic active bar color selection
    if val >= critical_thresh:
        active_color = "#f87171"  # red-400
    elif val >= warning_thresh:
        active_color = "#fbbf24"  # amber-400
    else:
        active_color = "#34d399"  # emerald-400
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={
            'suffix': f" {unit}",
            'font': {'size': 24, 'color': '#e2e8f0', 'family': 'Share Tech Mono'},
            'valueformat': '.1f'
        },
        title={
            'text': title.upper(), 
            'font': {'size': 12, 'color': '#94a3b8', 'family': 'Outfit'},
            'align': 'center'
        },
        gauge={
            'axis': {
                'range': [min_val, max_val], 
                'tickcolor': "#475569", 
                'tickwidth': 1,
                'tickfont': {'color': '#64748b', 'size': 10}
            },
            'bar': {'color': active_color, 'thickness': 0.25},
            'bgcolor': "#111827",
            'borderwidth': 1,
            'bordercolor': "#374151",
            'steps': [
                {'range': [min_val, warning_thresh], 'color': 'rgba(16, 185, 129, 0.15)'},
                {'range': [warning_thresh, critical_thresh], 'color': 'rgba(245, 158, 11, 0.15)'},
                {'range': [critical_thresh, max_val], 'color': 'rgba(239, 68, 68, 0.15)'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=15, r=15, t=40, b=10),
        height=140,
        font={'color': "#e2e8f0", 'family': "Outfit"}
    )
    
    return fig

def render_sensor_gauges(sensor_data: dict) -> None:
    """
    Renders a row of sensor dial gauges for the primary elevator metrics.
    
    Args:
        sensor_data: Dictionary of raw sensor readings.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_fig = render_radial_gauge(
            val=float(sensor_data["motor_current"]),
            min_val=0.0,
            max_val=30.0,
            title="Motor Current",
            unit="A",
            warning_thresh=MOTOR_CURRENT_THRESHOLD_WARNING,
            critical_thresh=MOTOR_CURRENT_THRESHOLD_CRITICAL
        )
        st.plotly_chart(current_fig, use_container_width=True, config={'displayModeBar': False})
        
    with col2:
        torque_fig = render_radial_gauge(
            val=float(sensor_data["torque"]),
            min_val=0.0,
            max_val=150.0,
            title="Motor Torque",
            unit="Nm",
            warning_thresh=TORQUE_THRESHOLD_WARNING,
            critical_thresh=TORQUE_THRESHOLD_CRITICAL
        )
        st.plotly_chart(torque_fig, use_container_width=True, config={'displayModeBar': False})
        
    with col3:
        temp_fig = render_radial_gauge(
            val=float(sensor_data["temperature"]),
            min_val=10.0,
            max_val=100.0,
            title="Chamber Temp",
            unit="°C",
            warning_thresh=TEMPERATURE_THRESHOLD_WARNING,
            critical_thresh=TEMPERATURE_THRESHOLD_CRITICAL
        )
        st.plotly_chart(temp_fig, use_container_width=True, config={'displayModeBar': False})
        
    with col4:
        vibration_fig = render_radial_gauge(
            val=float(sensor_data["vibration_magnitude"]),
            min_val=0.0,
            max_val=6.0,
            title="Vibration Mag",
            unit="g",
            warning_thresh=VIBRATION_THRESHOLD_WARNING,
            critical_thresh=VIBRATION_THRESHOLD_CRITICAL
        )
        st.plotly_chart(vibration_fig, use_container_width=True, config={'displayModeBar': False})
