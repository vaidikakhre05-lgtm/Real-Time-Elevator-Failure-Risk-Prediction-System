import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from src.config import MODELS_DIR, FEATURES_CONFIG
from src.data.synthetic_generator import generate_synthetic_data
from src.logger import setup_logger

logger = setup_logger("sidebar_component")

def init_session_state() -> None:
    """
    Initializes global Streamlit session state variables if they are not already set.
    """
    # 1. Active ML Model Configuration
    if "active_classifier" not in st.session_state:
        st.session_state["active_classifier"] = "random_forest"
        
    # 2. Simulation State Control
    if "simulation_active" not in st.session_state:
        st.session_state["simulation_active"] = False
        
    # 3. Live Sensor readings (Nominal default)
    if "current_sensor_data" not in st.session_state:
        st.session_state["current_sensor_data"] = {
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
        
    # 4. Simulation Engine wear parameters (for physical degradation simulation)
    if "simulation_wear" not in st.session_state:
        st.session_state["simulation_wear"] = {
            "cumulative_wear": 0.05,
            "door_wear": 0.01,
            "brake_wear": 0.0,
            "anomaly_active": False,
            "anomaly_type": None,
            "ticks": 0
        }
        
    # 5. History Lists for Trends
    if "sensor_history" not in st.session_state:
        st.session_state["sensor_history"] = []
        
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []

def render_sidebar() -> Optional[pd.DataFrame]:
    """
    Renders the unified industrial control panel sidebar.
    Allows model selection, real-time simulation controls, and batch CSV upload.
    
    Returns:
        Optional[pd.DataFrame]: Uploaded CSV DataFrame if present.
    """
    st.sidebar.markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #38bdf8; margin: 0; font-family: 'Outfit'; font-weight: 800;">ELEVATOR COMMAND</h2>
            <span style="font-family: 'Share Tech Mono'; color: #64748b; font-size: 0.8rem;">CONTROL CONTROL PANEL v1.0</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    init_session_state()
    
    # Section 1: ML Model Configuration
    st.sidebar.subheader("🤖 Model Configuration")
    model_choices = {
        "Random Forest Classifier (Default)": "random_forest",
        "Decision Tree Classifier": "decision_tree",
        "Logistic Regression": "logistic_regression"
    }
    
    selected_display = st.sidebar.selectbox(
        "Active Classifier",
        options=list(model_choices.keys()),
        index=list(model_choices.values()).index(st.session_state["active_classifier"])
    )
    st.session_state["active_classifier"] = model_choices[selected_display]
    
    st.sidebar.markdown("---")
    
    # Section 2: Real-time Simulation Engine Controls
    st.sidebar.subheader("⚡ Simulation Engine")
    
    # Play / Pause Simulation
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Start Live", use_container_width=True, disabled=st.session_state["simulation_active"]):
            st.session_state["simulation_active"] = True
            st.rerun()
    with col2:
        if st.button("⏸️ Stop Live", use_container_width=True, disabled=not st.session_state["simulation_active"]):
            st.session_state["simulation_active"] = False
            st.rerun()
            
    # Simulation Speed Slider
    st.session_state["simulation_speed"] = st.sidebar.slider(
        "Refresh Interval (s)",
        min_value=1.0,
        max_value=10.0,
        value=st.session_state.get("simulation_speed", 2.0),
        step=0.5
    )
    
    # Perform Maintenance
    if st.sidebar.button("🛠️ Perform Maintenance", use_container_width=True, help="Reset wear indexes and resolve active anomalies"):
        from src.simulation.maintenance_logic import perform_maintenance_reset
        curr_sensor, curr_wear = perform_maintenance_reset(
            st.session_state["current_sensor_data"],
            st.session_state["simulation_wear"]
        )
        st.session_state["current_sensor_data"] = curr_sensor
        st.session_state["simulation_wear"] = curr_wear
        st.toast("Maintenance completed! Elevator restored to nominal health.", icon="🛠️")
        st.rerun()

    # Reset Simulation
    if st.sidebar.button("🔄 Reset Fleet Logs", use_container_width=True):
        st.session_state["sensor_history"] = []
        st.session_state["prediction_history"] = []
        st.session_state["simulation_active"] = False
        st.session_state["simulation_wear"] = {
            "cumulative_wear": 0.05,
            "door_wear": 0.01,
            "brake_wear": 0.0,
            "anomaly_active": False,
            "anomaly_type": None,
            "ticks": 0
        }
        st.session_state["current_sensor_data"] = {
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
        st.toast("Simulation history reset successfully!", icon="🔄")
        st.rerun()
        
    st.sidebar.markdown("---")
    
    # Section 3: Batch Data Upload
    st.sidebar.subheader("📂 Batch Analysis")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Sensor Logs",
        type=["csv"],
        help="Upload an elevator sensor log matching the raw features structure."
    )
    
    uploaded_df = None
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(uploaded_df)} records!")
        except Exception as e:
            st.sidebar.error(f"Error parsing CSV: {e}")
            
    # Mock data generation button
    if st.sidebar.button("📊 Generate Dataset Mock", help="Generates and downloads a fresh 20,000 samples synthetic file"):
        try:
            with st.spinner("Generating dataset..."):
                generate_synthetic_data()
            st.sidebar.success("Generated data/synthetic/elevator_data.csv!")
        except Exception as e:
            st.sidebar.error(f"Generation error: {e}")
            
    return uploaded_df
