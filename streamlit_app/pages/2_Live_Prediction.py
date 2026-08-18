import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components.sidebar import render_sidebar
from src.data.feature_engineering import engineer_features
from src.constants import HEALTH_NORMAL, HEALTH_WARNING, HEALTH_CRITICAL

# Page config
st.set_page_config(
    page_title="Elevator Command - Live Prediction & Batch",
    page_icon="🛗",
    layout="wide"
)

# Load CSS
css_file = PROJECT_ROOT / "streamlit_app" / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render Sidebar (returns uploaded CSV if any)
uploaded_df = render_sidebar()

# Verify predictor is loaded
if "predictor" not in st.session_state:
    st.error("Predictor not loaded. Please return to Home page or run pipeline.")
    st.stop()

# Switch classifier if changed in sidebar
st.session_state["predictor"].change_classifier(st.session_state["active_classifier"])

st.markdown('<h1 class="dashboard-h1">🎛️ INFERENCE INTERFACE</h1>', unsafe_allow_html=True)

tab_manual, tab_batch = st.tabs(["🎚️ Manual Telemetry Controls", "📂 Batch Upload Analysis"])

# ========================================================
# TAB 1: MANUAL CONTROL PANEL
# ========================================================
with tab_manual:
    st.markdown('<h2 class="dashboard-h2">Adjust Elevator Sensors</h2>', unsafe_allow_html=True)
    st.write("Modify sensor sliders to test how different threshold violations affect risk classification.")
    
    col1, col2, col3 = st.columns(3)
    
    # We populate the default sliders with values from st.session_state["current_sensor_data"]
    curr = st.session_state["current_sensor_data"]
    
    with col1:
        st.markdown("##### ⚡ Electrical & Load")
        motor_current = st.slider("Motor Current (A)", 0.0, 30.0, float(curr["motor_current"]), 0.1)
        torque = st.slider("Motor Torque (Nm)", 0.0, 150.0, float(curr["torque"]), 0.5)
        passenger_load = st.slider("Passenger Load (kg)", 0.0, 1200.0, float(curr["passenger_load"]), 10.0)
        trips_per_hour = st.slider("Trips Per Hour", 0.0, 50.0, float(curr["trips_per_hour"]), 1.0)
        
    with col2:
        st.markdown("##### 🌐 Shaft Environment")
        temperature = st.slider("Chamber Temp (°C)", 10.0, 100.0, float(curr["temperature"]), 0.5)
        humidity = st.slider("Relative Humidity (%)", 0.0, 100.0, float(curr["humidity"]), 1.0)
        brake_status = st.selectbox("Mechanical Brake Shoe Status", ["Normal", "Worn"], index=0 if curr["brake_status"] == "Normal" else 1)
        days_since_last_maintenance = st.slider("Days Since Maintenance", 0.0, 365.0, float(curr["days_since_last_maintenance"]), 1.0)
        
    with col3:
        st.markdown("##### 📳 Motion & Vibration")
        vib_x = st.slider("Vibration X (g)", -3.0, 3.0, float(curr["vibration_x"]), 0.05)
        vib_y = st.slider("Vibration Y (g)", -3.0, 3.0, float(curr["vibration_y"]), 0.05)
        vib_z = st.slider("Vibration Z (g)", -3.0, 3.0, float(curr["vibration_z"]), 0.05)
        
        # Recalculate vector magnitude on the fly
        vibration_magnitude = np.sqrt(vib_x**2 + vib_y**2 + vib_z**2)
        st.markdown(
            f"""
            <div style="background-color: #111827; padding: 12px; border-radius: 8px; margin-top: 10px; border: 1px solid #1e293b;">
                <span style="font-size: 0.85rem; color: #94a3b8; font-family: monospace;">Vibration Magnitude (Vector Sum)</span><br>
                <strong style="font-size: 1.5rem; color: #38bdf8; font-family: 'Share Tech Mono';">{vibration_magnitude:.3f} g</strong>
            </div>
            """,
            unsafe_allow_html=True
        )
        operating_hours = st.number_input("Total Operating Hours", 0.0, 50000.0, float(curr["operating_hours"]), 10.0)
        
    # Re-calculate door cycles based on nominal ratio if not adjusted
    door_cycle_count = float(curr["door_cycle_count"]) + (operating_hours - float(curr["operating_hours"])) * 10
    if door_cycle_count < 0: door_cycle_count = 0.0
    
    # Save the manual sliders to the current session sensor data
    manual_sensor_data = {
        "motor_current": motor_current,
        "torque": torque,
        "temperature": temperature,
        "humidity": humidity,
        "vibration_x": vib_x,
        "vibration_y": vib_y,
        "vibration_z": vib_z,
        "vibration_magnitude": vibration_magnitude,
        "passenger_load": passenger_load,
        "door_cycle_count": door_cycle_count,
        "trips_per_hour": trips_per_hour,
        "operating_hours": operating_hours,
        "brake_status": brake_status,
        "days_since_last_maintenance": days_since_last_maintenance
    }
    
    # Apply button
    if st.button("🔌 Update Live Dashboard Sensor Values", use_container_width=True):
        st.session_state["current_sensor_data"] = manual_sensor_data
        st.toast("Dashboard telemetry updated!", icon="🔌")
        
    # Run immediate prediction
    pred = st.session_state["predictor"].predict_single(manual_sensor_data)
    
    st.markdown("---")
    
    col_out1, col_out2 = st.columns(2)
    
    with col_out1:
        st.markdown('<h2 class="dashboard-h2">🔮 Prediction Outcome</h2>', unsafe_allow_html=True)
        
        # Color coding
        risk_class = "kpi-normal"
        risk_color = "#34d399"
        p_class = pred["predicted_class"]
        if p_class == "Warning":
            risk_class = "kpi-warning"
            risk_color = "#fbbf24"
        elif p_class == "Critical":
            risk_class = "kpi-critical"
            risk_color = "#f87171"
            
        st.markdown(
            f"""
            <div class="glass-card {risk_class}">
                <div class="kpi-title">RISK CATEGORY</div>
                <div class="kpi-val" style="color: {risk_color};">{p_class.upper()}</div>
                <div class="kpi-title" style="margin-top: 15px;">PREDICTED DAYS UNTIL FAILURE</div>
                <div class="kpi-val" style="color: #38bdf8; font-size: 1.8rem;">{pred['predicted_days_until_failure']:.1f} Days</div>
                <div class="kpi-title" style="margin-top: 15px;">CABIN HEALTH SCORE</div>
                <div class="kpi-val" style="color: #c084fc; font-size: 1.8rem;">{pred['health_score']:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_out2:
        st.markdown('<h2 class="dashboard-h2">🎲 Class Probabilities</h2>', unsafe_allow_html=True)
        for cls, prob in pred["probabilities"].items():
            col_lbl, col_bar = st.columns([1, 4])
            with col_lbl:
                st.markdown(f"<strong style='font-family: monospace;'>{cls}</strong>", unsafe_allow_html=True)
            with col_bar:
                st.progress(prob)
                st.markdown(f"<span style='font-size: 0.8rem; color: #94a3b8;'>{prob*100:.2f}% Confidence</span>", unsafe_allow_html=True)

# ========================================================
# TAB 2: BATCH DATA UPLOAD
# ========================================================
with tab_batch:
    st.markdown('<h2 class="dashboard-h2">Batch Log File Processing</h2>', unsafe_allow_html=True)
    st.write(
        "Upload raw elevator logs to run model predictions in batch. "
        "The model will process the dataset, engineer features, scale values, and output predictions."
    )
    
    # If the user has uploaded a file in the sidebar, or we can upload here too
    if uploaded_df is not None:
        st.success("Log file loaded from sidebar command.")
        df_batch = uploaded_df.copy()
        
        required_cols = [
            "motor_current", "torque", "temperature", "humidity",
            "vibration_x", "vibration_y", "vibration_z", "vibration_magnitude",
            "passenger_load", "door_cycle_count", "trips_per_hour",
            "operating_hours", "brake_status", "days_since_last_maintenance"
        ]
        
        # Verify columns
        missing_cols = [col for col in required_cols if col not in df_batch.columns]
        if missing_cols:
            st.error(f"Invalid file schema. Missing required columns: {missing_cols}")
        else:
            st.write(f"Record count: **{len(df_batch)}** rows.")
            
            if st.button("🧠 Run Batch ML Inference", use_container_width=True):
                with st.spinner("Processing logs and executing ML models..."):
                    # 1. Feature Engineering (which handles rolling windows)
                    df_engineered = engineer_features(df_batch, is_training=False)
                    
                    # 2. Preprocess
                    preprocessor = st.session_state["predictor"].preprocessor
                    X, _ = preprocessor.transform(df_engineered)
                    
                    # 3. Classify
                    classifier = st.session_state["predictor"].classifier
                    class_preds_encoded = classifier.predict(X)
                    class_preds = preprocessor.label_encoder.inverse_transform(class_preds_encoded)
                    
                    probs = classifier.predict_proba(X)
                    
                    # 4. Regress
                    regressor = st.session_state["predictor"].regressor
                    rul_preds = regressor.predict(X)
                    rul_preds = np.clip(rul_preds, 0.0, 365.0)
                    
                    # 5. Build results DataFrame
                    df_results = df_batch.copy()
                    df_results["predicted_health_status"] = class_preds
                    df_results["predicted_days_until_failure"] = np.round(rul_preds, 2)
                    
                    # Add probabilities
                    for idx, cls in enumerate(preprocessor.label_encoder.classes_):
                        df_results[f"prob_{cls}"] = np.round(probs[:, idx], 4)
                        
                    # Calculate health scores
                    prob_normal = probs[:, preprocessor.label_encoder.transform(["Normal"])[0]] if "Normal" in preprocessor.label_encoder.classes_ else np.zeros(len(df_results))
                    prob_warning = probs[:, preprocessor.label_encoder.transform(["Warning"])[0]] if "Warning" in preprocessor.label_encoder.classes_ else np.zeros(len(df_results))
                    prob_critical = probs[:, preprocessor.label_encoder.transform(["Critical"])[0]] if "Critical" in preprocessor.label_encoder.classes_ else np.zeros(len(df_results))
                    
                    health_scores = (rul_preds / 365.0) * 50.0 + (prob_normal * 40.0) + (prob_warning * 15.0) - (prob_critical * 20.0)
                    df_results["health_score"] = np.round(np.clip(health_scores + 10.0, 0.0, 100.0), 2)
                    
                st.success("Batch predictions completed!")
                
                # Show results summary
                st.write("##### Prediction Summary Preview")
                st.dataframe(df_results[["operating_hours", "predicted_health_status", "predicted_days_until_failure", "health_score"]].head(10))
                
                # Distribution of predicted classes
                st.write("##### Risk Level Distribution")
                status_counts = df_results["predicted_health_status"].value_counts()
                st.bar_chart(status_counts)
                
                # Download button
                csv_download = df_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Predictions (CSV)",
                    data=csv_download,
                    file_name="elevator_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    else:
        st.info("Please upload a CSV file in the sidebar to begin batch analysis.")
        
        st.write("For testing, you can download the sample input generated in Phase 1:")
        sample_path = PROJECT_ROOT / "data" / "sample_input.csv"
        if sample_path.exists():
            with open(sample_path, "r") as f:
                st.download_button(
                    label="📥 Download Sample Input CSV",
                    data=f.read(),
                    file_name="elevator_sample_input.csv",
                    mime="text/csv"
                )
