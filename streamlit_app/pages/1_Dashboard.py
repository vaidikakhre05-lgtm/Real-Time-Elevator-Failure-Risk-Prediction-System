import streamlit as st
import time
import pandas as pd
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components.sidebar import render_sidebar
from streamlit_app.components.gauges import render_sensor_gauges
from streamlit_app.components.charts import render_live_trend_chart, render_probability_breakdown_chart, render_rul_history_chart
from streamlit_app.components.alerts import render_health_alerts
from src.simulation.sensor_generator import step_simulation

# Page config
st.set_page_config(
    page_title="Elevator Command - Live Dashboard",
    page_icon="🛗",
    layout="wide"
)

# Load CSS
css_file = PROJECT_ROOT / "streamlit_app" / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render Sidebar (loads state & sets active classifier)
render_sidebar()

# Verify predictor is loaded
if "predictor" not in st.session_state:
    st.error("Predictor not loaded. Please return to Home page or run pipeline.")
    st.stop()

# Switch classifier if changed in sidebar
st.session_state["predictor"].change_classifier(st.session_state["active_classifier"])

st.markdown('<h1 class="dashboard-h1">🛗 LIVE OPERATIONAL CONTROL ROOM</h1>', unsafe_allow_html=True)

# ========================================================
# SIMULATION LOOP STEP
# ========================================================
if st.session_state["simulation_active"]:
    # 1. Step the simulation
    new_sensor, new_wear = step_simulation(
        st.session_state["current_sensor_data"],
        st.session_state["simulation_wear"]
    )
    
    # 2. Get predictions
    pred_res = st.session_state["predictor"].predict_single(new_sensor)
    
    # 3. Save states
    st.session_state["current_sensor_data"] = new_sensor
    st.session_state["simulation_wear"] = new_wear
    
    # 4. Append to history
    ticks = new_wear["ticks"]
    
    # Add record to history
    sensor_rec = new_sensor.copy()
    sensor_rec["ticks"] = ticks
    st.session_state["sensor_history"].append(sensor_rec)
    
    pred_rec = {
        "ticks": ticks,
        "predicted_class": pred_res["predicted_class"],
        "predicted_days_until_failure": pred_res["predicted_days_until_failure"],
        "health_score": pred_res["health_score"]
    }
    # Save probabilities separately
    for cls, prob in pred_res["probabilities"].items():
        pred_rec[f"prob_{cls}"] = prob
    st.session_state["prediction_history"].append(pred_rec)
    
    # Limit history length to 50 items to prevent memory bloat
    if len(st.session_state["sensor_history"]) > 50:
        st.session_state["sensor_history"].pop(0)
    if len(st.session_state["prediction_history"]) > 50:
        st.session_state["prediction_history"].pop(0)

# Run prediction for static states (e.g. if simulation paused or manual sliders changed)
current_pred = st.session_state["predictor"].predict_single(st.session_state["current_sensor_data"])

# Ensure we have at least one record in history for charting
if not st.session_state["sensor_history"]:
    initial_rec = st.session_state["current_sensor_data"].copy()
    initial_rec["ticks"] = 0
    st.session_state["sensor_history"].append(initial_rec)
    
    initial_pred_rec = {
        "ticks": 0,
        "predicted_class": current_pred["predicted_class"],
        "predicted_days_until_failure": current_pred["predicted_days_until_failure"],
        "health_score": current_pred["health_score"]
    }
    for cls, prob in current_pred["probabilities"].items():
        initial_pred_rec[f"prob_{cls}"] = prob
    st.session_state["prediction_history"].append(initial_pred_rec)

# ========================================================
# RENDER LAYOUT
# ========================================================

# Row 1: Gauges (Telemetry Dial KPIs)
st.markdown('<h2 class="dashboard-h2">📊 Live Sensor Telemetry</h2>', unsafe_allow_html=True)
render_sensor_gauges(st.session_state["current_sensor_data"])

# Row 2: ML Inference KPIs
st.markdown('<h2 class="dashboard-h2">🧠 Predictive Maintenance Intelligence</h2>', unsafe_allow_html=True)

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)

# Set dynamic KPI card classes
risk_class = "kpi-normal"
risk_color = "#34d399"
pred_class = current_pred["predicted_class"]

if pred_class == "Warning":
    risk_class = "kpi-warning"
    risk_color = "#fbbf24"
elif pred_class == "Critical":
    risk_class = "kpi-critical"
    risk_color = "#f87171"

with col_kpi1:
    st.markdown(
        f"""
        <div class="glass-card {risk_class}">
            <div class="kpi-title">RISK LEVEL</div>
            <div class="kpi-val" style="color: {risk_color};">{pred_class.upper()}</div>
            <div style="font-size: 0.85rem; color: #94a3b8;">Current Failure Risk</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
with col_kpi2:
    rul_days = current_pred["predicted_days_until_failure"]
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 5px solid #38bdf8;">
            <div class="kpi-title">PREDICTED RUL</div>
            <div class="kpi-val" style="color: #38bdf8;">{rul_days:.1f} Days</div>
            <div style="font-size: 0.85rem; color: #94a3b8;">Days Until Failure</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
with col_kpi3:
    score = current_pred["health_score"]
    score_color = "#10b981" if score > 75 else "#f59e0b" if score > 40 else "#ef4444"
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 5px solid {score_color};">
            <div class="kpi-title">HEALTH SCORE</div>
            <div class="kpi-val" style="color: {score_color};">{score:.1f}%</div>
            <div style="font-size: 0.85rem; color: #94a3b8;">Overall Cabin Condition</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_kpi4:
    # Diagnostic overview: display classifier and simulation tick
    model_name = st.session_state["active_classifier"].replace("_", " ").upper()
    ticks = st.session_state["simulation_wear"]["ticks"]
    st.markdown(
        f"""
        <div class="glass-card" style="border-left: 5px solid #a855f7;">
            <div class="kpi-title">INFERENCE ENGINE</div>
            <div class="kpi-val" style="font-size: 1.25rem; color: #c084fc; line-height: 2.2rem;">
                {model_name}
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8;">Simulation Tick: <strong>{ticks}</strong></div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Alerts Area
render_health_alerts(pred_class, st.session_state["current_sensor_data"])

# Row 3: Trend Charts
col_chart1, col_chart2 = st.columns([2, 1])

# Convert histories to DataFrames for Plotly
hist_sensor_df = pd.DataFrame(st.session_state["sensor_history"])
hist_pred_df = pd.DataFrame(st.session_state["prediction_history"])

with col_chart1:
    # Multi-select for metrics to plot in live trends
    all_numeric_cols = [
        "motor_current", "torque", "temperature", "humidity", "vibration_magnitude", 
        "passenger_load", "days_since_last_maintenance", "operating_hours"
    ]
    selected_metrics = st.multiselect(
        "Select Telemetry Variables to Plot",
        options=all_numeric_cols,
        default=["motor_current", "vibration_magnitude", "temperature"]
    )
    
    live_chart = render_live_trend_chart(hist_sensor_df, selected_metrics)
    st.plotly_chart(live_chart, use_container_width=True)

with col_chart2:
    prob_chart = render_probability_breakdown_chart(current_pred["probabilities"])
    st.plotly_chart(prob_chart, use_container_width=True, config={'displayModeBar': False})
    
    rul_chart = render_rul_history_chart(hist_pred_df)
    st.plotly_chart(rul_chart, use_container_width=True, config={'displayModeBar': False})

# Row 4: Data Export
with st.expander("📥 Export Fleet Telemetry Logs", expanded=False):
    st.markdown("Download simulation runs and risk logs for reporting and compliance.")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        if not hist_sensor_df.empty:
            sensor_csv = hist_sensor_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Telemetry History (CSV)",
                data=sensor_csv,
                file_name=f"elevator_telemetry_tick_{ticks}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with col_d2:
        if not hist_pred_df.empty:
            # Combine histories for detailed report
            merged_history = pd.merge(hist_sensor_df, hist_pred_df, on="ticks")
            pred_csv = merged_history.to_csv(index=False)
            st.download_button(
                label="📥 Download Inference Report (CSV)",
                data=pred_csv,
                file_name=f"elevator_risk_report_tick_{ticks}.csv",
                mime="text/csv",
                use_container_width=True
            )

# Autorefresh / Rerun triggers
if st.session_state["simulation_active"]:
    time.sleep(st.session_state["simulation_speed"])
    st.rerun()
