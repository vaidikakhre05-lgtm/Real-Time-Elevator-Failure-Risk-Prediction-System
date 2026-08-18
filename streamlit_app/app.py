import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components.sidebar import render_sidebar, init_session_state
from src.models.predict import ElevatorPredictor

# Set page config
st.set_page_config(
    page_title="Elevator Predictive Maintenance System",
    page_icon="🛗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
init_session_state()

# Load custom CSS
css_file = PROJECT_ROOT / "streamlit_app" / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Instantiate the ML Predictor inside session state so we don't reload it constantly
if "predictor" not in st.session_state:
    try:
        st.session_state["predictor"] = ElevatorPredictor(
            classifier_name=st.session_state["active_classifier"]
        )
    except Exception as e:
        st.error(f"Failed to load machine learning models: {e}. Please run the pipeline first.")

# Sidebar rendering
render_sidebar()

# Main landing page content
st.markdown(
    """
    <div style="text-align: center; padding: 20px 0;">
        <h1 class="dashboard-h1">🛗 REAL-TIME ELEVATOR FAILURE RISK PREDICTION</h1>
        <p style="font-size: 1.25rem; color: #94a3b8; max-width: 800px; margin: 0 auto;">
            An industrial IoT predictive maintenance system leveraging machine learning to predict mechanical and electrical elevator failures before they occur.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<h2 class="dashboard-h2">🌐 System Overview</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        This project demonstrates an **end-to-end industrial MLOps application**.
        It simulates real-time telemetry from elevator cabin sensors, evaluates them against physical thresholds, 
        and applies machine learning classifiers and regressors to evaluate risk levels.
        
        ### Key Features:
        1. **Dual ML Task Inference**:
           * **Risk Level Classification**: Evaluates sensor states to classify current elevator health into `Normal`, `Warning`, or `Critical`.
           * **Regression (RUL)**: Predicts the exact **Remaining Useful Life (RUL)** or *Days Until Failure* to optimize maintenance schedules.
        2. **Physics-based Simulation**:
           * Natural degradation curves that wear over time (operating hours, door cycle count, brakes wear).
           * Custom simulated anomalies (overloads, friction heating, cabin misalignment) to verify predictive capabilities.
        3. **Multi-Model Support**:
           * Instantly swap classification models (Random Forest, Decision Tree, Logistic Regression) from the sidebar.
        """
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Enter Operational Live Dashboard", use_container_width=True):
        # In streamlit, navigation can be suggested or we instruct them to use the sidebar
        st.toast("Click on 'Dashboard' in the sidebar to view the live dashboard!", icon="🚀")

with col2:
    st.markdown('<h2 class="dashboard-h2">📊 Machine Learning Pipeline</h2>', unsafe_allow_html=True)
    
    # Render system summary metrics if they exist
    summary_path = PROJECT_ROOT / "reports" / "metrics" / "system_summary.json"
    import json
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary = json.load(f)
            
        st.markdown(
            f"""
            <div class="glass-card kpi-normal">
                <div class="kpi-title">Active Model</div>
                <div class="kpi-val" style="font-size: 1.6rem; color: #38bdf8;">
                    {summary['best_classifier']['name'].replace('_', ' ').upper()}
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 5px;">
                    Classification Accuracy: <strong>{summary['best_classifier']['accuracy']*100:.2f}%</strong><br>
                    Classification F1-Score: <strong>{summary['best_classifier']['f1_score']*100:.2f}%</strong>
                </div>
            </div>
            
            <div class="glass-card kpi-normal">
                <div class="kpi-title">RUL Regressor</div>
                <div class="kpi-val" style="font-size: 1.6rem; color: #818cf8;">
                    LINEAR REGRESSION
                </div>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 5px;">
                    R² Variance Score: <strong>{summary['regressor']['R2_Score']*100:.1f}%</strong><br>
                    Mean Absolute Error: <strong>{summary['regressor']['MAE']:.2f} Days</strong>
                </div>
            </div>
            
            <div class="glass-card" style="text-align: center; border-left: 5px solid #818cf8;">
                <div class="kpi-title">Overall System Composite Quality</div>
                <div class="kpi-val" style="color: #c084fc;">{summary['system_composite_score']:.1f} / 100.0</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Training metrics not found. Please run the model training pipeline first.")
        
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; font-family: monospace;">
        Developed for Technical Interviews & Portfolios | Built with Python, Scikit-Learn, Plotly, & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
