import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components.sidebar import render_sidebar

# Page config
st.set_page_config(
    page_title="Elevator Command - Model Comparison",
    page_icon="🛗",
    layout="wide"
)

# Load CSS
css_file = PROJECT_ROOT / "streamlit_app" / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Render Sidebar
render_sidebar()

st.markdown('<h1 class="dashboard-h1">⚖️ MODEL COMPARISON BOARD</h1>', unsafe_allow_html=True)

# 1. Load Metrics CSV
metrics_path = PROJECT_ROOT / "reports" / "model_comparison.csv"
regressor_metrics_path = PROJECT_ROOT / "reports" / "metrics" / "regressor_metrics.csv"

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<h2 class="dashboard-h2">Classification Models Comparison</h2>', unsafe_allow_html=True)
    if metrics_path.exists():
        df_metrics = pd.read_csv(metrics_path)
        
        # Style dataframe for display
        styled_df = df_metrics.style.format({
            "Accuracy": "{:.4%}",
            "Precision (Macro)": "{:.4%}",
            "Recall (Macro)": "{:.4%}",
            "F1-Score (Macro)": "{:.4%}",
            "ROC-AUC (Macro)": "{:.4f}"
        }).highlight_max(
            subset=["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)", "ROC-AUC (Macro)"],
            color="#064e3b" # Dark green highlight
        )
        st.dataframe(styled_df, use_container_width=True)
        
        # Download button for metrics table
        metrics_csv = df_metrics.to_csv(index=False)
        st.download_button(
            label="📥 Download Classification Metrics Table",
            data=metrics_csv,
            file_name="classifier_model_comparison.csv",
            mime="text/csv"
        )
    else:
        st.warning("Classifier metrics not found. Run pipeline runner to generate metrics.")

with col2:
    st.markdown('<h2 class="dashboard-h2">Regression Model (RUL)</h2>', unsafe_allow_html=True)
    if regressor_metrics_path.exists():
        df_reg = pd.read_csv(regressor_metrics_path)
        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 5px solid #818cf8;">
                <div style="font-family: 'Share Tech Mono'; font-size: 1.25rem; color: #818cf8;">LINEAR REGRESSION</div>
                <hr style="border-color: rgba(255,255,255,0.08); margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>Mean Absolute Error (MAE):</span>
                    <strong>{df_reg['MAE'].values[0]:.2f} Days</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span>Root Mean Squared Error (RMSE):</span>
                    <strong>{df_reg['RMSE'].values[0]:.2f} Days</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>Coefficient of Determination (R²):</span>
                    <strong>{df_reg['R2_Score'].values[0]:.4f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Regressor metrics not found. Run pipeline runner.")

st.markdown("---")

# 2. Select model to view figures
st.markdown('<h2 class="dashboard-h2">Model Evaluation Visualizations</h2>', unsafe_allow_html=True)
st.write("Toggle classification models below to inspect their confusion matrices and curves generated during Phase 1 training.")

model_select = st.selectbox(
    "Select Model for Detailed Analysis",
    options=["random_forest", "decision_tree", "logistic_regression"],
    format_func=lambda x: x.replace("_", " ").title()
)

fig_col1, fig_col2, fig_col3 = st.columns(3)

figures_dir = PROJECT_ROOT / "reports" / "figures"

with fig_col1:
    cm_img = figures_dir / f"confusion_matrix_{model_select}.png"
    if cm_img.exists():
        st.image(str(cm_img), caption="Confusion Matrix", use_container_width=True)
    else:
        st.info("Confusion Matrix image not found.")

with fig_col2:
    roc_img = figures_dir / f"roc_curve_{model_select}.png"
    if roc_img.exists():
        st.image(str(roc_img), caption="ROC Curve", use_container_width=True)
    else:
        st.info("ROC Curve image not found.")

with fig_col3:
    pr_img = figures_dir / f"pr_curve_{model_select}.png"
    if pr_img.exists():
        st.image(str(pr_img), caption="Precision-Recall Curve", use_container_width=True)
    else:
        st.info("PR Curve image not found.")

# Regressor predicted vs actual plot
st.markdown("---")
st.markdown('<h2 class="dashboard-h2">RUL Regressor Evaluation</h2>', unsafe_allow_html=True)
reg_img = figures_dir / "regressor_predicted_vs_actual_linear_regression.png"
if reg_img.exists():
    st.image(str(reg_img), caption="Linear Regression: Predicted vs. Actual Days Until Failure", width=600)
else:
    st.info("Regressor plot image not found.")
