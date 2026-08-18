import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components.sidebar import render_sidebar

# Page config
st.set_page_config(
    page_title="Elevator Command - Feature Importance",
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

# Verify predictor is loaded
if "predictor" not in st.session_state:
    st.error("Predictor not loaded. Please return to Home page or run pipeline.")
    st.stop()

# Switch classifier if changed in sidebar
st.session_state["predictor"].change_classifier(st.session_state["active_classifier"])

st.markdown('<h1 class="dashboard-h1">🔑 FEATURE IMPORTANCE & EXPLAINABILITY</h1>', unsafe_allow_html=True)
st.write(
    "Understanding which sensors contribute most to elevator failure predictions is crucial for operators. "
    "Below we extract importances and coefficients directly from the active machine learning model."
)

col_exp, col_heat = st.columns([3, 2])

with col_exp:
    st.markdown('<h2 class="dashboard-h2">Model Explainability Output</h2>', unsafe_allow_html=True)
    
    # Get active classifier model and columns
    model = st.session_state["predictor"].classifier
    preprocessor = st.session_state["predictor"].preprocessor
    feature_cols = preprocessor.feature_columns
    
    # Clean feature names for clean UI display
    clean_cols = [c.replace("_", " ").title() for c in feature_cols]
    
    if hasattr(model, "feature_importances_"):
        # Decision Tree or Random Forest
        importances = model.feature_importances_
        df_imp = pd.DataFrame({
            "Sensor Feature": clean_cols,
            "Importance (MDI)": importances
        }).sort_values(by="Importance (MDI)", ascending=True)
        
        fig = px.bar(
            df_imp,
            x="Importance (MDI)",
            y="Sensor Feature",
            orientation="h",
            title=f"FEATURE IMPORTANCES - {st.session_state['active_classifier'].replace('_', ' ').upper()}",
            color="Importance (MDI)",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            font={'color': "#e2e8f0", 'family': "Outfit"},
            height=450,
            margin=dict(l=40, r=20, t=40, b=30),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "**Mean Decrease in Impurity (MDI)** represents how much each feature decreases entropy/impurity "
            "across all trees in the forest. A higher value indicates the sensor is critical in partitioning Normal vs Risk states."
        )
        
    elif hasattr(model, "coef_"):
        # Logistic Regression
        coefs = model.coef_ # Shape (n_classes, n_features)
        classes = preprocessor.label_encoder.classes_
        
        st.write(f"##### Logistic Regression Coefficients (Log-Odds Impact)")
        
        # Select class to display coefficients for
        selected_class = st.selectbox("Select Target Class", options=classes)
        class_idx = list(classes).index(selected_class)
        
        df_coef = pd.DataFrame({
            "Sensor Feature": clean_cols,
            "Coefficient Magnitude": coefs[class_idx]
        }).sort_values(by="Coefficient Magnitude", ascending=True)
        
        fig = px.bar(
            df_coef,
            x="Coefficient Magnitude",
            y="Sensor Feature",
            orientation="h",
            title=f"COEFFICIENTS FOR CLASS: {selected_class.upper()}",
            color="Coefficient Magnitude",
            color_continuous_scale="RdBu"
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15, 23, 42, 0.5)',
            font={'color': "#e2e8f0", 'family': "Outfit"},
            height=450,
            margin=dict(l=40, r=20, t=40, b=30),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "Logistic Regression coefficients indicate the log-odds change in target class probability "
            "per unit increase of the scaled sensor feature. Positives increase likelihood; negatives decrease it."
        )
    else:
        st.warning("Active model does not support automated feature importance extraction.")

with col_heat:
    st.markdown('<h2 class="dashboard-h2">Telemetry Correlations</h2>', unsafe_allow_html=True)
    st.write("Heatmap showing correlations of sensor variables generated in Phase 1.")
    
    heatmap_img = PROJECT_ROOT / "reports" / "figures" / "correlation_heatmap.png"
    if heatmap_img.exists():
        st.image(str(heatmap_img), use_container_width=True)
    else:
        st.info("Correlation heatmap image not found.")
