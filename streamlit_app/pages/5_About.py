import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_app.components.sidebar import render_sidebar

# Page config
st.set_page_config(
    page_title="Elevator Command - Documentation",
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

st.markdown('<h1 class="dashboard-h1">📖 SYSTEM DOCUMENTATION & METRICS</h1>', unsafe_allow_html=True)

st.markdown(
    """
    ### Overview
    The **Real-Time Elevator Failure Risk Prediction System** is a predictive maintenance application designed to minimize elevator downtime 
    and prevent catastrophic failures in commercial high-rises. By monitoring high-frequency IoT sensor telemetry and operating indicators, 
    the system evaluates cabin degradation, detects anomalies, and calculates remaining useful life.
    
    ---
    
    ### 🏗️ Software Architecture
    The codebase is engineered following clean architecture principles, separating concern layers:
    
    ```
    ┌────────────────────────────────────────────────────────┐
    │                 Streamlit Web UI Layer                 │
    │  (app.py, Dashboard, Live Prediction, style.css, etc.)  │
    └───────────────────────────┬────────────────────────────┘
                                │ Calls
    ┌───────────────────────────▼────────────────────────────┐
    │                      Inference Layer                   │
    │         (ElevatorPredictor - predict.py)               │
    └───────────────────────────┬────────────────────────────┘
                                │ Uses
    ┌───────────────────────────▼────────────────────────────┐
    │                 Data Science & Model Layer             │
    │   (Preprocessing, Feature Engineering, joblib models)  │
    └───────────────────────────┬────────────────────────────┘
                                │ Evolved by
    ┌───────────────────────────▼────────────────────────────┐
    │                 IoT Simulation Engine                  │
    │       (step_simulation - sensor_generator.py)          │
    └────────────────────────────────────────────────────────┘
    ```
    
    ---
    
    ### ⚙️ Telemetry Dictionary
    
    * **Motor Current ($I$, Amps)**: Electrical current drawn by the hoisting motor. High current signifies motor stress, overloading, or high mechanical friction.
    * **Torque ($\tau$, Nm)**: Rotational force delivered by the motor. Strongly correlated with cabin passenger load and brake resistance.
    * **Chamber Temperature ($T$, °C)**: Thermal output of the motor hoist chamber. Overheating indicates bearings wear, lack of lubrication, or cooling fan failures.
    * **Vibration Magnitude ($V$, g)**: Composite acceleration vector sum calculated from 3-axis sensors:
      $$V = \\sqrt{V_x^2 + V_y^2 + V_z^2}$$
      High vibration implies rail misalignment, cable slack, or motor bearings wear.
    * **Passenger Load ($L$, kg)**: Current weight of passengers inside the cabin. Affects electrical drawing limits.
    * **Door Cycle Count ($C$)**: Cumulative count of door opening/closing actions. Important indicator for door actuator mechanical wear.
    * **Operating Hours ($H$)**: Total service life hours of the elevator hoist assembly.
    * **Brake Status**: Binary condition showing brake pad health ('Normal' vs 'Worn').
    * **Days Since Last Maintenance ($M$)**: Time elapsed since technical inspection. Used to evaluate maintenance schedule delays.
    
    ---
    
    ### 🧪 Predictive Physics Equations
    The simulated wear degradation factor ($W$) is modeled as:
    
    $$W = 0.35 \\left( \\frac{H}{H_{max}} \\right) + 0.45 \\left( \\frac{M}{M_{max}} \\right) + 0.10 \\left( \\frac{C}{C_{max}} \\right) + \\Delta_{brake} + \\epsilon$$
    
    Where:
    - $H_{max} = 15,000$ hours.
    - $M_{max} = 300$ days.
    - $C_{max} = 150,000$ cycles.
    - $\\Delta_{brake} = 0.25$ if brake is worn, else $0.0$.
    - $\\epsilon$ represents stochastic Gaussian noise $\\mathcal{N}(0, 0.05)$.
    
    Remaining Useful Life (RUL) days are calculated exponentially as:
    
    $$RUL_{base} = 365 \\cdot e^{-1.8 \\cdot W}$$
    
    The final RUL adjusts downwards based on active electrical, thermal, or vibration anomalies ($A_{max}$):
    
    $$RUL = RUL_{base} \\cdot (1 - 0.85 \\cdot A_{max})$$
    
    ---
    
    ### 🟢 Operator Action Playbook
    
    * **Green (Normal Health Score > 75%)**: Nominal state. Continue standard operations.
    * **Yellow (Warning Health Score 40%-75%)**: Prevention state. Scheduled maintenance required within 14 days. Inspect specific sensor violations (e.g. lubricate motor guide rails if vibration is high; adjust ventilation if temperature is high).
    * **Red (Critical Health Score < 40%)**: Intervention state. Immediately stop elevator operations and dispatch an emergency repair technician. Inspect bearings, cables, and mechanical brake caliper assemblies.
    """
)
