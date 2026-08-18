# Real-Time Elevator Failure Risk Prediction System

A production-quality, interview-ready **Predictive Maintenance System** for elevators. This application processes real-time IoT sensor telemetry, applies machine learning models to classify failure risks (`Normal`, `Warning`, `Critical`), and estimates the **Remaining Useful Life (RUL)** in days using regression.

---

## 🛗 Project Architecture

The system is designed following clean architecture and SOLID principles:

```
Real-Time-Elevator-Failure-Risk-Prediction-System/
│
├── data/                  # Telemetry datasets (Raw, Processed, Synthetic)
├── models/                # Saved scikit-learn models & preprocessors
├── reports/               # Performance charts & metrics comparisons
├── src/                   # Production modules
│   ├── data/              # Ingestion, synthetic generator, preprocessing
│   ├── models/            # Classifier & regressor training/evaluation scripts
│   ├── simulation/        # Real-time state simulators & maintenance logic
│   └── utils/             # Helpers, validation, custom performance metrics
├── streamlit_app/         # Multi-page dashboard interface
├── tests/                 # Comprehensive unit test suite
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## 🔬 Physics-Based Degradation & Telemetry Model

The elevator simulation implements physically consistent degradation pathways, rather than arbitrary random steps:
1. **Cumulative Wear ($W$)**: Models hoist degradation over operating hours ($H$), door cycles ($C$), and maintenance delays ($M$):
   $$W = 0.35 \left( \frac{H}{H_{max}} \right) + 0.45 \left( \frac{M}{M_{max}} \right) + 0.10 \left( \frac{C}{C_{max}} \right) + \Delta_{brake} + \mathcal{N}(0, 0.05)$$
2. **Current & Torque**: Elevate naturally with load weights and motor friction wear.
3. **Chamber Temperature**: Follows thermal inertia equations, heating up under heavy torque and cooling back toward ambient baselines.
4. **Vibration Magnitude**: Formulates the vector sum of X, Y, and Z axes ($V = \sqrt{V_x^2 + V_y^2 + V_z^2}$), spiking during bearing anomalies or rail misalignment.
5. **Remaining Useful Life (RUL)**: Computed exponentially from wear and reduced by active anomalies:
   $$\text{RUL} = 365 \cdot e^{-1.8 \cdot W} \cdot (1 - 0.85 \cdot A_{max})$$

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Data & Model Pipeline
Execute the full training pipeline (generates 20,000 synthetic samples, engineers features, trains all models, and exports evaluation figures):
```bash
python src/pipeline_runner.py
```

### 4. Launch the Dashboard
Start the Streamlit control room:
```bash
streamlit run streamlit_app/app.py
```

---

## 📊 Model Performance Results

The classifiers and regressor were trained on a stratified train/test split (16,000 train / 4,000 test) using `GridSearchCV` for hyperparameter optimization:

### Risk Level Classification (Normal, Warning, Critical)
* **Random Forest Classifier**: **99.95% Accuracy** (Macro F1-Score: **0.9994**, ROC-AUC: **1.0000**) - *Selected Model*
* **Decision Tree Classifier**: **99.83% Accuracy** (Macro F1-Score: **0.9978**)
* **Logistic Regression**: **96.45% Accuracy** (Macro F1-Score: **0.9631**)

### Remaining Useful Life (RUL) Regression
* **Linear Regression**: **$R^2$ Score of 0.9663** (Mean Absolute Error: **11.18 Days**)

### Overall Quality Score
* **System Composite Score**: **94.32 / 100.0**

All metric plots (confusion matrices, ROC/PR curves, predicted vs. actual) are saved under `reports/figures/`.

---

## 🛠️ Operational Control Room Layout

The Streamlit dashboard features a professional dark control room look containing:
1. **Sidebar**: Select active ML classifier, play/pause live simulation, set refresh speeds, or upload a custom CSV sensor log.
2. **Perform Maintenance Trigger**: A sidebar button that resets elevator wear factors, clears anomalies, and immediately restores the health score to 100%.
3. **Operational Dashboard**: Dial gauges showing current, temperature, torque, and vibration vector magnitudes; predictive status cards; and rolling timeline charts.
4. **Model Comparison Board**: Side-by-side performance grids and direct loading of confusion matrices/ROC curves.
5. **Explainability Board**: Dynamically extracts feature importances (MDI) from Random Forest or coefficients from Logistic Regression.

---

## 🧠 Interview Talking Points

If presenting this project in technical interviews, highlight the following design patterns:
* **Separation of Concerns**: UI widgets are decoupled from prediction models and simulators. Models can be retrained and swapped without breaking the Streamlit layout.
* **Physics Integration**: Showed how domain knowledge (physics-based degradation curves and vector math) was used to structure synthetic data generation, rather than feeding pure white noise.
* **Inference Guardrails**: Implemented schema validation in `src/utils/validation.py` to intercept physically impossible values (e.g. negative current) before calling `model.predict()`.
* **Stateful Simulation Loop**: Handled real-time auto-refresh, historical log buffers, and maintenance resets inside Streamlit using `st.session_state` and modular function callbacks.
