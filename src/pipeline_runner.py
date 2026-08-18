import os
import sys
from pathlib import Path

# Add project root to path for easy imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.helpers import create_directory_structure, save_json
from src.data.load_data import run_data_pipeline, get_or_generate_dataset
from src.visualization.plots import generate_eda_plots
from src.models.save_models import save_preprocessor_assets
from src.models.train_classifier import train_classifiers
from src.models.train_regressor import train_regressor
from src.models.evaluate_classifier import evaluate_classifiers
from src.models.evaluate_regressor import evaluate_regressor_model
from src.utils.metrics import calculate_predictive_maintenance_score
from src.logger import setup_logger

logger = setup_logger("pipeline_runner")

def run_full_pipeline() -> None:
    """
    Executes the entire end-to-end Machine Learning pipeline:
    - Setup directory structures
    - Load/Generate data
    - EDA Plot generation
    - Preprocessing and Feature Engineering
    - Model training (Logistic Regression, Decision Tree, Random Forest, Linear Regression)
    - Save models and preprocessor artifacts
    - Model Evaluation and metrics compilation
    """
    logger.info("=" * 60)
    logger.info("Starting Real-Time Elevator Failure Prediction ML Pipeline")
    logger.info("=" * 60)
    
    # 1. Initialize directory structure
    logger.info("Step 1: Setting up folder directory structures...")
    create_directory_structure(PROJECT_ROOT)
    
    # 2. Get/generate data and generate EDA plots
    logger.info("Step 2: Loading dataset and generating EDA plots...")
    df_raw = get_or_generate_dataset()
    generate_eda_plots(df_raw)
    
    # 3. Preprocessing & Feature Engineering
    logger.info("Step 3: Engineering features and preprocessing data...")
    X_train, X_test, y_train_dict, y_test_dict, preprocessor = run_data_pipeline()
    
    # 4. Save Preprocessor Assets
    logger.info("Step 4: Saving preprocessor assets...")
    save_preprocessor_assets(preprocessor)
    
    # 5. Train & Evaluate Classifiers
    logger.info("Step 5: Training classification models...")
    trained_classifiers = train_classifiers(X_train, y_train_dict["y_class"])
    
    logger.info("Step 6: Evaluating classification models...")
    classifier_comparison_df = evaluate_classifiers(
        trained_classifiers, 
        X_test, 
        y_test_dict["y_class"],
        preprocessor.label_encoder.classes_.tolist()
    )
    print("\nClassification Model Comparison Table:")
    print(classifier_comparison_df.to_string(index=False))
    
    # Get the best F1 model info
    best_classifier_row = classifier_comparison_df.loc[classifier_comparison_df["F1-Score (Macro)"].idxmax()]
    logger.info(f"Best Classifier: {best_classifier_row['Model']} (F1: {best_classifier_row['F1-Score (Macro)']:.4f})")
    
    # 6. Train & Evaluate Regressor
    logger.info("Step 7: Training regressor model (Linear Regression)...")
    regressor_model = train_regressor(X_train, y_train_dict["y_reg"])
    
    logger.info("Step 8: Evaluating regressor model...")
    regressor_metrics = evaluate_regressor_model(regressor_model, X_test, y_test_dict["y_reg"])
    print("\nRegressor Model Performance:")
    for metric, value in regressor_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
            
    # 7. System Composite Score
    logger.info("Step 9: Calculating system composite score...")
    accuracy = float(best_classifier_row["Accuracy"])
    f1 = float(best_classifier_row["F1-Score (Macro)"])
    r2 = float(regressor_metrics["R2_Score"])
    rmse = float(regressor_metrics["RMSE"])
    
    system_score = calculate_predictive_maintenance_score(accuracy, f1, r2, rmse)
    logger.info(f"System Composite Performance Score: {system_score:.2f} / 100.0")
    
    # Save composite summary
    summary = {
        "best_classifier": {
            "name": best_classifier_row["Model"],
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc": float(best_classifier_row["ROC-AUC (Macro)"])
        },
        "regressor": regressor_metrics,
        "system_composite_score": system_score
    }
    summary_path = PROJECT_ROOT / "reports" / "metrics" / "system_summary.json"
    save_json(summary, summary_path)
    logger.info(f"Saved system summary JSON to {summary_path}")
    
    logger.info("=" * 60)
    logger.info("ML Pipeline execution successfully completed!")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_full_pipeline()
