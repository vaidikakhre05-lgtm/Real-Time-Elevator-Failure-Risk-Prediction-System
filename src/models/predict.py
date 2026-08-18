import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from src.config import MODELS_DIR
from src.constants import HEALTH_CLASSES, HEALTH_NORMAL, HEALTH_WARNING, HEALTH_CRITICAL
from src.models.save_models import load_model_artifact
from src.data.feature_engineering import engineer_features
from src.data.preprocessing import DataPreprocessor
from src.logger import setup_logger

logger = setup_logger("predict")

class ElevatorPredictor:
    """
    End-to-end inference engine for real-time elevator failure risk prediction.
    Loads models, preprocessor assets, and performs inference on raw sensor readings.
    """
    def __init__(self, classifier_name: str = "random_forest"):
        self.classifier_name = classifier_name
        self.preprocessor: DataPreprocessor = None
        self.classifier: Any = None
        self.regressor: Any = None
        self._load_assets()
        
    def _load_assets(self) -> None:
        """
        Loads preprocessor and selected models from disk.
        """
        try:
            logger.info("Loading preprocessor and model assets...")
            # Load preprocessor
            self.preprocessor = load_model_artifact("preprocessor.pkl")
            
            # Load classifier
            classifier_file = f"{self.classifier_name}.pkl"
            self.classifier = load_model_artifact(classifier_file, subfolder="classifier")
            
            # Load regressor
            self.regressor = load_model_artifact("linear_regression.pkl", subfolder="regressor")
            logger.info("Assets loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading predictive assets: {e}")
            raise e
            
    def change_classifier(self, classifier_name: str) -> None:
        """
        Changes the active classifier.
        
        Args:
            classifier_name: Name of the classifier to load (logistic_regression, decision_tree, random_forest).
        """
        if classifier_name == self.classifier_name and self.classifier is not None:
            return
            
        logger.info(f"Switching classifier to: {classifier_name}")
        self.classifier_name = classifier_name
        classifier_file = f"{self.classifier_name}.pkl"
        self.classifier = load_model_artifact(classifier_file, subfolder="classifier")
        
    def predict_single(self, raw_sensor_reading: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts health state and days until failure for a single raw sensor reading.
        
        Args:
            raw_sensor_reading: Dict containing the raw sensor features.
            
        Returns:
            Dict containing predictions, probabilities, and health score.
        """
        # Convert dictionary to DataFrame
        df_raw = pd.DataFrame([raw_sensor_reading])
        
        # 1. Feature Engineering
        # In a single-sample inference, we need to handle rolling mean/std.
        # If the input doesn't have rolling features pre-computed, we compute them.
        # But wait! For rolling statistics, we might need a history.
        # If history isn't provided, we fall back to the current value for mean, and 0 for std.
        # Let's ensure the feature engineering handles this robustly!
        
        # Check if engineered features are already provided in the input (e.g., from simulation).
        # Otherwise, engineer them.
        required_cols = (
            self.preprocessor.feature_columns + 
            [self.preprocessor.label_encoder.classes_[0] if hasattr(self.preprocessor, "label_encoder") else ""]
        )
        
        # If the input has only raw features, we engineer them.
        # To avoid rolling errors on a single sample, we append it to a historical dataframe
        # or we just provide default values for rolling.
        df_feat = df_raw.copy()
        
        # Engineer features
        df_feat = engineer_features(df_feat, is_training=False)
        
        # 2. Preprocess (Scale & Encode)
        X, _ = self.preprocessor.transform(df_feat)
        
        # 3. Predict Health Category and Probabilities
        class_pred_encoded = self.classifier.predict(X)[0]
        class_pred = self.preprocessor.label_encoder.inverse_transform([class_pred_encoded])[0]
        
        probs = self.classifier.predict_proba(X)[0]
        class_probs = {
            self.preprocessor.label_encoder.classes_[i]: float(probs[i])
            for i in range(len(probs))
        }
        
        # 4. Predict Days Until Failure (Remaining Useful Life)
        rul_pred = self.regressor.predict(X)[0]
        rul_pred = float(np.clip(rul_pred, 0.0, 365.0))
        
        # 5. Compute overall health score (0 to 100)
        # Normal prob + portion of warning prob - critical prob
        prob_normal = class_probs.get(HEALTH_NORMAL, 0.0)
        prob_warning = class_probs.get(HEALTH_WARNING, 0.0)
        prob_critical = class_probs.get(HEALTH_CRITICAL, 0.0)
        
        # Health score calculation
        health_score = (rul_pred / 365.0) * 50.0 + (prob_normal * 40.0) + (prob_warning * 15.0) - (prob_critical * 20.0)
        health_score = float(np.clip(health_score + 10.0, 0.0, 100.0))
        
        return {
            "predicted_class": class_pred,
            "probabilities": class_probs,
            "predicted_days_until_failure": rul_pred,
            "health_score": health_score
        }
