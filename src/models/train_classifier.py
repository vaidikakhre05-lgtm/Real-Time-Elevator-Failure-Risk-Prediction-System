import numpy as np
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.config import RANDOM_STATE, CV_FOLDS
from src.logger import setup_logger
from src.models.save_models import save_model_artifact

logger = setup_logger("train_classifier")

def train_classifiers(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
    """
    Trains Logistic Regression, Decision Tree (GridSearchCV), and Random Forest (GridSearchCV)
    classifiers to predict elevator health status.
    
    Args:
        X_train: Preprocessed features.
        y_train: Encoded health status labels.
        
    Returns:
        Dict: Dictionary of trained models.
    """
    trained_models = {}
    
    # 1. Logistic Regression
    logger.info("Training Logistic Regression Classifier...")
    log_reg = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )
    log_reg.fit(X_train, y_train)
    trained_models["logistic_regression"] = log_reg
    save_model_artifact(log_reg, "logistic_regression.pkl", subfolder="classifier")
    
    # 2. Decision Tree with GridSearchCV
    logger.info("Training Decision Tree Classifier (GridSearchCV)...")
    dt_base = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    dt_param_grid = {
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"]
    }
    dt_grid = GridSearchCV(
        dt_base, 
        dt_param_grid, 
        cv=CV_FOLDS, 
        scoring="f1_macro", 
        n_jobs=-1
    )
    dt_grid.fit(X_train, y_train)
    logger.info(f"Decision Tree - Best Params: {dt_grid.best_params_}")
    trained_models["decision_tree"] = dt_grid.best_estimator_
    save_model_artifact(dt_grid.best_estimator_, "decision_tree.pkl", subfolder="classifier")
    
    # 3. Random Forest with GridSearchCV
    logger.info("Training Random Forest Classifier (GridSearchCV)...")
    rf_base = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    rf_param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [10, 15, None],
        "min_samples_split": [2, 5]
    }
    rf_grid = GridSearchCV(
        rf_base, 
        rf_param_grid, 
        cv=CV_FOLDS, 
        scoring="f1_macro", 
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)
    logger.info(f"Random Forest - Best Params: {rf_grid.best_params_}")
    trained_models["random_forest"] = rf_grid.best_estimator_
    save_model_artifact(rf_grid.best_estimator_, "random_forest.pkl", subfolder="classifier")
    
    logger.info("All classifiers trained and saved successfully.")
    return trained_models
