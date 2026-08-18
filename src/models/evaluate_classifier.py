import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize
from src.config import FIGURES_DIR, METRICS_DIR, REPORTS_DIR
from src.constants import HEALTH_CLASSES
from src.logger import setup_logger

logger = setup_logger("evaluate_classifier")

def evaluate_classifiers(
    models: Dict[str, Any], 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    classes: List[str] = HEALTH_CLASSES
) -> pd.DataFrame:
    """
    Evaluates all classification models, saves performance charts, and generates a comparison table.
    
    Args:
        models: Dictionary of model_name -> fitted estimator.
        X_test: Test features.
        y_test: Test labels (encoded).
        classes: List of original class names in order of their label encoding.
        
    Returns:
        pd.DataFrame: Performance metrics comparison table.
    """
    logger.info("Evaluating classifiers...")
    metrics_list = []
    n_classes = len(classes)
    
    # Binarize labels for multi-class ROC and Precision-Recall
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        
        # Multi-class ROC-AUC (One-vs-Rest)
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC for {name}: {e}")
            roc_auc = np.nan
            
        metrics_list.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (Macro)": prec,
            "Recall (Macro)": rec,
            "F1-Score (Macro)": f1,
            "ROC-AUC (Macro)": roc_auc
        })
        
        # 1. Generate & Save Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=classes, 
            yticklabels=classes
        )
        plt.title(f"Confusion Matrix - {name.replace('_', ' ').title()}")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_path = FIGURES_DIR / f"confusion_matrix_{name}.png"
        plt.savefig(cm_path, dpi=300)
        plt.close()
        logger.info(f"Saved Confusion Matrix to {cm_path}")
        
        # 2. Generate & Save ROC Curve
        plt.figure(figsize=(7, 6))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC = {roc_auc_val:.2f})")
            
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name.replace('_', ' ').title()}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = FIGURES_DIR / f"roc_curve_{name}.png"
        plt.savefig(roc_path, dpi=300)
        plt.close()
        logger.info(f"Saved ROC Curve to {roc_path}")
        
        # 3. Generate & Save Precision-Recall Curve
        plt.figure(figsize=(7, 6))
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_prob[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f"Class {classes[i]} (AUC = {pr_auc:.2f})")
            
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {name.replace('_', ' ').title()}")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        pr_path = FIGURES_DIR / f"pr_curve_{name}.png"
        plt.savefig(pr_path, dpi=300)
        plt.close()
        logger.info(f"Saved Precision-Recall Curve to {pr_path}")
        
    # Create Comparison DataFrame
    df_comparison = pd.DataFrame(metrics_list)
    
    # Save comparison report
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_path = METRICS_DIR / "classifier_comparison.csv"
    df_comparison.to_csv(comparison_path, index=False)
    logger.info(f"Saved Classifier Comparison to {comparison_path}")
    
    # Also save to main reports directory
    main_comparison_path = REPORTS_DIR / "model_comparison.csv"
    df_comparison.to_csv(main_comparison_path, index=False)
    
    return df_comparison
