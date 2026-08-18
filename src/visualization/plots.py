import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.config import FIGURES_DIR
from src.logger import setup_logger

logger = setup_logger("plots")

def generate_eda_plots(df: pd.DataFrame) -> None:
    """
    Generates Exploratory Data Analysis (EDA) plots and saves them to reports/figures/.
    Includes:
    - Correlation Heatmap
    - Feature Distributions (Vibration, Temp, Current, Torque)
    - Class Distribution (Health Status)
    
    Args:
        df: Input DataFrame with features and targets.
    """
    logger.info("Generating EDA plots...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Class Distribution
    if "health_status" in df.columns:
        plt.figure(figsize=(7, 5))
        order = ["Normal", "Warning", "Critical"]
        # Ensure order matches existing classes
        present_order = [o for o in order if o in df["health_status"].unique()]
        sns.countplot(
            data=df, 
            x="health_status", 
            order=present_order, 
            palette={"Normal": "green", "Warning": "orange", "Critical": "red"}
        )
        plt.title("Elevator Health Status Distribution")
        plt.xlabel("Health Status")
        plt.ylabel("Count")
        plt.tight_layout()
        class_dist_path = FIGURES_DIR / "class_distribution.png"
        plt.savefig(class_dist_path, dpi=300)
        plt.close()
        logger.info(f"Saved Class Distribution Plot to {class_dist_path}")
        
    # 2. Correlation Heatmap (numerical features)
    numerical_df = df.select_dtypes(include=[np.number])
    if not numerical_df.empty:
        plt.figure(figsize=(12, 10))
        # Use a subset of important columns to make heatmap readable
        important_cols = [
            "motor_current", "torque", "temperature", "humidity", 
            "vibration_magnitude", "passenger_load", "operating_hours", 
            "days_since_last_maintenance", "days_until_failure"
        ]
        cols_to_corr = [c for c in important_cols if c in numerical_df.columns]
        
        corr = numerical_df[cols_to_corr].corr()
        sns.heatmap(
            corr, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm", 
            vmin=-1, 
            vmax=1, 
            square=True, 
            linewidths=0.5
        )
        plt.title("Correlation Matrix of Sensor and Target Features")
        plt.tight_layout()
        heatmap_path = FIGURES_DIR / "correlation_heatmap.png"
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        logger.info(f"Saved Correlation Heatmap to {heatmap_path}")
        
    # 3. Feature Distributions (Grid of Subplots)
    features_to_plot = ["motor_current", "torque", "temperature", "vibration_magnitude"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features_to_plot):
        if feature in df.columns:
            sns.histplot(
                data=df, 
                x=feature, 
                kde=True, 
                ax=axes[idx], 
                color="indigo",
                bins=30
            )
            axes[idx].set_title(f"Distribution of {feature.replace('_', ' ').title()}")
            axes[idx].set_xlabel(feature.replace("_", " ").title())
            axes[idx].set_ylabel("Density/Count")
            
    plt.tight_layout()
    dist_path = FIGURES_DIR / "feature_distributions.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    logger.info(f"Saved Feature Distributions to {dist_path}")
    
    # 4. Pairplot (sample subset to keep execution extremely fast)
    pairplot_cols = ["motor_current", "temperature", "vibration_magnitude", "days_until_failure"]
    cols_to_pair = [c for c in pairplot_cols if c in df.columns]
    if len(cols_to_pair) > 1:
        # Sample 1000 rows to make it fast
        df_sample = df.sample(min(1000, len(df)), random_state=42)
        hue_col = "health_status" if "health_status" in df_sample.columns else None
        
        g = sns.pairplot(
            df_sample[cols_to_pair + ([hue_col] if hue_col else [])], 
            hue=hue_col, 
            palette={"Normal": "green", "Warning": "orange", "Critical": "red"} if hue_col else None,
            diag_kind="kde",
            plot_kws={"alpha": 0.5, "s": 20}
        )
        g.fig.suptitle("Pairplot of Sensor Features and Days Until Failure", y=1.02)
        pairplot_path = FIGURES_DIR / "pairplot.png"
        g.savefig(pairplot_path, dpi=300)
        plt.close()
        logger.info(f"Saved Pairplot to {pairplot_path}")
        
    logger.info("EDA plotting completed.")
