import os
import sys
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.save_models import load_model_artifact

def test_model_loading():
    """
    Checks that trained model objects can be successfully loaded and are valid scikit-learn models.
    """
    # Check if models exist first, otherwise skip
    model_file = PROJECT_ROOT / "models" / "preprocessor.pkl"
    if not model_file.exists():
        pytest.skip("Models not trained yet. Run pipeline first.")
        
    preprocessor = load_model_artifact("preprocessor.pkl")
    assert preprocessor is not None
    assert preprocessor.is_fit
    
    scaler = load_model_artifact("scaler.pkl")
    assert scaler is not None
    
    rf = load_model_artifact("random_forest.pkl", subfolder="classifier")
    assert hasattr(rf, "predict")
    
    lin_reg = load_model_artifact("linear_regression.pkl", subfolder="regressor")
    assert hasattr(lin_reg, "predict")
