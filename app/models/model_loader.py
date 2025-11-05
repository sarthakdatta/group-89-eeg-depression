"""
Model Loading and Inference Utilities
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

_model = None
_model_path = None


def load_model(model_path: str = "eeg_depression_model.pkl") -> None:
    """
    Load the trained EEG depression classification model.
    
    Args:
        model_path: Path to the saved model file (.pkl)
    """
    global _model, _model_path
    
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        _model = joblib.load(model_path)
        _model_path = str(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def get_model():
    """Get the loaded model instance."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model


def predict(features: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Make prediction using the loaded model.
    
    Args:
        features: Feature array (1D array of size n_features or 2D array of shape (n_samples, n_features))
        
    Returns:
        Tuple of (prediction, probability)
        - prediction: 0 (not depressed) or 1 (depressed)
        - probability: Array of probabilities [prob_not_depressed, prob_depressed]
    """
    model = get_model()
    
    # Ensure features are 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return int(prediction), probability


def is_model_loaded() -> bool:
    """Check if model is loaded."""
    return _model is not None
