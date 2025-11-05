"""
FastAPI Endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import logging
from typing import Dict, Any

from app.models.model_loader import predict, is_model_loaded
from app.preprocessing.eeg_processing import (
    process_eeg_array_to_features,
    load_mat,
    SAMPLING_RATE,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Status of the API and model
    """
    model_status = "loaded" if is_model_loaded() else "not loaded"
    
    return {
        "status": "healthy",
        "model_status": model_status,
        "message": "EEG Depression Classification API is running"
    }


@router.post("/predict")
async def predict_depression(
    file: UploadFile = File(..., description="EEG data file (.mat format)")
) -> Dict[str, Any]:
    """
    Predict depression from uploaded EEG file.
    
    Args:
        file: Uploaded .mat file containing EEG data
        
    Returns:
        Prediction result with class and probabilities
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file is available."
        )
    
    # Validate file type
    if not file.filename.endswith('.mat'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a .mat file."
        )
    
    try:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Load and process EEG data
            logger.info(f"Processing EEG file: {file.filename}")
            raw_data = load_mat(tmp_path, verbose=False)
            
            # Validate data shape
            if raw_data.ndim != 2:
                raise ValueError(f"Expected 2D array (channels x time), got {raw_data.ndim}D")
            
            if raw_data.shape[0] not in [128, 129]:
                raise ValueError(
                    f"Expected 128 or 129 channels, got {raw_data.shape[0]} channels"
                )
            
            # Extract features
            features = process_eeg_array_to_features(raw_data, SAMPLING_RATE)
            
            # Make prediction
            prediction, probability = predict(features)
            
            # Format response
            result = {
                "prediction": int(prediction),
                "prediction_label": "depressed" if prediction == 1 else "not_depressed",
                "probabilities": {
                    "not_depressed": float(probability[0]),
                    "depressed": float(probability[1])
                },
                "confidence": float(max(probability)),
                "file_name": file.filename,
                "status": "success"
            }
            
            logger.info(f"Prediction completed for {file.filename}: {result['prediction_label']}")
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.error(f"Error processing EEG file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing EEG file: {str(e)}"
        )


@router.post("/predict/array")
async def predict_depression_from_array(
    eeg_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Predict depression from EEG data array (for testing/development).
    
    Args:
        eeg_data: Dictionary with 'data' key containing 2D array (channels x time_samples)
                  and optional 'sampling_rate' (default: 250)
        
    Returns:
        Prediction result with class and probabilities
    """
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file is available."
        )
    
    try:
        # Extract data from request
        data_array = np.array(eeg_data.get("data"), dtype=np.float32)
        sampling_rate = eeg_data.get("sampling_rate", SAMPLING_RATE)
        
        # Validate data shape
        if data_array.ndim != 2:
            raise ValueError(f"Expected 2D array (channels x time), got {data_array.ndim}D")
        
        # Extract features
        features = process_eeg_array_to_features(data_array, sampling_rate)
        
        # Make prediction
        prediction, probability = predict(features)
        
        # Format response
        result = {
            "prediction": int(prediction),
            "prediction_label": "depressed" if prediction == 1 else "not_depressed",
            "probabilities": {
                "not_depressed": float(probability[0]),
                "depressed": float(probability[1])
            },
            "confidence": float(max(probability)),
            "status": "success"
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing EEG array: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing EEG data: {str(e)}"
        )
