"""
Streamlit GUI for EEG Depression Classification
"""

import streamlit as st
import numpy as np
import joblib
from pathlib import Path
import requests
import time
from app.preprocessing.eeg_processing import (
    process_eeg_array_to_features,
    load_mat,
    SAMPLING_RATE,
)
from app.models.model_loader import load_model, predict, is_model_loaded

# Page configuration
st.set_page_config(
    page_title="EEG Depression Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-depressed {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .prediction-not-depressed {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'api_mode' not in st.session_state:
    st.session_state.api_mode = False

def load_model_locally():
    """Load model directly in Streamlit."""
    try:
        if not st.session_state.model_loaded:
            with st.spinner("Loading model..."):
                load_model("eeg_depression_model.pkl")
                st.session_state.model_loaded = True
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False

def process_and_predict(file_path, use_api=False, api_url="http://localhost:8000"):
    """Process EEG file and make prediction."""
    try:
        # Load EEG data
        with st.spinner("Loading EEG data..."):
            raw_data = load_mat(file_path, verbose=False)
            st.success(f"✅ Loaded EEG data: {raw_data.shape[0]} channels × {raw_data.shape[1]} samples")
        
        # Validate data
        if raw_data.ndim != 2:
            st.error(f"Expected 2D array, got {raw_data.ndim}D")
            return None
        
        if raw_data.shape[0] not in [128, 129]:
            st.warning(f"Expected 128 or 129 channels, got {raw_data.shape[0]}")
        
        if use_api:
            # Use API endpoint
            return predict_via_api(file_path, api_url)
        else:
            # Process locally
            with st.spinner("Processing EEG data and extracting features..."):
                progress_bar = st.progress(0)
                
                # Extract features
                features = process_eeg_array_to_features(raw_data, SAMPLING_RATE)
                progress_bar.progress(50)
                
                # Make prediction
                if not is_model_loaded():
                    if not load_model_locally():
                        return None
                
                prediction, probability = predict(features)
                progress_bar.progress(100)
                
                return {
                    "prediction": prediction,
                    "probability": probability,
                    "confidence": float(max(probability)),
                    "status": "success"
                }
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def predict_via_api(file_path, api_url):
    """Send file to FastAPI backend for prediction."""
    try:
        with st.spinner("Sending data to API..."):
            with open(file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(
                    f"{api_url}/api/v1/predict",
                    files=files,
                    timeout=60
                )
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the FastAPI server is running.")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def display_results(result):
    """Display prediction results in a nice format."""
    if result is None:
        return
    
    prediction = result["prediction"]
    probability = result["probability"]
    confidence = result.get("confidence", max(probability))
    
    # Determine styling
    is_depressed = prediction == 1
    box_class = "prediction-depressed" if is_depressed else "prediction-not-depressed"
    emoji = "🔴" if is_depressed else "🟢"
    label = "Depressed" if is_depressed else "Not Depressed"
    
    # Display main prediction
    st.markdown(f"""
    <div class="prediction-box {box_class}">
        <h2 style="text-align: center; margin-bottom: 1rem;">
            {emoji} Prediction: <strong>{label}</strong>
        </h2>
        <div style="text-align: center; font-size: 1.5rem; margin-bottom: 1rem;">
            Confidence: <strong>{confidence*100:.1f}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display probability breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Not Depressed",
            value=f"{probability[0]*100:.2f}%",
            delta=f"{probability[0]*100:.2f}% probability"
        )
        st.progress(float(probability[0]))
    
    with col2:
        st.metric(
            label="Depressed",
            value=f"{probability[1]*100:.2f}%",
            delta=f"{probability[1]*100:.2f}% probability"
        )
        st.progress(float(probability[1]))
    
    # Additional info
    st.info(f"""
    **Model Prediction Summary:**
    - Classification: **{label}**
    - Confidence Score: **{confidence*100:.1f}%**
    - Not Depressed Probability: **{probability[0]*100:.2f}%**
    - Depressed Probability: **{probability[1]*100:.2f}%**
    """)

# Main UI
def main():
    # Header
    st.markdown('<p class="main-header">🧠 EEG Depression Classification</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Choose mode
        mode = st.radio(
            "Select Mode:",
            ["Standalone (Load Model Locally)", "API Mode (Connect to FastAPI)"],
            help="Standalone mode loads the model directly. API mode connects to your FastAPI backend."
        )
        
        use_api = mode == "API Mode (Connect to FastAPI)"
        
        if use_api:
            api_url = st.text_input(
                "API URL:",
                value="http://localhost:8000",
                help="URL of your FastAPI server"
            )
            
            # Test API connection
            if st.button("🔌 Test API Connection"):
                try:
                    response = requests.get(f"{api_url}/api/v1/health", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        st.success("✅ API Connected!")
                        st.json(data)
                    else:
                        st.error(f"❌ API returned status {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Cannot connect: {str(e)}")
        else:
            # Load model button
            if st.button("🔄 Load Model"):
                if load_model_locally():
                    st.success("✅ Model loaded successfully!")
        
        st.markdown("---")
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. Upload a `.mat` file containing EEG data
        2. File should have 128 or 129 channels
        3. Wait for processing
        4. View prediction results
        """)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This application classifies depressive symptoms
        using EEG data from the MODMA dataset.
        
        **Model:** Logistic Regression
        **Features:** Power Spectral Density (Welch's method)
        **Frequency Bands:** Delta, Theta, Alpha, Beta, Gamma
        """)
    
    # Main content area
    st.header("📤 Upload EEG Data")
    
    uploaded_file = st.file_uploader(
        "Choose a .mat file",
        type=['mat'],
        help="Upload EEG data in MATLAB .mat format"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Process button
            if st.button("🚀 Analyze EEG Data", type="primary", use_container_width=True):
                result = process_and_predict(tmp_path, use_api=use_api, api_url=api_url if use_api else None)
                
                if result:
                    st.markdown("---")
                    st.header("📊 Results")
                    display_results(result)
                    
                    # Download results as JSON
                    import json
                    results_json = json.dumps(result, indent=2)
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=results_json,
                        file_name="prediction_results.json",
                        mime="application/json"
                    )
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    else:
        # Show example/demo
        st.info("👆 Please upload a .mat file to get started")
        
        with st.expander("📖 Learn More"):
            st.markdown("""
            ### How It Works
            
            1. **Preprocessing**: The EEG signal is filtered and cleaned
               - Bandpass filter (1-45 Hz)
               - Notch filter (50 Hz line noise removal)
               - Edge trimming
            
            2. **Feature Extraction**: Power spectral density is calculated
               - 5 frequency bands: Delta, Theta, Alpha, Beta, Gamma
               - 128 channels × 5 bands = 640 features
            
            3. **Classification**: Trained model predicts depression status
               - Binary classification: Depressed (1) or Not Depressed (0)
               - Probability scores for each class
            
            ### Data Requirements
            - Format: MATLAB .mat file
            - Channels: 128 or 129 channels
            - Sampling Rate: 250 Hz (default)
            - Duration: Minimum ~60 seconds (after trimming)
            """)

if __name__ == "__main__":
    main()

