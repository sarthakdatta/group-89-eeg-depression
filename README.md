# EEG Depression Classification

An application for classifying patients' depressive symptoms using EEG data from the MODMA dataset.

## Project Overview

This project consists of:
- **Model Development**: Machine learning model trained on MODMA dataset EEG data
- **FastAPI Backend**: REST API for EEG depression classification
- **Streamlit GUI**: User-friendly web interface for uploading and analyzing EEG data

## Team Members

- **Takehiro Ishikawa** (Model Development Lead)
- **Sayak Datta** (Backend/Deployment Lead)

## Features

- 🧠 EEG signal preprocessing (bandpass filtering, notch filtering, noise removal)
- 📊 Feature extraction using Welch's method (5 frequency bands across 128 channels)
- 🤖 Deep learning classification (MLP, Keras)
- 🌐 RESTful API with FastAPI
- 💻 Interactive web interface with Streamlit
- 🐳 Docker support for easy deployment

## Installation

### Prerequisites

- Python 3.11+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd group-89-eeg-depression
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure the model file is present:
```bash
ls eeg_depression_model.keras
```

## Usage

### Option 1: Streamlit GUI (Recommended for Users)

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

Or use the helper script:
```bash
python run_streamlit.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**
- Upload `.mat` files with EEG data
- Two modes: Standalone (loads model locally) or API mode (connects to FastAPI)
- Visual results with probability breakdowns
- Download prediction results as JSON

### Option 2: FastAPI Backend

Start the API server:

```bash
python run_api.py
```

Or:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

**API Endpoints:**
- `GET /api/v1/health` - Health check and model status
- `POST /api/v1/predict` - Upload `.mat` file for prediction
- `POST /api/v1/predict/array` - Send EEG data as JSON array

**Interactive API Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Option 3: Docker Deployment

Build and run with Docker:

```bash
docker build -t eeg-depression-api .
docker run -p 8000:8000 eeg-depression-api
```

## Data Format

- **File Format**: MATLAB `.mat` file
- **Channels**: 128 or 129 channels (129th channel is automatically removed if present)
- **Sampling Rate**: 250 Hz (default)
- **Minimum Duration**: ~60 seconds (after trimming 30 seconds from each end)

## Processing Pipeline

1. **Preprocessing**:
   - Detrend (remove DC offset)
   - Bandpass filter (1-45 Hz)
   - Notch filter (50 Hz line noise removal)
   - Trim edges (first/last 30 seconds)

2. **Feature Extraction**:
   - Sliding window extraction (2-second windows, 50% overlap)
   - Noise removal (z-score thresholding)
   - Welch's method for power spectral density
   - 5 frequency bands: Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz)
   - 640 features total (128 channels × 5 bands)

3. **Classification**:
   - Trained MLP model
   - Binary classification: Depressed (1) or Not Depressed (0)
   - Probability scores for each class

## Project Structure

```
group-89-eeg-depression/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   └── endpoints.py        # API endpoints
│   ├── core/
│   │   └── config.py           # Configuration
│   ├── models/
│   │   └── model_loader.py     # Model loading and inference
│   └── preprocessing/
│       └── eeg_processing.py  # EEG preprocessing functions
├── streamlit_app.py            # Streamlit GUI
├── run_api.py                  # FastAPI runner script
├── run_streamlit.py            # Streamlit runner script
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── eeg_depression_model.keras  # Trained model
└── README.md                   # This file
```

## Testing

### Test the API

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Predict from file (if you have a .mat file)
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "file=@path/to/your/eeg_file.mat"
```

### Test the Streamlit App

1. Run `streamlit run streamlit_app.py`
2. Upload a `.mat` file
3. Click "Analyze EEG Data"
4. View results

## Model Information

- **Model Type**: MLP (Keras / TensorFlow)
- **Training Dataset**: MODMA (Multimodal Open Dataset for Mental Disorder Analysis)
- **Features**: 640 (128 channels × 5 frequency bands)
- **Output**: Binary classification (Depressed / Not Depressed) with probability scores
## Deployment

This application can be deployed to:
- **Render**: Supports Docker containers
- **Streamlit Cloud**: For Streamlit apps
- **Any cloud platform** supporting Docker containers

## References

- MODMA Dataset: Cai, H., et al. (2022). A multi-modal open dataset for mental-disorder analysis. Scientific Data, 9, 178.
- World Health Organization. Depressive disorder (depression). https://www.who.int/news-room/fact-sheets/detail/depression


