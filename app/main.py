"""
FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

from app.core.config import settings
from app.api.endpoints import router
from app.models.model_loader import load_model

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API for classifying patients' depressive symptoms using EEG data from the MODMA dataset",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix=settings.api_prefix)


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    try:
        logger.info("Loading EEG depression classification model...")
        load_model(settings.model_path)
        logger.info("Model loaded successfully!")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.warning("API will start but predictions will fail until model is available.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("API will start but predictions will fail until model is available.")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "EEG Depression Classification API",
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )
