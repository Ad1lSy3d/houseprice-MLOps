
"""
FastAPI Backend for EstateIQ – House Price Intelligence Platform.

Provides RESTful endpoints for health checks, model metadata & evaluation metrics,
and real-time property price estimations. Serves the responsive frontend dashboard.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from backend.schemas import (
    AppStatusResponseSchema,
    HealthResponseSchema,
    ModelsResponseSchema,
    PredictionResponseSchema,
    PropertyFeaturesSchema,
)
from src.predict import (
    InvalidFeatureError,
    ModelNotFoundError,
    get_default_model_path,
    load_evaluation_report,
    load_model,
    predict_price,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

APP_VERSION = "0.1.0"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="EstateIQ – House Price Intelligence Platform",
    description="Machine Learning API for estimating California housing property values and tracking model metrics.",
    version=APP_VERSION,
)

# Prometheus monitoring instrumentation
# Automatically tracks HTTP requests, response status codes and request duration.
Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=True,
)

# Enable CORS for local development and client integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend directory for static assets
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_model=None, summary="Application status or dashboard view")
def root(request: Request) -> Any:
    """
    Root endpoint:
    - Returns application status JSON for API clients and automated tests.
    - Serves the frontend dashboard HTML if requested by a web browser.
    """
    accept_header = request.headers.get("accept", "")

    if "text/html" in accept_header and "application/json" not in accept_header:
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

    return {
        "app": "EstateIQ – House Price Intelligence Platform",
        "status": "online",
        "version": APP_VERSION,
        "description": "Smart Property Valuation Powered by Machine Learning",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "models": "/models",
            "dashboard": "/dashboard",
            "docs": "/docs",
            "metrics": "/metrics",
        },
    }


@app.get("/dashboard", summary="Access the interactive EstateIQ dashboard")
def dashboard() -> FileResponse:
    """Serve the interactive HTML dashboard."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend index.html not found.",
        )
    return FileResponse(index_file)


@app.get("/health", response_model=HealthResponseSchema, summary="System health check")
def health_check() -> HealthResponseSchema:
    """
    Health check endpoint reporting model readiness and system operational status.
    Does NOT train models on startup.
    """
    model_path = get_default_model_path()
    model_loaded = model_path.exists()
    report = load_evaluation_report()
    best_model_name = report.get("best_model", "Unknown")

    return HealthResponseSchema(
        status="healthy",
        model_loaded=model_loaded,
        best_model_name=best_model_name if model_loaded else "Model file missing",
        version=APP_VERSION,
    )


@app.get("/models", response_model=ModelsResponseSchema, summary="Model comparison & evaluation metrics")
def get_models() -> Any:
    """
    Return all evaluated models, actual test metrics (MAE, RMSE, R²),
    hyperparameters, and feature importances from reports/evaluation_results.json.
    """
    try:
        report = load_evaluation_report()
        if not report or "models" not in report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Evaluation report not found. Please train models first: 'uv run python src/train.py'",
            )
        return report
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error loading model metrics: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model evaluation metrics: {exc}",
        ) from exc


@app.post("/predict", response_model=PredictionResponseSchema, summary="Predict property valuation")
def predict(features: PropertyFeaturesSchema) -> Any:
    """
    Accepts 8 raw housing features, validates numeric domain bounds, applies
    feature engineering, and returns the estimated property price in USD
    along with empirical residual uncertainty range.
    """
    try:
        result = predict_price(features.model_dump())
        return result
    except ModelNotFoundError as exc:
        logger.error("Model file not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except InvalidFeatureError as exc:
        logger.warning("Invalid feature input: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


# Convenient root static file helpers
@app.get("/style.css", include_in_schema=False)
def get_style() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")


@app.get("/script.js", include_in_schema=False)
def get_script() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "script.js", media_type="application/javascript")
