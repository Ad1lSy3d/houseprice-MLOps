"""
Integration and API tests for EstateIQ FastAPI backend.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_api_status_endpoint():
    """Verify GET / returns application status information."""
    response = client.get("/", headers={"Accept": "application/json"})
    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert data["status"] == "online"
    assert "endpoints" in data


def test_health_endpoint():
    """Verify GET /health returns operational health and model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "HistGradientBoostingRegressor" in data["best_model_name"] or "RandomForestRegressor" in data["best_model_name"]


def test_models_endpoint():
    """Verify GET /models returns actual evaluation metrics and feature importances."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "best_model" in data
    assert "models" in data
    assert "best_model_metrics" in data
    assert "best_model_feature_importances" in data

    # Verify both candidate models are represented
    assert "RandomForestRegressor" in data["models"]
    assert "HistGradientBoostingRegressor" in data["models"]


def test_predict_endpoint_valid():
    """Verify POST /predict returns estimated valuation for valid property characteristics."""
    payload = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction_raw"] > 0
    assert data["prediction_usd"] > 0
    assert "prediction_usd_formatted" in data
    assert "uncertainty" in data
    assert "engineered_features" in data
    assert data["uncertainty"]["lower_bound_usd"] <= data["prediction_usd"]


def test_predict_endpoint_invalid_values():
    """Verify POST /predict returns 422 for negative or invalid numerical inputs."""
    bad_payload = {
        "MedInc": -5.0,  # Invalid negative
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_endpoint_missing_fields():
    """Verify POST /predict returns 422 when required fields are missing."""
    incomplete_payload = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_dashboard_endpoint():
    """Verify GET /dashboard serves the frontend HTML."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "EstateIQ" in response.text

