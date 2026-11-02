"""
Tests for Streamlit application structure, presets, direct model inference, and deployment configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from sklearn.pipeline import Pipeline
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root and streamlit_app to sys.path to test functions directly
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "streamlit_app") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "streamlit_app"))

from backend.schemas import PropertyFeaturesSchema
from src.predict import predict_price
from streamlit_app.app import (
    PRESETS,
    load_cached_model,
    load_cached_report,
    resolve_backend_url,
)


def test_streamlit_files_exist():
    """Verify Streamlit app files and requirements exist."""
    assert (PROJECT_ROOT / "streamlit_app" / "app.py").exists()
    assert (PROJECT_ROOT / "streamlit_app" / "requirements.txt").exists()


def test_streamlit_requirements_complete():
    """Verify all dependencies required for direct inference are declared in requirements.txt."""
    req_file = PROJECT_ROOT / "streamlit_app" / "requirements.txt"
    content = req_file.read_text(encoding="utf-8")
    required_packages = ["streamlit", "scikit-learn", "joblib", "numpy", "pandas", "pydantic"]
    for pkg in required_packages:
        assert pkg in content, f"Missing required dependency '{pkg}' in streamlit_app/requirements.txt"


def test_streamlit_presets_valid_against_schema():
    """Verify that all presets in Streamlit app satisfy backend Pydantic validation schema."""
    assert len(PRESETS) >= 4

    for preset_name, features in PRESETS.items():
        validated = PropertyFeaturesSchema(**features)
        assert validated.MedInc > 0
        assert 32.0 <= validated.Latitude <= 43.0
        assert -125.0 <= validated.Longitude <= -114.0


def test_resolve_backend_url_environment(monkeypatch):
    """Verify backend URL resolution with custom environment variable."""
    monkeypatch.setenv("BACKEND_URL", "https://estateiq-api.onrender.com/")
    assert resolve_backend_url() == "https://estateiq-api.onrender.com"


def test_render_blueprint_valid():
    """Verify that render.yaml blueprint is syntactically valid and properly configured."""
    render_file = PROJECT_ROOT / "render.yaml"
    assert render_file.exists()

    with open(render_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    services = config.get("services", [])
    assert len(services) > 0
    api_service = services[0]
    assert api_service.get("type") == "web"
    assert api_service.get("runtime") == "docker"
    assert api_service.get("healthCheckPath") == "/health"


def test_direct_model_loading_and_caching():
    """Verify direct model loading succeeds and Streamlit caching reuses instance."""
    model_1, err_1 = load_cached_model()
    assert err_1 is None
    assert model_1 is not None
    assert isinstance(model_1, Pipeline)

    # Calling again should return cached instance
    model_2, err_2 = load_cached_model()
    assert err_2 is None
    assert model_1 is model_2

    # Verify cached report loads
    report = load_cached_report()
    assert "best_model" in report
    assert "models" in report


def test_direct_prediction_from_streamlit():
    """Verify direct in-process prediction using loaded model and sample housing inputs."""
    model, err = load_cached_model()
    assert err is None
    assert model is not None

    sample_input = PRESETS["San Francisco Bay Area"]
    result = predict_price(sample_input, model=model)

    assert "prediction_raw" in result and result["prediction_raw"] > 0
    assert "prediction_usd" in result and result["prediction_usd"] > 0
    assert result["prediction_usd_formatted"].startswith("$")
    assert "uncertainty" in result
    assert result["uncertainty"]["lower_bound_usd"] <= result["prediction_usd"] <= result["uncertainty"]["upper_bound_usd"]
    assert "engineered_features" in result
    assert result["engineered_features"]["rooms_per_household"] > 0
    assert result["engineered_features"]["bedrooms_per_room"] > 0
    assert result["engineered_features"]["population_per_household"] > 0
    assert "disclaimer" in result


def test_direct_prediction_all_presets():
    """Verify all California presets yield strictly positive valuations and valid bounds."""
    model, err = load_cached_model()
    assert err is None
    assert model is not None

    for preset_name, features in PRESETS.items():
        result = predict_price(features, model=model)
        assert result["prediction_usd"] > 0, f"Preset '{preset_name}' produced non-positive prediction"
        assert result["uncertainty"]["lower_bound_usd"] >= 0
        assert result["uncertainty"]["upper_bound_usd"] > result["prediction_usd"]
