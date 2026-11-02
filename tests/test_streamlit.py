"""
Tests for Streamlit application structure, presets, and deployment configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add streamlit_app to sys.path to test functions directly
sys.path.insert(0, str(PROJECT_ROOT / "streamlit_app"))

from backend.schemas import PropertyFeaturesSchema
from streamlit_app.app import PRESETS, resolve_backend_url


def test_streamlit_files_exist():
    """Verify Streamlit app files and requirements exist."""
    assert (PROJECT_ROOT / "streamlit_app" / "app.py").exists()
    assert (PROJECT_ROOT / "streamlit_app" / "requirements.txt").exists()


def test_streamlit_presets_valid_against_schema():
    """Verify that all presets in Streamlit app satisfy backend Pydantic validation schema."""
    assert len(PRESETS) >= 4

    for preset_name, features in PRESETS.items():
        # Validate against Pydantic schema
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

