"""
Unit tests for prediction module and uncertainty estimation.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from src.predict import (
    InvalidFeatureError,
    ModelNotFoundError,
    load_model,
    predict_price,
)


@pytest.fixture
def valid_features() -> dict[str, float]:
    return {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }


def test_predict_price_valid_input(valid_features: dict[str, float]):
    """Verify prediction output structure, USD scaling, and uncertainty intervals."""
    res = predict_price(valid_features)

    assert "prediction_raw" in res
    assert "prediction_usd" in res
    assert "prediction_usd_formatted" in res
    assert "uncertainty" in res
    assert "engineered_features" in res
    assert "model_name" in res
    assert "disclaimer" in res

    # Target unit conversion ($100,000 scale)
    expected_usd = res["prediction_raw"] * 100_000.0
    assert pytest.approx(res["prediction_usd"], rel=1e-2) == expected_usd
    assert res["prediction_usd_formatted"].startswith("$")

    # Uncertainty verification: lower <= estimate <= upper
    unc = res["uncertainty"]
    assert unc["lower_bound_usd"] <= res["prediction_usd"]
    assert unc["upper_bound_usd"] >= res["prediction_usd"]
    assert unc["method"] == "empirical_test_rmse"

    # Engineered features
    eng = res["engineered_features"]
    assert "rooms_per_household" in eng
    assert "bedrooms_per_room" in eng
    assert "population_per_household" in eng
    assert eng["rooms_per_household"] > 0
    assert eng["bedrooms_per_room"] > 0


def test_predict_price_missing_feature(valid_features: dict[str, float]):
    """Verify that omitting a required feature raises InvalidFeatureError."""
    bad_features = valid_features.copy()
    del bad_features["MedInc"]
    with pytest.raises(InvalidFeatureError) as exc_info:
        predict_price(bad_features)
    assert "Missing required housing features" in str(exc_info.value)


def test_predict_price_negative_values(valid_features: dict[str, float]):
    """Verify that negative physical counts raise InvalidFeatureError."""
    bad_features = valid_features.copy()
    bad_features["AveRooms"] = -3.5
    with pytest.raises(InvalidFeatureError) as exc_info:
        predict_price(bad_features)
    assert "must be strictly positive" in str(exc_info.value)


def test_predict_price_out_of_bounds_geography(valid_features: dict[str, float]):
    """Verify that coordinates outside California bounds raise InvalidFeatureError."""
    bad_features = valid_features.copy()
    bad_features["Latitude"] = 55.0  # Outside California
    with pytest.raises(InvalidFeatureError) as exc_info:
        predict_price(bad_features)
    assert "Latitude 55.0 is outside California range" in str(exc_info.value)


def test_load_model_missing_file(tmp_path: Path):
    """Verify that attempting to load a nonexistent model raises ModelNotFoundError."""
    missing_file = tmp_path / "nonexistent_model.joblib"
    with pytest.raises(ModelNotFoundError) as exc_info:
        load_model(missing_file)
    assert "Trained model not found" in str(exc_info.value)

