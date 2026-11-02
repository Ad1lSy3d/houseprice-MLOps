"""
Prediction Module for EstateIQ.

Loads the persisted best model pipeline, validates input features, applies
the exact feature engineering used during training, computes the estimated
property value converted into USD ($100,000 target scale), and derives a
statistically defensible uncertainty interval based on out-of-sample test RMSE.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.etl import (
    ALL_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    engineer_features,
    get_project_root,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# California Housing target scale factor ($100,000 per target unit)
USD_SCALE_FACTOR = 100_000.0


class ModelNotFoundError(FileNotFoundError):
    """Raised when the requested trained model file cannot be found."""
    pass


class InvalidFeatureError(ValueError):
    """Raised when input features violate validation rules."""
    pass


def get_default_model_path() -> Path:
    """Return default path to best model."""
    return get_project_root() / "models" / "best_model.joblib"


def get_default_report_path() -> Path:
    """Return default path to evaluation report."""
    return get_project_root() / "reports" / "evaluation_results.json"


def load_model(model_path: Path | str | None = None) -> Pipeline:
    """
    Load trained scikit-learn Pipeline from disk.

    Args:
        model_path: Optional path to joblib model file.

    Returns:
        Fitted scikit-learn Pipeline.

    Raises:
        ModelNotFoundError: If the model file is missing.
    """
    path = Path(model_path) if model_path else get_default_model_path()
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        raise ModelNotFoundError(
            f"Trained model not found at '{path}'. Please train the model first by running: 'uv run python src/train.py'"
        )

    return joblib.load(path)


def load_evaluation_report(report_path: Path | str | None = None) -> dict[str, Any]:
    """
    Load evaluation metrics and metadata from reports/evaluation_results.json.
    """
    path = Path(report_path) if report_path else get_default_report_path()
    if not path.is_absolute():
        path = get_project_root() / path

    if not path.exists():
        logger.warning("Evaluation report not found at %s. Using fallback estimates.", path)
        return {
            "best_model": "HistGradientBoostingRegressor",
            "best_model_metrics": {"rmse": 0.4529, "mae": 0.3008, "r2_score": 0.8435},
        }

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_raw_features(features: dict[str, Any]) -> dict[str, float]:
    """
    Validate that all 8 required features are present, numeric, and within plausible domain bounds.
    """
    missing = [feat for feat in RAW_FEATURE_NAMES if feat not in features]
    if missing:
        raise InvalidFeatureError(f"Missing required housing features: {missing}")

    validated: dict[str, float] = {}
    for feat in RAW_FEATURE_NAMES:
        try:
            val = float(features[feat])
        except (ValueError, TypeError) as exc:
            raise InvalidFeatureError(f"Feature '{feat}' must be numeric. Received: {features[feat]}") from exc

        # Physical positivity checks
        if feat in ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]:
            if val <= 0:
                raise InvalidFeatureError(f"Feature '{feat}' must be strictly positive (> 0). Received: {val}")

        # Geographical bounds for California
        if feat == "Latitude" and not (32.0 <= val <= 43.0):
            raise InvalidFeatureError(
                f"Latitude {val} is outside California range [32.0, 43.0]."
            )
        if feat == "Longitude" and not (-125.0 <= val <= -114.0):
            raise InvalidFeatureError(
                f"Longitude {val} is outside California range [-125.0, -114.0]."
            )

        validated[feat] = val

    return validated


def predict_price(
    features: dict[str, Any] | pd.DataFrame,
    model_path: Path | str | None = None,
    report_path: Path | str | None = None,
    model: Pipeline | None = None,
) -> dict[str, Any]:
    """
    Estimate property value using the trained model.

    Args:
        features: Dictionary containing the 8 raw housing features, or a DataFrame.
        model_path: Optional path to joblib model file.
        report_path: Optional path to evaluation report.
        model: Optional pre-loaded scikit-learn Pipeline instance.

    Returns:
        Dictionary containing prediction results, USD conversions, engineered features,
        and defensible uncertainty range based on out-of-sample RMSE.
    """
    # 1. Validate inputs
    if isinstance(features, dict):
        validated_dict = validate_raw_features(features)
        raw_df = pd.DataFrame([validated_dict])
    elif isinstance(features, pd.DataFrame):
        for col in RAW_FEATURE_NAMES:
            if col not in features.columns:
                raise InvalidFeatureError(f"DataFrame missing required column: '{col}'")
        raw_df = features[RAW_FEATURE_NAMES].copy()
        validated_dict = raw_df.iloc[0].to_dict()
    else:
        raise InvalidFeatureError("Input features must be a dict or a pandas DataFrame.")

    # 2. Apply feature engineering
    augmented_df = engineer_features(raw_df)

    # 3. Ensure exact feature column alignment
    X = augmented_df[ALL_FEATURE_NAMES]

    # 4. Load model if not already provided
    if model is None:
        model = load_model(model_path)

    # 5. Predict (raw units: $100,000s)
    raw_preds = model.predict(X)
    raw_val = float(raw_preds[0])
    # House value cannot be negative in physical terms
    raw_val = max(0.0, raw_val)

    # 6. Convert to USD
    prediction_usd = float(raw_val * USD_SCALE_FACTOR)

    # 7. Defensible Uncertainty Estimation:
    # Based on out-of-sample empirical RMSE on held-out test data.
    # Root Mean Squared Error measures the standard deviation of residuals;
    # ±1 RMSE provides a defensible one-standard-error empirical prediction interval.
    report = load_evaluation_report(report_path)
    best_metrics = report.get("best_model_metrics", {})
    rmse = float(best_metrics.get("rmse", 0.4529))
    uncertainty_usd = float(rmse * USD_SCALE_FACTOR)

    lower_usd = max(0.0, prediction_usd - uncertainty_usd)
    upper_usd = prediction_usd + uncertainty_usd

    engineered_dict = {
        "rooms_per_household": round(float(augmented_df["rooms_per_household"].iloc[0]), 3),
        "bedrooms_per_room": round(float(augmented_df["bedrooms_per_room"].iloc[0]), 3),
        "population_per_household": round(float(augmented_df["population_per_household"].iloc[0]), 3),
    }

    return {
        "prediction_raw": round(raw_val, 4),
        "prediction_usd": round(prediction_usd, 2),
        "prediction_usd_formatted": f"${prediction_usd:,.2f}",
        "model_name": report.get("best_model", "HistGradientBoostingRegressor"),
        "uncertainty": {
            "lower_bound_usd": round(lower_usd, 2),
            "upper_bound_usd": round(upper_usd, 2),
            "lower_bound_formatted": f"${lower_usd:,.2f}",
            "upper_bound_formatted": f"${upper_usd:,.2f}",
            "uncertainty_usd": round(uncertainty_usd, 2),
            "method": "empirical_test_rmse",
            "description": (
                f"Range reflects ±1 test RMSE (${uncertainty_usd:,.0f}), representing "
                "the standard deviation of out-of-sample model prediction residuals."
            ),
        },
        "engineered_features": engineered_dict,
        "input_features": validated_dict,
        "disclaimer": (
            "Educational prediction based on 1990 California Census housing dataset. "
            "Not an appraisal or real-time property valuation service."
        ),
    }


if __name__ == "__main__":
    # Quick standalone demonstration with a realistic Bay Area profile
    sample_input = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
    result = predict_price(sample_input)
    print("--- Sample Prediction Output ---")
    print(json.dumps(result, indent=2))

