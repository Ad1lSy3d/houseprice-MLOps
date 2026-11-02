"""
Unit tests for model training pipeline and MLflow tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from src.etl import ALL_FEATURE_NAMES, TARGET_NAME, extract_data, engineer_features
from src.train import run_training


@pytest.fixture
def sample_feature_store(tmp_path: Path) -> Path:
    """Create a lightweight feature store CSV for fast test execution."""
    raw_df = extract_data().iloc[:300].copy()
    proc_df = engineer_features(raw_df)
    feat_csv = tmp_path / "sample_features.csv"
    proc_df.to_csv(feat_csv, index=False)
    return feat_csv


def test_model_training_and_artifacts(tmp_path: Path, sample_feature_store: Path):
    """Verify models train, evaluate metrics, and save artifacts properly."""
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    mlruns_dir = tmp_path / "mlruns"

    report = run_training(
        feature_store_path=sample_feature_store,
        models_dir=models_dir,
        reports_dir=reports_dir,
        mlruns_dir=mlruns_dir,
    )

    # 1. Verify model files saved
    assert (models_dir / "random_forest.joblib").exists()
    assert (models_dir / "gradient_boosting.joblib").exists()
    assert (models_dir / "best_model.joblib").exists()

    # 2. Verify evaluation report JSON
    rep_file = reports_dir / "evaluation_results.json"
    assert rep_file.exists()
    with open(rep_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "best_model" in data
    assert data["best_model"] in ["RandomForestRegressor", "HistGradientBoostingRegressor"]
    assert "models" in data
    assert "RandomForestRegressor" in data["models"]
    assert "HistGradientBoostingRegressor" in data["models"]

    # Verify metrics structure
    for model_key in ["RandomForestRegressor", "HistGradientBoostingRegressor"]:
        m = data["models"][model_key]["metrics"]
        assert "mae" in m and m["mae"] > 0
        assert "rmse" in m and m["rmse"] > 0
        assert "r2_score" in m
        assert "feature_importances" in data["models"][model_key]

    # 3. Verify MLflow tracking directory populated
    assert mlruns_dir.exists()
    assert any(mlruns_dir.iterdir())

