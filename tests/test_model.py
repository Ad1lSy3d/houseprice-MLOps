"""
Unit tests for model training pipeline and MLflow tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import pytest

from src.etl import ALL_FEATURE_NAMES, TARGET_NAME, extract_data, engineer_features
from src.predict import predict_price
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
    """Verify models train, evaluate metrics, and save local artifacts properly."""
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    mlruns_dir = tmp_path / "mlruns"

    report = run_training(
        feature_store_path=sample_feature_store,
        models_dir=models_dir,
        reports_dir=reports_dir,
        mlruns_dir=mlruns_dir,
        experiment_name="EstateIQ_Model_Training_Test",
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
        assert "mae_usd" in m and m["mae_usd"] > 0
        assert "rmse_usd" in m and m["rmse_usd"] > 0
        assert "feature_importances" in data["models"][model_key]

    # 3. Verify comparison files & plots
    assert (reports_dir / "model_comparison.json").exists()
    assert (reports_dir / "model_comparison.png").exists()
    assert (reports_dir / "histgradientboostingregressor_feature_importance.png").exists()
    assert (reports_dir / "randomforestregressor_feature_importance.png").exists()


def test_mlflow_experiment_and_runs(tmp_path: Path, sample_feature_store: Path):
    """Verify MLflow experiment creation, parameters, metrics, and artifact logging."""
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    mlruns_dir = tmp_path / "mlruns"
    exp_name = "EstateIQ_Model_Training_Verify"
    db_path = mlruns_dir / "test_mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"

    run_training(
        feature_store_path=sample_feature_store,
        models_dir=models_dir,
        reports_dir=reports_dir,
        mlruns_dir=mlruns_dir,
        experiment_name=exp_name,
        tracking_uri=tracking_uri,
    )

    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Verify experiment exists
    exp = client.get_experiment_by_name(exp_name)
    assert exp is not None
    assert exp.name == exp_name

    # 2. Verify separate runs for candidate models
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 2

    run_names = {r.data.tags.get("mlflow.runName") for r in runs}
    assert "HistGradientBoostingRegressor" in run_names
    assert "RandomForestRegressor" in run_names

    # 3. Check parameters, metrics, tags, and artifacts on each run
    for run in runs:
        params = run.data.params
        assert "model_name" in params
        assert "framework" in params and params["framework"] == "scikit-learn"
        assert "scaler" in params and params["scaler"] == "StandardScaler"
        assert "random_state" in params and params["random_state"] == "42"
        assert "dataset_name" in params and params["dataset_name"] == "California Housing"

        metrics = run.data.metrics
        assert "test_mae" in metrics and metrics["test_mae"] > 0
        assert "test_rmse" in metrics and metrics["test_rmse"] > 0
        assert "test_r2_score" in metrics
        assert "test_mae_usd" in metrics and metrics["test_mae_usd"] > 0
        assert "test_rmse_usd" in metrics and metrics["test_rmse_usd"] > 0

        tags = run.data.tags
        assert "is_best_model" in tags
        assert tags["is_best_model"] in ["true", "false"]

        # Check logged artifacts
        artifacts = client.list_artifacts(run.info.run_id)
        artifact_paths = {a.path for a in artifacts}
        assert "plots" in artifact_paths
        assert "model" in artifact_paths
        assert "evaluation" in artifact_paths


def test_repeated_training_creates_new_runs(tmp_path: Path, sample_feature_store: Path):
    """Verify repeated training creates new runs instead of overwriting history."""
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    mlruns_dir = tmp_path / "mlruns"
    exp_name = "EstateIQ_Model_Training_Repeated"
    db_path = mlruns_dir / "repeated.db"
    tracking_uri = f"sqlite:///{db_path}"

    # First execution -> 2 runs
    run_training(
        feature_store_path=sample_feature_store,
        models_dir=models_dir,
        reports_dir=reports_dir,
        mlruns_dir=mlruns_dir,
        experiment_name=exp_name,
        tracking_uri=tracking_uri,
    )
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(exp_name)
    runs_1 = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs_1) == 2

    # Second execution -> 4 runs total
    run_training(
        feature_store_path=sample_feature_store,
        models_dir=models_dir,
        reports_dir=reports_dir,
        mlruns_dir=mlruns_dir,
        experiment_name=exp_name,
        tracking_uri=tracking_uri,
    )
    runs_2 = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs_2) == 4


def test_existing_model_prediction_still_works(tmp_path: Path, sample_feature_store: Path):
    """Verify that after MLflow training, predict_price continues working flawlessly."""
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    mlruns_dir = tmp_path / "mlruns"

    run_training(
        feature_store_path=sample_feature_store,
        models_dir=models_dir,
        reports_dir=reports_dir,
        mlruns_dir=mlruns_dir,
        experiment_name="EstateIQ_Model_Training_Predict",
    )

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

    result = predict_price(
        sample_input,
        model_path=models_dir / "best_model.joblib",
        report_path=reports_dir / "evaluation_results.json",
    )

    assert result["prediction_raw"] > 0
    assert result["prediction_usd"] > 0
    assert result["prediction_usd_formatted"].startswith("$")
    assert "uncertainty" in result
    assert result["uncertainty"]["lower_bound_usd"] <= result["prediction_usd"] <= result["uncertainty"]["upper_bound_usd"]
    assert "engineered_features" in result
    assert "rooms_per_household" in result["engineered_features"]
    assert "bedrooms_per_room" in result["engineered_features"]
    assert "population_per_household" in result["engineered_features"]
    assert "disclaimer" in result
