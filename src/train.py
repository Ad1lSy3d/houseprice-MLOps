"""
Model Training and MLflow Tracking Pipeline for EstateIQ.

Trains and evaluates:
1. Random Forest Regressor
2. HistGradientBoosting Regressor

Evaluates both using MAE, RMSE, and R2 on a held-out test set (random_state=42),
tracks metrics, parameters, and model artifacts with MLflow, selects the best
model by lowest RMSE, and persists all models and evaluation reports.
"""

from __future__ import annotations

import argparse
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

# Configure MLflow filesystem store compatibility
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.etl import ALL_FEATURE_NAMES, TARGET_NAME, get_project_root, run_etl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "EstateIQ_House_Price_Experiment"


def load_dataset(feature_store_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load curated dataset from feature store, running ETL first if missing.
    """
    if not feature_store_path.exists():
        logger.info("Feature store not found at %s. Triggering ETL pipeline...", feature_store_path)
        run_etl(feature_store_path)

    df = pd.read_csv(feature_store_path)
    X = df[ALL_FEATURE_NAMES].copy()
    y = df[TARGET_NAME].copy()
    return X, y


def train_and_evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tracking_uri: str,
) -> dict[str, Any]:
    """
    Fit model pipeline, evaluate test metrics, log to MLflow, and return results.
    """
    logger.info("Training %s...", model_name)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(root_mean_squared_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    logger.info(
        "%s Test Evaluation: MAE=%.4f ($%.2f), RMSE=%.4f ($%.2f), R²=%.4f",
        model_name,
        mae,
        mae * 100_000,
        rmse,
        rmse * 100_000,
        r2,
    )

    # Feature importances
    feature_importances: dict[str, float] = {}
    regressor = pipeline.named_steps["regressor"]
    if hasattr(regressor, "feature_importances_"):
        raw_imp = regressor.feature_importances_
        feature_importances = {feat: float(imp) for feat, imp in zip(ALL_FEATURE_NAMES, raw_imp)}
    else:
        # Use permutation importance for HistGradientBoosting
        perm_res = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        # Normalize positive mean importances to sum to 1.0 for comparability
        raw_imp = np.clip(perm_res.importances_mean, a_min=0, a_max=None)
        total = raw_imp.sum() or 1.0
        norm_imp = raw_imp / total
        feature_importances = {feat: float(imp) for feat, imp in zip(ALL_FEATURE_NAMES, norm_imp)}

    # MLflow logging
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = model_name.replace(" ", "_")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2_score": r2,
            "test_mae_usd": mae * 100_000,
            "test_rmse_usd": rmse * 100_000,
        })
        mlflow.set_tags({
            "model_type": model_name,
            "framework": "scikit-learn",
            "features_count": len(ALL_FEATURE_NAMES),
            "target": TARGET_NAME,
        })
        mlflow.sklearn.log_model(pipeline, artifact_path="model", serialization_format="cloudpickle")

    return {
        "model_name": model_name,
        "pipeline": pipeline,
        "params": params,
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "mae_usd": round(mae * 100_000, 2),
            "rmse_usd": round(rmse * 100_000, 2),
        },
        "feature_importances": feature_importances,
    }


def run_training(
    feature_store_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    mlruns_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Execute full training pipeline:
    1. Load data
    2. Split train/test (80/20, random_state=42)
    3. Train & evaluate Random Forest and HistGradientBoosting
    4. Save models & reports
    5. Log runs to MLflow
    """
    root = get_project_root()
    feat_path = Path(feature_store_path) if feature_store_path else root / "feature_store" / "housing_features.csv"
    if not feat_path.is_absolute():
        feat_path = root / feat_path

    mod_dir = Path(models_dir) if models_dir else root / "models"
    if not mod_dir.is_absolute():
        mod_dir = root / mod_dir
    mod_dir.mkdir(parents=True, exist_ok=True)

    rep_dir = Path(reports_dir) if reports_dir else root / "reports"
    if not rep_dir.is_absolute():
        rep_dir = root / rep_dir
    rep_dir.mkdir(parents=True, exist_ok=True)

    ml_dir = Path(mlruns_dir) if mlruns_dir else root / "mlruns"
    if not ml_dir.is_absolute():
        ml_dir = root / ml_dir
    ml_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"file://{ml_dir.resolve()}"

    # Load dataset
    X, y = load_dataset(feat_path)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Dataset split: Train=%d, Test=%d", len(X_train), len(X_test))

    # Model 1: Random Forest
    rf_params = {
        "n_estimators": 100,
        "max_depth": 20,
        "random_state": 42,
        "n_jobs": -1,
    }
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(**rf_params)),
    ])
    rf_results = train_and_evaluate_model(
        model_name="RandomForestRegressor",
        pipeline=rf_pipeline,
        params=rf_params,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tracking_uri=tracking_uri,
    )

    # Model 2: HistGradientBoosting
    hgb_params = {
        "max_iter": 150,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "random_state": 42,
    }
    hgb_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", HistGradientBoostingRegressor(**hgb_params)),
    ])
    hgb_results = train_and_evaluate_model(
        model_name="HistGradientBoostingRegressor",
        pipeline=hgb_pipeline,
        params=hgb_params,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tracking_uri=tracking_uri,
    )

    # Save both models
    rf_model_path = mod_dir / "random_forest.joblib"
    hgb_model_path = mod_dir / "gradient_boosting.joblib"
    joblib.dump(rf_results["pipeline"], rf_model_path)
    joblib.dump(hgb_results["pipeline"], hgb_model_path)
    logger.info("Saved models to %s and %s", rf_model_path, hgb_model_path)

    # Select best model by lowest test RMSE
    if hgb_results["metrics"]["rmse"] <= rf_results["metrics"]["rmse"]:
        best_results = hgb_results
        best_key = "HistGradientBoostingRegressor"
    else:
        best_results = rf_results
        best_key = "RandomForestRegressor"

    best_model_path = mod_dir / "best_model.joblib"
    joblib.dump(best_results["pipeline"], best_model_path)
    logger.info("Best model selected: %s (RMSE=%.4f). Saved to %s", best_key, best_results["metrics"]["rmse"], best_model_path)

    # Prepare evaluation report
    report = {
        "experiment_name": EXPERIMENT_NAME,
        "best_model": best_key,
        "selection_metric": "lowest_test_rmse",
        "models": {
            "RandomForestRegressor": {
                "metrics": rf_results["metrics"],
                "params": rf_params,
                "feature_importances": rf_results["feature_importances"],
                "model_file": "models/random_forest.joblib",
            },
            "HistGradientBoostingRegressor": {
                "metrics": hgb_results["metrics"],
                "params": hgb_params,
                "feature_importances": hgb_results["feature_importances"],
                "model_file": "models/gradient_boosting.joblib",
            },
        },
        "best_model_metrics": best_results["metrics"],
        "best_model_feature_importances": best_results["feature_importances"],
        "dataset_split": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "random_state": 42,
            "test_size": 0.2,
        },
        "features": ALL_FEATURE_NAMES,
    }

    report_path = rep_dir / "evaluation_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report saved to %s", report_path)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EstateIQ Model Training & MLflow Tracking")
    parser.add_argument("--feature-store", type=str, default=None, help="Path to housing_features.csv")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to directory for saved models")
    parser.add_argument("--reports-dir", type=str, default=None, help="Path to directory for evaluation reports")
    parser.add_argument("--mlruns-dir", type=str, default=None, help="Path to MLflow mlruns directory")
    args = parser.parse_args()

    run_training(
        feature_store_path=args.feature_store,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        mlruns_dir=args.mlruns_dir,
    )
