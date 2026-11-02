"""
Model Training and MLflow Tracking Pipeline for EstateIQ.

Trains and evaluates:
1. HistGradientBoosting Regressor
2. Random Forest Regressor

Evaluates both using MAE, RMSE, and R2 on a held-out test set (random_state=42),
tracks metrics, parameters, and model artifacts with MLflow (using SQLite tracking),
selects the best model by lowest RMSE, and persists all models and evaluation reports.
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

# Configure MLflow environment defaults
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
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

DEFAULT_EXPERIMENT_NAME = "EstateIQ_Model_Training"


def get_tracking_uri(mlruns_dir: Path) -> str:
    """
    Return tracking URI from environment variable or default to SQLite backend.
    """
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    db_path = (mlruns_dir / "mlflow.db").resolve()
    return f"sqlite:///{db_path}"


def get_experiment_name() -> str:
    """
    Return experiment name from environment variable or default to EstateIQ_Model_Training.
    """
    return os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)


def init_mlflow(tracking_uri: str, experiment_name: str, mlruns_dir: Path) -> str:
    """
    Initialize MLflow tracking URI and ensure the named experiment exists.
    Returns the experiment ID.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        artifact_location = str((mlruns_dir / "artifacts").resolve())
        exp_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
        logger.info("Created MLflow experiment '%s' (ID: %s)", experiment_name, exp_id)
    else:
        exp_id = experiment.experiment_id
        logger.info("Using existing MLflow experiment '%s' (ID: %s)", experiment_name, exp_id)

    mlflow.set_experiment(experiment_name)
    return exp_id


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


def generate_feature_importance_plot(
    feature_importances: dict[str, float],
    model_name: str,
    output_path: Path,
) -> Path:
    """
    Render and save a horizontal bar chart of feature importances.
    """
    sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_feats]
    values = [x[1] for x in sorted_feats]

    plt.figure(figsize=(9, 5))
    plt.barh(names[::-1], values[::-1], color="#2563eb", edgecolor="#1d4ed8", alpha=0.85)
    plt.title(f"Feature Importance - {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Relative Importance / Weight", fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_model_comparison_plot(
    models_metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> Path:
    """
    Render and save a multi-metric bar chart comparing evaluated models.
    """
    model_names = list(models_metrics.keys())
    rmses = [models_metrics[m]["rmse"] for m in model_names]
    maes = [models_metrics[m]["mae"] for m in model_names]
    r2s = [models_metrics[m]["r2_score"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar(x - width, rmses, width, label="RMSE", color="#ef4444", alpha=0.9)
    plt.bar(x, maes, width, label="MAE", color="#3b82f6", alpha=0.9)
    plt.bar(x + width, r2s, width, label="R² Score", color="#10b981", alpha=0.9)

    plt.xlabel("Candidate Models", fontsize=10, fontweight="bold")
    plt.ylabel("Metric Score", fontsize=10, fontweight="bold")
    plt.title("Model Performance Comparison (MAE, RMSE, R²)", fontsize=12, fontweight="bold")
    plt.xticks(x, model_names, fontsize=10)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def train_and_evaluate_model(
    model_name: str,
    model_type: str,
    pipeline: Pipeline,
    hyperparams: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    tracking_uri: str,
    experiment_name: str,
    models_dir: Path,
    reports_dir: Path,
    target_model_file: Path,
) -> dict[str, Any]:
    """
    Fit model pipeline, evaluate test metrics, log run to MLflow, and return results.
    """
    logger.info("Training %s (%s)...", model_name, model_type)
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

    # Calculate feature importances
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
        raw_imp = np.clip(perm_res.importances_mean, a_min=0, a_max=None)
        total = raw_imp.sum() or 1.0
        norm_imp = raw_imp / total
        feature_importances = {feat: float(imp) for feat, imp in zip(ALL_FEATURE_NAMES, norm_imp)}

    # Save model binary locally
    joblib.dump(pipeline, target_model_file)
    logger.info("Saved %s binary to %s", model_name, target_model_file)

    # Generate feature importance plot artifact
    slug = model_name.lower().replace(" ", "_")
    feat_plot_path = reports_dir / f"{slug}_feature_importance.png"
    generate_feature_importance_plot(feature_importances, model_name, feat_plot_path)

    # MLflow logging
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = model_name.replace(" ", "_")
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # 1. Log parameters
        combined_params: dict[str, Any] = {
            "model_name": model_name,
            "model_type": model_type,
            "framework": "scikit-learn",
            "scaler": "StandardScaler",
            "random_state": 42,
            "test_size": 0.2,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "dataset_name": "California Housing",
            "features_count": len(ALL_FEATURE_NAMES),
            "target_name": TARGET_NAME,
        }
        for k, v in hyperparams.items():
            combined_params[f"param_{k}"] = v

        mlflow.log_params(combined_params)

        # 2. Log evaluation metrics
        mlflow.log_metrics({
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2_score": r2,
            "test_mae_usd": mae * 100_000,
            "test_rmse_usd": rmse * 100_000,
        })

        # 3. Set run tags
        mlflow.set_tags({
            "model_name": model_name,
            "model_type": model_type,
            "framework": "scikit-learn",
            "features_count": str(len(ALL_FEATURE_NAMES)),
            "target": TARGET_NAME,
        })

        # 4. Log feature importance plot artifact
        mlflow.log_artifact(str(feat_plot_path), artifact_path="plots")

        # 5. Log serialized model file artifact
        mlflow.log_artifact(str(target_model_file), artifact_path="model")

        # 6. Log MLflow model flavor
        try:
            mlflow.sklearn.log_model(
                pipeline,
                name="model_pipeline",
                serialization_format="cloudpickle",
            )
        except Exception as exc:
            logger.warning("MLflow sklearn model log note: %s", exc)

        logger.info("Logged MLflow run for %s (Run ID: %s)", model_name, run_id)

    return {
        "model_name": model_name,
        "model_type": model_type,
        "run_id": run_id,
        "pipeline": pipeline,
        "params": hyperparams,
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "mae_usd": round(mae * 100_000, 2),
            "rmse_usd": round(rmse * 100_000, 2),
        },
        "feature_importances": feature_importances,
        "feature_plot_path": str(feat_plot_path),
        "model_file": str(target_model_file),
    }


def run_training(
    feature_store_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    mlruns_dir: Path | str | None = None,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """
    Execute full training and MLflow tracking pipeline:
    1. Load data from feature store
    2. Split train/test (80/20, random_state=42)
    3. Initialize MLflow with persistent SQLite backend
    4. Train & evaluate HistGradientBoosting and Random Forest models
    5. Save models, comparison plots, and evaluation reports
    6. Log evaluation artifacts to MLflow runs and mark best model
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

    exp_name = experiment_name or get_experiment_name()
    track_uri = tracking_uri or get_tracking_uri(ml_dir)

    # Initialize MLflow experiment
    init_mlflow(track_uri, exp_name, ml_dir)

    # Load dataset
    X, y = load_dataset(feat_path)

    # Train / test split (80/20 split, seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Dataset split: Train=%d, Test=%d", len(X_train), len(X_test))

    # Model 1: HistGradientBoostingRegressor
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
    hgb_model_path = mod_dir / "gradient_boosting.joblib"
    hgb_results = train_and_evaluate_model(
        model_name="HistGradientBoostingRegressor",
        model_type="Histogram-based Gradient Boosting Regressor",
        pipeline=hgb_pipeline,
        hyperparams=hgb_params,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tracking_uri=track_uri,
        experiment_name=exp_name,
        models_dir=mod_dir,
        reports_dir=rep_dir,
        target_model_file=hgb_model_path,
    )

    # Model 2: RandomForestRegressor
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
    rf_model_path = mod_dir / "random_forest.joblib"
    rf_results = train_and_evaluate_model(
        model_name="RandomForestRegressor",
        model_type="Random Forest Ensemble Regressor",
        pipeline=rf_pipeline,
        hyperparams=rf_params,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        tracking_uri=track_uri,
        experiment_name=exp_name,
        models_dir=mod_dir,
        reports_dir=rep_dir,
        target_model_file=rf_model_path,
    )

    # Select best model by lowest test RMSE
    if hgb_results["metrics"]["rmse"] <= rf_results["metrics"]["rmse"]:
        best_results = hgb_results
        best_key = "HistGradientBoostingRegressor"
    else:
        best_results = rf_results
        best_key = "RandomForestRegressor"

    best_model_path = mod_dir / "best_model.joblib"
    joblib.dump(best_results["pipeline"], best_model_path)
    logger.info(
        "Best model selected: %s (RMSE=%.4f). Saved to %s",
        best_key,
        best_results["metrics"]["rmse"],
        best_model_path,
    )

    # Generate comparison plot
    comparison_metrics = {
        "HistGradientBoosting": hgb_results["metrics"],
        "RandomForest": rf_results["metrics"],
    }
    comparison_plot_path = rep_dir / "model_comparison.png"
    generate_model_comparison_plot(comparison_metrics, comparison_plot_path)

    # Save model comparison summary JSON
    comparison_summary = {
        "experiment_name": exp_name,
        "best_model": best_key,
        "selection_metric": "lowest_test_rmse",
        "models": {
            "HistGradientBoostingRegressor": {
                "run_id": hgb_results["run_id"],
                "metrics": hgb_results["metrics"],
                "params": hgb_params,
            },
            "RandomForestRegressor": {
                "run_id": rf_results["run_id"],
                "metrics": rf_results["metrics"],
                "params": rf_params,
            },
        },
    }
    comp_json_path = rep_dir / "model_comparison.json"
    with open(comp_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    # Prepare complete evaluation report
    report = {
        "experiment_name": exp_name,
        "best_model": best_key,
        "selection_metric": "lowest_test_rmse",
        "models": {
            "HistGradientBoostingRegressor": {
                "run_id": hgb_results["run_id"],
                "metrics": hgb_results["metrics"],
                "params": hgb_params,
                "feature_importances": hgb_results["feature_importances"],
                "model_file": "models/gradient_boosting.joblib",
            },
            "RandomForestRegressor": {
                "run_id": rf_results["run_id"],
                "metrics": rf_results["metrics"],
                "params": rf_params,
                "feature_importances": rf_results["feature_importances"],
                "model_file": "models/random_forest.joblib",
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

    # Log summary artifacts and best-model tags to MLflow runs
    client = MlflowClient(tracking_uri=track_uri)
    for res, is_best in [(hgb_results, best_key == "HistGradientBoostingRegressor"),
                         (rf_results, best_key == "RandomForestRegressor")]:
        run_id = res["run_id"]
        client.set_tag(run_id, "is_best_model", "true" if is_best else "false")
        client.set_tag(run_id, "best_model_selection", best_key)
        # Log evaluation and comparison artifacts into the run
        client.log_artifact(run_id, str(report_path), artifact_path="evaluation")
        client.log_artifact(run_id, str(comp_json_path), artifact_path="evaluation")
        client.log_artifact(run_id, str(comparison_plot_path), artifact_path="evaluation")
        if is_best:
            client.log_artifact(run_id, str(best_model_path), artifact_path="model")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EstateIQ Model Training & MLflow Tracking")
    parser.add_argument("--feature-store", type=str, default=None, help="Path to housing_features.csv")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to directory for saved models")
    parser.add_argument("--reports-dir", type=str, default=None, help="Path to directory for evaluation reports")
    parser.add_argument("--mlruns-dir", type=str, default=None, help="Path to MLflow mlruns directory")
    parser.add_argument("--experiment-name", type=str, default=None, help="MLflow experiment name")
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI")
    args = parser.parse_args()

    run_training(
        feature_store_path=args.feature_store,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        mlruns_dir=args.mlruns_dir,
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
    )
