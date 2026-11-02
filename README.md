# EstateIQ – House Price Intelligence Platform

EstateIQ is an end-to-end Machine Learning application and real-estate analytics platform that estimates residential property valuations using California census and geographical data. Built as a modular, maintainable foundation for MLOps, it pairs rigorous model experimentation and tracking (via MLflow and Scikit-learn Pipelines) with a high-performance FastAPI backend and a modern vanilla web dashboard.

---

## Architecture Overview

EstateIQ follows a decoupled, production-ready ML architecture designed to ease future extensions into CI/CD, containerization, and automated registry workflows:

```
                                  ┌────────────────────────┐
                                  │  California Housing   │
                                  │     (Scikit-Learn)     │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                   [src/etl.py Pipeline]
                                  Clean, Validate & Engineer
                                              │
                                              ▼
                                 ┌───────────────────────────┐
                                 │ feature_store/            │
                                 │   housing_features.csv    │
                                 └─────────────┬─────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
             [RandomForestRegressor]                      [HistGradientBoosting]
                       │                                               │
                       └───────────────────────┬───────────────────────┘
                                               │ Evaluated (MAE, RMSE, R²)
                                               ▼
                                   [src/train.py Pipeline]
                                               ├─────────────────────────────┐
                                               ▼                             ▼
                                   ┌───────────────────────┐     ┌───────────────────────┐
                                   │       MLflow UI       │     │ models/               │
                                   │  (mlruns/ experiment) │     │   best_model.joblib   │
                                   └───────────────────────┘     └───────────┬───────────┘
                                                                             │
                                                                             ▼
                                                                  [src/predict.py]
                                                              Feature Transform & Scaling
                                                                             │
                                                                             ▼
                                                                   [backend/main.py]
                                                                    FastAPI REST API
                                                                             │
                                                        ┌────────────────────┴────────────────────┐
                                                        ▼                                         ▼
                                                [frontend/ UI]                             [/docs API]
                                            Interactive Dashboard                          Swagger Docs
```

---

## Dataset & Domain Features

### California Housing Dataset
Derived from the 1990 U.S. Census, the dataset encompasses 20,640 California block groups, each averaging 1,425 individuals.

> [!NOTE]
> The target variable (`MedHouseVal`) is natively expressed in units of **$100,000**. EstateIQ scales all model predictions and residuals by $100,000 to output readable, currency-formatted USD values. In the historical 1990 census, median values above $500,000 were top-coded.

### Base Features
1. `MedInc`: Median income within the block group (in tens of thousands USD, e.g. 8.32 = $83,200).
2. `HouseAge`: Median age of houses in the block group (years).
3. `AveRooms`: Average number of rooms per household.
4. `AveBedrms`: Average number of bedrooms per household.
5. `Population`: Total population residing in the block group.
6. `AveOccup`: Average number of household members.
7. `Latitude`: Block group latitude (California: 32.0°N to 43.0°N).
8. `Longitude`: Block group longitude (California: -125.0°W to -114.0°W).

### Engineered Domain Features
To capture spatial and property-density dynamics without data leakage:
- `rooms_per_household`: Average rooms per household (`AveRooms`).
- `bedrooms_per_room`: Ratio of bedrooms to total rooms (`AveBedrms / AveRooms`). Indicates property density and home style.
- `population_per_household`: Occupancy density per household (`AveOccup`).

---

## Models & Evaluation

EstateIQ trains and benchmarks two non-linear tree-based ensembles on an 80/20 train-test split (`random_state=42`) wrapped in Scikit-learn `StandardScaler` preprocessing pipelines:

1. **Random Forest Regressor**: 100 estimators, max depth 20.
2. **HistGradientBoosting Regressor**: 150 iterations, learning rate 0.1, max leaf nodes 31.

### Test Set Benchmark Results

| Model Architecture | MAE (Score) | MAE (USD) | RMSE (Score) | RMSE (USD) | R² Score | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoosting** | **0.3008** | **$30,079.38** | **0.4529** | **$45,290.80** | **0.8435** | **Selected Best** |
| Random Forest | 0.3286 | $32,857.09 | 0.5057 | $50,569.71 | 0.8048 | Evaluated Candidate |

*Model selection is automated based on the lowest test RMSE.*

### Defensible Uncertainty Estimation
Rather than presenting arbitrary heuristic confidence intervals, EstateIQ provides an empirical prediction range based on the out-of-sample **Root Mean Squared Error (RMSE)**:
$$\text{Uncertainty Bounds} = [\max(0, \hat{y} - \text{RMSE}), \hat{y} + \text{RMSE}]$$
For our best model, this corresponds to approximately **±$45,291**, representing $\pm 1$ standard deviation of model residuals on previously unseen data.

---

## Project Structure

```text
houseprice-MLOps/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI application, static file serving, REST endpoints
│   └── schemas.py           # Pydantic v2 validation models
├── frontend/
│   ├── index.html           # Real-estate analytics dashboard structure
│   ├── style.css            # Responsive modern theme with charts and animations
│   └── script.js            # Vanilla JS fetching /models, presets, and /predict
├── src/
│   ├── __init__.py
│   ├── etl.py               # Data extraction, cleaning, and feature engineering
│   ├── train.py             # Model training, evaluation, and MLflow logging
│   └── predict.py           # Prediction inference engine and USD converter
├── feature_store/
│   └── housing_features.csv # Processed dataset with engineered features
├── models/
│   ├── best_model.joblib    # Selected champion model pipeline
│   ├── gradient_boosting.joblib
│   └── random_forest.joblib
├── mlruns/                  # MLflow experiment tracking records and artifacts
├── reports/
│   └── evaluation_results.json # Full benchmark metrics, params, and feature importances
├── tests/
│   ├── __init__.py
│   ├── test_api.py          # FastAPI testclient integration tests
│   ├── test_etl.py          # ETL and feature engineering tests
│   ├── test_model.py        # Model training and artifact tests
│   └── test_predict.py      # Prediction and error validation tests
├── pyproject.toml           # uv project configuration and dependencies
├── uv.lock                  # Pinned dependency lockfile
├── .gitignore               # Ignored environments and temporary files
└── README.md                # Project documentation
```

---

## Environment Setup (Using `uv`)

EstateIQ relies exclusively on [uv](https://docs.astral.sh/uv/) for deterministic Python environment and dependency management.

### 1. Prerequisites
- Python 3.12
- `uv` installed on your machine (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 2. Virtual Environment Creation & Synchronization
```bash
# Create Python 3.12 virtual environment
uv venv --python 3.12

# Sync all pinned dependencies from uv.lock
uv sync
```

---

## Execution Guide

All execution commands use `uv run` to guarantee that dependencies run inside the isolated virtual environment.

### 1. Run the ETL Pipeline
Extracts raw data, computes domain features, and saves `feature_store/housing_features.csv`:
```bash
uv run python src/etl.py
```

### 2. Train Models & Track with MLflow
Trains both candidate models, evaluates them on the test set, logs parameters and artifacts to MLflow, and saves the best model:
```bash
uv run python src/train.py
```

### 3. Launch the MLflow UI
Inspect experiment runs, compare hyperparameters, and visualize model metrics:
```bash
uv run mlflow ui --port 5000
```
Open [http://localhost:5000](http://localhost:5000) in your browser and select the `EstateIQ_House_Price_Experiment`.

### 4. Run the Prediction Module in CLI (Optional)
Test the standalone inference pipeline with sample data:
```bash
uv run python src/predict.py
```

### 5. Launch the FastAPI Application & Dashboard
Start the Uvicorn web server:
```bash
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## Accessing the Application

Once the server is running:

- **Web Dashboard**: Navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) or [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard).
- **Interactive API Documentation (Swagger)**: Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
- **Alternative Redoc Documentation**: Navigate to [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc).
- **Health Check Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).
- **Models & Evaluation Metrics**: [http://127.0.0.1:8000/models](http://127.0.0.1:8000/models).

---

## Running the Automated Test Suite

Run the full pytest suite:
```bash
uv run pytest -v
```

The test suite covers:
- Raw California data extraction and column validation.
- Feature engineering calculations (including zero-division safety).
- Missing and negative value imputation.
- Feature store generation in temporary directories.
- Fast model pipeline training, metric computation, and artifact persistence.
- Prediction USD scaling, uncertainty calculation, and bounds enforcement.
- Validation of negative or invalid coordinates.
- FastAPI health, model metrics, prediction schemas, and 422 error handlers.

---

## API Reference

### `GET /health`
Returns system operational status and model readiness.

**Sample Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "best_model_name": "HistGradientBoostingRegressor",
  "version": "0.1.0"
}
```

### `GET /models`
Returns real test benchmark metrics (MAE, RMSE, R²), hyperparameters, and feature importances for all candidate models.

### `POST /predict`
Estimates valuation for a property given 8 block group features.

**Sample Request:**
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322.0,
  "AveOccup": 2.5555,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

**Sample Response:**
```json
{
  "prediction_raw": 4.2448,
  "prediction_usd": 424482.35,
  "prediction_usd_formatted": "$424,482.35",
  "model_name": "HistGradientBoostingRegressor",
  "uncertainty": {
    "lower_bound_usd": 379192.35,
    "upper_bound_usd": 469772.35,
    "lower_bound_formatted": "$379,192.35",
    "upper_bound_formatted": "$469,772.35",
    "uncertainty_usd": 45290.0,
    "method": "empirical_test_rmse",
    "description": "Range reflects ±1 test RMSE ($45,290), representing the standard deviation of out-of-sample model prediction residuals."
  },
  "engineered_features": {
    "rooms_per_household": 6.984,
    "bedrooms_per_room": 0.147,
    "population_per_household": 2.555
  },
  "input_features": { ... },
  "disclaimer": "Educational prediction based on 1990 California Census housing dataset. Not an appraisal or real-time property valuation service."
}
```

---

## Educational Disclaimer

EstateIQ is built for educational, academic, and demonstration purposes using the 1990 California Census dataset. It should not be used as a licensed appraisal or live real-estate appraisal service.

