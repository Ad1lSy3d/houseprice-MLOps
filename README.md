# EstateIQ – House Price Intelligence Platform
### Production MLOps System: CI/CD, Containerization, Kubernetes, Terraform, Prometheus/Grafana Monitoring, Streamlit Frontend & MLflow Experiment Tracking

EstateIQ is an enterprise-grade Machine Learning Operations (MLOps) platform that estimates California residential property values. Built on Scikit-learn, FastAPI, and uv, it implements the complete production machine learning lifecycle: automated ETL & feature engineering, experiment tracking and model registry with MLflow, containerization via Docker, infrastructure as code with Terraform, high-availability Kubernetes clustering on Minikube, real-time observability with Prometheus and Grafana, and an interactive cloud-ready Streamlit frontend.

---

## 1. End-to-End System Architecture

```text
                                      ┌──────────────────────────────────────────────┐
                                      │        Streamlit Community Cloud (UI)        │
                                      │            (streamlit_app/app.py)            │
                                      └──────────────────────┬───────────────────────┘
                                                             │
                                                             │ HTTPS / REST (Configurable BACKEND_URL)
                                                             ▼
                                      ┌──────────────────────────────────────────────┐
                                      │         Public FastAPI Cloud Hosting         │
                                      │             (Render Blueprint IaC)           │
                                      └──────────────────────────────────────────────┘
                                                             ▲
                                                             │ (Local Minikube Cluster Architecture)
                                                             │
                                      ┌──────────────────────────────────────────────┐
                                      │       Kubernetes Service (NodePort 30080)    │
                                      │           (estateiq-service: http:8000)      │
                                      └──────────────────────┬───────────────────────┘
                                                             │
                                   ┌─────────────────────────┴─────────────────────────┐
                                   ▼                                                   ▼
                       ┌───────────────────────┐                           ┌───────────────────────┐
                       │  EstateIQ Replica 1   │                           │  EstateIQ Replica 2   │ (3 Replicas Total)
                       │  FastAPI + ML Model   │                           │  FastAPI + ML Model   │
                       └───────────┬───────────┘                           └───────────┬───────────┘
                                   │                                                   │
                                   └─────────────────────────┬─────────────────────────┘
                                                             │ Scrapes /metrics (every 15s)
                                                             ▼
                                              ┌─────────────────────────────┐
                                              │  Prometheus ServiceMonitor  │
                                              │    (release: monitoring)    │
                                              └──────────────┬──────────────┘
                                                             │
                                                             ▼
                                              ┌─────────────────────────────┐
                                              │    Prometheus TSDB Engine   │
                                              │       (NodePort 30090)      │
                                              └──────────────┬──────────────┘
                                                             │ Queries Metrics
                                                             ▼
                                              ┌─────────────────────────────┐
                                              │      Grafana Dashboard      │
                                              │       (NodePort 30300)      │
                                              │  (Auto-provisioned Config)  │
                                              └─────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════════════════════════════
                          OFFLINE DATA & MLFLOW TRACKING PIPELINE
══════════════════════════════════════════════════════════════════════════════════════════════════════

  [California Census Data] ──> [src/etl.py] ──> [feature_store/housing_features.csv]
                                                     │
                                                     ▼
                                            [src/train.py] ──> [models/best_model.joblib]
                                                     │     ──> [reports/evaluation_results.json]
                                                     │     ──> [reports/model_comparison.png]
                                                     ▼
                                      [MLflow Tracking: EstateIQ_Model_Training]
                                      (SQLite Backend: sqlite:///mlruns/mlflow.db)
                                      ├── Run 1: HistGradientBoostingRegressor (Best Model)
                                      └── Run 2: RandomForestRegressor
                                      └── Artifacts: models, plots, evaluation JSONs
```

---

## 2. Repository Structure

```text
houseprice-MLOps/
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI pipeline (lint & pytest)
├── backend/
│   ├── __init__.py
│   ├── main.py                        # FastAPI application with Prometheus instrumentation
│   └── schemas.py                     # Pydantic v2 validation models
├── feature_store/
│   └── housing_features.csv           # Curated feature dataset with engineered ratios
├── frontend/                          # Local HTML/CSS/JS dashboard
│   ├── index.html
│   ├── style.css
│   └── script.js
├── k8s/                               # Raw Kubernetes resource manifests
│   ├── deployment.yaml                # 3-replica EstateIQ deployment definition
│   ├── service.yaml                   # NodePort 30080 service definition with named port http
│   └── servicemonitor.yaml            # Prometheus Operator ServiceMonitor resource
├── models/
│   ├── best_model.joblib              # Persisted winning model (HistGradientBoosting, 556 KB)
│   └── gradient_boosting.joblib       # Standalone HistGradientBoosting model (556 KB)
├── monitoring/
│   └── grafana/
│       ├── estateiq-dashboard.json    # Complete 12-panel Grafana dashboard JSON
│       └── dashboard-configmap.yaml   # ConfigMap with grafana_dashboard=1 label
├── reports/
│   ├── evaluation_results.json        # Test metrics, params, and feature importances
│   ├── model_comparison.json         # Comparative summary between candidate models
│   ├── model_comparison.png          # Performance comparison bar chart
│   ├── histgradientboostingregressor_feature_importance.png
│   └── randomforestregressor_feature_importance.png
├── src/
│   ├── __init__.py
│   ├── etl.py                         # ETL and feature engineering pipeline
│   ├── train.py                       # MLflow tracking pipeline & model training
│   └── predict.py                     # Prediction logic, USD scaling, empirical uncertainty
├── streamlit_app/
│   ├── app.py                         # Streamlit cloud frontend
│   └── requirements.txt               # Standalone dependencies for Streamlit Cloud
├── terraform/
│   ├── main.tf                        # Kubernetes Deployment & Service resources in HCL
│   ├── variables.tf                   # Deployment configuration variables
│   ├── outputs.tf                     # NodePort service URL output
│   └── .terraform.lock.hcl            # Provider lockfile
├── tests/
│   ├── test_api.py                    # FastAPI route testing
│   ├── test_etl.py                    # Data extraction & transformation tests
│   ├── test_model.py                  # MLflow experiment, runs, parameters & artifact tests
│   ├── test_monitoring.py             # Prometheus metrics & ServiceMonitor tests
│   ├── test_predict.py                # Validation rules & prediction boundary tests
│   └── test_streamlit.py              # Streamlit configuration & preset tests
├── .dockerignore                      # Docker context exclusions
├── .gitignore                         # Git exclusions (.venv, mlruns, *.db, *.tfstate, random_forest.joblib)
├── docker-compose.yml                 # Multi-container orchestration definition
├── Dockerfile                         # Production-grade multi-stage container build
├── pyproject.toml                     # Python project dependencies managed via uv
├── render.yaml                        # Infrastructure-as-code blueprint for Render.com hosting
├── uv.lock                            # Deterministic uv dependency lockfile
└── README.md                          # Platform documentation
```

---

## 3. Environment Setup & Dependency Management

EstateIQ uses `uv` for fast, reproducible Python packaging and virtual environment management.

### Prerequisites
- Python 3.12
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`)
- Docker Desktop or Docker Engine
- Minikube & kubectl
- Terraform v1.5+
- Helm v3+

### Virtual Environment Initialization
```bash
# Clone the repository
git clone https://github.com/Ad1lSy3d/houseprice-MLOps.git
cd houseprice-MLOps

# Initialize virtual environment with Python 3.12
uv venv --python 3.12

# Synchronize all project dependencies reproducibly from uv.lock
uv sync
```

---

## 4. MLflow Experiment Tracking & Model Registry

EstateIQ includes enterprise experiment tracking using **MLflow** with a persistent local **SQLite** database backend.

### Experiment Configuration
- **Experiment Name**: `EstateIQ_Model_Training` (configurable via `MLFLOW_EXPERIMENT_NAME`)
- **Backend Store URI**: `sqlite:///mlruns/mlflow.db` (configurable via `MLFLOW_TRACKING_URI`)
- **Artifact Location**: `mlruns/artifacts/`

### Executing Model Training
To run the automated ETL, model training, parameter logging, metric recording, and artifact generation:
```bash
uv run python src/train.py
```

### Logged Experiment Details
Each training run evaluates both candidate models with a reproducible 80/20 train/test split (`random_state=42`) and logs:
- **Separate Runs**: Dedicated runs for `HistGradientBoostingRegressor` and `RandomForestRegressor`.
- **Logged Parameters**:
  - Model Name & Model Type
  - Framework (`scikit-learn`) & Preprocessing Scaler (`StandardScaler`)
  - Random Seed (`random_state=42`)
  - Hyperparameters (`learning_rate`, `max_iter`, `max_leaf_nodes`, `n_estimators`, `max_depth`, `n_jobs`)
  - Dataset Information (`California Housing`, 11 features, 16,512 train samples, 4,128 test samples)
- **Evaluation Metrics**:
  - Out-of-sample Test MAE & Test RMSE (target scale $100,000)
  - Test MAE in USD & Test RMSE in USD
  - Test $R^2$ Score
- **Artifacts Recorded**:
  - `evaluation/evaluation_results.json`: Full evaluation results and feature importances
  - `evaluation/model_comparison.json`: Comparative metrics summary
  - `evaluation/model_comparison.png`: Comparison bar chart across MAE, RMSE, and $R^2$
  - `plots/<model>_feature_importance.png`: Feature importance horizontal bar chart
  - `model/<model>.joblib` & `model/best_model.joblib`: Serialized Scikit-learn Pipeline
  - `model_pipeline`: Native MLflow Scikit-learn model flavor
- **Best Model Selection**:
  - Automatically identifies the winning model based on lowest test RMSE.
  - Adds tags `is_best_model: "true"` and `best_model_selection: "HistGradientBoostingRegressor"`.
  - Persists the winner to `models/best_model.joblib` for zero-downtime serving.

### Evaluated Model Performance Summary

| Candidate Model | Test MAE | Test RMSE | $R^2$ Score | Test MAE (USD) | Test RMSE (USD) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HistGradientBoostingRegressor** | **0.3008** | **0.4529** | **0.8435** | **$30,079.38** | **$45,290.80** | **Selected Best Model** |
| RandomForestRegressor | 0.3286 | 0.5057 | 0.8048 | $32,857.09 | $50,569.71 | Candidate Model |

### Launching the MLflow UI
Launch the interactive MLflow dashboard:
```bash
# Standard command (port 5000):
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000

# On macOS where ControlCenter/AirPlay listens on port 5000, use port 5001:
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5001
```
Open your browser at:
- **http://localhost:5000** (or **http://localhost:5001**)

Inside the UI, navigate to the **`EstateIQ_Model_Training`** experiment to view side-by-side run comparisons, parameter diffs, metric charts, and visual artifact plots.

---

## 5. Docker Containerization

The backend is packaged into a production-grade container image using Python 3.12-slim and `uv`.

### Build and Run Commands
```bash
# Build the Docker image
docker build -t estateiq:latest .

# Run the container locally mapping port 8000
docker run -d --name estateiq-app -p 8000:8000 estateiq:latest

# Verify container health
curl -s http://localhost:8000/health
```

---

## 6. Kubernetes Clustering & Terraform IaC

The application is deployed to Kubernetes as a 3-replica high-availability deployment, managed declaratively via Terraform.

### Terraform Commands
```bash
cd terraform

# Initialize Terraform Kubernetes provider
terraform init

# Inspect execution plan
terraform plan

# Apply infrastructure changes (manages 3 replicas and NodePort 30080)
terraform apply -auto-approve
```

### Direct Kubernetes Commands (Minikube)
```bash
# Load Docker image into Minikube
minikube image load estateiq:latest --overwrite=true

# Restart deployment to pull updated image
kubectl rollout restart deployment/estateiq-deployment
kubectl rollout status deployment/estateiq-deployment

# Verify all replicas are running
kubectl get pods -l app=estateiq
```

---

## 7. Prometheus & Grafana Monitoring

EstateIQ is instrumented with `prometheus-fastapi-instrumentator`, exposing standard HTTP metrics on `/metrics`.

### ServiceMonitor Integration
Prometheus discovers and scrapes EstateIQ via the Kubernetes `ServiceMonitor` (`k8s/servicemonitor.yaml`):
- Namespace: `default`
- Service Selector: `app: estateiq`
- Port: `http` (targetPort 8000)
- Metric Path: `/metrics`
- Scrape Interval: `15s`
- Label: `release: monitoring`

### Grafana Dashboard (`monitoring/grafana/estateiq-dashboard.json`)
Auto-provisioned via the Grafana sidecar using `monitoring/grafana/dashboard-configmap.yaml` with label `grafana_dashboard: "1"`.

#### Monitored Panels & PromQL Queries:
1. **Total HTTP Requests**:
   ```promql
   sum(http_requests_total{job="estateiq-service"})
   ```
2. **Kubernetes Pod Availability**:
   ```promql
   sum(kube_deployment_status_replicas_available{deployment="estateiq-deployment", namespace="default"})
   ```
3. **Failed Requests (4xx / 5xx)**:
   ```promql
   sum(http_requests_total{job="estateiq-service", status=~"[45].*"}) or vector(0)
   ```
4. **P95 Request Latency**:
   ```promql
   histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="estateiq-service"}[5m])) by (le))
   ```
5. **Request Rate Over Time (req/s)**:
   ```promql
   sum by (handler, method) (rate(http_requests_total{job="estateiq-service"}[1m]))
   ```
6. **Average Request Duration by Endpoint**:
   ```promql
   sum(rate(http_request_duration_seconds_sum{job="estateiq-service"}[5m])) by (handler) / sum(rate(http_request_duration_seconds_count{job="estateiq-service"}[5m])) by (handler)
   ```
7. **HTTP Status Code Distribution**:
   ```promql
   sum by (status) (http_requests_total{job="estateiq-service"})
   ```
8. **CPU Usage per Pod**:
   ```promql
   sum by (pod) (rate(container_cpu_usage_seconds_total{pod=~"estateiq-deployment-.*", container="estateiq", namespace="default"}[1m]))
   ```
9. **Memory Working Set per Pod**:
   ```promql
   sum by (pod) (container_memory_working_set_bytes{pod=~"estateiq-deployment-.*", container="estateiq", namespace="default"})
   ```

---

## 8. Streamlit Frontend & Public Deployment

EstateIQ features an intuitive Streamlit frontend located in `streamlit_app/`.

### Architecture: Direct In-Process Cloud Inference
On Streamlit Community Cloud, EstateIQ loads the trained model artifact (`models/best_model.joblib`) directly into memory using `@st.cache_resource`. This eliminates the need for any third-party hosting service (like Render) or external backend URLs:
- **Direct Predictions**: Uses `src/predict.py` directly for in-memory, zero-latency inference and feature engineering.
- **Self-Contained Benchmarks**: Loads model comparison metrics and feature importances directly from `reports/evaluation_results.json`.
- **Optional FastAPI Gateway**: The frontend includes an optional toggle for local development or Kubernetes/Docker MLOps demonstrations.

### Running Streamlit Locally
```bash
# Run Streamlit frontend
uv run streamlit run streamlit_app/app.py
```

### Public Deployment on Streamlit Community Cloud
1. Push repository changes to GitHub (`main` branch).
2. Connect to [share.streamlit.io](https://share.streamlit.io/).
3. App settings:
   - **Repository**: `Ad1lSy3d/houseprice-MLOps`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app/app.py`
4. Deploy: Streamlit Cloud installs dependencies from `streamlit_app/requirements.txt` and starts the app with direct model execution. No secret or `BACKEND_URL` is required.

---

## 9. Verification & Test Execution Guide

Run the comprehensive pytest suite covering API endpoints, ETL pipeline, model training, MLflow tracking, prediction bounds, Prometheus metrics, and Streamlit direct inference:
```bash
uv run pytest -v
```

### Test Suite Execution Output:
```text
tests/test_api.py::test_api_status_endpoint PASSED                       [  3%]
tests/test_api.py::test_health_endpoint PASSED                           [  6%]
tests/test_api.py::test_models_endpoint PASSED                           [  9%]
tests/test_api.py::test_predict_endpoint_valid PASSED                    [ 12%]
tests/test_api.py::test_predict_endpoint_invalid_values PASSED           [ 16%]
tests/test_api.py::test_predict_endpoint_missing_fields PASSED           [ 19%]
tests/test_api.py::test_dashboard_endpoint PASSED                        [ 22%]
tests/test_etl.py::test_extract_data PASSED                              [ 25%]
tests/test_etl.py::test_engineer_features PASSED                         [ 29%]
tests/test_etl.py::test_clean_data PASSED                                [ 32%]
tests/test_etl.py::test_run_etl_creates_feature_store PASSED             [ 35%]
tests/test_model.py::test_model_training_and_artifacts PASSED            [ 38%]
tests/test_model.py::test_mlflow_experiment_and_runs PASSED              [ 41%]
tests/test_model.py::test_repeated_training_creates_new_runs PASSED      [ 45%]
tests/test_model.py::test_existing_model_prediction_still_works PASSED   [ 48%]
tests/test_monitoring.py::test_metrics_endpoint_exposed PASSED           [ 51%]
tests/test_monitoring.py::test_grafana_dashboard_json_structure PASSED   [ 54%]
tests/test_monitoring.py::test_servicemonitor_manifest PASSED            [ 58%]
tests/test_predict.py::test_predict_price_valid_input PASSED             [ 61%]
tests/test_predict.py::test_predict_price_missing_feature PASSED         [ 64%]
tests/test_predict.py::test_predict_price_negative_values PASSED         [ 67%]
tests/test_predict.py::test_predict_price_out_of_bounds_geography PASSED [ 70%]
tests/test_predict.py::test_load_model_missing_file PASSED               [ 74%]
tests/test_streamlit.py::test_streamlit_files_exist PASSED               [ 77%]
tests/test_streamlit.py::test_streamlit_requirements_complete PASSED     [ 80%]
tests/test_streamlit.py::test_streamlit_presets_valid_against_schema PASSED [ 83%]
tests/test_resolve_backend_url_environment PASSED                         [ 87%]
tests/test_render_blueprint_valid PASSED                                  [ 90%]
tests/test_streamlit.py::test_direct_model_loading_and_caching PASSED    [ 93%]
tests/test_streamlit.py::test_direct_prediction_from_streamlit PASSED    [ 96%]
tests/test_streamlit.py::test_direct_prediction_all_presets PASSED       [100%]

======================== 31 passed, 3 warnings in 9.32s ========================
```

---

## 10. Screenshots & Evidence Checklist

Capture these screenshots for your project submission or portfolio:

1. **MLflow Experiment & Runs UI**: MLflow Web UI showing the `EstateIQ_Model_Training` experiment, runs for `HistGradientBoostingRegressor` and `RandomForestRegressor`, metrics (MAE, RMSE, $R^2$), and logged artifacts (plots, model, evaluation JSON).
2. **MLflow Feature Importance Artifact**: MLflow run artifact viewer showing the logged `plots/histgradientboostingregressor_feature_importance.png` plot.
3. **Docker Container Run**: Terminal showing `docker ps` with container `estateiq:latest` listening on `0.0.0.0:8000`.
4. **Kubernetes Deployment**: Terminal showing `kubectl get pods -o wide` and `kubectl get svc estateiq-service` with 3 healthy replicas.
5. **Terraform Apply Execution**: Terminal output showing `Apply complete! Resources: 0 added, 2 changed, 0 destroyed.`
6. **Prometheus Targets UI**: Prometheus Web UI (`http://<minikube-ip>:30090/targets`) displaying `serviceMonitor/default/estateiq-servicemonitor/0` with state **UP** (3/3 endpoints).
7. **Grafana Monitoring Dashboard**: Grafana UI (`http://<minikube-ip>:30300/d/estateiq-monitoring`) displaying all 12 live panels with request counts, latency quantiles, and status distribution.
8. **Streamlit Prediction Interface**: Browser showing `streamlit_app/app.py` with Bay Area preset valuation result, empirical uncertainty bounds, and engineered features.
9. **FastAPI Swagger Documentation**: Browser showing `http://localhost:8000/docs` with all endpoints (`/predict`, `/models`, `/health`, `/metrics`).
10. **GitHub Actions CI Run**: Green checkmark showing all 27 pytest tests passing on GitHub Actions.

---

## 11. Viva & Theoretical Concepts Reference

- **Why SQLite-Backed MLflow Tracking over Raw Filesystem?**
  MLflow v3 natively supports database backends via SQLAlchemy. An SQLite backend (`sqlite:///mlruns/mlflow.db`) provides atomic transactions, query indexing, robust concurrency for parallel training workers, and persistent experiment metadata across sessions.
- **Why Scikit-learn Pipeline over loose transformers?**
  Pipelines prevent data leakage by strictly fitting scalers on the training split only and applying pre-fitted transformations during cross-validation and production inference.
- **Why HistGradientBoostingRegressor?**
  Bins continuous numerical features into 256 discrete bins, achieving orders of magnitude faster training and inference with native handling of non-linear interactions compared to standard gradient boosting.
- **Why ServiceMonitor over static Prometheus scrape configs?**
  Prometheus Operator utilizes Kubernetes Custom Resource Definitions (CRDs). A `ServiceMonitor` decouples metric scraping targets from Prometheus server configuration, automatically tracking dynamically scaling pod IPs and endpoints.
- **Why Named Port `http` in Service?**
  ServiceMonitors reference endpoints by the symbolic port name defined in `spec.ports[].name` rather than hardcoded container port numbers, adhering to declarative Kubernetes standards.
- **Why Empirical Out-of-Sample RMSE for Uncertainty?**
  Avoids fabricating arbitrary confidence percentages; directly communicates the standard deviation of regression residuals observed on held-out test data ($\pm 1 \text{ RMSE} \approx \$45,291$).
