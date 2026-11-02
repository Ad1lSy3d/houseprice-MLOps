"""
EstateIQ – House Price Intelligence Platform
Public Streamlit Web Frontend

Direct in-process Scikit-learn inference engine for Streamlit Community Cloud,
with optional FastAPI connectivity for local Docker, Kubernetes and MLOps demonstrations.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path for reliable module resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import streamlit as st
from sklearn.pipeline import Pipeline

from src.predict import (
    RAW_FEATURE_NAMES,
    InvalidFeatureError,
    ModelNotFoundError,
    load_evaluation_report,
    load_model,
    predict_price,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="EstateIQ – House Price Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished real-estate aesthetics
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #38BDF8;
        font-weight: 700;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0.2rem 0;
    }
    .metric-sub {
        font-size: 0.9rem;
        color: #94A3B8;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.85rem 1.2rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #92400E;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Model Loading & Report Caching
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_cached_model() -> tuple[Pipeline | None, str | None]:
    """
    Load and cache the trained Scikit-learn Pipeline from models/best_model.joblib.
    Cached across all user interactions to eliminate repeated disk I/O.
    """
    try:
        model_path = PROJECT_ROOT / "models" / "best_model.joblib"
        model = load_model(model_path)
        return model, None
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return None, str(exc)


@st.cache_data(show_spinner=False)
def load_cached_report() -> dict[str, Any]:
    """
    Load and cache model evaluation benchmark report.
    """
    try:
        report_path = PROJECT_ROOT / "reports" / "evaluation_results.json"
        return load_evaluation_report(report_path)
    except Exception as exc:
        logger.error("Failed to load evaluation report: %s", exc)
        return {
            "best_model": "HistGradientBoostingRegressor",
            "best_model_metrics": {"rmse": 0.4529, "mae": 0.3008, "r2_score": 0.8435},
        }


# ==============================================================================
# Preset Profiles (Matching Existing Dashboard)
# ==============================================================================
PRESETS = {
    "San Francisco Bay Area": {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.9841,
        "AveBedrms": 1.0238,
        "Population": 322.0,
        "AveOccup": 2.5555,
        "Latitude": 37.88,
        "Longitude": -122.23,
    },
    "Los Angeles Coastal": {
        "MedInc": 5.6431,
        "HouseAge": 32.0,
        "AveRooms": 5.8200,
        "AveBedrms": 1.0100,
        "Population": 1100.0,
        "AveOccup": 2.7500,
        "Latitude": 33.95,
        "Longitude": -118.42,
    },
    "Central Valley (Inland)": {
        "MedInc": 2.4500,
        "HouseAge": 25.0,
        "AveRooms": 4.8500,
        "AveBedrms": 1.0500,
        "Population": 2150.0,
        "AveOccup": 3.2000,
        "Latitude": 36.75,
        "Longitude": -119.78,
    },
    "San Diego Suburban": {
        "MedInc": 4.5000,
        "HouseAge": 18.0,
        "AveRooms": 5.4000,
        "AveBedrms": 1.0300,
        "Population": 1850.0,
        "AveOccup": 2.9000,
        "Latitude": 32.85,
        "Longitude": -117.15,
    },
}

# ==============================================================================
# Helper Functions: Optional FastAPI Gateway
# ==============================================================================
def resolve_backend_url() -> str:
    """Resolve backend URL from env vars, secrets, or fallback."""
    if "BACKEND_URL" in os.environ and os.environ["BACKEND_URL"].strip():
        return os.environ["BACKEND_URL"].strip().rstrip("/")
    try:
        if "BACKEND_URL" in st.secrets and st.secrets["BACKEND_URL"].strip():
            return st.secrets["BACKEND_URL"].strip().rstrip("/")
    except Exception:
        pass
    return "http://localhost:8000"


def check_backend_health(url: str) -> tuple[bool, str]:
    """Test connectivity to optional FastAPI /health endpoint."""
    if not url:
        return False, "No URL configured"
    try:
        resp = requests.get(f"{url}/health", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            model_name = data.get("best_model_name", "Online")
            return True, f"Online ({model_name})"
        return False, f"HTTP {resp.status_code}"
    except Exception:
        return False, "Offline"


def fetch_model_benchmarks(url: str) -> dict[str, Any] | None:
    """Retrieve model evaluation results from /models endpoint."""
    try:
        resp = requests.get(f"{url}/models", timeout=2.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def call_predict_api(url: str, payload: dict[str, float]) -> tuple[dict[str, Any] | None, str | None]:
    """Send prediction request to optional FastAPI backend."""
    try:
        resp = requests.post(
            f"{url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=8.0,
        )
        if resp.status_code == 200:
            return resp.json(), None
        elif resp.status_code == 422:
            err_detail = resp.json().get("detail", "Validation error")
            return None, f"Input Validation Error: {err_detail}"
        else:
            return None, f"Server Error (HTTP {resp.status_code}): {resp.text}"
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to backend at {url}. Ensure FastAPI is running."
    except requests.exceptions.Timeout:
        return None, "Request timed out waiting for the prediction API."
    except Exception as exc:
        return None, f"Unexpected error: {str(exc)}"


# ==============================================================================
# Sidebar: Model Status & Controls
# ==============================================================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/tandpfun/skill-icons/main/icons/Python-Dark.svg", width=42)
    st.title("EstateIQ Control Panel")

    st.subheader("Model Engine")
    model, model_error = load_cached_model()
    benchmarks_data = load_cached_report()
    best_model_name = benchmarks_data.get("best_model", "HistGradientBoostingRegressor")

    if model is not None:
        st.success(f"Model Loaded: {best_model_name}", icon="✅")
        st.caption("Direct In-Process Inference • Model cached in memory")
    else:
        st.error("Model Unavailable", icon="🚨")
        if model_error:
            st.caption(f"Error: {model_error}")
        st.caption("Ensure `models/best_model.joblib` exists in repository.")

    with st.expander("FastAPI Gateway (Optional / Local MLOps)"):
        st.caption("For local Kubernetes/Docker cluster demonstration:")
        default_url = resolve_backend_url()
        backend_url = st.text_input(
            "Service URL",
            value=default_url,
            help="Optional FastAPI service URL (e.g. http://localhost:8000).",
        ).strip().rstrip("/")
        use_remote_api = st.checkbox(
            "Route predictions via FastAPI",
            value=False,
            help="Check to route predictions through the external FastAPI REST API instead of direct in-memory model execution.",
        )
        if backend_url:
            is_healthy, health_msg = check_backend_health(backend_url)
            if is_healthy:
                st.caption(f"API Gateway: {health_msg} ✅")
            else:
                st.caption(f"API Gateway: {health_msg} (Fallback to In-Process)")

    st.divider()

    st.subheader("Regional Presets")
    selected_preset = st.selectbox(
        "Populate with California Profile:",
        options=list(PRESETS.keys()),
        index=0,
    )

    if st.button("Apply Selected Preset", use_container_width=True):
        preset_vals = PRESETS[selected_preset]
        for k, v in preset_vals.items():
            st.session_state[f"input_{k}"] = v
        st.rerun()

    st.divider()
    st.caption("EstateIQ MLOps Platform • v0.1.0")

# ==============================================================================
# Main Application Content
# ==============================================================================
st.markdown('<div class="main-header">🏡 EstateIQ – House Price Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Automated Property Valuation Powered by Machine Learning & MLOps Pipelines</div>',
    unsafe_allow_html=True,
)

# Initialize session state with default preset if uninitialized
initial_data = PRESETS["San Francisco Bay Area"]
for k, v in initial_data.items():
    if f"input_{k}" not in st.session_state:
        st.session_state[f"input_{k}"] = v

# Form Layout: Left Column = Inputs, Right Column = Real-time Valuation Card
col_form, col_result = st.columns([1.1, 1.0], gap="large")

with col_form:
    st.subheader("Property Characteristics")
    st.caption("Enter the eight block-group census attributes:")

    with st.form("prediction_form"):
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            med_inc = st.number_input(
                "Median Income ($10,000s)",
                min_value=0.1,
                max_value=30.0,
                value=float(st.session_state["input_MedInc"]),
                step=0.1,
                format="%.4f",
                help="Median income in block group (e.g. 8.32 = $83,200/yr).",
            )
        with row1_col2:
            house_age = st.number_input(
                "Median House Age (years)",
                min_value=1.0,
                max_value=120.0,
                value=float(st.session_state["input_HouseAge"]),
                step=1.0,
                format="%.1f",
                help="Median age of houses in block group.",
            )

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            ave_rooms = st.number_input(
                "Average Rooms per Home",
                min_value=0.5,
                max_value=50.0,
                value=float(st.session_state["input_AveRooms"]),
                step=0.1,
                format="%.4f",
                help="Average total rooms per household.",
            )
        with row2_col2:
            ave_bedrms = st.number_input(
                "Average Bedrooms per Home",
                min_value=0.2,
                max_value=20.0,
                value=float(st.session_state["input_AveBedrms"]),
                step=0.05,
                format="%.4f",
                help="Average bedrooms per household.",
            )

        row3_col1, row3_col2 = st.columns(2)
        with row3_col1:
            population = st.number_input(
                "Block Population",
                min_value=1.0,
                max_value=50000.0,
                value=float(st.session_state["input_Population"]),
                step=50.0,
                format="%.0f",
                help="Total population residing in block group.",
            )
        with row3_col2:
            ave_occup = st.number_input(
                "Average Occupancy (people/home)",
                min_value=0.5,
                max_value=50.0,
                value=float(st.session_state["input_AveOccup"]),
                step=0.1,
                format="%.4f",
                help="Average occupants per household.",
            )

        row4_col1, row4_col2 = st.columns(2)
        with row4_col1:
            latitude = st.number_input(
                "Latitude (°N)",
                min_value=32.0,
                max_value=43.0,
                value=float(st.session_state["input_Latitude"]),
                step=0.01,
                format="%.4f",
                help="California latitude bounds: 32.0° to 43.0°.",
            )
        with row4_col2:
            longitude = st.number_input(
                "Longitude (°W)",
                min_value=-125.0,
                max_value=-114.0,
                value=float(st.session_state["input_Longitude"]),
                step=0.01,
                format="%.4f",
                help="California longitude bounds: -125.0° to -114.0°.",
            )

        submitted = st.form_submit_button(
            "Estimate Property Value",
            type="primary",
            use_container_width=True,
        )

with col_result:
    st.subheader("Valuation Estimate")

    features_payload = {
        "MedInc": float(med_inc),
        "HouseAge": float(house_age),
        "AveRooms": float(ave_rooms),
        "AveBedrms": float(ave_bedrms),
        "Population": float(population),
        "AveOccup": float(ave_occup),
        "Latitude": float(latitude),
        "Longitude": float(longitude),
    }

    if submitted or "last_prediction" in st.session_state:
        pred_data: dict[str, Any] | None = None
        err_msg: str | None = None

        if use_remote_api:
            with st.spinner("Querying EstateIQ FastAPI backend..."):
                pred_data, err_msg = call_predict_api(backend_url, features_payload)
        else:
            with st.spinner("Calculating property valuation..."):
                try:
                    if model is None:
                        err_msg = (
                            "Model artifact not loaded. Please verify that "
                            "`models/best_model.joblib` exists in the repository."
                        )
                    else:
                        pred_data = predict_price(features_payload, model=model)
                except InvalidFeatureError as val_err:
                    err_msg = f"Input Validation Error: {str(val_err)}"
                except Exception as exc:
                    err_msg = f"Inference Error: {str(exc)}"

        if err_msg:
            st.error(err_msg, icon="🚨")
        elif pred_data:
            st.session_state["last_prediction"] = pred_data
            price_formatted = pred_data.get("prediction_usd_formatted", "$0")
            model_name = pred_data.get("model_name", "Machine Learning Model")
            unc = pred_data.get("uncertainty", {})
            eng = pred_data.get("engineered_features", {})

            # Featured Metric Display
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated Market Valuation</div>
                    <div class="metric-value">{price_formatted}</div>
                    <div class="metric-sub">Champion Model: <strong>{model_name}</strong></div>
                    <div style="margin-top: 0.8rem; font-size: 0.95rem; color: #38BDF8;">
                        Empirical Uncertainty Range (±1 RMSE):<br>
                        <strong>{unc.get('lower_bound_formatted', '$0')} – {unc.get('upper_bound_formatted', '$0')}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Engineered Features Breakdown
            st.caption("Automated Feature Engineering Ratios:")
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric(
                    "Bedrooms / Room",
                    f"{eng.get('bedrooms_per_room', 0):.3f}",
                    help="Ratio of bedrooms to total rooms.",
                )
            with mcol2:
                st.metric(
                    "Rooms / Household",
                    f"{eng.get('rooms_per_household', 0):.2f}",
                    help="Total room count per household.",
                )
            with mcol3:
                st.metric(
                    "Occupancy / Home",
                    f"{eng.get('population_per_household', 0):.2f}",
                    help="Average residents per home.",
                )

            # Defensible Uncertainty Notice
            st.info(
                f"🛡️ **Defensible Uncertainty Estimation**: {unc.get('description', 'Derived from test RMSE.')}",
                icon="ℹ️",
            )
    else:
        st.info("👈 Review the property attributes on the left and click **Estimate Property Value** to generate a real-time ML prediction.", icon="💡")

    # Mandatory Educational Disclaimer
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Educational Disclaimer:</strong> This valuation is generated using a regression model trained on the
            <strong>1990 U.S. Census California Housing dataset</strong>. It is designed for educational, demonstration,
            and research purposes only. It does not reflect current real-estate valuations or certified appraisals.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# Analytics & Model Architecture Tabs
# ==============================================================================
st.divider()

tab_models, tab_dataset, tab_arch = st.tabs(["📊 Model Benchmark Intelligence", "📑 Dataset & Limitations", "⚙️ MLOps Architecture"])

with tab_models:
    st.subheader("Model Performance Benchmarks")
    st.caption("Champion and candidate models evaluated on held-out test data (random_state=42):")

    benchmarks = None
    if use_remote_api:
        benchmarks = fetch_model_benchmarks(backend_url)
    if not benchmarks:
        benchmarks = load_cached_report()

    if benchmarks and "models" in benchmarks:
        models_dict = benchmarks["models"]
        best_name = benchmarks.get("best_model", "")

        table_rows = []
        for name, info in models_dict.items():
            metrics = info.get("metrics", {})
            is_best = "★ Best Champion" if name == best_name else "Evaluated"
            table_rows.append({
                "Architecture": name,
                "MAE ($)": f"${metrics.get('mae_usd', 0):,.2f}",
                "RMSE ($)": f"${metrics.get('rmse_usd', 0):,.2f}",
                "R² Score": f"{metrics.get('r2_score', 0):.4f}",
                "Status": is_best,
            })

        st.dataframe(table_rows, use_container_width=True)

        # Feature Importances of champion model
        feat_imp = benchmarks.get("best_model_feature_importances", {})
        if feat_imp:
            st.subheader(f"Feature Importances ({best_name})")
            st.bar_chart(feat_imp)
    else:
        st.info("Model benchmark report is unavailable.")

with tab_dataset:
    st.subheader("California Housing Census Dataset (1990)")
    st.markdown(
        """
        - **Source**: 1990 U.S. Census data available via `sklearn.datasets.fetch_california_housing`.
        - **Sample Size**: 20,640 California census block groups.
        - **Target Variable**: Median house value for households in each block group (measured in units of $100,000).
        - **Known Limitations**:
          - Values were top-coded (capped) at $500,000 in the original 1990 census.
          - Attributes represent geographic block group averages rather than individual home listings.
          - Economic inflation and 30+ years of California demographic shifts make this unsuitable for actual real-estate transactions.
        """
    )

with tab_arch:
    st.subheader("End-to-End MLOps System Architecture")
    st.markdown(
        """
        ```
        [Streamlit Community Cloud]
               │
               ├── (Direct In-Process Inference) ──> [Scikit-learn Pipeline (Embedded)]
               │                                          StandardScaler + HistGradientBoosting
               │
               └── (Optional REST API Mode) ──> [FastAPI Backend (Kubernetes / Docker)]
                                                      └── [Prometheus & Grafana Monitoring]
        ```
        - **Direct In-Process Mode**: Standalone cloud inference using `@st.cache_resource` for zero-latency responses without requiring a separate web service.
        - **MLOps Cluster Mode**: Enterprise 3-replica Kubernetes deployment with Terraform and Prometheus/Grafana observability.
        - **Experimentation**: MLflow tracking with persistent SQLite backend logging hyperparameters, metrics, and serialized artifacts.
        """
    )
