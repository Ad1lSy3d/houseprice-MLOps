"""
Pydantic Schemas for EstateIQ FastAPI backend.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class PropertyFeaturesSchema(BaseModel):
    """Input features required for house price prediction."""

    MedInc: float = Field(
        ...,
        gt=0.0,
        le=30.0,
        description="Median income in block group (in tens of thousands USD, e.g. 8.32 = $83,200)",
        examples=[8.3252],
    )
    HouseAge: float = Field(
        ...,
        ge=1.0,
        le=120.0,
        description="Median house age in block group (years)",
        examples=[41.0],
    )
    AveRooms: float = Field(
        ...,
        gt=0.0,
        le=150.0,
        description="Average number of rooms per household",
        examples=[6.9841],
    )
    AveBedrms: float = Field(
        ...,
        gt=0.0,
        le=50.0,
        description="Average number of bedrooms per household",
        examples=[1.0238],
    )
    Population: float = Field(
        ...,
        gt=0.0,
        le=50000.0,
        description="Block group population",
        examples=[322.0],
    )
    AveOccup: float = Field(
        ...,
        gt=0.0,
        le=1500.0,
        description="Average number of household members",
        examples=[2.5555],
    )
    Latitude: float = Field(
        ...,
        ge=32.0,
        le=43.0,
        description="Geographical latitude within California bounds (32.0 to 43.0)",
        examples=[37.88],
    )
    Longitude: float = Field(
        ...,
        ge=-125.0,
        le=-114.0,
        description="Geographical longitude within California bounds (-125.0 to -114.0)",
        examples=[-122.23],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.9841,
                "AveBedrms": 1.0238,
                "Population": 322.0,
                "AveOccup": 2.5555,
                "Latitude": 37.88,
                "Longitude": -122.23,
            }
        }
    }


class UncertaintySchema(BaseModel):
    lower_bound_usd: float
    upper_bound_usd: float
    lower_bound_formatted: str
    upper_bound_formatted: str
    uncertainty_usd: float
    method: str
    description: str


class PredictionResponseSchema(BaseModel):
    prediction_raw: float
    prediction_usd: float
    prediction_usd_formatted: str
    model_name: str
    uncertainty: UncertaintySchema
    engineered_features: dict[str, float]
    input_features: dict[str, float]
    disclaimer: str


class ModelMetricsSchema(BaseModel):
    mae: float
    rmse: float
    r2_score: float
    mae_usd: float
    rmse_usd: float


class ModelDetailSchema(BaseModel):
    metrics: ModelMetricsSchema
    params: dict[str, Any]
    feature_importances: dict[str, float]
    model_file: str


class ModelsResponseSchema(BaseModel):
    experiment_name: str
    best_model: str
    selection_metric: str
    models: dict[str, ModelDetailSchema]
    best_model_metrics: ModelMetricsSchema
    best_model_feature_importances: dict[str, float]


class HealthResponseSchema(BaseModel):
    status: str
    model_loaded: bool
    best_model_name: str
    version: str


class AppStatusResponseSchema(BaseModel):
    app: str
    status: str
    version: str
    description: str
    endpoints: dict[str, str]

