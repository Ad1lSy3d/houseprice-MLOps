"""
Unit tests for ETL and feature engineering pipeline.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.etl import (
    ALL_FEATURE_NAMES,
    ENGINEERED_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    TARGET_NAME,
    clean_data,
    engineer_features,
    extract_data,
    run_etl,
)


def test_extract_data():
    """Verify California Housing extraction from scikit-learn."""
    df = extract_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 20000
    for feat in RAW_FEATURE_NAMES:
        assert feat in df.columns
    assert TARGET_NAME in df.columns


def test_engineer_features():
    """Verify feature engineering calculations and edge case handling."""
    sample_df = pd.DataFrame({
        "MedInc": [5.0, 3.0],
        "HouseAge": [25.0, 15.0],
        "AveRooms": [5.0, 0.0],  # Edge case: zero rooms
        "AveBedrms": [1.0, 0.5],
        "Population": [1000.0, 500.0],
        "AveOccup": [3.0, 2.5],
        "Latitude": [34.0, 37.0],
        "Longitude": [-118.0, -122.0],
    })

    result_df = engineer_features(sample_df)

    # Check engineered columns exist
    for col in ENGINEERED_FEATURE_NAMES:
        assert col in result_df.columns

    # Test rooms_per_household
    assert result_df.loc[0, "rooms_per_household"] == 5.0

    # Test bedrooms_per_room (1.0 / 5.0 = 0.2)
    assert pytest.approx(result_df.loc[0, "bedrooms_per_room"]) == 0.2

    # Zero-division edge case handled gracefully (filled with 0.0)
    assert result_df.loc[1, "bedrooms_per_room"] == 0.0

    # Test population_per_household
    assert result_df.loc[0, "population_per_household"] == 3.0


def test_clean_data():
    """Verify cleaning handles nulls and invalid negative physical values."""
    dirty_df = pd.DataFrame({
        "MedInc": [4.0, -2.0, 6.0],  # Negative income invalid
        "HouseAge": [20.0, np.nan, 30.0],  # Missing value
        "AveRooms": [5.0, 6.0, 7.0],
        "AveBedrms": [1.0, 1.2, 1.1],
        "Population": [800.0, 900.0, 1000.0],
        "AveOccup": [2.5, 3.0, 2.8],
        "Latitude": [34.0, 35.0, 36.0],
        "Longitude": [-118.0, -119.0, -120.0],
    })

    cleaned = clean_data(dirty_df)

    assert not cleaned.isnull().any().any()
    assert (cleaned["MedInc"] > 0).all()
    assert not np.isnan(cleaned.loc[1, "HouseAge"])


def test_run_etl_creates_feature_store(tmp_path: Path):
    """Verify running ETL generates housing_features.csv in configured output path."""
    target_csv = tmp_path / "custom_features.csv"
    output_path = run_etl(output_path=target_csv)

    assert output_path.exists()
    assert output_path == target_csv

    loaded = pd.read_csv(output_path)
    assert len(loaded) > 20000
    for col in ALL_FEATURE_NAMES + [TARGET_NAME]:
        assert col in loaded.columns

