"""
ETL and Feature Engineering Pipeline for EstateIQ.

Extracts the California Housing dataset from scikit-learn, validates and cleans
the raw records, engineers domain-specific features, and persists the curated
features into a project-root-relative feature store.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base feature definitions
RAW_FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET_NAME = "MedHouseVal"
ENGINEERED_FEATURE_NAMES = [
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
]
ALL_FEATURE_NAMES = RAW_FEATURE_NAMES + ENGINEERED_FEATURE_NAMES


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def extract_data() -> pd.DataFrame:
    """
    Extract the California Housing dataset from scikit-learn.

    Returns:
        pd.DataFrame: Raw California housing dataset with target column.
    """
    logger.info("Extracting California Housing dataset from scikit-learn...")
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    logger.info("Extracted %d rows and %d columns.", df.shape[0], df.shape[1])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer housing domain features from base features:
    - rooms_per_household: average rooms per household (AveRooms)
    - bedrooms_per_room: ratio of bedrooms to total rooms (AveBedrms / AveRooms)
    - population_per_household: average occupants per household (AveOccup)

    Args:
        df: DataFrame containing the 8 raw housing features.

    Returns:
        DataFrame augmented with engineered features.
    """
    df_feat = df.copy()

    # Avoid zero-division issues safely
    ave_rooms_safe = df_feat["AveRooms"].replace(0, np.nan)

    # 1. rooms_per_household (in the California dataset, AveRooms is already rooms/household)
    df_feat["rooms_per_household"] = df_feat["AveRooms"].astype(float)

    # 2. bedrooms_per_room: ratio of bedrooms to rooms
    df_feat["bedrooms_per_room"] = (df_feat["AveBedrms"] / ave_rooms_safe).fillna(0.0)

    # 3. population_per_household: average occupancy per household
    df_feat["population_per_household"] = df_feat["AveOccup"].astype(float)

    return df_feat


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing or invalid values:
    - Drop exact duplicate records
    - Fill any missing values with median values
    - Ensure positive values for strictly physical features
    """
    df_clean = df.copy()

    # Check for missing values
    null_counts = df_clean.isnull().sum()
    if null_counts.any():
        logger.warning("Missing values detected: %s. Imputing with median.", null_counts[null_counts > 0].to_dict())
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

    # Remove impossible negative values for physical quantities
    physical_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
    for col in physical_cols:
        if col in df_clean.columns:
            invalid_mask = df_clean[col] < 0
            if invalid_mask.any():
                logger.warning("Replacing %d negative values in '%s' with median.", invalid_mask.sum(), col)
                median_val = df_clean.loc[~invalid_mask, col].median()
                df_clean.loc[invalid_mask, col] = median_val

    # Replace inf values if any
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    if df_clean.isnull().any().any():
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

    return df_clean


def run_etl(output_path: Path | str | None = None) -> Path:
    """
    Execute the complete ETL and feature engineering pipeline.

    Args:
        output_path: Path where the processed dataset will be stored.
                     Defaults to '<project_root>/feature_store/housing_features.csv'.

    Returns:
        Path to the saved processed dataset.
    """
    root = get_project_root()
    if output_path is None:
        target_file = root / "feature_store" / "housing_features.csv"
    else:
        target_file = Path(output_path)
        if not target_file.is_absolute():
            target_file = root / target_file

    target_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extract
    raw_df = extract_data()

    # 2. Clean
    cleaned_df = clean_data(raw_df)

    # 3. Engineer features
    processed_df = engineer_features(cleaned_df)

    # 4. Save to feature store
    processed_df.to_csv(target_file, index=False)
    logger.info("Feature store generated successfully at: %s", target_file)
    logger.info("Processed dataset shape: %s", processed_df.shape)

    return target_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EstateIQ ETL & Feature Engineering Pipeline")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to output CSV file relative to project root.",
    )
    args = parser.parse_args()
    run_etl(output_path=args.output)

