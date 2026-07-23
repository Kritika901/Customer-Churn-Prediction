from __future__ import annotations

import pandas as pd

from src.utils import (
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    TARGET_COLUMN,
    ensure_dirs,
    get_logger,
)

logger = get_logger(__name__)


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn CSV file.

    Args:
        path: Optional custom path to the CSV. Defaults to
            ``data/raw/Telco-Customer-Churn.csv``.

    Returns:
        Raw pandas DataFrame.

    Raises:
        FileNotFoundError: If the dataset cannot be located.
    """
    file_path = path or RAW_DATA_FILE
    try:
        df = pd.read_csv(file_path)
        logger.info("Loaded raw dataset with shape %s from %s", df.shape, file_path)
        return df
    except FileNotFoundError as exc:
        logger.error(
            "Raw dataset not found at %s. Download the IBM Telco Customer "
            "Churn dataset from Kaggle and place it there.",
            file_path,
        )
        raise exc


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct known data type issues in the Telco dataset.

    ``TotalCharges`` is shipped as a string column because a handful of
    rows contain blank spaces instead of numeric values (for customers
    with zero tenure). This function coerces it to a numeric dtype.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with corrected dtypes.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if "SeniorCitizen" in df.columns:
        # Stored as 0/1 int, keep as-is but ensure it's an integer dtype
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    logger.info("Data types corrected.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Numeric columns are imputed with the median; this mainly affects the
    ``TotalCharges`` column after type coercion.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing values handled.
    """
    df = df.copy()
    missing_before = df.isnull().sum().sum()

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Filled missing values in '%s' with median=%.2f", col, median_val)

    categorical_cols = df.select_dtypes(exclude=["number"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("Filled missing values in '%s' with mode='%s'", col, mode_val)

    missing_after = df.isnull().sum().sum()
    logger.info("Missing values reduced from %d to %d.", missing_before, missing_after)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows from the dataset."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    logger.info("Removed %d duplicate rows (%d -> %d).", before - after, before, after)
    return df


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that carry no predictive signal (e.g. customerID)."""
    df = df.copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Dropped 'customerID' identifier column.")
    return df


def clean_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the target column has no missing/invalid values."""
    df = df.copy()
    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    return df


def run_preprocessing(save: bool = True) -> pd.DataFrame:
    """
    Execute the full preprocessing pipeline end-to-end.

    Args:
        save: If True, persist the cleaned dataset to
            ``data/processed/processed_data.csv``.

    Returns:
        The cleaned DataFrame.
    """
    ensure_dirs()
    df = load_raw_data()
    df = fix_data_types(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = clean_target_column(df)
    df = drop_identifier_columns(df)

    if save:
        df.to_csv(PROCESSED_DATA_FILE, index=False)
        logger.info("Processed dataset saved to %s", PROCESSED_DATA_FILE)

    return df


if __name__ == "__main__":
    cleaned_df = run_preprocessing()
    print(cleaned_df.head())
    print(f"\nFinal shape: {cleaned_df.shape}")
