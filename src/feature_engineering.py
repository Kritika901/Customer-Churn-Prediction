from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import (
    ENCODERS_FILE,
    FEATURE_COLUMNS_FILE,
    SCALER_FILE,
    TARGET_COLUMN,
    get_logger,
    save_artifact,
)

logger = get_logger(__name__)

# Binary categorical columns -> Label Encoding
BINARY_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Multi-category nominal columns -> One-Hot Encoding
ONE_HOT_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map the target column ('Yes'/'No') to binary integers (1/0)."""
    df = df.copy()
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).map({"Yes": 1, "No": 0})
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def label_encode_binary(
    df: pd.DataFrame, fit: bool = True, encoders: dict | None = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Label-encode binary categorical columns.

    Args:
        df: Input DataFrame.
        fit: Whether to fit new encoders (True for training) or reuse
            existing ones (False for inference).
        encoders: Dictionary of pre-fitted encoders, required when
            ``fit=False``.

    Returns:
        Tuple of (transformed DataFrame, dictionary of fitted encoders).
    """
    df = df.copy()
    encoders = encoders or {}
    cols_present = [c for c in BINARY_COLUMNS if c in df.columns]

    for col in cols_present:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = df[col].astype(str).map(
                lambda val, le=le: le.transform([val])[0] if val in le.classes_ else -1
            )

    logger.info("Label encoded columns: %s", cols_present)
    return df, encoders


def one_hot_encode(
    df: pd.DataFrame, reference_columns: list | None = None
) -> Tuple[pd.DataFrame, list]:
    """
    One-hot encode nominal multi-category columns.

    Args:
        df: Input DataFrame.
        reference_columns: Full set of expected columns after encoding
            (from training). Used to align inference-time data so it has
            exactly the same columns as the training set.

    Returns:
        Tuple of (encoded DataFrame, list of resulting feature columns).
    """
    cols_present = [c for c in ONE_HOT_COLUMNS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_present, drop_first=True)

    if reference_columns is not None:
        # Align columns: add missing ones as 0, drop unexpected extras
        for col in reference_columns:
            if col not in df.columns:
                df[col] = 0
        extra_cols = [c for c in df.columns if c not in reference_columns and c != TARGET_COLUMN]
        df = df.drop(columns=extra_cols, errors="ignore")
        ordered_cols = [c for c in reference_columns if c in df.columns]
        df = df[ordered_cols + ([TARGET_COLUMN] if TARGET_COLUMN in df.columns else [])]

    logger.info("One-hot encoded columns: %s", cols_present)
    return df, list(df.columns)


def scale_numeric_features(
    df: pd.DataFrame, fit: bool = True, scaler: StandardScaler | None = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numeric features using StandardScaler (zero mean, unit variance).

    Args:
        df: Input DataFrame.
        fit: Whether to fit a new scaler (training) or reuse an existing
            one (inference).
        scaler: A pre-fitted StandardScaler, required when ``fit=False``.

    Returns:
        Tuple of (scaled DataFrame, fitted StandardScaler).
    """
    df = df.copy()
    cols_present = [c for c in NUMERIC_COLUMNS if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols_present] = scaler.fit_transform(df[cols_present])
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when fit=False.")
        df[cols_present] = scaler.transform(df[cols_present])

    logger.info("Scaled numeric columns: %s", cols_present)
    return df, scaler


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target vector y."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def train_test_splitter(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split preserving the churn class ratio."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Train/test split -> X_train: %s, X_test: %s", X_train.shape, X_test.shape
    )
    return X_train, X_test, y_train, y_test


def run_feature_engineering(
    df: pd.DataFrame, save: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """
    Execute the complete feature engineering pipeline for model training.

    Args:
        df: Cleaned DataFrame (output of preprocessing.run_preprocessing).
        save: Whether to persist encoders, scaler, and feature columns.

    Returns:
        X_train, X_test, y_train, y_test, and a dict bundling all fitted
        artifacts (encoders, scaler, feature_columns).
    """
    df = encode_target(df)
    df, encoders = label_encode_binary(df, fit=True)
    df, feature_columns_with_target = one_hot_encode(df)

    X, y = split_features_target(df)
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_splitter(X, y)

    X_train, scaler = scale_numeric_features(X_train, fit=True)
    X_test, _ = scale_numeric_features(X_test, fit=False, scaler=scaler)

    if save:
        save_artifact(encoders, ENCODERS_FILE)
        save_artifact(scaler, SCALER_FILE)
        save_artifact(feature_columns, FEATURE_COLUMNS_FILE)
        logger.info("Saved encoders, scaler, and feature columns to models/.")

    artifacts = {
        "encoders": encoders,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }
    return X_train, X_test, y_train, y_test, artifacts


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing

    cleaned = run_preprocessing(save=False)
    X_tr, X_te, y_tr, y_te, _ = run_feature_engineering(cleaned)
    print("X_train shape:", X_tr.shape)
    print("X_test shape:", X_te.shape)
