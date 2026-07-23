"""
utils.py
--------
Common utility functions shared across the Customer Churn Prediction
pipeline: logging setup, path handling, and joblib save/load helpers.

Author: ML Engineering Team
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib

# --------------------------------------------------------------------------- #
# Project paths (single source of truth for every script in the project)
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

RAW_DATA_FILE: Path = DATA_RAW_DIR / "Telco-Customer-Churn.csv"
PROCESSED_DATA_FILE: Path = DATA_PROCESSED_DIR / "processed_data.csv"
MODEL_FILE: Path = MODELS_DIR / "churn_model.pkl"
SCALER_FILE: Path = MODELS_DIR / "scaler.pkl"
ENCODERS_FILE: Path = MODELS_DIR / "encoders.pkl"
FEATURE_COLUMNS_FILE: Path = MODELS_DIR / "feature_columns.pkl"
METRICS_FILE: Path = REPORTS_DIR / "metrics.json"

TARGET_COLUMN: str = "Churn"


def ensure_dirs() -> None:
    """Create every project directory required for the pipeline to run."""
    for directory in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def get_logger(name: str, log_file: str = "pipeline.log") -> logging.Logger:
    """
    Create (or fetch) a logger that writes to both the console and a
    rotating log file under reports/.

    Args:
        name: Name of the logger, typically ``__name__`` of the caller.
        log_file: File name (relative to reports/) to persist logs to.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    ensure_dirs()
    logger = logging.getLogger(name)

    if logger.handlers:  # Avoid duplicate handlers on repeated calls
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(REPORTS_DIR / log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def save_artifact(obj: Any, path: Path) -> None:
    """Persist any Python object to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path) -> Any:
    """Load a joblib-serialized artifact from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artifact not found at '{path}'. Did you run `python src/train.py` first?"
        )
    return joblib.load(path)
