from __future__ import annotations

from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")  # Headless backend, safe for scripts/servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from src.utils import FIGURES_DIR, get_logger

logger = get_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with accuracy, precision, recall, and f1 score.
    """
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    logger.info("Computed metrics: %s", metrics)
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model"
) -> str:
    """
    Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        model_name: Name used in the plot title and output file.

    Returns:
        Path to the saved figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_name}")

    out_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", out_path)
    return str(out_path)


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "model"
) -> tuple[str, float]:
    """
    Plot and save an ROC curve, returning the AUC score.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities for the positive class.
        model_name: Name used in the plot title and output file.

    Returns:
        Tuple of (path to saved figure, AUC score).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")

    out_path = FIGURES_DIR / f"roc_curve_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve to %s (AUC=%.4f)", out_path, roc_auc)
    return str(out_path), round(roc_auc, 4)


def plot_feature_importance(
    model: Any, feature_names: list[str], model_name: str = "model", top_n: int = 15
) -> str | None:
    """
    Plot and save a feature importance bar chart for tree-based models,
    or absolute coefficient magnitudes for linear models.

    Args:
        model: A fitted scikit-learn estimator.
        feature_names: List of feature column names.
        model_name: Name used in the plot title and output file.
        top_n: Number of top features to display.

    Returns:
        Path to the saved figure, or None if the model exposes neither
        ``feature_importances_`` nor ``coef_``.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model %s has no feature importance attribute.", model_name)
        return None

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="teal")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances - {model_name}")

    out_path = FIGURES_DIR / f"feature_importance_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance plot to %s", out_path)
    return str(out_path)


def full_evaluation(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Run a complete evaluation for a fitted model: metrics, confusion
    matrix, ROC curve, and feature importance.

    Args:
        model: A fitted scikit-learn classifier.
        X_test: Test feature matrix.
        y_test: Test target vector.
        model_name: Identifier for saved artifacts/plots.

    Returns:
        Dictionary containing all computed metrics and plot paths.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred)
    cm_path = plot_confusion_matrix(y_test, y_pred, model_name)
    roc_path, auc_score = plot_roc_curve(y_test, y_prob, model_name)
    fi_path = plot_feature_importance(model, list(X_test.columns), model_name)

    metrics["roc_auc"] = auc_score
    metrics["confusion_matrix_path"] = cm_path
    metrics["roc_curve_path"] = roc_path
    metrics["feature_importance_path"] = fi_path

    return metrics
