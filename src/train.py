from __future__ import annotations

import json
from typing import Any, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.evaluate import full_evaluation
from src.feature_engineering import run_feature_engineering
from src.preprocessing import run_preprocessing
from src.utils import (
    METRICS_FILE,
    MODEL_FILE,
    ensure_dirs,
    get_logger,
    save_artifact,
)

logger = get_logger(__name__)

RANDOM_STATE = 42

MODEL_REGISTRY: Dict[str, Any] = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
    ),
}


def train_all_models(X_train, y_train) -> Dict[str, Any]:
    """Fit every model in ``MODEL_REGISTRY`` on the training data."""
    trained = {}
    for name, model in MODEL_REGISTRY.items():
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)
        trained[name] = model
        logger.info("%s training complete.", name)
    return trained


def select_best_model(
    results: Dict[str, Dict[str, Any]]
) -> tuple[str, Dict[str, Any]]:
    
    """Select the best-performing model based on ROC-AUC, tie-broken by F1."""
    
    best_name = max(
        results, key=lambda name: (results[name]["roc_auc"], results[name]["f1_score"])
    )
    logger.info("Best model selected: %s (%s)", best_name, results[best_name])
    return best_name, results[best_name]


def run_training_pipeline() -> None:
    """Execute the full training pipeline and persist all artifacts."""
    ensure_dirs()
    logger.info("=== Starting Customer Churn training pipeline ===")

    try:
        cleaned_df = run_preprocessing(save=True)
    except FileNotFoundError:
        logger.error(
            "Cannot proceed without the raw dataset. Place "
            "'Telco-Customer-Churn.csv' inside data/raw/ and re-run."
        )
        return

    X_train, X_test, y_train, y_test, artifacts = run_feature_engineering(
        cleaned_df, save=True
    )

    trained_models = train_all_models(X_train, y_train)

    all_metrics: Dict[str, Dict[str, Any]] = {}
    for name, model in trained_models.items():
        logger.info("Evaluating %s ...", name)
        metrics = full_evaluation(model, X_test, y_test, model_name=name)
        all_metrics[name] = metrics

    best_name, best_metrics = select_best_model(all_metrics)
    best_model = trained_models[best_name]

    save_artifact(best_model, MODEL_FILE)
    logger.info("Best model ('%s') saved to %s", best_name, MODEL_FILE)

    report = {
        "best_model": best_name,
        "best_model_metrics": best_metrics,
        "all_models": all_metrics,
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=str)
    logger.info("Full comparison report saved to %s", METRICS_FILE)

    print("\n================ TRAINING SUMMARY ================")
    for name, metrics in all_metrics.items():
        print(
            f"{name:20s} | Acc: {metrics['accuracy']:.3f} | "
            f"Prec: {metrics['precision']:.3f} | Rec: {metrics['recall']:.3f} | "
            f"F1: {metrics['f1_score']:.3f} | AUC: {metrics['roc_auc']:.3f}"
        )
    print(f"\nBest model: {best_name}")
    print(f"Model saved to: {MODEL_FILE}")
    print("====================================================\n")


if __name__ == "__main__":
    run_training_pipeline()
