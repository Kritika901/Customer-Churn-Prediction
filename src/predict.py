from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.feature_engineering import (
    label_encode_binary,
    one_hot_encode,
    scale_numeric_features,
)
from src.utils import (
    ENCODERS_FILE,
    FEATURE_COLUMNS_FILE,
    MODEL_FILE,
    SCALER_FILE,
    get_logger,
    load_artifact,
)

logger = get_logger(__name__)


class ChurnPredictor:
    """
    Wraps the trained model and all preprocessing artifacts so that raw
    customer records (single or batch) can be turned into churn
    predictions with a single call.
    """

    def __init__(self) -> None:
        self.model = load_artifact(MODEL_FILE)
        self.scaler = load_artifact(SCALER_FILE)
        self.encoders = load_artifact(ENCODERS_FILE)
        self.feature_columns = load_artifact(FEATURE_COLUMNS_FILE)
        logger.info("ChurnPredictor initialized with %d features.", len(self.feature_columns))

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the exact same encoding/scaling pipeline used in training."""
        df = df.copy()

        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
        if "Churn" in df.columns:
            df = df.drop(columns=["Churn"])
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median() if df["TotalCharges"].notna().any() else 0)

        df, _ = label_encode_binary(df, fit=False, encoders=self.encoders)
        df, _ = one_hot_encode(df, reference_columns=self.feature_columns)
        df, _ = scale_numeric_features(df, fit=False, scaler=self.scaler)

        # Guarantee exact column order expected by the model
        df = df.reindex(columns=self.feature_columns, fill_value=0)
        return df

    def predict_single(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict churn for a single customer.

        Args:
            customer: Dictionary of raw feature values (same schema as
                the original dataset columns, minus customerID/Churn).

        Returns:
            Dictionary with prediction label, churn probability,
            confidence, and the top contributing factors.
        """
        df = pd.DataFrame([customer])
        X = self._prepare_features(df)

        prob_churn = float(self.model.predict_proba(X)[0, 1])
        prediction = "Churn" if prob_churn >= 0.5 else "No Churn"
        confidence = prob_churn if prediction == "Churn" else 1 - prob_churn

        top_factors = self._top_factors(X)

        return {
            "prediction": prediction,
            "churn_probability": round(prob_churn * 100, 2),
            "confidence": round(confidence * 100, 2),
            "top_factors": top_factors,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn for a batch of customers (e.g. an uploaded CSV).

        Args:
            df: Raw DataFrame with the original Telco schema.

        Returns:
            A copy of the input DataFrame with two extra columns:
            ``Churn_Prediction`` and ``Churn_Probability``.
        """
        result_df = df.copy()
        X = self._prepare_features(df)

        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = np.where(probabilities >= 0.5, "Churn", "No Churn")

        result_df["Churn_Prediction"] = predictions
        result_df["Churn_Probability"] = np.round(probabilities * 100, 2)
        return result_df

    def _top_factors(self, X: pd.DataFrame, top_n: int = 5) -> list[Dict[str, Any]]:
        """
        Approximate the top contributing factors for a single prediction
        using the model's global feature importance (or coefficients),
        weighted by the customer's own (scaled) feature values.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            return []

        contribution = np.abs(X.values[0]) * importances
        top_idx = np.argsort(contribution)[::-1][:top_n]

        factors = [
            {"feature": self.feature_columns[i], "impact": round(float(contribution[i]), 4)}
            for i in top_idx
        ]
        return factors


if __name__ == "__main__":
    predictor = ChurnPredictor()
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.5,
        "TotalCharges": 452.5,
    }
    print(predictor.predict_single(sample_customer))
