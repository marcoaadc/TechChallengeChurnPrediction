"""Transformador custom para feature engineering de Churn."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria features derivadas para o modelo de Churn.

    Features geradas:
        - total_services_count: quantidade de serviços ativos (Yes)
        - tenure_to_charges_ratio: tenure / MonthlyCharges
        - has_no_support: 1 se não tem OnlineSecurity nem TechSupport
        - is_new_customer: 1 se tenure < 6 meses
    """

    SERVICE_COLS = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "ChurnFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["total_services_count"] = X[self.SERVICE_COLS].apply(lambda row: sum(1 for v in row if v == "Yes"), axis=1)
        X["tenure_to_charges_ratio"] = X["tenure"] / (X["MonthlyCharges"] + 1e-6)
        X["has_no_support"] = ((X["OnlineSecurity"] == "No") & (X["TechSupport"] == "No")).astype(int)
        X["is_new_customer"] = (X["tenure"] < 6).astype(int)
        return X
