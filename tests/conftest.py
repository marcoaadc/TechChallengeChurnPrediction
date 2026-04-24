"""Fixtures compartilhadas para os testes."""

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture()
def client():
    """TestClient do FastAPI com modelo carregado."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def sample_customer():
    """Dados de exemplo de um cliente para teste de predição."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 840.50,
    }
