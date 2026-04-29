"""Fixtures compartilhadas para os testes."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture()
def client():
    """TestClient com modelo e preprocessor MOCKADOS (testes unitários rápidos)."""
    mock_model = MagicMock()
    mock_model.return_value = MagicMock()
    mock_model.return_value.item.return_value = 0.0

    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = np.zeros((1, 44))

    app.state.model = mock_model
    app.state.preprocessor = mock_preprocessor

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture()
def integration_client():
    """TestClient que carrega o modelo real (.pt) — apenas para teste E2E."""
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
