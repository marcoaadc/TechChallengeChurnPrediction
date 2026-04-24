"""Testes da API FastAPI: unitários (mockados) + integração E2E."""


# --- Testes unitários (modelo mockado, rápidos) ---


def test_health_endpoint(client):
    """Smoke test: /health retorna status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_invalid_data(client):
    """Predição com dados incompletos retorna erro 422."""
    response = client.post("/predict", json={"gender": "Female"})
    assert response.status_code == 422


def test_predict_invalid_senior_citizen(client, sample_customer):
    """SeniorCitizen fora do range [0, 1] retorna erro 422."""
    sample_customer["SeniorCitizen"] = 5
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 422


# --- Teste de integração E2E (modelo real carregado) ---


def test_predict_integration(integration_client, sample_customer):
    """E2E: predição com modelo real retorna probabilidade válida."""
    response = integration_client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert isinstance(data["churn_prediction"], bool)
    assert "model_version" in data
