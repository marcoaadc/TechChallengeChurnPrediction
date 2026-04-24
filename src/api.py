"""API FastAPI para inferência do modelo de Churn."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, Request

from src.model import ChurnMLP
from src.schemas import CustomerInput, HealthResponse, PredictionOutput
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

model: ChurnMLP | None = None
preprocessor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo e preprocessor na inicialização."""
    global model, preprocessor

    model_path = MODELS_DIR / "mlp_churn.pt"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = ChurnMLP(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Modelo carregado de '%s'", model_path)

    preprocessor = joblib.load(preprocessor_path)
    logger.info("Preprocessor carregado de '%s'", preprocessor_path)

    yield


app = FastAPI(
    title="Churn Prediction API",
    description="API de previsão de Churn usando MLP (PyTorch)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_latency(request: Request, call_next):
    """Middleware que loga a latência de cada request."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s — %.1fms", request.method, request.url.path, elapsed_ms)
    return response


@app.get("/health", response_model=HealthResponse)
def health():
    """Verifica se a API e o modelo estão operacionais."""
    return HealthResponse(
        status="ok" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    """Recebe dados de um cliente e retorna a probabilidade de churn."""
    input_df = pd.DataFrame([customer.model_dump()])

    X_processed = preprocessor.transform(input_df)
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probability = torch.sigmoid(logits).item()

    return PredictionOutput(
        churn_probability=round(probability, 4),
        churn_prediction=probability >= 0.5,
    )
