"""API FastAPI para inferência do modelo de Churn."""

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from src.feature_engineering import ChurnFeatureEngineer
from src.model import ChurnMLP
from src.schemas import CustomerInput, HealthResponse, PredictionOutput
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Carrega modelo e preprocessor na inicialização via app.state."""
    model_path = MODELS_DIR / "mlp_churn.pt"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"

    app.state.model = None
    app.state.preprocessor = None
    app.state.optimal_threshold = 0.5

    try:
        if not model_path.exists():
            logger.error("Arquivo do modelo não encontrado: '%s'", model_path)
        elif not preprocessor_path.exists():
            logger.error("Arquivo do preprocessor não encontrado: '%s'", preprocessor_path)
        else:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
            mlp = ChurnMLP(
                input_dim=checkpoint["input_dim"],
                hidden_dims=checkpoint["hidden_dims"],
                dropout=checkpoint["dropout"],
            )
            mlp.load_state_dict(checkpoint["model_state_dict"])
            mlp.eval()
            app.state.model = mlp
            app.state.optimal_threshold = checkpoint.get("optimal_threshold", 0.5)
            logger.info("Modelo carregado de '%s' (threshold=%.2f)", model_path, app.state.optimal_threshold)

            app.state.preprocessor = joblib.load(preprocessor_path)
            logger.info("Preprocessor carregado de '%s'", preprocessor_path)
    except Exception:
        logger.exception("Falha ao carregar modelo ou preprocessor")

    yield


app = FastAPI(
    title="Churn Prediction API",
    description="API de previsão de Churn usando MLP (PyTorch)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_latency(request: Request, call_next: Callable) -> Response:
    """Middleware que loga a latência de cada request."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s — %.1fms", request.method, request.url.path, elapsed_ms)
    return response


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Verifica se a API e o modelo estão operacionais."""
    model_loaded = hasattr(request.app.state, "model") and request.app.state.model is not None
    return HealthResponse(
        status="ok" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput, request: Request) -> PredictionOutput:
    """Recebe dados de um cliente e retorna a probabilidade de churn."""
    model = getattr(request.app.state, "model", None)
    preprocessor = getattr(request.app.state, "preprocessor", None)

    if model is None or preprocessor is None:
        logger.error("Tentativa de predição sem modelo/preprocessor carregado")
        raise HTTPException(status_code=503, detail="Modelo não está disponível. Tente novamente mais tarde.")

    try:
        input_df = pd.DataFrame([customer.model_dump()])
        feature_engineer = ChurnFeatureEngineer()
        input_df = feature_engineer.transform(input_df)
        X_processed = preprocessor.transform(input_df)
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    except Exception as exc:
        logger.exception("Erro ao pré-processar entrada")
        raise HTTPException(status_code=422, detail=f"Erro no pré-processamento dos dados: {exc}") from exc

    try:
        with torch.no_grad():
            logits = model(X_tensor)
            probability = torch.sigmoid(logits).item()
    except Exception as exc:
        logger.exception("Erro durante a inferência do modelo")
        raise HTTPException(status_code=500, detail="Erro interno durante a inferência.") from exc

    threshold = getattr(request.app.state, "optimal_threshold", 0.5)

    return PredictionOutput(
        churn_probability=round(probability, 4),
        churn_prediction=probability >= threshold,
    )
