"""Schemas Pydantic para validação de request/response da API."""

from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Dados de entrada de um cliente para predição de churn."""

    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[70.35])
    TotalCharges: float = Field(..., ge=0, examples=[840.5])


class PredictionOutput(BaseModel):
    """Resultado da predição de churn."""

    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: bool
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    """Resposta do endpoint de health check."""

    status: str
    model_loaded: bool
