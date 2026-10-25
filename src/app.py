
import os
import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "model.joblib"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH

HTTP_REQUESTS = Counter(
    "churn_api_http_requests",
    "Number of HTTP requests handled by the churn API.",
    ("method", "endpoint", "status_code"),
)
HTTP_REQUEST_DURATION = Histogram(
    "churn_api_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "endpoint"),
)
PREDICTIONS = Counter(
    "churn_api_predictions",
    "Number of successful churn predictions by outcome and risk tier.",
    ("prediction", "risk_tier"),
)
PREDICTION_PROBABILITY = Histogram(
    "churn_api_prediction_probability",
    "Distribution of predicted churn probabilities.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="An ML API for predicting customer churn risk.",
    version="1.0.0",
)


@app.middleware("http")
async def record_http_metrics(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)

    started_at = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        HTTP_REQUESTS.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()
        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(time.perf_counter() - started_at)


class CustomerFeatures(BaseModel):
    tenure_months: float = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0, le=500)
    total_charges: float = Field(..., ge=0, le=50000)
    support_tickets: int = Field(..., ge=0, le=50)
    contract_type_code: int = Field(..., ge=0, le=2)
    payment_delay_days: int = Field(..., ge=0, le=90)


class ChurnPredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]


def get_model():
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run python src/train.py first."
        )

    return joblib.load(MODEL_PATH)


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if MODEL_PATH.is_file() else "model_missing"
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict_churn(payload: CustomerFeatures):
    model = get_model()
    input_df = pd.DataFrame([payload.model_dump()])

    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(probability >= 0.5)

    if probability >= 0.7:
        risk_tier = "HIGH"
    elif probability >= 0.4:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    PREDICTIONS.labels(
        prediction=str(prediction),
        risk_tier=risk_tier,
    ).inc()
    PREDICTION_PROBABILITY.observe(probability)

    return ChurnPredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        risk_tier=risk_tier,
    )
