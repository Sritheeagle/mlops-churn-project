
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "model.joblib"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH

app = FastAPI(
    title="Customer Churn Prediction API",
    description="An ML API for predicting customer churn risk.",
    version="1.0.0",
)


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

    return ChurnPredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        risk_tier=risk_tier,
    )
