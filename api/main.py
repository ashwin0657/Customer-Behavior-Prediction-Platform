from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Customer Churn Prediction API", version="1.0")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
model = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))


class CustomerFeatures(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    AvgOrderValue: float
    TenureDays: float
    UniqueProducts: float
    ReturnRate: float
    PurchaseFrequencyRate: float


class ChurnPrediction(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: str


def risk_bucket(p: float) -> str:
    if p >= 0.7:
        return "High Risk"
    if p >= 0.4:
        return "Medium Risk"
    return "Low Risk"


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running. See /docs for usage."}


@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(customer: CustomerFeatures):
    try:
        X = pd.DataFrame([customer.dict()])[feature_cols]
        proba = model.predict_proba(X)[0][1]
        return ChurnPrediction(
            churn_probability=round(float(proba), 4),
            will_churn=bool(proba >= 0.5),
            risk_level=risk_bucket(proba),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/customer/{customer_id}")
def get_customer_prediction(customer_id: int):
    df = pd.read_csv(os.path.join(DATA_DIR, "customer_features.csv"))
    row = df[df["CustomerID"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    X = row[feature_cols]
    proba = model.predict_proba(X)[0][1]
    return {
        "customer_id": customer_id,
        "churn_probability": round(float(proba), 4),
        "will_churn": bool(proba >= 0.5),
        "risk_level": risk_bucket(proba),
    }
