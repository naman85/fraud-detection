from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection API")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))


# ===============================
# Request Schema
# ===============================

class Transaction(BaseModel):
    type: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


# ===============================
# Health Check
# ===============================

@app.get("/")
def home():
    return {"status": "API running successfully"}


# ===============================
# Prediction Endpoint
# ===============================

@app.post("/predict")
def predict(transaction: Transaction):

    features = np.array([[
        transaction.type,
        transaction.amount,
        transaction.oldbalanceOrg,
        transaction.newbalanceOrig,
        transaction.oldbalanceDest,
        transaction.newbalanceDest
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    risk_level = "High Risk" if probability > 0.7 else "Low Risk"

    return {
        "fraud_prediction": int(prediction),
        "risk_score": float(probability),
        "risk_level": risk_level
    }