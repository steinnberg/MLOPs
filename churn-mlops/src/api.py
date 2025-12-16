import os
import pandas as pd
import mlflow
import mlflow.pyfunc

from fastapi import FastAPI, HTTPException
from src.schema import PredictRequest

app = FastAPI(title="Churn MLOps API", version="1.0")

MODEL_URI = os.getenv("MODEL_URI")  # ex: "runs:/<run_id>/model"
_model = None

@app.on_event("startup")
def load_model():
    global _model
    if not MODEL_URI:
        print("⚠️ MODEL_URI not set. Example: runs:/<run_id>/model")
        return
    _model = mlflow.pyfunc.load_model(MODEL_URI)
    print(f"✅ Loaded model from: {MODEL_URI}")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_uri": MODEL_URI}

@app.post("/predict")
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Set MODEL_URI and restart.")
    # 1 row dataframe
    X = pd.DataFrame([req.features])
    # sklearn pipeline returns predict / predict_proba via pyfunc wrapper (predict gives class)
    y_pred = _model.predict(X)
    # y_pred could be array-like
    pred = int(y_pred[0])
    return {"churn_pred": pred}

@app.get("/")
def root():
    return {
        "message": "Churn MLOps API is running",
        "docs": "/docs"
    }
