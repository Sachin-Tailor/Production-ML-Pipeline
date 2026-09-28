"""
Production ML Serving API
High-Performance, Containerized Customer Churn Inference Service using FastAPI & Pydantic.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ml-serving-api")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PIPELINE_PATH = os.path.join(BASE_DIR, "model_pipeline.joblib")
MODEL_METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

# Global state
app_state: Dict = {
    "pipeline": None,
    "metadata": None,
    "start_time": time.time(),
}


def load_artifacts():
    """Load model pipeline and metadata from disk, auto-training if absent."""
    if not (os.path.exists(MODEL_PIPELINE_PATH) and os.path.exists(MODEL_METADATA_PATH)):
        logger.warning("Artifacts missing. Triggering initial model training pipeline...")
        from train import train
        train()

    logger.info("Loading model pipeline from %s", MODEL_PIPELINE_PATH)
    app_state["pipeline"] = joblib.load(MODEL_PIPELINE_PATH)

    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
        app_state["metadata"] = json.load(f)
    logger.info("Model artifacts loaded successfully. Version: %s", app_state["metadata"].get("version"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Initializing API application lifespan...")
    load_artifacts()
    yield
    logger.info("Shutting down API application...")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Enterprise-grade low-latency inference service for Customer Retention & Risk Scoring",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend/dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web assets
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ==========================================
# Pydantic Request & Response Schemas
# ==========================================

class CustomerFeatures(BaseModel):
    credit_score: int = Field(..., ge=300, le=850, description="Customer credit score (300-850)", examples=[650])
    country: Literal["France", "Germany", "Spain"] = Field(..., description="Customer residence country", examples=["Germany"])
    gender: Literal["Female", "Male"] = Field(..., description="Customer gender", examples=["Female"])
    age: int = Field(..., ge=18, le=100, description="Customer age in years", examples=[42])
    tenure: int = Field(..., ge=0, le=10, description="Years customer has been with bank", examples=[3])
    balance: float = Field(..., ge=0.0, description="Account balance", examples=[85000.50])
    num_of_products: int = Field(..., ge=1, le=4, description="Number of bank products used", examples=[2])
    has_credit_card: int = Field(..., ge=0, le=1, description="1 if customer has credit card, else 0", examples=[1])
    is_active_member: int = Field(..., ge=0, le=1, description="1 if active banking member, else 0", examples=[0])
    estimated_salary: float = Field(..., ge=0.0, description="Annual estimated salary", examples=[120000.0])


class PredictionResult(BaseModel):
    churn_prediction: int = Field(..., description="0 = Retained, 1 = Churned")
    churn_label: str = Field(..., description="Human-readable retention outcome")
    churn_probability: float = Field(..., description="Calculated probability of churning (0.0 to 1.0)")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Risk tier based on probability")


class SinglePredictionResponse(BaseModel):
    success: bool = True
    model_version: str
    prediction: PredictionResult


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_length=1, description="List of customers for inference")


class BatchPredictionResponse(BaseModel):
    success: bool = True
    model_version: str
    count: int
    predictions: List[PredictionResult]


def compute_risk_tier(probability: float) -> str:
    """Categorize churn probability into business actionable risk tiers."""
    if probability < 0.35:
        return "Low"
    elif probability < 0.65:
        return "Medium"
    return "High"


def run_inference(features_list: List[CustomerFeatures]) -> List[PredictionResult]:
    """Helper function to execute pipeline inference."""
    pipeline = app_state["pipeline"]
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference",
        )

    # Convert pydantic models to pandas DataFrame
    df = pd.DataFrame([f.model_dump() for f in features_list])

    preds = pipeline.predict(df)
    probas = pipeline.predict_proba(df)[:, 1]

    results = []
    for pred, proba in zip(preds, probas):
        prob = round(float(proba), 4)
        results.append(
            PredictionResult(
                churn_prediction=int(pred),
                churn_label="Churned" if int(pred) == 1 else "Retained",
                churn_probability=prob,
                risk_level=compute_risk_tier(prob),
            )
        )
    return results


# ==========================================
# API Endpoints
# ==========================================

@app.get("/", tags=["General"])
def root(request: Request):
    """Serve modern interactive dashboard for browsers, or API JSON for curl/tests."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        index_file = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
    return {
        "service": "Production ML Inference API",
        "model": "Customer Churn Prediction",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/dashboard", tags=["UI"])
def dashboard():
    """Direct route for interactive visual dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Dashboard UI not found")


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Healthcheck endpoint for orchestrators (Kubernetes / Docker / Cloud)."""
    is_loaded = app_state["pipeline"] is not None
    uptime_seconds = round(time.time() - app_state["start_time"], 2)

    return {
        "status": "healthy" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "model_version": app_state["metadata"].get("version") if app_state["metadata"] else "unknown",
        "uptime_seconds": uptime_seconds,
    }


@app.get("/model/metadata", tags=["Model Info"])
def get_model_metadata():
    """Retrieve current model version, performance metrics, and feature schemas."""
    if app_state["metadata"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is not loaded",
        )
    return app_state["metadata"]


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Inference"])
def predict_single(payload: CustomerFeatures):
    """
    Perform low-latency churn risk inference on a single customer.
    Returns prediction outcome, probability, and risk tier (Low/Medium/High).
    """
    results = run_inference([payload])
    version = app_state["metadata"].get("version", "1.0.0") if app_state["metadata"] else "1.0.0"

    return SinglePredictionResponse(
        success=True,
        model_version=version,
        prediction=results[0],
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(payload: BatchPredictionRequest):
    """
    High-throughput batch inference endpoint.
    Accepts an array of customer records and returns vectorized predictions.
    """
    results = run_inference(payload.customers)
    version = app_state["metadata"].get("version", "1.0.0") if app_state["metadata"] else "1.0.0"

    return BatchPredictionResponse(
        success=True,
        model_version=version,
        count=len(results),
        predictions=results,
    )
