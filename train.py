"""
Production ML Training Pipeline
Customer Churn Prediction with Scikit-Learn Pipeline & Metadata Tracking.
"""

import json
import os
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PIPELINE_PATH = os.path.join(MODEL_DIR, "model_pipeline.joblib")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

NUMERIC_FEATURES = [
    "credit_score",
    "age",
    "tenure",
    "balance",
    "num_of_products",
    "estimated_salary",
]

CATEGORICAL_FEATURES = [
    "country",
    "gender",
    "has_credit_card",
    "is_active_member",
]

TARGET_COL = "churn"


def generate_synthetic_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic customer churn data with realistic business correlations."""
    np.random.seed(random_state)

    credit_score = np.random.normal(650, 100, n_samples).clip(350, 850).astype(int)
    country = np.random.choice(["France", "Germany", "Spain"], size=n_samples, p=[0.5, 0.25, 0.25])
    gender = np.random.choice(["Female", "Male"], size=n_samples, p=[0.45, 0.55])
    age = np.random.gamma(shape=9, scale=4, size=n_samples).clip(18, 92).astype(int)
    tenure = np.random.randint(0, 11, size=n_samples)
    balance = np.where(
        np.random.rand(n_samples) > 0.3,
        np.random.normal(110000, 35000, n_samples).clip(1000, 250000),
        0.0,
    )
    num_of_products = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.50, 0.45, 0.04, 0.01])
    has_credit_card = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
    is_active_member = np.random.choice([0, 1], size=n_samples, p=[0.48, 0.52])
    estimated_salary = np.random.uniform(15000, 180000, size=n_samples).round(2)

    # Churn probability calculation based on known risk drivers
    logit = (
        -2.5
        + (age > 45) * 1.2
        + (num_of_products >= 3) * 1.8
        + (is_active_member == 0) * 0.9
        + (country == "Germany") * 0.7
        + (balance > 120000) * 0.5
        - (credit_score > 720) * 0.4
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "credit_score": credit_score,
        "country": country,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "num_of_products": num_of_products,
        "has_credit_card": has_credit_card,
        "is_active_member": is_active_member,
        "estimated_salary": estimated_salary,
        "churn": churn,
    })
    return df


def build_pipeline() -> Pipeline:
    """Construct an end-to-end preprocessing and modeling pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def train(data_path: str = None) -> dict:
    """Train the model pipeline, evaluate metrics, and persist artifacts."""
    print("=" * 60)
    print("Starting Production ML Training Pipeline: Customer Churn")
    print("=" * 60)

    if data_path and os.path.exists(data_path):
        print(f"Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Generating realistic synthetic churn dataset...")
        df = generate_synthetic_data()

    print(f"Total dataset shape: {df.shape}")
    print(f"Churn distribution:\n{df[TARGET_COL].value_counts(normalize=True).round(3)}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    print("\nTraining preprocessing & classification pipeline...")
    pipeline.fit(X_train, y_train)

    # Predictions & Probabilities
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics evaluation
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    print("\nModel Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  - {k.upper()}: {v}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    # Save trained pipeline artifact
    joblib.dump(pipeline, MODEL_PIPELINE_PATH)
    print(f"Model pipeline saved to: {MODEL_PIPELINE_PATH}")

    # Save model metadata & schema
    metadata = {
        "model_name": "customer_churn_random_forest",
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "problem_type": "binary_classification",
        "metrics": metrics,
        "features": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "target_mapping": {0: "Retained", 1: "Churned"},
        "sample_count": len(df),
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {MODEL_METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    train()
