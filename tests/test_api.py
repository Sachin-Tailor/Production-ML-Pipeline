"""
Integration & Unit Tests for FastAPI Serving Endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    # Context manager triggers FastAPI lifespan (loads model artifacts)
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data


def test_dashboard_endpoint(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "RetentionAI" in response.text
    assert "<form" in response.text


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_version" in data


def test_model_metadata_endpoint(client):
    response = client.get("/model/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "accuracy" in data["metrics"]
    assert "f1_score" in data["metrics"]
    assert "features" in data


def test_predict_single_valid_customer(client):
    payload = {
        "credit_score": 680,
        "country": "Germany",
        "gender": "Female",
        "age": 42,
        "tenure": 4,
        "balance": 95000.0,
        "num_of_products": 2,
        "has_credit_card": 1,
        "is_active_member": 0,
        "estimated_salary": 85000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "prediction" in data

    pred = data["prediction"]
    assert pred["churn_prediction"] in [0, 1]
    assert pred["churn_label"] in ["Retained", "Churned"]
    assert 0.0 <= pred["churn_probability"] <= 1.0
    assert pred["risk_level"] in ["Low", "Medium", "High"]


def test_predict_validation_error_invalid_age(client):
    # Age < 18 should fail pydantic validation
    payload = {
        "credit_score": 700,
        "country": "France",
        "gender": "Male",
        "age": 12,  # Invalid age
        "tenure": 1,
        "balance": 1000.0,
        "num_of_products": 1,
        "has_credit_card": 1,
        "is_active_member": 1,
        "estimated_salary": 50000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_validation_error_invalid_country(client):
    # Country not in Literal["France", "Germany", "Spain"]
    payload = {
        "credit_score": 700,
        "country": "India",  # Not in allowed categories
        "gender": "Male",
        "age": 30,
        "tenure": 2,
        "balance": 20000.0,
        "num_of_products": 1,
        "has_credit_card": 1,
        "is_active_member": 1,
        "estimated_salary": 60000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_prediction(client):
    payload = {
        "customers": [
            {
                "credit_score": 750,
                "country": "France",
                "gender": "Male",
                "age": 28,
                "tenure": 5,
                "balance": 0.0,
                "num_of_products": 1,
                "has_credit_card": 1,
                "is_active_member": 1,
                "estimated_salary": 65000.0,
            },
            {
                "credit_score": 580,
                "country": "Germany",
                "gender": "Female",
                "age": 55,
                "tenure": 2,
                "balance": 130000.0,
                "num_of_products": 3,
                "has_credit_card": 0,
                "is_active_member": 0,
                "estimated_salary": 90000.0,
            },
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
