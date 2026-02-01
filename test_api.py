import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_sales():
    payload = {
        "price": 100.0,
        "discount": 0.15,
        "promotion_intensity": 7.0,
        "footfall": 500,
        "ad_spend": 1000.0,
        "competitor_price": 105.0,
        "stock_level": 200,
        "weather_index": 8.0,
        "customer_sentiment": 7.5,
        "return_rate": 0.05
    }

    response = client.post("/api/predict", json=payload)

    # In a fresh clone (no dataset/model), prediction endpoints should return 503.
    if response.status_code == 503:
        pytest.skip("Model not loaded (dataset/model not provided)")

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"]["predicted_sales"] > 0

def test_pricing_optimization():
    payload = {
        "min_price": 80.0,
        "max_price": 120.0,
        "footfall": 400
    }

    response = client.post("/api/optimize", json=payload)

    if response.status_code == 503:
        pytest.skip("Model not loaded (dataset/model not provided)")

    assert response.status_code == 200
    data = response.json()
    assert "optimal_price" in data
    assert 80 <= data["optimal_price"] <= 120

def test_statistics():
    response = client.get("/api/stats")
    # Will return 404 if dataset is not present, which is acceptable
    assert response.status_code in [200, 404]

def test_invalid_input():
    payload = {"price": -10}  # Invalid negative price
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422  # Validation error
