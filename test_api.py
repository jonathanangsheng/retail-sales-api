import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_footfall():
    payload = {
        "price": 100.0,
        "discount": 0.15,
        "promotion_intensity": 7.0,
        "ad_spend": 1000.0,
        "competitor_price": 105.0,
        "stock_level": 200,
        "weather_index": 8.0,
        "customer_sentiment": 7.5,
        "return_rate": 0.05
    }
    
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"]["predicted_footfall"] > 0

def test_pricing_optimization():
    payload = {
        "min_price": 80.0,
        "max_price": 120.0
    }
    
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "optimal_price" in data
    assert 80 <= data["optimal_price"] <= 120

def test_statistics():
    response = client.get("/api/stats")
    assert response.status_code in [200, 404]

def test_invalid_price():
    payload = {"price": -10, "discount": 0.1}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422
