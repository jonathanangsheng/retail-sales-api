# Retail Sales Analytics API

Simple ML-powered API for sales prediction and pricing optimization.

## Features
- 🎯 Sales prediction using Random Forest
- 💰 Pricing optimization for maximum revenue
- 📊 Statistical analysis
- ✅ Input validation
- 🧪 Unit tests

## Setup (3 Steps)

### 1. Install Python 3.11+
Download from https://python.org

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add dataset
Download from Kaggle and save as `retail_sales.csv` in project folder

## Run
```bash
uvicorn main:app --reload
```

Open: http://localhost:8000/docs

## Test
```bash
pytest test_api.py -v
```

## API Endpoints

- `POST /api/predict` - Predict sales
- `POST /api/optimize` - Find optimal price
- `GET /api/stats` - Dataset statistics
- `GET /api/model/info` - ML model details

## Example Usage

### Predict Sales
```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "price": 100,
    "discount": 0.15,
    "promotion_intensity": 7,
    "footfall": 500,
    "ad_spend": 1000,
    "competitor_price": 105,
    "stock_level": 200,
    "weather_index": 8,
    "customer_sentiment": 7.5,
    "return_rate": 0.05
  }'
```

### Optimize Pricing
```bash
curl -X POST "http://localhost:8000/api/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "min_price": 80,
    "max_price": 120,
    "footfall": 400
  }'
```

## Tech Stack
- FastAPI - REST API framework
- Scikit-learn - Machine learning
- Pandas - Data processing
- Pydantic - Data validation
- Pytest - Testing
