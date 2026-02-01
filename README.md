# Retail Store Analytics API

Predict customer footfall and optimize pricing strategies using machine learning.

## 🎯 What This API Does

Since the dataset has no "sales" column, we predict **footfall** (customer visits) instead:

**OUTPUT 1:** Predict how many customers will visit based on:
- Price, discounts, promotions
- Weather conditions
- Customer sentiment
- Competition

**OUTPUT 2:** Find the optimal price that maximizes customer footfall

## 🚀 Quick Start

1. **Download dataset** from Kaggle and save as `retail_sales.csv`

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run API:**
```bash
uvicorn main:app --reload
```

4. **Open browser:**
http://localhost:8000/docs

## 📊 API Endpoints

- `POST /api/predict` - Predict customer footfall
- `POST /api/optimize` - Find optimal price for max footfall
- `GET /api/stats` - Dataset statistics
- `GET /api/model/info` - Model details

## 🧪 Example Request
```json
{
  "price": 100,
  "discount": 0.15,
  "promotion_intensity": 7,
  "ad_spend": 1000,
  "competitor_price": 105,
  "stock_level": 200,
  "weather_index": 8,
  "customer_sentiment": 7.5,
  "return_rate": 0.05
}
```

## 🎓 For Interview

**Key Points:**
- Target: Footfall (customer visits), not sales
- Features: 9 inputs (price, discount, promotions, etc.)
- Model: Random Forest Regressor
- Use Case: Retail traffic optimization

## ✅ Requirements Met

- ✅ RESTful API (FastAPI)
- ✅ ML/AI (Random Forest)
- ✅ 2 outputs (prediction + optimization)
- ✅ Data validation (Pydantic)
- ✅ Tests (Pytest)
- ✅ Documentation (auto-generated)
