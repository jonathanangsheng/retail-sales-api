from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List, Dict
import pandas as pd
import numpy as np
from ml_model import predictor

app = FastAPI(
    title="Retail Sales Analytics API",
    description="ML-powered sales prediction and pricing optimization",
    version="1.0.0"
)

# Data models for validation
class SalesPredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
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
        }]
    })

    price: float = Field(..., gt=0)
    discount: float = Field(..., ge=0, le=1)
    promotion_intensity: float = Field(..., ge=0, le=10)
    footfall: int = Field(..., gt=0)
    ad_spend: float = Field(..., ge=0)
    competitor_price: float = Field(..., gt=0)
    stock_level: int = Field(..., gt=0)
    weather_index: float = Field(..., ge=0, le=10)
    customer_sentiment: float = Field(..., ge=0, le=10)
    return_rate: float = Field(..., ge=0, le=1)


class PricingRequest(BaseModel):
    min_price: float = Field(..., gt=0)
    max_price: float = Field(..., gt=0)
    discount: float = Field(0.1, ge=0, le=1)
    promotion_intensity: float = Field(5.0, ge=0, le=10)
    footfall: int = Field(400, gt=0)
    ad_spend: float = Field(800.0, ge=0)
    competitor_price: float = Field(100.0, gt=0)
    stock_level: int = Field(150, gt=0)
    weather_index: float = Field(7.0, ge=0, le=10)
    customer_sentiment: float = Field(7.0, ge=0, le=10)
    return_rate: float = Field(0.05, ge=0, le=1)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    if not predictor.load():
        print("Model not found. Training new model...")
        try:
            predictor.train()
        except FileNotFoundError:
            print("⚠️  Warning: retail_sales.csv not found!")
            print("Please download dataset from Kaggle and place it in the project folder")

@app.get("/")
async def root():
    return {
        "message": "Retail Sales Analytics API",
        "docs": "/docs",
        "endpoints": {
            "predict_sales": "/api/predict",
            "optimize_pricing": "/api/optimize",
            "statistics": "/api/stats"
        }
    }

@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None
    }

# OUTPUT 1: Sales Prediction with ML
@app.post("/api/predict")
async def predict_sales(request: SalesPredictionRequest):
    """
    🎯 OUTPUT 1: Predict sales using Random Forest ML model
    
    Returns:
    - Predicted sales volume
    - Confidence interval
    - Top 5 features affecting this prediction
    """
    try:
        features = request.dict()
        result = predictor.predict(features)
        importance = predictor.get_feature_importance()
        
        return {
            "prediction": result,
            "top_factors": importance,
            "interpretation": f"Expected to sell {result['predicted_sales']} units"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# OUTPUT 2: Pricing Optimization
@app.post("/api/optimize")
async def optimize_pricing(request: PricingRequest):
    """
    💰 OUTPUT 2: Find optimal price for maximum revenue
    
    Analyzes different price points to find the one that
    maximizes revenue (price × predicted_sales)
    """
    try:
        # Test 50 price points
        prices = np.linspace(request.min_price, request.max_price, 50)
        results = []
        
        fixed_features = {
            'discount': request.discount,
            'promotion_intensity': request.promotion_intensity,
            'footfall': request.footfall,
            'ad_spend': request.ad_spend,
            'competitor_price': request.competitor_price,
            'stock_level': request.stock_level,
            'weather_index': request.weather_index,
            'customer_sentiment': request.customer_sentiment,
            'return_rate': request.return_rate
        }
        
        for price in prices:
            features = {'price': price, **fixed_features}
            pred = predictor.predict(features)
            revenue = price * pred['predicted_sales']
            
            results.append({
                'price': round(price, 2),
                'predicted_sales': pred['predicted_sales'],
                'revenue': round(revenue, 2)
            })
        
        # Find optimal
        optimal = max(results, key=lambda x: x['revenue'])
        
        return {
            "optimal_price": optimal['price'],
            "expected_sales": optimal['predicted_sales'],
            "expected_revenue": optimal['revenue'],
            "analysis": f"Best price is ${optimal['price']} for ${optimal['revenue']} revenue",
            "price_curve": results[::5]  # Every 5th point for visualization
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_statistics():
    """Get dataset statistics"""
    try:
        df = pd.read_csv('retail_sales.csv')
        
        return {
            "dataset_size": len(df),
            "sales_statistics": {
                "mean": float(df['sales'].mean()),
                "median": float(df['sales'].median()),
                "min": float(df['sales'].min()),
                "max": float(df['sales'].max()),
                "std": float(df['sales'].std())
            },
            "price_statistics": {
                "mean": float(df['price'].mean()),
                "median": float(df['price'].median()),
                "min": float(df['price'].min()),
                "max": float(df['price'].max())
            },
            "correlations": {
                "price_vs_sales": float(df['price'].corr(df['sales'])),
                "discount_vs_sales": float(df['discount'].corr(df['sales'])),
                "footfall_vs_sales": float(df['footfall'].corr(df['sales']))
            }
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")

@app.get("/api/model/info")
async def model_info():
    """Get ML model information"""
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Random Forest Regressor",
        "n_estimators": 100,
        "features": predictor.feature_names,
        "feature_importance": predictor.get_feature_importance()
    }
