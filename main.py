from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
from ml_model import predictor

app = FastAPI(
    title="Retail Store Analytics API",
    description="ML-powered footfall prediction and pricing optimization for retail",
    version="1.0.0"
)

# Request models
class FootfallPredictionRequest(BaseModel):
    price: float = Field(..., gt=0, description="Product price")
    discount: float = Field(..., ge=0, le=1, description="Discount (0-1)")
    promotion_intensity: float = Field(..., ge=0, le=10, description="Promotion strength (0-10)")
    ad_spend: float = Field(..., ge=0, description="Advertising spend")
    competitor_price: float = Field(..., gt=0, description="Competitor price")
    stock_level: int = Field(..., gt=0, description="Stock available")
    weather_index: float = Field(..., ge=0, le=10, description="Weather conditions (0-10)")
    customer_sentiment: float = Field(..., ge=0, le=10, description="Customer satisfaction (0-10)")
    return_rate: float = Field(..., ge=0, le=1, description="Return rate (0-1)")

class PricingOptimizationRequest(BaseModel):
    min_price: float = Field(..., gt=0, description="Minimum price to test")
    max_price: float = Field(..., gt=0, description="Maximum price to test")
    discount: float = Field(0.1, ge=0, le=1)
    promotion_intensity: float = Field(5.0, ge=0, le=10)
    ad_spend: float = Field(800.0, ge=0)
    competitor_price: float = Field(100.0, gt=0)
    stock_level: int = Field(150, gt=0)
    weather_index: float = Field(7.0, ge=0, le=10)
    customer_sentiment: float = Field(7.0, ge=0, le=10)
    return_rate: float = Field(0.05, ge=0, le=1)

# Startup
@app.on_event("startup")
async def startup_event():
    """Load or train model on startup"""
    if not predictor.load():
        print("⚠️  No trained model found. Training new model...")
        try:
            predictor.train()
        except FileNotFoundError:
            print("❌ retail_sales.csv not found!")
            print("   Download from: https://www.kaggle.com/datasets/mabubakrsiddiq/retail-store-product-sales-simulation-dataset")
        except Exception as e:
            print(f"❌ Training failed: {e}")

@app.get("/")
async def root():
    return {
        "message": "Retail Store Analytics API",
        "description": "Predict customer footfall and optimize pricing strategies",
        "endpoints": {
            "docs": "/docs",
            "predict_footfall": "/api/predict",
            "optimize_pricing": "/api/optimize",
            "statistics": "/api/stats",
            "model_info": "/api/model/info"
        }
    }

@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "model_target": predictor.target
    }

# OUTPUT 1: Footfall Prediction
@app.post("/api/predict")
async def predict_footfall(request: FootfallPredictionRequest):
    """
    🎯 OUTPUT 1: Predict Store Footfall (Customer Visits)
    
    Uses Random Forest ML model to predict how many customers will visit
    based on pricing, promotions, weather, and other factors.
    
    Returns:
    - predicted_footfall: Expected number of customer visits
    - confidence_interval: Lower and upper bounds
    - top_factors: Key features driving this prediction
    """
    try:
        features = request.dict()
        result = predictor.predict(features)
        importance = predictor.get_feature_importance()
        
        return {
            "prediction": result,
            "top_driving_factors": importance[:5],
            "business_insight": f"Expect approximately {result['predicted_footfall']} customer visits with current strategy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# OUTPUT 2: Pricing Optimization
@app.post("/api/optimize")
async def optimize_pricing(request: PricingOptimizationRequest):
    """
    💰 OUTPUT 2: Optimal Pricing for Maximum Footfall
    
    Analyzes different price points to find the optimal price that
    maximizes customer visits (footfall). Lower prices typically
    increase traffic, but there's an optimal balance.
    """
    try:
        # Test 50 price points
        prices = np.linspace(request.min_price, request.max_price, 50)
        results = []
        
        fixed_features = {
            'discount': request.discount,
            'promotion_intensity': request.promotion_intensity,
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
            
            results.append({
                'price': round(float(price), 2),
                'predicted_footfall': int(pred['predicted_footfall'])
            })
        
        # Find optimal (max footfall)
        optimal = max(results, key=lambda x: x['predicted_footfall'])
        
        return {
            "optimal_price": optimal['price'],
            "expected_footfall": optimal['predicted_footfall'],
            "analysis": f"Price of ${optimal['price']:.2f} maximizes customer visits at {optimal['predicted_footfall']} customers",
            "price_sensitivity_curve": results[::5],  # Every 5th point
            "insight": "Lower prices generally increase footfall, but consider profit margins"
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
            "footfall_statistics": {
                "mean": float(df['footfall'].mean()),
                "median": float(df['footfall'].median()),
                "min": int(df['footfall'].min()),
                "max": int(df['footfall'].max()),
                "std": float(df['footfall'].std())
            },
            "price_statistics": {
                "mean": float(df['price'].mean()),
                "median": float(df['price'].median()),
                "min": float(df['price'].min()),
                "max": float(df['price'].max())
            },
            "key_correlations": {
                "price_vs_footfall": float(df['price'].corr(df['footfall'])),
                "discount_vs_footfall": float(df['discount'].corr(df['footfall'])),
                "promotion_vs_footfall": float(df['promotion_intensity'].corr(df['footfall'])),
                "weather_vs_footfall": float(df['weather_index'].corr(df['footfall']))
            }
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/info")
async def model_info():
    """Get ML model details"""
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "Random Forest Regressor",
        "target_variable": predictor.target,
        "n_features": len(predictor.feature_names),
        "features": predictor.feature_names,
        "feature_importance": predictor.get_feature_importance(),
        "business_application": "Predicts customer store visits based on pricing and marketing strategies"
    }
