import os
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, ConfigDict, model_validator

from ml_model import predictor


app = FastAPI(
    title="Retail Store Analytics API",
    description="ML-powered footfall prediction and pricing optimization for retail",
    version="1.0.0",
)


# ----------------------------
# Helpers
# ----------------------------

def _first_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _dataset_path() -> Optional[str]:
    return _first_existing_path(
        [
            "retail_sales.csv",
            "Retail_Sales.csv",
            os.path.join("data", "retail_sales.csv"),
            os.path.join("data", "Retail_Sales.csv"),
        ]
    )


def _read_dataset_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    return df


def _ensure_model_loaded():
    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure dataset exists and restart so training can run, or provide footfall_model.joblib.",
        )


# ----------------------------
# Request/Response Models (Pydantic v2)
# ----------------------------

class FootfallPredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "price": 100.0,
                "discount": 0.15,
                "promotion_intensity": 7.0,
                "ad_spend": 1000.0,
                "competitor_price": 105.0,
                "stock_level": 200,
                "weather_index": 8.0,
                "customer_sentiment": 7.5,
                "return_rate": 0.05,
            }
        }
    )

    price: float = Field(..., gt=0, description="Product price")
    discount: float = Field(..., ge=0, le=1, description="Discount (0 to 1)")
    promotion_intensity: float = Field(..., ge=0, le=10, description="Promotion strength (0 to 10)")
    ad_spend: float = Field(..., ge=0, description="Advertising spend")
    competitor_price: float = Field(..., gt=0, description="Competitor price")
    stock_level: int = Field(..., ge=0, description="Stock available (0 allowed)")
    weather_index: float = Field(..., ge=0, le=10, description="Weather conditions (0 to 10)")
    customer_sentiment: float = Field(..., ge=0, le=10, description="Customer satisfaction (0 to 10)")
    return_rate: float = Field(..., ge=0, le=1, description="Return rate (0 to 1)")


class PricingOptimizationRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "min_price": 60.0,
                "max_price": 140.0,
                "discount": 0.10,
                "promotion_intensity": 5.0,
                "ad_spend": 800.0,
                "competitor_price": 100.0,
                "stock_level": 150,
                "weather_index": 7.0,
                "customer_sentiment": 7.0,
                "return_rate": 0.05,
            }
        }
    )

    min_price: float = Field(..., gt=0, description="Minimum price to test")
    max_price: float = Field(..., gt=0, description="Maximum price to test")

    discount: float = Field(0.10, ge=0, le=1, description="Fixed discount while sweeping price")
    promotion_intensity: float = Field(5.0, ge=0, le=10)
    ad_spend: float = Field(800.0, ge=0)
    competitor_price: float = Field(100.0, gt=0)
    stock_level: int = Field(150, ge=0)
    weather_index: float = Field(7.0, ge=0, le=10)
    customer_sentiment: float = Field(7.0, ge=0, le=10)
    return_rate: float = Field(0.05, ge=0, le=1)

    @model_validator(mode="after")
    def validate_range(self):
        if self.max_price <= self.min_price:
            raise ValueError("max_price must be greater than min_price")
        return self


class HealthResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_target": "footfall",
            }
        }
    )

    status: str
    model_loaded: bool
    model_target: str


class FootfallPredictionResult(BaseModel):
    predicted_footfall: int
    confidence_lower: int
    confidence_upper: int
    std_dev: float
    interpretation: str


class FeatureImportanceItem(BaseModel):
    feature: str
    importance_percent: float
    rank: int


class PredictResponse(BaseModel):
    prediction: FootfallPredictionResult
    top_driving_factors: List[FeatureImportanceItem]
    business_insight: str


class OptimizeCurvePoint(BaseModel):
    price: float
    predicted_footfall: int


class OptimizeResponse(BaseModel):
    optimal_price: float
    expected_footfall: int
    analysis: str
    price_sensitivity_curve: List[OptimizeCurvePoint]
    insight: str


# ----------------------------
# Startup
# ----------------------------

@app.on_event("startup")
async def startup_event():
    if predictor.load():
        return

    path = _dataset_path()
    if not path:
        print("Dataset not found. Expected retail_sales.csv (or Retail_Sales.csv) in root or /data")
        return

    print("No trained model found. Training new footfall model...")
    try:
        predictor.train(csv_path=path)
    except Exception as e:
        print(f"Training failed: {e}")


# ----------------------------
# Routes
# ----------------------------

@app.get("/")
async def root():
    return {
        "message": "Retail Store Analytics API",
        "description": "Predict customer footfall and optimize pricing strategies",
        "endpoints": {
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/api/health",
            "predict_footfall": "/api/predict",
            "optimize_pricing": "/api/optimize",
            "statistics": "/api/stats",
            "model_info": "/api/model/info",
        },
    }


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.model is not None,
        model_target=getattr(predictor, "target", "footfall"),
    )


@app.post(
    "/api/predict",
    response_model=PredictResponse,
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": {
                            "predicted_footfall": 512,
                            "confidence_lower": 460,
                            "confidence_upper": 565,
                            "std_dev": 27.3812,
                            "interpretation": "Expected about 512 customer visits",
                        },
                        "top_driving_factors": [
                            {"feature": "price", "importance_percent": 21.3, "rank": 1},
                            {"feature": "promotion_intensity", "importance_percent": 17.8, "rank": 2},
                            {"feature": "discount", "importance_percent": 12.4, "rank": 3},
                        ],
                        "business_insight": "Expect approximately 512 customer visits with current strategy",
                    }
                }
            },
        }
    },
)
async def predict_footfall(request: FootfallPredictionRequest = Body(...)):
    _ensure_model_loaded()

    try:
        features = request.model_dump()
        result = predictor.predict(features)
        importance = predictor.get_feature_importance()

        return PredictResponse(
            prediction=FootfallPredictionResult(**result),
            top_driving_factors=[FeatureImportanceItem(**x) for x in importance[:5]],
            business_insight=f"Expect approximately {result['predicted_footfall']} customer visits with current strategy",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/optimize",
    response_model=OptimizeResponse,
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "optimal_price": 78.0,
                        "expected_footfall": 620,
                        "analysis": "Price of $78.00 maximizes customer visits at 620 customers",
                        "price_sensitivity_curve": [
                            {"price": 60.0, "predicted_footfall": 600},
                            {"price": 70.0, "predicted_footfall": 615},
                            {"price": 80.0, "predicted_footfall": 620},
                            {"price": 90.0, "predicted_footfall": 610},
                        ],
                        "insight": "Lower prices often increase footfall, but consider profit margins",
                    }
                }
            },
        }
    },
)
async def optimize_pricing(request: PricingOptimizationRequest = Body(...)):
    _ensure_model_loaded()

    try:
        prices = np.linspace(request.min_price, request.max_price, 50)
        results = []

        fixed = {
            "discount": request.discount,
            "promotion_intensity": request.promotion_intensity,
            "ad_spend": request.ad_spend,
            "competitor_price": request.competitor_price,
            "stock_level": request.stock_level,
            "weather_index": request.weather_index,
            "customer_sentiment": request.customer_sentiment,
            "return_rate": request.return_rate,
        }

        for p in prices:
            features = {"price": float(p), **fixed}
            pred = predictor.predict(features)
            results.append(
                {
                    "price": round(float(p), 2),
                    "predicted_footfall": int(pred["predicted_footfall"]),
                }
            )

        optimal = max(results, key=lambda x: x["predicted_footfall"])

        return OptimizeResponse(
            optimal_price=float(optimal["price"]),
            expected_footfall=int(optimal["predicted_footfall"]),
            analysis=f"Price of ${optimal['price']:.2f} maximizes customer visits at {optimal['predicted_footfall']} customers",
            price_sensitivity_curve=[OptimizeCurvePoint(**x) for x in results[::5]],
            insight="Lower prices often increase footfall, but consider profit margins",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics():
    path = _dataset_path()
    if not path:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = _read_dataset_clean(path)

        if "footfall" not in df.columns:
            raise HTTPException(status_code=500, detail="Dataset missing footfall column")

        return {
            "dataset_path": path,
            "dataset_size": int(len(df)),
            "target_column": "footfall",
            "footfall_statistics": {
                "mean": float(df["footfall"].mean()),
                "median": float(df["footfall"].median()),
                "min": int(df["footfall"].min()),
                "max": int(df["footfall"].max()),
                "std": float(df["footfall"].std()),
            },
            "price_statistics": {
                "mean": float(df["price"].mean()),
                "median": float(df["price"].median()),
                "min": float(df["price"].min()),
                "max": float(df["price"].max()),
            },
            "key_correlations": {
                "price_vs_footfall": float(df["price"].corr(df["footfall"])),
                "discount_vs_footfall": float(df["discount"].corr(df["footfall"])),
                "promotion_vs_footfall": float(df["promotion_intensity"].corr(df["footfall"])),
                "weather_vs_footfall": float(df["weather_index"].corr(df["footfall"])),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/info")
async def model_info():
    _ensure_model_loaded()
    return {
        "model_type": "Random Forest Regressor",
        "target_variable": getattr(predictor, "target", "footfall"),
        "n_features": len(predictor.feature_names),
        "features": predictor.feature_names,
        "train_r2": predictor.train_score,
        "test_r2": predictor.test_score,
        "feature_importance": predictor.get_feature_importance(),
        "business_application": "Predicts customer store visits (footfall) based on pricing and marketing strategy inputs",
    }
