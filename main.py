from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List, Optional
import pandas as pd
import numpy as np
import os

from ml_model import predictor

app = FastAPI(
    title="Retail Sales Analytics API",
    description="ML-powered sales prediction and pricing optimization",
    version="1.0.0",
)

# ----------------------------
# Pydantic request + response models
# ----------------------------

class SalesPredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "price": 100.0,
                    "discount": 0.15,
                    "promotion_intensity": 7.0,
                    "footfall": 500,
                    "ad_spend": 1000.0,
                    "competitor_price": 105.0,
                    "stock_level": 200,
                    "weather_index": 8.0,
                    "customer_sentiment": 7.5,
                    "return_rate": 0.05,
                }
            ]
        }
    )

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
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "min_price": 80.0,
                    "max_price": 120.0,
                    "discount": 0.10,
                    "promotion_intensity": 5.0,
                    "footfall": 400,
                    "ad_spend": 800.0,
                    "competitor_price": 100.0,
                    "stock_level": 150,
                    "weather_index": 7.0,
                    "customer_sentiment": 7.0,
                    "return_rate": 0.05,
                }
            ]
        }
    )

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

    @model_validator(mode="after")
    def validate_price_range(self):
        if self.max_price <= self.min_price:
            raise ValueError("max_price must be greater than min_price")
        return self


class PredictionResult(BaseModel):
    predicted_sales: float
    confidence_lower: float
    confidence_upper: float
    std_dev: float


class FeatureFactor(BaseModel):
    feature: str
    importance: float
    rank: int


class SalesPredictionResponse(BaseModel):
    prediction: PredictionResult
    top_factors: List[FeatureFactor]
    interpretation: str


class PriceCurvePoint(BaseModel):
    price: float
    predicted_sales: float
    revenue: float


class PricingOptimizationResponse(BaseModel):
    optimal_price: float
    expected_sales: float
    expected_revenue: float
    analysis: str
    price_curve: List[PriceCurvePoint]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ----------------------------
# Helpers
# ----------------------------

def _first_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _ensure_model_loaded():
    if predictor.model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Provide retail_sales.csv and restart, "
                "or include a pre-trained sales_model.joblib."
            ),
        )


def _read_dataset_clean(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    # Drop accidental index columns
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    return df


# ----------------------------
# Startup: load or train model
# ----------------------------

@app.on_event("startup")
async def startup_event():
    if predictor.load():
        return

    print("Model not found. Training new model...")

    dataset_path = _first_existing_path(
        [
            "retail_sales.csv",
            os.path.join("data", "retail_sales.csv"),
        ]
    )

    if dataset_path is None:
        print("⚠️  Warning: dataset not found!")
        print("Please download dataset from Kaggle and place it in the project folder")
        print("Expected filename: retail_sales.csv")
        return

    try:
        predictor.train(csv_path=dataset_path)
    except TypeError:
        predictor.train()
    except Exception as e:
        print(f"⚠️  Warning: model training failed: {e}")


# ----------------------------
# Routes
# ----------------------------

@app.get("/")
async def root():
    return {
        "message": "Retail Sales Analytics API",
        "docs": "/docs",
        "endpoints": {
            "predict_sales": "/api/predict",
            "optimize_pricing": "/api/optimize",
            "statistics": "/api/stats",
        },
    }


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy", model_loaded=predictor.model is not None)


@app.post(
    "/api/predict",
    response_model=SalesPredictionResponse,
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "prediction": {
                            "predicted_sales": 320.5,
                            "confidence_lower": 280.1,
                            "confidence_upper": 360.9,
                            "std_dev": 20.2,
                        },
                        "top_factors": [
                            {"feature": "price", "importance": 0.31, "rank": 1},
                            {"feature": "footfall", "importance": 0.22, "rank": 2},
                            {"feature": "discount", "importance": 0.12, "rank": 3},
                        ],
                        "interpretation": "Expected to sell 320.5 units",
                    }
                }
            },
        }
    },
)
async def predict_sales(request: SalesPredictionRequest = Body(...)) -> SalesPredictionResponse:
    _ensure_model_loaded()

    try:
        features = request.model_dump()
        result = predictor.predict(features)
        importance = predictor.get_feature_importance()

        return SalesPredictionResponse(
            prediction=PredictionResult(**result),
            top_factors=[FeatureFactor(**x) for x in importance],
            interpretation=f"Expected to sell {result['predicted_sales']} units",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/optimize",
    response_model=PricingOptimizationResponse,
    responses={
        200: {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "example": {
                        "optimal_price": 104.0,
                        "expected_sales": 310.0,
                        "expected_revenue": 32240.0,
                        "analysis": "Best price is $104.0 for $32240.0 revenue",
                        "price_curve": [
                            {"price": 80.0, "predicted_sales": 380.0, "revenue": 30400.0},
                            {"price": 90.0, "predicted_sales": 350.0, "revenue": 31500.0},
                            {"price": 100.0, "predicted_sales": 320.0, "revenue": 32000.0},
                            {"price": 110.0, "predicted_sales": 290.0, "revenue": 31900.0},
                            {"price": 120.0, "predicted_sales": 260.0, "revenue": 31200.0},
                        ],
                    }
                }
            },
        }
    },
)
async def optimize_pricing(request: PricingRequest = Body(...)) -> PricingOptimizationResponse:
    _ensure_model_loaded()

    try:
        prices = np.linspace(request.min_price, request.max_price, 50)
        results = []

        base_features = request.model_dump()
        base_features.pop("min_price", None)
        base_features.pop("max_price", None)

        for price in prices:
            test_features = base_features.copy()
            test_features["price"] = float(price)

            pred = predictor.predict(test_features)
            revenue = float(price) * float(pred["predicted_sales"])

            results.append(
                {
                    "price": round(float(price), 2),
                    "predicted_sales": float(pred["predicted_sales"]),
                    "revenue": round(float(revenue), 2),
                }
            )

        optimal = max(results, key=lambda x: x["revenue"])

        return PricingOptimizationResponse(
            optimal_price=float(optimal["price"]),
            expected_sales=float(optimal["predicted_sales"]),
            expected_revenue=float(optimal["revenue"]),
            analysis=f"Best price is ${optimal['price']} for ${optimal['revenue']} revenue",
            price_curve=[PriceCurvePoint(**x) for x in results[::5]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics():
    try:
        dataset_path = _first_existing_path(
            [
                "retail_sales.csv",
                os.path.join("data", "retail_sales.csv"),
            ]
        )
        if dataset_path is None:
            raise FileNotFoundError("Dataset not found")

        df = _read_dataset_clean(dataset_path)

        if predictor.target_column and predictor.target_column in df.columns:
            target_col = predictor.target_column
        else:
            target_col = predictor.detect_target_column(df)

        return {
            "dataset_path": dataset_path,
            "dataset_size": int(len(df)),
            "target_column": target_col,
            "target_statistics": {
                "mean": float(df[target_col].mean()),
                "median": float(df[target_col].median()),
                "min": float(df[target_col].min()),
                "max": float(df[target_col].max()),
                "std": float(df[target_col].std()),
            },
            "price_statistics": {
                "mean": float(df["price"].mean()),
                "median": float(df["price"].median()),
                "min": float(df["price"].min()),
                "max": float(df["price"].max()),
            },
            "correlations": {
                "price_vs_target": float(df["price"].corr(df[target_col])),
                "discount_vs_target": float(df["discount"].corr(df[target_col])),
                "footfall_vs_target": float(df["footfall"].corr(df[target_col])),
            },
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset not found")


@app.get("/api/model/info")
async def model_info():
    _ensure_model_loaded()
    return {
        "model_type": "Random Forest Regressor",
        "n_estimators": 200,
        "features": predictor.feature_names,
        "target_column": predictor.target_column,
        "feature_importance": predictor.get_feature_importance(),
    }
