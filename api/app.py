from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import Predictor
from train_model import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Exnet.ai Trading Signal API",
    description="ML-powered stock prediction API with SHAP explanations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = Predictor()

class StockFeatures(BaseModel):
    VIX: Optional[float] = Field(default=20.0, description="VIX volatility index")
    FNG: Optional[float] = Field(default=50.0, description="Fear & Greed index")
    RSI: Optional[float] = Field(default=50.0, description="RSI indicator")
    AnnVolatility: Optional[float] = Field(default=20.0, description="Annual volatility")
    Momentum125: Optional[float] = Field(default=0.0, description="125-day momentum")
    PriceStrength: Optional[float] = Field(default=0.0, description="Price strength indicator")
    VolumeBreadth: Optional[float] = Field(default=1.0, description="Volume breadth")
    CallPut: Optional[float] = Field(default=50.0, description="Call/Put ratio")
    NewsScore: Optional[float] = Field(default=5.0, description="News sentiment score")
    MACD: Optional[float] = Field(default=0.0, description="MACD indicator")
    BollingerBandWidth: Optional[float] = Field(default=0.1, description="Bollinger band width")
    GDP_lag: Optional[float] = Field(default=2.0, description="GDP with lag")
    UNRATE: Optional[float] = Field(default=4.0, description="Unemployment rate")
    CPIAUCSL: Optional[float] = Field(default=2.0, description="CPI inflation")
    PAYEMS: Optional[float] = Field(default=150000, description="Non-farm payrolls")
    FEDFUNDS: Optional[float] = Field(default=2.0, description="Federal funds rate")

class PredictionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    features: StockFeatures
    prediction_days: int = Field(default=5, description="Prediction horizon in days")
    include_shap: bool = Field(default=True, description="Include SHAP explanation")

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]

class PredictionResponse(BaseModel):
    ticker: str
    prediction_days: int
    prediction_probability: List[float]
    predicted_class: int
    confidence: float
    shap_explanation: Optional[Dict[str, Any]] = None
    status: str = "success"
    error_message: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    summary: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    version: str = "1.0.0"

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = {}
    for days in [5, 20, 60]:
        models_loaded[f"{days}d"] = days in predictor.models
    
    return HealthResponse(
        status="healthy" if any(models_loaded.values()) else "no_models_loaded",
        models_loaded=models_loaded
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_signal(request: PredictionRequest):
    """Predict trading signal for a single stock"""
    try:
        features_dict = request.features.dict()
        features_df = pd.DataFrame([features_dict])
        
        if request.prediction_days not in predictor.models:
            raise HTTPException(
                status_code=400, 
                detail=f"Model for {request.prediction_days}-day prediction not loaded"
            )
        
        proba = predictor.predict_proba(features_df, request.prediction_days)
        
        if proba is None:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        predicted_class = int(np.argmax(proba))
        confidence = float(np.max(proba))
        
        shap_explanation = None
        if request.include_shap:
            shap_result = predictor.get_signal_shap_explanation(features_df, request.prediction_days)
            if shap_result:
                shap_explanation = {
                    "top_features": shap_result["top_features"],
                    "feature_values": shap_result["feature_values"]
                }
        
        return PredictionResponse(
            ticker=request.ticker,
            prediction_days=request.prediction_days,
            prediction_probability=proba.tolist(),
            predicted_class=predicted_class,
            confidence=confidence,
            shap_explanation=shap_explanation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return PredictionResponse(
            ticker=request.ticker,
            prediction_days=request.prediction_days,
            prediction_probability=[0.5, 0.5],
            predicted_class=0,
            confidence=0.5,
            status="error",
            error_message=str(e)
        )

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict_signals(request: BatchPredictionRequest):
    """Predict trading signals for multiple stocks"""
    results = []
    successful_predictions = 0
    failed_predictions = 0
    
    for pred_request in request.predictions:
        try:
            result = await predict_signal(pred_request)
            results.append(result)
            
            if result.status == "success":
                successful_predictions += 1
            else:
                failed_predictions += 1
                
        except Exception as e:
            logger.error(f"Batch prediction error for {pred_request.ticker}: {str(e)}")
            results.append(PredictionResponse(
                ticker=pred_request.ticker,
                prediction_days=pred_request.prediction_days,
                prediction_probability=[0.5, 0.5],
                predicted_class=0,
                confidence=0.5,
                status="error",
                error_message=str(e)
            ))
            failed_predictions += 1
    
    summary = {
        "total_requests": len(request.predictions),
        "successful_predictions": successful_predictions,
        "failed_predictions": failed_predictions,
        "success_rate": successful_predictions / len(request.predictions) if request.predictions else 0
    }
    
    return BatchPredictionResponse(results=results, summary=summary)

@app.get("/models")
async def list_available_models():
    """List available prediction models"""
    available_models = {}
    
    for days in predictor.models.keys():
        available_models[f"{days}d"] = {
            "prediction_days": days,
            "features_count": len(predictor.feature_columns.get(days, [])),
            "has_shap_explainer": days in predictor.shap_explainers
        }
    
    return {
        "available_models": available_models,
        "total_models": len(available_models)
    }

@app.post("/load_model")
async def load_model(
    prediction_days: int,
    model_path: str,
    scaler_path: str,
    features_path: str,
    shap_path: str,
    background_tasks: BackgroundTasks
):
    """Load a trained model (admin endpoint)"""
    try:
        background_tasks.add_task(
            predictor.load_model,
            prediction_days,
            model_path,
            scaler_path,
            features_path,
            shap_path
        )
        
        return {
            "message": f"Loading model for {prediction_days}-day prediction",
            "status": "loading"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/feature_categories")
async def get_feature_categories():
    """Get information about feature categories"""
    sample_features = [
        'VIX', 'RSI', 'MACD', 'GDP_lag', 'UNRATE', 
        'interaction_VIX_RSI', 'regime_high_vol', 'RSI_sq_transformed'
    ]
    
    categories = predictor._categorize_features(sample_features)
    
    category_info = {}
    for category, features in categories.items():
        category_info[category] = {
            "description": get_category_description(category),
            "example_features": features[:5],
            "total_features": len(features)
        }
    
    return {"feature_categories": category_info}

def get_category_description(category: str) -> str:
    """Get description for feature category"""
    descriptions = {
        "proprietary": "Custom indicators like VIX, Fear & Greed, RSI, momentum measures",
        "macro": "Macroeconomic indicators from FRED (GDP, unemployment, inflation, etc.)",
        "technical": "Traditional technical analysis indicators (moving averages, oscillators)",
        "interaction": "Cross-feature interactions capturing complex relationships",
        "regime": "Market regime indicators (high/low volatility, bull/bear markets)",
        "transformed": "Non-linear transformations of base features (squared, log, etc.)"
    }
    return descriptions.get(category, "Unknown category")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
