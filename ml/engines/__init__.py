#!/usr/bin/env python3
"""
ML Engines Module
Enterprise machine learning engines for DevSkyy platform

Available Engines:
- TextGenerationEngine: Multi-model text generation with brand voice consistency
- FashionTrendPredictor: Time series forecasting and trend prediction
"""

from .text_generation import (
    TextGenerationEngine,
    TextGenerationRequest,
    TextGenerationResult,
    AIModel,
    OutputFormat,
    FinishReason,
    ModelSelector
)

from .fashion_trend_predictor import (
    FashionTrendPredictor,
    FashionTrendData,
    TrendPrediction,
    TrendCategory,
    GrowthTrajectory,
    Seasonality,
    TrendRecommendation
)

__all__ = [
    # Text Generation
    'TextGenerationEngine',
    'TextGenerationRequest',
    'TextGenerationResult',
    'AIModel',
    'OutputFormat',
    'FinishReason',
    'ModelSelector',
    # Fashion Trend Prediction
    'FashionTrendPredictor',
    'FashionTrendData',
    'TrendPrediction',
    'TrendCategory',
    'GrowthTrajectory',
    'Seasonality',
    'TrendRecommendation'
]

__version__ = '2.0.0'
