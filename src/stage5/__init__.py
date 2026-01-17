"""
Stage 5: ML Model Predictions

Runs signals through ML models based on direction and vol_regime.
V3 uses probability-based classification models.

Models: models_{direction}_{regime}_{time}.pkl
- direction: up (LONG) or down (SHORT)
- regime: high, mid, low (vol_regime)
- time: 60 or 300 seconds
"""

from src.stage5.predictor import (
    MLPredictor,
    PredictionResult,
)

from src.stage5.predictor_v3 import (
    MLPredictorV3,
    PredictionResultV3,
)

__all__ = [
    "MLPredictor",
    "PredictionResult",
    "MLPredictorV3",
    "PredictionResultV3",
]
