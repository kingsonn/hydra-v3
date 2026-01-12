"""
Stage 5: ML Model Predictor

Runs signals through ML models based on direction and vol_regime.
Calculates percentile scores using rolling prediction buffer.
"""
import json
import pickle
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import structlog

from src.stage3.models import Direction


logger = structlog.get_logger(__name__)


# Paths
ML_MODELS_DIR = Path("ml_models")
FEATURE_COLUMNS_PATH = Path("feature_columns.json")
PERCENTILE_BUFFER_PATH = Path("model_buffer_for_percentile.json")

# Buffer limits
MAX_BUFFER_SIZE = 1000

# Feature columns (7 numerical + 8 one-hot)
NUMERICAL_FEATURES = [
    "MOI_250ms",
    "MOI_1s", 
    "delta_velocity",
    "AggressionPersistence",
    "absorption_z",
    "dist_lvn",
    "vol_5m",
]

PAIR_COLUMNS = [
    "pair_ADAUSDT",
    "pair_BNBUSDT",
    "pair_BTCUSDT",
    "pair_DOGEUSDT",
    "pair_ETHUSDT",
    "pair_LTCUSDT",
    "pair_SOLUSDT",
    "pair_XRPUSDT",
]

# Valid symbols for one-hot encoding
VALID_SYMBOLS = [col.replace("pair_", "") for col in PAIR_COLUMNS]


@dataclass
class PredictionResult:
    """Result from ML model predictions"""
    # Raw predictions
    pred_60: float = 0.0
    pred_300: float = 0.0
    
    # Percentiles (0-100)
    percentile_60: float = 0.0
    percentile_300: float = 0.0
    
    # Model names used
    model_60: str = ""
    model_300: str = ""
    
    # Feature vector used (for debugging)
    features: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pred_60": self.pred_60,
            "pred_300": self.pred_300,
            "percentile_60": self.percentile_60,
            "percentile_300": self.percentile_300,
            "model_60": self.model_60,
            "model_300": self.model_300,
        }


class MLPredictor:
    """
    Stage 5 ML Model Predictor
    
    Matches signal direction + vol_regime to appropriate models,
    creates feature vectors, runs predictions, and calculates percentiles.
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or ML_MODELS_DIR
        
        # Loaded models: {model_name: model_ensemble}
        self._models: Dict[str, Any] = {}
        
        # Percentile buffers: {model_name: [predictions]}
        self._percentile_buffers: Dict[str, List[float]] = {}
        
        # Load models and buffers
        self._load_models()
        self._load_percentile_buffers()
        
        logger.info(
            "ml_predictor_initialized",
            models_loaded=len(self._models),
            buffers_loaded=len(self._percentile_buffers),
        )
    
    def _load_models(self) -> None:
        """Load all ML models from disk"""
        if not self.models_dir.exists():
            logger.warning("ml_models_dir_not_found", path=str(self.models_dir))
            return
        
        for pkl_file in self.models_dir.glob("*.pkl"):
            model_name = pkl_file.stem  # e.g., "models_up_mid_60"
            try:
                with open(pkl_file, "rb") as f:
                    self._models[model_name] = pickle.load(f)
                logger.debug("model_loaded", name=model_name)
            except Exception as e:
                logger.error("model_load_failed", name=model_name, error=str(e))
    
    def _load_percentile_buffers(self) -> None:
        """Load percentile buffers from JSON"""
        if not PERCENTILE_BUFFER_PATH.exists():
            logger.warning("percentile_buffer_not_found", path=str(PERCENTILE_BUFFER_PATH))
            return
        
        try:
            with open(PERCENTILE_BUFFER_PATH, "r") as f:
                self._percentile_buffers = json.load(f)
            logger.info(
                "percentile_buffers_loaded",
                models=list(self._percentile_buffers.keys()),
            )
        except Exception as e:
            logger.error("percentile_buffer_load_failed", error=str(e))
    
    def _save_percentile_buffers(self) -> None:
        """Save percentile buffers to JSON"""
        try:
            with open(PERCENTILE_BUFFER_PATH, "w") as f:
                json.dump(self._percentile_buffers, f, indent=2)
        except Exception as e:
            logger.error("percentile_buffer_save_failed", error=str(e))
    
    def _get_model_name(self, direction: Direction, vol_regime: str, time: int) -> str:
        """
        Get model name based on direction, vol_regime, and time.
        
        Args:
            direction: LONG or SHORT
            vol_regime: HIGH, MID, or LOW
            time: 60 or 300
        
        Returns:
            Model name like "models_up_mid_60"
        """
        dir_str = "up" if direction == Direction.LONG else "down"
        regime_str = vol_regime.lower()  # HIGH -> high
        return f"models_{dir_str}_{regime_str}_{time}"
    
    def _create_feature_vector(
        self,
        symbol: str,
        moi_250ms: float,
        moi_1s: float,
        delta_velocity: float,
        aggression_persistence: float,
        absorption_z: float,
        dist_lvn: float,
        vol_5m: float,
    ) -> np.ndarray:
        """
        Create feature vector with 7 numerical features + 8 one-hot encoded columns.
        
        Returns:
            numpy array of shape (1, 15)
        """
        # 7 numerical features
        features = [
            moi_250ms,
            moi_1s,
            delta_velocity,
            aggression_persistence,
            absorption_z,
            dist_lvn,
            vol_5m,
        ]
        
        # 8 one-hot encoded columns for symbol
        for valid_symbol in VALID_SYMBOLS:
            if symbol == valid_symbol:
                features.append(1.0)
            else:
                features.append(0.0)
        
        return np.array([features])
    
    def _calculate_percentile(self, value: float, buffer: List[float]) -> float:
        """
        Calculate percentile of value within buffer.
        
        Returns:
            Percentile (0-100)
        """
        if not buffer:
            return 50.0  # Default to median if no history
        
        arr = np.array(buffer)
        # Count how many values are less than the given value
        percentile = (np.sum(arr < value) / len(arr)) * 100.0
        return float(percentile)
    
    def _update_buffer(self, model_name: str, prediction: float) -> None:
        """
        Add prediction to rolling buffer, maintaining max size of 1000.
        """
        if model_name not in self._percentile_buffers:
            self._percentile_buffers[model_name] = []
        
        buffer = self._percentile_buffers[model_name]
        buffer.append(prediction)
        
        # Keep only last MAX_BUFFER_SIZE entries
        if len(buffer) > MAX_BUFFER_SIZE:
            self._percentile_buffers[model_name] = buffer[-MAX_BUFFER_SIZE:]
    
    def predict(
        self,
        direction: Direction,
        vol_regime: str,
        symbol: str,
        moi_250ms: float,
        moi_1s: float,
        delta_velocity: float,
        aggression_persistence: float,
        absorption_z: float,
        dist_lvn: float,
        vol_5m: float,
    ) -> PredictionResult:
        """
        Run ML predictions for both 60s and 300s models.
        
        Args:
            direction: Signal direction (LONG or SHORT)
            vol_regime: Volatility regime (HIGH, MID, LOW)
            symbol: Trading pair (e.g., "BTCUSDT")
            ... feature values ...
        
        Returns:
            PredictionResult with predictions and percentiles
        """
        result = PredictionResult()
        
        # Get model names
        model_60_name = self._get_model_name(direction, vol_regime, 60)
        model_300_name = self._get_model_name(direction, vol_regime, 300)
        result.model_60 = model_60_name
        result.model_300 = model_300_name
        
        # Check if models exist
        if model_60_name not in self._models:
            logger.warning("model_not_found", name=model_60_name)
            return result
        if model_300_name not in self._models:
            logger.warning("model_not_found", name=model_300_name)
            return result
        
        # Create feature vector
        feature_vector = self._create_feature_vector(
            symbol=symbol,
            moi_250ms=moi_250ms,
            moi_1s=moi_1s,
            delta_velocity=delta_velocity,
            aggression_persistence=aggression_persistence,
            absorption_z=absorption_z,
            dist_lvn=dist_lvn,
            vol_5m=vol_5m,
        )
        result.features = feature_vector[0].tolist()
        
        # Run predictions (using last model in ensemble)
        try:
            model_60 = self._models[model_60_name]
            if isinstance(model_60, list):
                result.pred_60 = float(model_60[-1].predict(feature_vector)[0])
            else:
                result.pred_60 = float(model_60.predict(feature_vector)[0])
        except Exception as e:
            logger.error("prediction_failed", model=model_60_name, error=str(e))
        
        try:
            model_300 = self._models[model_300_name]
            if isinstance(model_300, list):
                result.pred_300 = float(model_300[-1].predict(feature_vector)[0])
            else:
                result.pred_300 = float(model_300.predict(feature_vector)[0])
        except Exception as e:
            logger.error("prediction_failed", model=model_300_name, error=str(e))
        
        # Calculate percentiles from buffers
        buffer_60 = self._percentile_buffers.get(model_60_name, [])
        buffer_300 = self._percentile_buffers.get(model_300_name, [])
        
        result.percentile_60 = self._calculate_percentile(result.pred_60, buffer_60)
        result.percentile_300 = self._calculate_percentile(result.pred_300, buffer_300)
        
        # Update buffers with new predictions
        self._update_buffer(model_60_name, result.pred_60)
        self._update_buffer(model_300_name, result.pred_300)
        
        # Save buffers to disk (could be optimized to batch saves)
        self._save_percentile_buffers()
        
        logger.debug(
            "ml_prediction_complete",
            symbol=symbol,
            direction=direction.value,
            vol_regime=vol_regime,
            pred_60=result.pred_60,
            pred_300=result.pred_300,
            pct_60=result.percentile_60,
            pct_300=result.percentile_300,
        )
        
        return result
    
    def get_buffer_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all percentile buffers"""
        stats = {}
        for model_name, buffer in self._percentile_buffers.items():
            if buffer:
                arr = np.array(buffer)
                stats[model_name] = {
                    "count": len(buffer),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }
            else:
                stats[model_name] = {"count": 0}
        return stats
