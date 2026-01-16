"""
Stage 5: ML Model Predictor V2

Enhanced predictor with:
1. 29 features vs 7 original
2. Ensemble averaging across all fold models
3. Better feature engineering
4. Fee-aware predictions
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
import structlog

from src.stage3.models import Direction


logger = structlog.get_logger(__name__)


# Paths - V2 models
ML_MODELS_DIR = Path("models_v2")
FEATURE_COLUMNS_PATH = Path("feature_columns_v2.json")
PERCENTILE_BUFFER_PATH = Path("model_buffer_for_percentile.json")

# Buffer limits
MAX_BUFFER_SIZE = 1000

# Valid symbols for one-hot encoding
VALID_SYMBOLS = [
    "ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
    "ETHUSDT", "LTCUSDT", "SOLUSDT", "XRPUSDT"
]

# V2 Feature columns (29 numerical + 8 one-hot = 37 total)
NUMERICAL_FEATURES_V2 = [
    # Order flow (7)
    "MOI_250ms", "MOI_1s", "MOI_5s", "MOI_z",
    "delta_velocity", "delta_velocity_5s", "AggressionPersistence",
    
    # Order flow momentum (3)
    "MOI_roc_1s", "MOI_roc_5s", "MOI_acceleration",
    
    # Absorption (3)
    "absorption_z", "price_impact_z", "MOI_flip_rate",
    
    # Volatility (4)
    "vol_1m", "vol_5m", "vol_ratio", "vol_rank",
    
    # Structure (4)
    "dist_lvn", "dist_poc", "dist_lvn_atr", "dist_poc_atr",
    
    # Time (3)
    "hour_sin", "hour_cos", "is_weekend",
    
    # Trade intensity (2)
    "trade_intensity", "trade_intensity_z",
    
    # Cumulative (2)
    "cum_delta_1m", "cum_delta_5m",
]

# V1 Feature columns (for backward compatibility)
NUMERICAL_FEATURES_V1 = [
    "MOI_250ms", "MOI_1s", "delta_velocity", "AggressionPersistence",
    "absorption_z", "dist_lvn", "vol_5m",
]


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
    
    # Ensemble info
    ensemble_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pred_60": self.pred_60,
            "pred_300": self.pred_300,
            "percentile_60": self.percentile_60,
            "percentile_300": self.percentile_300,
            "model_60": self.model_60,
            "model_300": self.model_300,
            "ensemble_size": self.ensemble_size,
        }


@dataclass
class FeatureState:
    """
    Rolling state for computing features that require history.
    One instance per symbol.
    """
    # Rolling buffers (use deque for efficiency)
    moi_1s_buffer: deque = field(default_factory=lambda: deque(maxlen=100))
    moi_250ms_buffer: deque = field(default_factory=lambda: deque(maxlen=100))
    absorption_buffer: deque = field(default_factory=lambda: deque(maxlen=500))
    price_impact_buffer: deque = field(default_factory=lambda: deque(maxlen=500))
    trade_count_buffer: deque = field(default_factory=lambda: deque(maxlen=500))
    ret_buffer: deque = field(default_factory=lambda: deque(maxlen=3600))  # 15min
    signed_qty_buffer: deque = field(default_factory=lambda: deque(maxlen=1200))  # 5min
    
    # Last values for computing derivatives
    last_moi_1s: float = 0.0
    last_delta_velocity: float = 0.0
    last_moi_1s_4_ago: float = 0.0  # For 1s rate of change
    last_moi_1s_20_ago: float = 0.0  # For 5s rate of change
    
    # Sign flip tracking
    last_moi_sign: int = 0
    sign_flips: deque = field(default_factory=lambda: deque(maxlen=240))  # 60s
    
    def update(self, moi_250ms: float, moi_1s: float, absorption_raw: float, 
               price_impact: float, trade_count: float, ret: float, signed_qty: float):
        """Update rolling buffers with new values"""
        # Track sign flips
        current_sign = 1 if moi_1s > 0 else (-1 if moi_1s < 0 else 0)
        if self.last_moi_sign != 0 and current_sign != 0 and current_sign != self.last_moi_sign:
            self.sign_flips.append(1)
        else:
            self.sign_flips.append(0)
        self.last_moi_sign = current_sign
        
        # Store history for rate of change
        if len(self.moi_1s_buffer) >= 4:
            self.last_moi_1s_4_ago = self.moi_1s_buffer[-4]
        if len(self.moi_1s_buffer) >= 20:
            self.last_moi_1s_20_ago = self.moi_1s_buffer[-20]
        
        # Update derivatives before adding new value
        self.last_delta_velocity = moi_1s - self.last_moi_1s
        self.last_moi_1s = moi_1s
        
        # Update buffers
        self.moi_250ms_buffer.append(moi_250ms)
        self.moi_1s_buffer.append(moi_1s)
        self.absorption_buffer.append(absorption_raw)
        self.price_impact_buffer.append(price_impact)
        self.trade_count_buffer.append(trade_count)
        self.ret_buffer.append(ret)
        self.signed_qty_buffer.append(signed_qty)
    
    def get_moi_std(self) -> float:
        """Rolling std of MOI_1s"""
        if len(self.moi_1s_buffer) < 10:
            return 1.0
        return float(np.std(list(self.moi_1s_buffer))) + 1e-6
    
    def get_moi_flip_rate(self) -> float:
        """Sign flips per minute"""
        return sum(self.sign_flips)
    
    def get_absorption_z(self) -> float:
        """Z-scored absorption"""
        if len(self.absorption_buffer) < 50:
            return 0.0
        arr = np.array(list(self.absorption_buffer))
        mean = np.mean(arr)
        std = np.std(arr) + 1e-6
        return float((arr[-1] - mean) / std)
    
    def get_price_impact_z(self) -> float:
        """Z-scored price impact"""
        if len(self.price_impact_buffer) < 50:
            return 0.0
        arr = np.array(list(self.price_impact_buffer))
        mean = np.mean(arr)
        std = np.std(arr) + 1e-6
        return float((arr[-1] - mean) / std)
    
    def get_vol(self, window: int) -> float:
        """Rolling volatility"""
        if len(self.ret_buffer) < window:
            return 0.0001
        arr = list(self.ret_buffer)[-window:]
        return float(np.std(arr)) + 1e-8
    
    def get_trade_intensity(self) -> float:
        """Average trade count over 100 bars"""
        if len(self.trade_count_buffer) < 100:
            return 0.0
        return float(np.mean(list(self.trade_count_buffer)[-100:]))
    
    def get_trade_intensity_z(self) -> float:
        """Z-scored trade count"""
        if len(self.trade_count_buffer) < 50:
            return 0.0
        arr = np.array(list(self.trade_count_buffer))
        mean = np.mean(arr)
        std = np.std(arr) + 1e-6
        return float((arr[-1] - mean) / std)
    
    def get_cum_delta(self, window: int) -> float:
        """Cumulative signed qty over window"""
        if len(self.signed_qty_buffer) < window:
            return 0.0
        return float(sum(list(self.signed_qty_buffer)[-window:]))


class MLPredictorV2:
    """
    Stage 5 ML Model Predictor V2
    
    Enhanced with:
    - 29 features (vs 7 original)
    - Ensemble averaging across fold models
    - Rolling feature state per symbol
    """
    
    def __init__(self, models_dir: Optional[Path] = None, use_v2_features: bool = True):
        self.models_dir = models_dir or ML_MODELS_DIR
        self.use_v2_features = use_v2_features
        
        # Loaded models: {model_name: List[model]}
        self._models: Dict[str, List] = {}
        
        # Percentile buffers: {model_name: [predictions]}
        self._percentile_buffers: Dict[str, List[float]] = {}
        
        # Feature state per symbol
        self._feature_states: Dict[str, FeatureState] = {}
        
        # Load models and buffers
        self._load_models()
        self._load_percentile_buffers()
        
        logger.info(
            "ml_predictor_v2_initialized",
            models_loaded=len(self._models),
            buffers_loaded=len(self._percentile_buffers),
            use_v2_features=use_v2_features,
        )
    
    def _load_models(self) -> None:
        """Load all ML models from disk"""
        if not self.models_dir.exists():
            logger.warning("ml_models_dir_not_found", path=str(self.models_dir))
            return
        
        for pkl_file in self.models_dir.glob("*.pkl"):
            model_name = pkl_file.stem
            try:
                with open(pkl_file, "rb") as f:
                    loaded = pickle.load(f)
                    # Handle both list of models (ensemble) and single model
                    if isinstance(loaded, list):
                        self._models[model_name] = loaded
                    else:
                        self._models[model_name] = [loaded]
                logger.debug("model_loaded", name=model_name, ensemble_size=len(self._models[model_name]))
            except Exception as e:
                logger.error("model_load_failed", name=model_name, error=str(e))
    
    def _load_percentile_buffers(self) -> None:
        """Load percentile buffers from JSON"""
        if not PERCENTILE_BUFFER_PATH.exists():
            return
        
        try:
            with open(PERCENTILE_BUFFER_PATH, "r") as f:
                self._percentile_buffers = json.load(f)
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
        """Get model name based on direction, vol_regime, and time"""
        dir_str = "up" if direction == Direction.LONG else "down"
        regime_str = vol_regime.lower()
        return f"models_{dir_str}_{regime_str}_{time}"
    
    def _get_or_create_feature_state(self, symbol: str) -> FeatureState:
        """Get or create feature state for symbol"""
        if symbol not in self._feature_states:
            self._feature_states[symbol] = FeatureState()
        return self._feature_states[symbol]
    
    def _create_feature_vector_v1(
        self,
        symbol: str,
        moi_250ms: float,
        moi_1s: float,
        delta_velocity: float,
        aggression_persistence: float,
        absorption_z: float,
        dist_lvn: float,
        vol_5m: float,
    ) -> pd.DataFrame:
        """Create V1 feature vector (7 features + 8 one-hot)"""
        feature_dict = {
            "MOI_250ms": [moi_250ms],
            "MOI_1s": [moi_1s],
            "delta_velocity": [delta_velocity],
            "AggressionPersistence": [aggression_persistence],
            "absorption_z": [absorption_z],
            "dist_lvn": [dist_lvn],
            "vol_5m": [vol_5m],
        }
        
        for valid_symbol in VALID_SYMBOLS:
            col_name = f"pair_{valid_symbol}"
            feature_dict[col_name] = [1.0 if symbol == valid_symbol else 0.0]
        
        return pd.DataFrame(feature_dict)
    
    def _create_feature_vector_v2(
        self,
        symbol: str,
        state: FeatureState,
        # Raw inputs
        moi_250ms: float,
        moi_1s: float,
        moi_5s: float,
        delta_velocity: float,
        aggression_persistence: float,
        dist_lvn: float,
        dist_poc: float,
        atr_5m: float,
        hour: int,
        is_weekend: bool,
    ) -> pd.DataFrame:
        """
        Create V2 feature vector (29 features + 8 one-hot).
        Uses rolling state for derived features.
        """
        moi_std = state.get_moi_std()
        vol_1m = state.get_vol(240)
        vol_5m = state.get_vol(1200)
        vol_15m = state.get_vol(3600)
        
        feature_dict = {
            # Order flow (7)
            "MOI_250ms": [moi_250ms],
            "MOI_1s": [moi_1s],
            "MOI_5s": [moi_5s],
            "MOI_z": [abs(moi_1s) / moi_std],
            "delta_velocity": [delta_velocity],
            "delta_velocity_5s": [moi_1s - state.last_moi_1s_20_ago if state.last_moi_1s_20_ago else 0.0],
            "AggressionPersistence": [aggression_persistence],
            
            # Order flow momentum (3)
            "MOI_roc_1s": [np.clip((moi_1s - state.last_moi_1s_4_ago) / (abs(state.last_moi_1s_4_ago) + 1e-6), -10, 10) if state.last_moi_1s_4_ago else 0.0],
            "MOI_roc_5s": [np.clip((moi_1s - state.last_moi_1s_20_ago) / (abs(state.last_moi_1s_20_ago) + 1e-6), -10, 10) if state.last_moi_1s_20_ago else 0.0],
            "MOI_acceleration": [delta_velocity - state.last_delta_velocity],
            
            # Absorption (3)
            "absorption_z": [state.get_absorption_z()],
            "price_impact_z": [state.get_price_impact_z()],
            "MOI_flip_rate": [state.get_moi_flip_rate()],
            
            # Volatility (4)
            "vol_1m": [vol_1m],
            "vol_5m": [vol_5m],
            "vol_ratio": [vol_1m / vol_5m if vol_5m > 0 else 1.0],
            "vol_rank": [0.5],  # Would need longer history to compute properly
            
            # Structure (4)
            "dist_lvn": [dist_lvn],
            "dist_poc": [dist_poc],
            "dist_lvn_atr": [dist_lvn / atr_5m if atr_5m > 0 else 0.0],
            "dist_poc_atr": [dist_poc / atr_5m if atr_5m > 0 else 0.0],
            
            # Time (3)
            "hour_sin": [np.sin(2 * np.pi * hour / 24)],
            "hour_cos": [np.cos(2 * np.pi * hour / 24)],
            "is_weekend": [1 if is_weekend else 0],
            
            # Trade intensity (2)
            "trade_intensity": [state.get_trade_intensity()],
            "trade_intensity_z": [state.get_trade_intensity_z()],
            
            # Cumulative (2)
            "cum_delta_1m": [state.get_cum_delta(240)],
            "cum_delta_5m": [state.get_cum_delta(1200)],
        }
        
        # One-hot encode symbol
        for valid_symbol in VALID_SYMBOLS:
            col_name = f"pair_{valid_symbol}"
            feature_dict[col_name] = [1.0 if symbol == valid_symbol else 0.0]
        
        return pd.DataFrame(feature_dict)
    
    def _calculate_percentile(self, value: float, buffer: List[float]) -> float:
        """Calculate percentile of value within buffer"""
        if not buffer:
            return 50.0
        arr = np.array(buffer)
        percentile = (np.sum(arr < value) / len(arr)) * 100.0
        return float(percentile)
    
    def _update_buffer(self, model_name: str, prediction: float) -> None:
        """Add prediction to rolling buffer"""
        if model_name not in self._percentile_buffers:
            self._percentile_buffers[model_name] = []
        
        buffer = self._percentile_buffers[model_name]
        buffer.append(prediction)
        
        if len(buffer) > MAX_BUFFER_SIZE:
            self._percentile_buffers[model_name] = buffer[-MAX_BUFFER_SIZE:]
    
    def _ensemble_predict(self, models: List, features: pd.DataFrame) -> float:
        """Average predictions across all models in ensemble"""
        predictions = []
        for model in models:
            try:
                pred = model.predict(features)[0]
                predictions.append(pred)
            except Exception:
                continue
        
        if not predictions:
            return 0.0
        return float(np.mean(predictions))
    
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
        # V2 additional inputs (optional)
        moi_5s: float = 0.0,
        dist_poc: float = 0.0,
        atr_5m: float = 0.0,
        hour: int = 12,
        is_weekend: bool = False,
        # For state updates
        absorption_raw: float = 0.0,
        price_impact: float = 0.0,
        trade_count: float = 0.0,
        ret: float = 0.0,
        signed_qty: float = 0.0,
    ) -> PredictionResult:
        """
        Run ML predictions for both 60s and 300s models.
        
        Supports both V1 (7 features) and V2 (29 features) models.
        """
        result = PredictionResult()
        
        try:
            # Get/update feature state
            state = self._get_or_create_feature_state(symbol)
            state.update(moi_250ms, moi_1s, absorption_raw, price_impact, 
                        trade_count, ret, signed_qty)
            
            # Get model names
            model_60_name = self._get_model_name(direction, vol_regime, 60)
            model_300_name = self._get_model_name(direction, vol_regime, 300)
            result.model_60 = model_60_name
            result.model_300 = model_300_name
            
            # Check if models exist
            if model_60_name not in self._models or model_300_name not in self._models:
                logger.warning("model_not_found", model_60=model_60_name, model_300=model_300_name)
                return result
            
            # Create feature vector (V1 for backward compatibility)
            if self.use_v2_features and len(self._models.get(model_60_name, [])) > 0:
                # Check if model expects V2 features by checking feature count
                first_model = self._models[model_60_name][0]
                expected_features = getattr(first_model, 'n_features_in_', 15)
                
                if expected_features > 20:  # V2 model
                    feature_df = self._create_feature_vector_v2(
                        symbol, state, moi_250ms, moi_1s, moi_5s, delta_velocity,
                        aggression_persistence, dist_lvn, dist_poc, atr_5m, hour, is_weekend
                    )
                else:
                    feature_df = self._create_feature_vector_v1(
                        symbol, moi_250ms, moi_1s, delta_velocity,
                        aggression_persistence, absorption_z, dist_lvn, vol_5m
                    )
            else:
                feature_df = self._create_feature_vector_v1(
                    symbol, moi_250ms, moi_1s, delta_velocity,
                    aggression_persistence, absorption_z, dist_lvn, vol_5m
                )
            
            result.features = feature_df.iloc[0].tolist()
            
            # Ensemble predictions
            models_60 = self._models[model_60_name]
            models_300 = self._models[model_300_name]
            
            result.pred_60 = self._ensemble_predict(models_60, feature_df)
            result.pred_300 = self._ensemble_predict(models_300, feature_df)
            result.ensemble_size = len(models_60)
            
            # Calculate percentiles
            buffer_60 = self._percentile_buffers.get(model_60_name, [])
            buffer_300 = self._percentile_buffers.get(model_300_name, [])
            
            result.percentile_60 = self._calculate_percentile(result.pred_60, buffer_60)
            result.percentile_300 = self._calculate_percentile(result.pred_300, buffer_300)
            
            # Update buffers
            self._update_buffer(model_60_name, result.pred_60)
            self._update_buffer(model_300_name, result.pred_300)
            
            # Periodic save
            if sum(len(b) for b in self._percentile_buffers.values()) % 100 == 0:
                self._save_percentile_buffers()
            
            logger.debug(
                "ml_prediction_v2_complete",
                symbol=symbol,
                direction=direction.value,
                vol_regime=vol_regime,
                pred_60=result.pred_60,
                pred_300=result.pred_300,
                pct_60=result.percentile_60,
                pct_300=result.percentile_300,
                ensemble_size=result.ensemble_size,
            )
            
            return result
            
        except Exception as e:
            logger.error("predict_fatal_error", symbol=symbol, error=str(e))
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
