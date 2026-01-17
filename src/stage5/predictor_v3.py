"""
Stage 5: ML Model Predictor V3

Real-time compatible predictor with:
1. Incremental bar builder using deques (O(1) updates)
2. Cross-sectional ranking across symbols
3. Compatible with V3 models (binary classification for TP hit probability)
4. Ensemble averaging across fold models
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

ML_MODELS_DIR = Path("models_v3")
FEATURE_COLUMNS_PATH = Path("feature_columns_v3.json")
PERCENTILE_BUFFER_PATH = Path("model_buffer_v3.json")
MAX_BUFFER_SIZE = 1000

VALID_SYMBOLS = ["ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT", "XRPUSDT"]
VOL_REGIMES = ["low", "mid", "high"]
HORIZONS = [60, 300]


@dataclass
class PredictionResultV3:
    """Result from V3 ML predictions"""
    prob_60: float = 0.0  # Probability of TP hit in 60s
    prob_300: float = 0.0  # Probability of TP hit in 300s
    percentile_60: float = 0.0
    percentile_300: float = 0.0
    model_60: str = ""
    model_300: str = ""
    ensemble_size: int = 0
    cross_sectional_rank: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prob_60": self.prob_60,
            "prob_300": self.prob_300,
            "percentile_60": self.percentile_60,
            "percentile_300": self.percentile_300,
            "model_60": self.model_60,
            "model_300": self.model_300,
            "ensemble_size": self.ensemble_size,
            "cross_sectional_rank": self.cross_sectional_rank,
        }


@dataclass
class IncrementalBarState:
    """Rolling state for O(1) feature computation per symbol"""
    prices: deque = field(default_factory=lambda: deque(maxlen=14400))
    signed_qty: deque = field(default_factory=lambda: deque(maxlen=4800))
    qty: deque = field(default_factory=lambda: deque(maxlen=4800))
    trade_count: deque = field(default_factory=lambda: deque(maxlen=4800))
    highs: deque = field(default_factory=lambda: deque(maxlen=4800))
    lows: deque = field(default_factory=lambda: deque(maxlen=4800))
    moi_signs: deque = field(default_factory=lambda: deque(maxlen=960))
    volume_profile: Dict[float, float] = field(default_factory=dict)
    vp_qty: deque = field(default_factory=lambda: deque(maxlen=4800))
    vp_bins: deque = field(default_factory=lambda: deque(maxlen=4800))
    
    # Cache computed values
    last_moi_1s: float = 0.0
    last_delta_vel: float = 0.0


class MLPredictorV3:
    """
    Real-time ML Predictor V3
    
    Features:
    - Incremental bar builder with O(1) updates
    - Cross-sectional ranking across all symbols
    - Binary classification (probability of TP hit)
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or ML_MODELS_DIR
        self._models: Dict[str, List] = {}
        self._calibrators: Dict[str, Any] = {}  # Probability calibrators
        self._percentile_buffers: Dict[str, List[float]] = {}
        self._bar_states: Dict[str, IncrementalBarState] = {}
        self._feature_columns: List[str] = []
        self._cross_sectional_cache: Dict[str, Dict[str, float]] = {}  # {feature: {symbol: value}}
        
        self._load_models()
        self._load_feature_columns()
        self._load_percentile_buffers()
        
        logger.info("predictor_v3_initialized", models=len(self._models), features=len(self._feature_columns))
    
    def _load_models(self):
        """Load only the 12 correct models (direction_regime_horizon format)"""
        if not self.models_dir.exists():
            logger.warning("models_dir_not_found", path=str(self.models_dir))
            return
        
        # Only load models with correct naming: models_{up/down}_{low/mid/high}_{60/300}.pkl
        for direction in ["up", "down"]:
            for regime in VOL_REGIMES:
                for horizon in HORIZONS:
                    model_name = f"models_{direction}_{regime}_{horizon}"
                    pkl_path = self.models_dir / f"{model_name}.pkl"
                    
                    if not pkl_path.exists():
                        logger.warning("model_not_found", name=model_name)
                        continue
                    
                    try:
                        with open(pkl_path, "rb") as f:
                            loaded = pickle.load(f)
                            # Handle new format with calibrator
                            if isinstance(loaded, dict) and "models" in loaded:
                                self._models[model_name] = loaded["models"]
                                if loaded.get("calibrator"):
                                    self._calibrators[model_name] = loaded["calibrator"]
                            else:
                                self._models[model_name] = loaded if isinstance(loaded, list) else [loaded]
                        logger.debug("model_loaded", name=model_name, ensemble_size=len(self._models[model_name]))
                    except Exception as e:
                        logger.error("model_load_failed", name=model_name, error=str(e))
        
        logger.info("models_loaded", count=len(self._models), calibrators=len(self._calibrators))
    
    def _load_feature_columns(self):
        if FEATURE_COLUMNS_PATH.exists():
            with open(FEATURE_COLUMNS_PATH) as f:
                self._feature_columns = json.load(f)
    
    def _load_percentile_buffers(self):
        if PERCENTILE_BUFFER_PATH.exists():
            try:
                with open(PERCENTILE_BUFFER_PATH) as f:
                    self._percentile_buffers = json.load(f)
            except:
                pass
    
    def _save_percentile_buffers(self):
        try:
            with open(PERCENTILE_BUFFER_PATH, "w") as f:
                json.dump(self._percentile_buffers, f)
        except:
            pass
    
    def _get_state(self, symbol: str) -> IncrementalBarState:
        if symbol not in self._bar_states:
            self._bar_states[symbol] = IncrementalBarState()
        return self._bar_states[symbol]
    
    def update_bar(
        self,
        symbol: str,
        price: float,
        qty: float,
        signed_qty: float,
        trade_count: int,
        high: float,
        low: float,
        bin_size: float = 10.0,
    ) -> Dict[str, float]:
        """
        Update bar state and compute features incrementally.
        Call this every 250ms bar.
        """
        state = self._get_state(symbol)
        
        # Update deques
        state.prices.append(price)
        state.signed_qty.append(signed_qty)
        state.qty.append(qty)
        state.trade_count.append(trade_count)
        state.highs.append(high)
        state.lows.append(low)
        
        # Volume profile update
        price_bin = round(price / bin_size) * bin_size
        state.volume_profile[price_bin] = state.volume_profile.get(price_bin, 0) + qty
        state.vp_qty.append(qty)
        state.vp_bins.append(price_bin)
        if len(state.vp_qty) == state.vp_qty.maxlen:
            old_bin = state.vp_bins[0]
            state.volume_profile[old_bin] = max(0, state.volume_profile.get(old_bin, 0) - state.vp_qty[0])
        
        return self._compute_features(state, symbol, price, signed_qty)
    
    def _compute_features(self, state: IncrementalBarState, symbol: str, price: float, signed_qty: float) -> Dict[str, float]:
        """Compute all features from rolling state"""
        f = {}
        sq = list(state.signed_qty)
        n = len(sq)
        
        # Order flow
        f["MOI_250ms"] = signed_qty
        f["MOI_1s"] = sum(sq[-4:]) if n >= 4 else sum(sq)
        f["MOI_5s"] = sum(sq[-20:]) if n >= 20 else sum(sq)
        f["MOI_20s"] = sum(sq[-80:]) if n >= 80 else sum(sq)
        
        if n >= 100:
            moi_arr = np.array(sq[-100:])
            moi_std = np.std(moi_arr) + 1e-6
            f["MOI_std"] = moi_std
            f["MOI_z"] = abs(f["MOI_1s"]) / moi_std
            f["AggressionPersistence"] = np.mean(np.abs(moi_arr)) / moi_std
        else:
            f["MOI_std"], f["MOI_z"], f["AggressionPersistence"] = 1.0, 0.0, 0.0
        
        # Delta velocity
        prev_moi = sum(sq[-8:-4]) if n >= 8 else 0
        f["delta_velocity"] = f["MOI_1s"] - prev_moi if n >= 5 else 0
        f["delta_velocity_5s"] = f["MOI_1s"] - (sum(sq[-24:-20]) if n >= 24 else 0)
        
        # MOI momentum
        f["MOI_roc_1s"] = np.clip((f["MOI_1s"] - prev_moi) / (abs(prev_moi) + 1e-6), -10, 10) if n >= 8 else 0
        prev_5s = sum(sq[-24:-20]) if n >= 24 else 0
        f["MOI_roc_5s"] = np.clip((f["MOI_1s"] - prev_5s) / (abs(prev_5s) + 1e-6), -10, 10) if n >= 24 else 0
        f["MOI_acceleration"] = f["delta_velocity"] - state.last_delta_vel
        state.last_delta_vel = f["delta_velocity"]
        
        # Flip rate
        sign = 1 if f["MOI_1s"] > 0 else (-1 if f["MOI_1s"] < 0 else 0)
        state.moi_signs.append(sign)
        signs = list(state.moi_signs)
        f["MOI_flip_rate"] = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1] and signs[i] and signs[i-1])
        
        # Volatility
        prices = list(state.prices)
        if len(prices) >= 2:
            rets = np.diff(prices) / np.array(prices[:-1])
            f["vol_1m"] = np.std(rets[-240:]) if len(rets) >= 240 else np.std(rets) + 1e-8
            f["vol_5m"] = np.std(rets[-1200:]) if len(rets) >= 1200 else np.std(rets) + 1e-8
            f["vol_ratio"] = f["vol_1m"] / (f["vol_5m"] + 1e-8)
            f["vol_rank"] = 0.5  # Would need history
        else:
            f["vol_1m"], f["vol_5m"], f["vol_ratio"], f["vol_rank"] = 0.0001, 0.0001, 1.0, 0.5
        
        # ATR
        if len(state.closes if hasattr(state, 'closes') else prices) >= 2:
            highs, lows = np.array(list(state.highs)), np.array(list(state.lows))
            closes = np.array(prices)
            tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
            f["ATR_5m"] = np.mean(tr[-1200:]) if len(tr) >= 1200 else np.mean(tr) + 1e-8
        else:
            f["ATR_5m"] = price * 0.001
        f["ATR_5m_pct"] = f["ATR_5m"] / price
        
        # Absorption
        qty_list = list(state.qty)
        if len(prices) >= 2 and len(qty_list) >= 2:
            pc = np.abs(np.diff(prices[-500:])) + 1e-8
            qs = np.array(qty_list[-len(pc):])
            if len(pc) == len(qs) and len(pc) >= 50:
                absorption = qs / pc
                f["absorption_z"] = (absorption[-1] - np.mean(absorption)) / (np.std(absorption) + 1e-6)
                pi = pc / (qs + 1e-6)
                f["price_impact_z"] = (pi[-1] - np.mean(pi)) / (np.std(pi) + 1e-6)
            else:
                f["absorption_z"], f["price_impact_z"] = 0.0, 0.0
        else:
            f["absorption_z"], f["price_impact_z"] = 0.0, 0.0
        
        # Structure (LVN/POC)
        if state.volume_profile:
            vp = state.volume_profile
            poc = max(vp, key=vp.get)
            valid = {k: v for k, v in vp.items() if v >= max(vp.values()) * 0.1}
            lvn = min(valid, key=valid.get) if valid else poc
            f["dist_poc"] = abs(price - poc)
            f["dist_lvn"] = abs(price - lvn)
            f["dist_poc_atr"] = f["dist_poc"] / (f["ATR_5m"] + 1e-6)
            f["dist_lvn_atr"] = f["dist_lvn"] / (f["ATR_5m"] + 1e-6)
        else:
            f["dist_poc"], f["dist_lvn"], f["dist_poc_atr"], f["dist_lvn_atr"] = 0, 0, 0, 0
        
        # Trade intensity
        tc = list(state.trade_count)
        if len(tc) >= 100:
            f["trade_intensity"] = np.mean(tc[-100:])
            f["trade_intensity_z"] = (tc[-1] - np.mean(tc[-500:])) / (np.std(tc[-500:]) + 1e-6) if len(tc) >= 500 else 0
        else:
            f["trade_intensity"], f["trade_intensity_z"] = 0, 0
        
        # Cumulative delta
        f["cum_delta_1m"] = sum(sq[-240:]) if n >= 240 else sum(sq)
        f["cum_delta_5m"] = sum(sq[-1200:]) if n >= 1200 else sum(sq)
        
        # Time (placeholder - should be passed in)
        import datetime
        now = datetime.datetime.now()
        f["hour_sin"] = np.sin(2 * np.pi * now.hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * now.hour / 24)
        f["is_weekend"] = 1 if now.weekday() >= 5 else 0
        
        # Store for cross-sectional
        for key in ["MOI_z", "vol_ratio", "absorption_z", "AggressionPersistence", "MOI_flip_rate", "trade_intensity_z", "cum_delta_5m"]:
            if key not in self._cross_sectional_cache:
                self._cross_sectional_cache[key] = {}
            self._cross_sectional_cache[key][symbol] = f.get(key, 0)
        
        return f
    
    def _compute_cross_sectional_ranks(self, symbol: str, features: Dict[str, float]) -> Dict[str, float]:
        """Compute cross-sectional ranks relative to other symbols"""
        ranks = {}
        for key in ["MOI_z", "vol_ratio", "absorption_z", "AggressionPersistence", "MOI_flip_rate", "trade_intensity_z", "cum_delta_5m"]:
            cache = self._cross_sectional_cache.get(key, {})
            if len(cache) > 1:
                values = sorted(cache.values())
                val = features.get(key, 0)
                rank = sum(1 for v in values if v < val) / len(values)
                ranks[f"{key}_rank"] = rank
            else:
                ranks[f"{key}_rank"] = 0.5
        
        # MOI_z_relative
        cache = self._cross_sectional_cache.get("MOI_z", {})
        if len(cache) > 1:
            vals = list(cache.values())
            mean, std = np.mean(vals), np.std(vals) + 1e-6
            ranks["MOI_z_relative"] = (features.get("MOI_z", 0) - mean) / std
        else:
            ranks["MOI_z_relative"] = 0
        
        # momentum_rank
        cache = self._cross_sectional_cache.get("cum_delta_5m", {})
        if len(cache) > 1:
            values = sorted(cache.values())
            val = features.get("cum_delta_5m", 0)
            ranks["momentum_rank"] = sum(1 for v in values if v < val) / len(values)
        else:
            ranks["momentum_rank"] = 0.5
        
        return ranks
    
    def _get_model_name(self, direction: Direction, vol_regime: str, horizon: int) -> str:
        dir_str = "up" if direction == Direction.LONG else "down"
        return f"models_{dir_str}_{vol_regime.lower()}_{horizon}"
    
    def _ensemble_predict(self, models: List, features: pd.DataFrame) -> float:
        preds = []
        for m in models:
            try:
                if hasattr(m, 'predict_proba'):
                    preds.append(m.predict_proba(features)[0, 1])
                else:
                    preds.append(m.predict(features)[0])
            except:
                continue
        return float(np.mean(preds)) if preds else 0.0
    
    def _calc_percentile(self, val: float, buffer: List[float]) -> float:
        if not buffer:
            return 50.0
        return float(sum(1 for v in buffer if v < val) / len(buffer) * 100)
    
    def _update_buffer(self, name: str, val: float):
        if name not in self._percentile_buffers:
            self._percentile_buffers[name] = []
        self._percentile_buffers[name].append(val)
        if len(self._percentile_buffers[name]) > MAX_BUFFER_SIZE:
            self._percentile_buffers[name] = self._percentile_buffers[name][-MAX_BUFFER_SIZE:]
    
    def predict(
        self,
        direction: Direction,
        vol_regime: str,
        symbol: str,
        features: Dict[str, float],
    ) -> PredictionResultV3:
        """
        Run prediction using pre-computed features.
        
        Args:
            direction: LONG or SHORT
            vol_regime: "low", "mid", or "high"
            symbol: Trading pair
            features: Dict of feature values from update_bar()
        """
        result = PredictionResultV3()
        
        try:
            # Add cross-sectional ranks
            cs_ranks = self._compute_cross_sectional_ranks(symbol, features)
            features.update(cs_ranks)
            
            # Build feature vector
            feat_vec = []
            for col in self._feature_columns:
                feat_vec.append(features.get(col, 0.0))
            
            feat_df = pd.DataFrame([feat_vec], columns=self._feature_columns)
            
            # Get models
            model_60 = self._get_model_name(direction, vol_regime, 60)
            model_300 = self._get_model_name(direction, vol_regime, 300)
            result.model_60 = model_60
            result.model_300 = model_300
            
            if model_60 not in self._models or model_300 not in self._models:
                return result
            
            # Predict
            raw_prob_60 = self._ensemble_predict(self._models[model_60], feat_df)
            raw_prob_300 = self._ensemble_predict(self._models[model_300], feat_df)
            
            # Apply calibration if available
            if model_60 in self._calibrators:
                result.prob_60 = float(self._calibrators[model_60].predict([raw_prob_60])[0])
            else:
                result.prob_60 = raw_prob_60
            
            if model_300 in self._calibrators:
                result.prob_300 = float(self._calibrators[model_300].predict([raw_prob_300])[0])
            else:
                result.prob_300 = raw_prob_300
            
            result.ensemble_size = len(self._models[model_60])
            
            # Percentiles
            result.percentile_60 = self._calc_percentile(result.prob_60, self._percentile_buffers.get(model_60, []))
            result.percentile_300 = self._calc_percentile(result.prob_300, self._percentile_buffers.get(model_300, []))
            
            # Update buffers
            self._update_buffer(model_60, result.prob_60)
            self._update_buffer(model_300, result.prob_300)
            
            # Cross-sectional rank for this prediction
            result.cross_sectional_rank = cs_ranks.get("MOI_z_rank", 0.5)
            
            # Periodic save
            if sum(len(b) for b in self._percentile_buffers.values()) % 100 == 0:
                self._save_percentile_buffers()
            
            logger.debug("prediction_v3", symbol=symbol, prob_60=result.prob_60, prob_300=result.prob_300)
            
        except Exception as e:
            logger.error("predict_error", symbol=symbol, error=str(e))
        
        return result
