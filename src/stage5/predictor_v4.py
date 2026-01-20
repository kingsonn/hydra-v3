"""
Stage 5: ML Predictor V4.1 - Outcome-Based Gate

V4.1 Changes:
- R = 0.6 × ATR_1h + fees (more stable than ATR_5m)
- Horizon = 900s (15 minutes) instead of 300s
- ATR_1h computed from 14400 bars (1 hour)

Core design:
1. Single model per direction/regime (no separate filter model)
2. Uses LAGGED order flow features to avoid signal detection leakage
3. Direction is INPUT from Stage 3, not inferred from features
4. Outputs calibrated probability of TP hit

Usage in pipeline:
- Stage 3 fires signal with direction
- Stage 5 predicts P(TP|signal, direction)
- Gate threshold: 0.55 (55% = positive expectancy with 2:1 R:R)
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

ML_MODELS_DIR = Path("models_v4")
FEATURE_COLUMNS_PATH = Path("models_v4/feature_columns_v4.json")

VALID_SYMBOLS = ["ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT", "XRPUSDT"]
VOL_REGIMES = ["low", "mid", "high"]
HORIZON_SEC = 900  # V4.1: 15 minutes (extended from 300s)
ATR_MULT = 0.6    # V4.1: R = 0.6 × ATR_1h + fees

# Gate threshold: With 2:1 R:R, need >33% win rate to break even
# We use 55% for positive expectancy buffer
PROB_THRESHOLD = 0.55


@dataclass
class PredictionResultV4:
    """Result from V4 ML prediction"""
    prob_tp: float = 0.0          # P(TP hit before SL)
    prob_calibrated: float = 0.0  # Calibrated probability
    passes_gate: bool = False     # Whether signal should be taken
    model_key: str = ""
    ensemble_size: int = 0
    vol_regime: str = "mid"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prob_tp": self.prob_tp,
            "prob_calibrated": self.prob_calibrated,
            "passes_gate": self.passes_gate,
            "model_key": self.model_key,
            "ensemble_size": self.ensemble_size,
            "vol_regime": self.vol_regime,
        }


@dataclass
class FeatureState:
    """Rolling state for incremental feature computation"""
    prices: deque = field(default_factory=lambda: deque(maxlen=14400))
    signed_qty: deque = field(default_factory=lambda: deque(maxlen=4800))
    qty: deque = field(default_factory=lambda: deque(maxlen=4800))
    trade_count: deque = field(default_factory=lambda: deque(maxlen=4800))
    highs: deque = field(default_factory=lambda: deque(maxlen=4800))
    lows: deque = field(default_factory=lambda: deque(maxlen=4800))
    volume_profile: Dict[float, float] = field(default_factory=dict)
    vp_qty: deque = field(default_factory=lambda: deque(maxlen=4800))
    vp_bins: deque = field(default_factory=lambda: deque(maxlen=4800))
    
    # Lagged feature buffers (1-minute = 240 bars at 250ms)
    moi_z_history: deque = field(default_factory=lambda: deque(maxlen=300))
    cum_delta_history: deque = field(default_factory=lambda: deque(maxlen=300))
    delta_vel_history: deque = field(default_factory=lambda: deque(maxlen=300))
    
    # Regime tracking
    last_vol_regime: str = "mid"
    bars_in_regime: int = 0


class MLPredictorV4:
    """
    V4 Outcome-Based ML Predictor
    
    Answers: P(+2R before -1R | Stage 3 signal with direction d)
    
    Key design:
    - Uses LAGGED order flow features (1-minute lag)
    - Direction is taken from Stage 3 signal (not inferred)
    - Single model per direction/regime
    - Calibrated probability output
    """
    
    def __init__(self, models_dir: Optional[Path] = None, threshold: float = PROB_THRESHOLD):
        self.models_dir = models_dir or ML_MODELS_DIR
        self.threshold = threshold
        self._models: Dict[str, List] = {}
        self._calibrators: Dict[str, Any] = {}
        self._feature_columns: List[str] = []
        self._feature_states: Dict[str, FeatureState] = {}
        
        self._load_models()
        self._load_feature_columns()
        
        logger.info("predictor_v4_initialized", 
                    models=len(self._models), 
                    features=len(self._feature_columns),
                    threshold=self.threshold)
    
    def _load_models(self):
        """Load V4 outcome models: models_v4/<SYMBOL>/outcome_<direction>_<regime>_300.pkl"""
        if not self.models_dir.exists():
            logger.warning("models_dir_not_found", path=str(self.models_dir))
            return
        
        for symbol in VALID_SYMBOLS:
            symbol_dir = self.models_dir / symbol
            if not symbol_dir.exists():
                continue
            
            for direction in ["long", "short"]:
                for regime in VOL_REGIMES:
                    model_file = f"outcome_{direction}_{regime}_{HORIZON_SEC}.pkl"
                    pkl_path = symbol_dir / model_file
                    model_key = f"{symbol}_{direction}_{regime}"
                    
                    if not pkl_path.exists():
                        continue
                    
                    try:
                        with open(pkl_path, "rb") as f:
                            loaded = pickle.load(f)
                            if isinstance(loaded, dict) and "models" in loaded:
                                self._models[model_key] = loaded["models"]
                                if loaded.get("calibrator"):
                                    self._calibrators[model_key] = loaded["calibrator"]
                            else:
                                self._models[model_key] = loaded if isinstance(loaded, list) else [loaded]
                        logger.debug("model_loaded", key=model_key, ensemble_size=len(self._models[model_key]))
                    except Exception as e:
                        logger.error("model_load_failed", key=model_key, error=str(e))
        
        logger.info("v4_models_loaded", count=len(self._models), calibrators=len(self._calibrators))
    
    def _load_feature_columns(self):
        """Load feature columns from training config"""
        if FEATURE_COLUMNS_PATH.exists():
            with open(FEATURE_COLUMNS_PATH) as f:
                self._feature_columns = json.load(f)
        else:
            # Default V4.1 features (outcome-predicting, no signal leakage)
            self._feature_columns = [
                "vol_5m", "vol_ratio", "vol_rank", "ATR_1h_pct",
                "absorption_z", "price_impact_z",
                "trade_intensity", "trade_intensity_z",
                "dist_poc_atr", "dist_lvn_atr",
                "hour_sin", "hour_cos", "is_weekend",
                "MOI_z_lag", "cum_delta_5m_lag", "delta_velocity_lag",
                "bars_in_regime",
            ]
    
    def _get_state(self, symbol: str) -> FeatureState:
        if symbol not in self._feature_states:
            self._feature_states[symbol] = FeatureState()
        return self._feature_states[symbol]
    
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
        Update bar state and compute features.
        Call every 250ms bar.
        
        Returns dict of features for prediction.
        """
        state = self._get_state(symbol)
        
        # Update rolling buffers
        state.prices.append(price)
        state.signed_qty.append(signed_qty)
        state.qty.append(qty)
        state.trade_count.append(trade_count)
        state.highs.append(high)
        state.lows.append(low)
        
        # Volume profile
        price_bin = round(price / bin_size) * bin_size
        state.volume_profile[price_bin] = state.volume_profile.get(price_bin, 0) + qty
        state.vp_qty.append(qty)
        state.vp_bins.append(price_bin)
        if len(state.vp_qty) == state.vp_qty.maxlen:
            old_bin = state.vp_bins[0]
            state.volume_profile[old_bin] = max(0, state.volume_profile.get(old_bin, 0) - state.vp_qty[0])
        
        return self._compute_features(state, symbol, price)
    
    def _compute_features(self, state: FeatureState, symbol: str, price: float) -> Dict[str, float]:
        """Compute V4 features with LAGGED order flow"""
        f = {}
        prices = list(state.prices)
        sq = list(state.signed_qty)
        qty_list = list(state.qty)
        tc_list = list(state.trade_count)
        n = len(prices)
        
        # ========== VOLATILITY ==========
        if n >= 2:
            rets = np.diff(prices) / np.array(prices[:-1])
            f["vol_1m"] = np.std(rets[-240:]) if len(rets) >= 240 else np.std(rets) + 1e-8
            f["vol_5m"] = np.std(rets[-1200:]) if len(rets) >= 1200 else np.std(rets) + 1e-8
            f["vol_ratio"] = f["vol_1m"] / (f["vol_5m"] + 1e-8)
            # Vol rank approximation
            if len(rets) >= 1000:
                vol_history = pd.Series(rets).rolling(1200).std().dropna().values
                f["vol_rank"] = np.sum(vol_history < f["vol_5m"]) / len(vol_history)
            else:
                f["vol_rank"] = 0.5
        else:
            f["vol_1m"] = f["vol_5m"] = 0.0001
            f["vol_ratio"] = 1.0
            f["vol_rank"] = 0.5
        
        # Vol regime
        if f["vol_rank"] < 0.33:
            vol_regime = "low"
        elif f["vol_rank"] < 0.67:
            vol_regime = "mid"
        else:
            vol_regime = "high"
        
        # Regime duration tracking
        if vol_regime != state.last_vol_regime:
            state.bars_in_regime = 0
            state.last_vol_regime = vol_regime
        state.bars_in_regime += 1
        f["bars_in_regime"] = state.bars_in_regime
        f["vol_regime"] = vol_regime
        
        # ========== ATR ==========
        if n >= 2:
            highs = np.array(list(state.highs))
            lows = np.array(list(state.lows))
            closes = np.array(prices)
            tr = np.maximum(highs[1:] - lows[1:], 
                          np.maximum(np.abs(highs[1:] - closes[:-1]), 
                                    np.abs(lows[1:] - closes[:-1])))
            f["ATR_5m"] = np.mean(tr[-1200:]) if len(tr) >= 1200 else np.mean(tr) + 1e-8
            # V4.1: ATR_1h for more stable R calculation (14400 bars = 1 hour)
            f["ATR_1h"] = np.mean(tr[-14400:]) if len(tr) >= 14400 else f["ATR_5m"]
        else:
            f["ATR_5m"] = price * 0.001
            f["ATR_1h"] = price * 0.001
        f["ATR_5m_pct"] = f["ATR_5m"] / price
        f["ATR_1h_pct"] = f["ATR_1h"] / price
        
        # ========== ABSORPTION / PRICE IMPACT ==========
        if n >= 50 and len(qty_list) >= 50:
            pc = np.abs(np.diff(prices[-500:])) + 1e-8
            qs = np.array(qty_list[-len(pc):])
            if len(pc) == len(qs):
                absorption = qs / pc
                f["absorption_z"] = (absorption[-1] - np.mean(absorption)) / (np.std(absorption) + 1e-6)
                pi = pc / (qs + 1e-6)
                f["price_impact_z"] = (pi[-1] - np.mean(pi)) / (np.std(pi) + 1e-6)
            else:
                f["absorption_z"] = f["price_impact_z"] = 0.0
        else:
            f["absorption_z"] = f["price_impact_z"] = 0.0
        
        # ========== TRADE INTENSITY ==========
        if len(tc_list) >= 100:
            f["trade_intensity"] = np.mean(tc_list[-100:])
            f["trade_intensity_z"] = (tc_list[-1] - np.mean(tc_list[-500:])) / (np.std(tc_list[-500:]) + 1e-6) if len(tc_list) >= 500 else 0
        else:
            f["trade_intensity"] = f["trade_intensity_z"] = 0
        
        # ========== STRUCTURE (POC/LVN) ==========
        if state.volume_profile:
            vp = state.volume_profile
            poc = max(vp, key=vp.get)
            valid = {k: v for k, v in vp.items() if v >= max(vp.values()) * 0.1}
            lvn = min(valid, key=valid.get) if valid else poc
            f["dist_poc_atr"] = abs(price - poc) / (f["ATR_5m"] + 1e-6)
            f["dist_lvn_atr"] = abs(price - lvn) / (f["ATR_5m"] + 1e-6)
        else:
            f["dist_poc_atr"] = f["dist_lvn_atr"] = 0
        
        # ========== TIME ==========
        import datetime
        now = datetime.datetime.now()
        f["hour_sin"] = np.sin(2 * np.pi * now.hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * now.hour / 24)
        f["is_weekend"] = 1 if now.weekday() >= 5 else 0
        
        # ========== CURRENT ORDER FLOW (for history buffer) ==========
        n_sq = len(sq)
        moi_1s = sum(sq[-4:]) if n_sq >= 4 else sum(sq)
        if n_sq >= 100:
            moi_arr = np.array(sq[-100:])
            moi_std = np.std(moi_arr) + 1e-6
            current_moi_z = abs(moi_1s) / moi_std
        else:
            current_moi_z = 0.0
        
        current_cum_delta = sum(sq[-1200:]) if n_sq >= 1200 else sum(sq)
        prev_moi = sum(sq[-8:-4]) if n_sq >= 8 else 0
        current_delta_vel = moi_1s - prev_moi if n_sq >= 5 else 0
        
        # Add to history buffers
        state.moi_z_history.append(current_moi_z)
        state.cum_delta_history.append(current_cum_delta)
        state.delta_vel_history.append(current_delta_vel)
        
        # ========== LAGGED ORDER FLOW (1-minute = 240 bars) ==========
        LAG_INDEX = 240
        if len(state.moi_z_history) > LAG_INDEX:
            f["MOI_z_lag"] = state.moi_z_history[-LAG_INDEX]
            f["cum_delta_5m_lag"] = state.cum_delta_history[-LAG_INDEX]
            f["delta_velocity_lag"] = state.delta_vel_history[-LAG_INDEX]
        else:
            f["MOI_z_lag"] = f["cum_delta_5m_lag"] = f["delta_velocity_lag"] = 0.0
        
        return f
    
    def _get_model_key(self, symbol: str, direction: Direction, vol_regime: str) -> str:
        """Get model key for prediction"""
        dir_str = "long" if direction == Direction.LONG else "short"
        regime = vol_regime.lower()
        return f"{symbol}_{dir_str}_{regime}"
    
    def _ensemble_predict(self, models: List, features: pd.DataFrame) -> float:
        """Average predictions across ensemble"""
        preds = []
        for m in models:
            try:
                if hasattr(m, 'predict_proba'):
                    preds.append(m.predict_proba(features)[0, 1])
                else:
                    preds.append(float(np.clip(m.predict(features)[0], 0, 1)))
            except Exception as e:
                logger.debug("ensemble_member_failed", error=str(e))
                continue
        return float(np.mean(preds)) if preds else 0.0
    
    def predict(
        self,
        direction: Direction,
        symbol: str,
        features: Dict[str, float],
    ) -> PredictionResultV4:
        """
        Predict P(TP|signal) for a Stage 3 signal.
        
        Args:
            direction: LONG or SHORT (from Stage 3 signal)
            symbol: Trading pair
            features: Features from update_bar()
        
        Returns:
            PredictionResultV4 with probability and gate decision
        """
        result = PredictionResultV4()
        
        if direction == Direction.NONE:
            return result
        
        try:
            vol_regime = features.get("vol_regime", "mid")
            result.vol_regime = vol_regime
            
            # Build feature vector
            feat_vec = [features.get(col, 0.0) for col in self._feature_columns]
            feat_df = pd.DataFrame([feat_vec], columns=self._feature_columns)
            
            # Get model
            model_key = self._get_model_key(symbol, direction, vol_regime)
            result.model_key = model_key
            
            if model_key not in self._models:
                logger.debug("model_not_found", key=model_key)
                # Default to 50% if no model (neutral)
                result.prob_tp = 0.5
                result.prob_calibrated = 0.5
                result.passes_gate = False
                return result
            
            models = self._models[model_key]
            result.ensemble_size = len(models)
            
            # Raw prediction
            raw_prob = self._ensemble_predict(models, feat_df)
            result.prob_tp = raw_prob
            
            # Calibrated prediction
            if model_key in self._calibrators:
                result.prob_calibrated = float(np.clip(
                    self._calibrators[model_key].predict([raw_prob])[0], 0, 1
                ))
            else:
                result.prob_calibrated = raw_prob
            
            # Gate decision
            result.passes_gate = result.prob_calibrated >= self.threshold
            
            logger.debug("prediction_v4",
                        symbol=symbol,
                        direction=direction.value,
                        prob_tp=f"{result.prob_tp:.3f}",
                        prob_cal=f"{result.prob_calibrated:.3f}",
                        passes=result.passes_gate)
            
        except Exception as e:
            logger.error("predict_error", symbol=symbol, error=str(e))
        
        return result
    
    def get_threshold(self) -> float:
        """Get current gate threshold"""
        return self.threshold
    
    def set_threshold(self, threshold: float):
        """Update gate threshold"""
        self.threshold = threshold
        logger.info("threshold_updated", new_threshold=threshold)


# Backward compatibility alias
def create_predictor_v4(models_dir: Optional[Path] = None) -> MLPredictorV4:
    """Factory function for V4 predictor"""
    return MLPredictorV4(models_dir)
