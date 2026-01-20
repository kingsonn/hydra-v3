"""
ML-Based Regime Detection Model
===============================

Classifies market regime into:
- TRENDING_UP: Strong bullish trend with structure
- TRENDING_DOWN: Strong bearish trend with structure
- RANGING: Sideways/consolidation with mean-reversion behavior
- VOLATILE: High volatility without clear direction
- BREAKOUT: Transitional state, potential new trend forming

Features used:
- Price action: returns, volatility, range metrics
- Trend indicators: EMA slopes, crossovers, ADX
- Volume/OI: volume profile, OI changes
- Structure: higher highs/lows, swing patterns
- Funding/Liquidations: market sentiment indicators

Model: Gradient Boosting (XGBoost/LightGBM) with isotonic calibration
"""
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class RegimeLabel(Enum):
    """Market regime labels"""
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    VOLATILE = 3
    BREAKOUT = 4


@dataclass
class RegimeFeatures:
    """Feature vector for regime classification"""
    # Price returns at multiple timeframes
    return_1h: float = 0.0
    return_4h: float = 0.0
    return_24h: float = 0.0
    return_48h: float = 0.0
    
    # Volatility metrics
    volatility_1h: float = 0.0      # Std of returns
    volatility_4h: float = 0.0
    volatility_24h: float = 0.0
    atr_14: float = 0.0             # ATR normalized by price
    atr_expansion: float = 1.0      # Short ATR / Long ATR
    
    # Range metrics
    range_24h_pct: float = 0.0      # (high-low)/close
    range_vs_atr: float = 1.0       # Range / ATR ratio
    price_position_in_range: float = 0.5  # 0=at low, 1=at high
    
    # Trend indicators
    ema_20_slope: float = 0.0       # EMA slope (change per bar)
    ema_50_slope: float = 0.0
    ema_20_50_spread: float = 0.0   # (EMA20 - EMA50) / price
    price_vs_ema_20: float = 0.0    # (price - EMA20) / price
    price_vs_ema_50: float = 0.0
    adx_14: float = 0.0             # ADX value (0-100)
    di_plus: float = 0.0            # +DI
    di_minus: float = 0.0           # -DI
    
    # RSI
    rsi_14: float = 50.0
    rsi_divergence: float = 0.0     # Price making HH but RSI making LH (bearish div)
    
    # Structure
    higher_highs: int = 0           # Count in last N bars
    higher_lows: int = 0
    lower_highs: int = 0
    lower_lows: int = 0
    swing_range: float = 0.0        # Recent swing high - swing low
    
    # Volume/OI
    volume_sma_ratio: float = 1.0   # Current volume / SMA(volume)
    oi_change_1h: float = 0.0
    oi_change_4h: float = 0.0
    oi_change_24h: float = 0.0
    
    # Funding/Liquidations
    funding_rate: float = 0.0
    funding_z: float = 0.0
    liq_imbalance_1h: float = 0.0   # (long_liq - short_liq) / total
    liq_imbalance_4h: float = 0.0
    cascade_active: bool = False
    
    # Cross-asset
    btc_correlation_24h: float = 0.0  # Correlation with BTC
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([
            self.return_1h,
            self.return_4h,
            self.return_24h,
            self.return_48h,
            self.volatility_1h,
            self.volatility_4h,
            self.volatility_24h,
            self.atr_14,
            self.atr_expansion,
            self.range_24h_pct,
            self.range_vs_atr,
            self.price_position_in_range,
            self.ema_20_slope,
            self.ema_50_slope,
            self.ema_20_50_spread,
            self.price_vs_ema_20,
            self.price_vs_ema_50,
            self.adx_14,
            self.di_plus,
            self.di_minus,
            self.rsi_14,
            self.rsi_divergence,
            self.higher_highs,
            self.higher_lows,
            self.lower_highs,
            self.lower_lows,
            self.swing_range,
            self.volume_sma_ratio,
            self.oi_change_1h,
            self.oi_change_4h,
            self.oi_change_24h,
            self.funding_rate,
            self.funding_z,
            self.liq_imbalance_1h,
            self.liq_imbalance_4h,
            1.0 if self.cascade_active else 0.0,
            self.btc_correlation_24h,
        ])
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for model"""
        return [
            "return_1h", "return_4h", "return_24h", "return_48h",
            "volatility_1h", "volatility_4h", "volatility_24h",
            "atr_14", "atr_expansion",
            "range_24h_pct", "range_vs_atr", "price_position_in_range",
            "ema_20_slope", "ema_50_slope", "ema_20_50_spread",
            "price_vs_ema_20", "price_vs_ema_50",
            "adx_14", "di_plus", "di_minus",
            "rsi_14", "rsi_divergence",
            "higher_highs", "higher_lows", "lower_highs", "lower_lows",
            "swing_range", "volume_sma_ratio",
            "oi_change_1h", "oi_change_4h", "oi_change_24h",
            "funding_rate", "funding_z",
            "liq_imbalance_1h", "liq_imbalance_4h", "cascade_active",
            "btc_correlation_24h",
        ]


@dataclass
class RegimePrediction:
    """Regime prediction result"""
    regime: RegimeLabel
    confidence: float  # 0-1
    probabilities: Dict[str, float]  # Probability per regime
    features_used: Optional[Dict[str, float]] = None


class RegimeMLModel:
    """
    ML-based market regime classifier.
    
    Training Pipeline:
    1. Collect historical 1H bar data with labels
    2. Extract features using RegimeFeatures
    3. Train gradient boosting classifier
    4. Calibrate probabilities with isotonic regression
    5. Save model and feature statistics
    
    Inference Pipeline:
    1. Extract features from current market state
    2. Normalize features using training statistics
    3. Predict regime with calibrated probabilities
    4. Return prediction with confidence
    """
    
    MODEL_VERSION = "1.0.0"
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.calibrator = None
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.model_path = model_path
        self.is_loaded = False
        
        # Feature importance (updated after training)
        self.feature_importance: Dict[str, float] = {}
        
        # Training metadata
        self.training_metadata: Dict[str, Any] = {}
        
        if model_path and model_path.exists():
            self.load(model_path)
    
    def extract_features(
        self,
        bars_1h: List[Dict],  # List of OHLCV bars
        current_price: float,
        oi_data: Optional[Dict] = None,
        funding_data: Optional[Dict] = None,
        liquidation_data: Optional[Dict] = None,
    ) -> RegimeFeatures:
        """
        Extract features from raw market data.
        
        Args:
            bars_1h: List of hourly bars with keys: open, high, low, close, volume
            current_price: Current market price
            oi_data: Open interest data dict
            funding_data: Funding rate data dict
            liquidation_data: Liquidation data dict
            
        Returns:
            RegimeFeatures object
        """
        if len(bars_1h) < 50:
            logger.warning("insufficient_bars_for_features", count=len(bars_1h))
            return RegimeFeatures()
        
        # Convert to numpy arrays
        closes = np.array([b['close'] for b in bars_1h])
        highs = np.array([b['high'] for b in bars_1h])
        lows = np.array([b['low'] for b in bars_1h])
        volumes = np.array([b.get('volume', 0) for b in bars_1h])
        
        features = RegimeFeatures()
        
        # ===== Price Returns =====
        features.return_1h = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0
        features.return_4h = (closes[-1] - closes[-4]) / closes[-4] if len(closes) > 4 else 0
        features.return_24h = (closes[-1] - closes[-24]) / closes[-24] if len(closes) > 24 else 0
        features.return_48h = (closes[-1] - closes[-48]) / closes[-48] if len(closes) > 48 else 0
        
        # ===== Volatility =====
        returns = np.diff(closes) / closes[:-1]
        features.volatility_1h = np.std(returns[-1:]) if len(returns) > 0 else 0
        features.volatility_4h = np.std(returns[-4:]) if len(returns) > 4 else 0
        features.volatility_24h = np.std(returns[-24:]) if len(returns) > 24 else 0
        
        # ATR
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        atr_14 = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
        features.atr_14 = atr_14 / current_price  # Normalized
        
        atr_short = np.mean(tr[-7:]) if len(tr) >= 7 else atr_14
        atr_long = np.mean(tr[-28:]) if len(tr) >= 28 else atr_14
        features.atr_expansion = atr_short / atr_long if atr_long > 0 else 1.0
        
        # ===== Range Metrics =====
        high_24h = np.max(highs[-24:])
        low_24h = np.min(lows[-24:])
        features.range_24h_pct = (high_24h - low_24h) / current_price
        features.range_vs_atr = (high_24h - low_24h) / atr_14 if atr_14 > 0 else 0
        features.price_position_in_range = (current_price - low_24h) / (high_24h - low_24h) if high_24h > low_24h else 0.5
        
        # ===== Trend Indicators =====
        # EMA calculation
        def ema(data, period):
            alpha = 2 / (period + 1)
            result = np.zeros_like(data)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        ema_20 = ema(closes, 20)
        ema_50 = ema(closes, 50)
        
        features.ema_20_slope = (ema_20[-1] - ema_20[-5]) / (5 * current_price) if len(ema_20) > 5 else 0
        features.ema_50_slope = (ema_50[-1] - ema_50[-5]) / (5 * current_price) if len(ema_50) > 5 else 0
        features.ema_20_50_spread = (ema_20[-1] - ema_50[-1]) / current_price
        features.price_vs_ema_20 = (current_price - ema_20[-1]) / current_price
        features.price_vs_ema_50 = (current_price - ema_50[-1]) / current_price
        
        # ADX calculation
        plus_dm = np.zeros(len(highs) - 1)
        minus_dm = np.zeros(len(highs) - 1)
        for i in range(len(highs) - 1):
            up_move = highs[i+1] - highs[i]
            down_move = lows[i] - lows[i+1]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        atr_for_adx = tr if len(tr) > 0 else np.ones(1)
        
        # Smoothed values (14-period)
        period = 14
        if len(plus_dm) >= period:
            smooth_plus_dm = np.mean(plus_dm[-period:])
            smooth_minus_dm = np.mean(minus_dm[-period:])
            smooth_atr = np.mean(atr_for_adx[-period:])
            
            di_plus = 100 * smooth_plus_dm / smooth_atr if smooth_atr > 0 else 0
            di_minus = 100 * smooth_minus_dm / smooth_atr if smooth_atr > 0 else 0
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus) if (di_plus + di_minus) > 0 else 0
            
            features.di_plus = di_plus
            features.di_minus = di_minus
            features.adx_14 = dx  # Simplified - would need smoothing for true ADX
        
        # ===== RSI =====
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        
        if len(gains) >= 14:
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            features.rsi_14 = 100 - (100 / (1 + rs))
        
        # RSI Divergence (simplified)
        if len(closes) > 20:
            price_hh = closes[-1] > np.max(closes[-20:-1])
            rsi_values = []
            for i in range(20):
                if i + 14 <= len(returns):
                    g = np.mean(np.where(returns[-(i+14):-(i) if i > 0 else None] > 0, returns[-(i+14):-(i) if i > 0 else None], 0))
                    l = np.mean(np.where(returns[-(i+14):-(i) if i > 0 else None] < 0, -returns[-(i+14):-(i) if i > 0 else None], 0))
                    rs = g / l if l > 0 else 100
                    rsi_values.append(100 - (100 / (1 + rs)))
            if len(rsi_values) > 1:
                rsi_lh = rsi_values[0] < max(rsi_values[1:])
                features.rsi_divergence = -1.0 if price_hh and rsi_lh else 0.0
        
        # ===== Structure =====
        # Find swing points (simplified)
        swing_highs = []
        swing_lows = []
        for i in range(2, len(closes) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append((i, lows[i]))
        
        # Count structure
        if len(swing_highs) >= 2:
            for i in range(1, min(5, len(swing_highs))):
                if swing_highs[-i][1] > swing_highs[-i-1][1]:
                    features.higher_highs += 1
                else:
                    features.lower_highs += 1
        
        if len(swing_lows) >= 2:
            for i in range(1, min(5, len(swing_lows))):
                if swing_lows[-i][1] > swing_lows[-i-1][1]:
                    features.higher_lows += 1
                else:
                    features.lower_lows += 1
        
        if swing_highs and swing_lows:
            recent_high = max(h[1] for h in swing_highs[-3:]) if swing_highs else current_price
            recent_low = min(l[1] for l in swing_lows[-3:]) if swing_lows else current_price
            features.swing_range = (recent_high - recent_low) / current_price
        
        # ===== Volume =====
        if len(volumes) > 20 and np.mean(volumes[-20:]) > 0:
            features.volume_sma_ratio = volumes[-1] / np.mean(volumes[-20:])
        
        # ===== OI Data =====
        if oi_data:
            features.oi_change_1h = oi_data.get('change_1h', 0)
            features.oi_change_4h = oi_data.get('change_4h', 0)
            features.oi_change_24h = oi_data.get('change_24h', 0)
        
        # ===== Funding Data =====
        if funding_data:
            features.funding_rate = funding_data.get('rate', 0)
            features.funding_z = funding_data.get('z_score', 0)
        
        # ===== Liquidation Data =====
        if liquidation_data:
            features.liq_imbalance_1h = liquidation_data.get('imbalance_1h', 0)
            features.liq_imbalance_4h = liquidation_data.get('imbalance_4h', 0)
            features.cascade_active = liquidation_data.get('cascade_active', False)
        
        return features
    
    def normalize_features(self, features: RegimeFeatures) -> np.ndarray:
        """Normalize features using training statistics"""
        raw = features.to_array()
        
        if not self.feature_stats:
            return raw
        
        normalized = np.zeros_like(raw)
        names = RegimeFeatures.feature_names()
        
        for i, name in enumerate(names):
            if name in self.feature_stats:
                mean = self.feature_stats[name]['mean']
                std = self.feature_stats[name]['std']
                if std > 0:
                    normalized[i] = (raw[i] - mean) / std
                else:
                    normalized[i] = raw[i] - mean
            else:
                normalized[i] = raw[i]
        
        return normalized
    
    def predict(self, features: RegimeFeatures) -> RegimePrediction:
        """
        Predict market regime from features.
        
        Args:
            features: RegimeFeatures object
            
        Returns:
            RegimePrediction with regime, confidence, and probabilities
        """
        if self.model is None:
            # Fallback to rule-based if no model
            return self._rule_based_predict(features)
        
        # Normalize and predict
        X = self.normalize_features(features).reshape(1, -1)
        
        try:
            # Get raw probabilities
            proba = self.model.predict_proba(X)[0]
            
            # Apply calibration if available
            if self.calibrator:
                proba = self.calibrator.predict_proba(proba.reshape(1, -1))[0]
            
            # Get prediction
            pred_idx = np.argmax(proba)
            regime = RegimeLabel(pred_idx)
            confidence = float(proba[pred_idx])
            
            probabilities = {
                label.name: float(proba[label.value])
                for label in RegimeLabel
            }
            
            return RegimePrediction(
                regime=regime,
                confidence=confidence,
                probabilities=probabilities,
                features_used={
                    name: float(val) 
                    for name, val in zip(RegimeFeatures.feature_names(), features.to_array())
                }
            )
            
        except Exception as e:
            logger.error("regime_prediction_failed", error=str(e))
            return self._rule_based_predict(features)
    
    def _rule_based_predict(self, features: RegimeFeatures) -> RegimePrediction:
        """Fallback rule-based regime classification"""
        probs = {label.name: 0.0 for label in RegimeLabel}
        
        # Trending detection
        trend_score = 0.0
        if features.adx_14 > 25:
            trend_score += 0.3
        if abs(features.ema_20_50_spread) > 0.01:
            trend_score += 0.2
        if features.higher_highs >= 2 or features.lower_lows >= 2:
            trend_score += 0.2
        if abs(features.return_24h) > 0.03:
            trend_score += 0.2
        
        # Direction
        direction_score = (
            features.ema_20_slope * 100 +
            features.return_24h * 10 +
            (features.higher_highs - features.lower_lows) * 0.1 +
            (features.higher_lows - features.lower_highs) * 0.1
        )
        
        # Volatility detection
        vol_score = 0.0
        if features.atr_expansion > 1.5:
            vol_score += 0.3
        if features.volatility_24h > 0.02:
            vol_score += 0.2
        if features.cascade_active:
            vol_score += 0.3
        
        # Range detection
        range_score = 0.0
        if features.adx_14 < 20:
            range_score += 0.3
        if features.range_vs_atr < 3:
            range_score += 0.2
        if abs(features.return_24h) < 0.01:
            range_score += 0.2
        if 0.3 < features.price_position_in_range < 0.7:
            range_score += 0.2
        
        # Breakout detection
        breakout_score = 0.0
        if features.atr_expansion > 1.3 and features.adx_14 > 20:
            breakout_score += 0.3
        if abs(features.return_1h) > 0.015:
            breakout_score += 0.2
        if features.volume_sma_ratio > 1.5:
            breakout_score += 0.2
        if abs(features.price_vs_ema_20) > 0.02:
            breakout_score += 0.2
        
        # Assign probabilities
        if trend_score > 0.5:
            if direction_score > 0:
                probs["TRENDING_UP"] = trend_score
            else:
                probs["TRENDING_DOWN"] = trend_score
        
        probs["RANGING"] = range_score
        probs["VOLATILE"] = vol_score
        probs["BREAKOUT"] = breakout_score
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}
        else:
            probs["RANGING"] = 1.0
        
        # Get best
        best_regime = max(probs, key=probs.get)
        regime = RegimeLabel[best_regime]
        confidence = probs[best_regime]
        
        return RegimePrediction(
            regime=regime,
            confidence=confidence,
            probabilities=probs,
        )
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        use_lightgbm: bool = True,
    ) -> Dict[str, float]:
        """
        Train the regime classification model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            validation_split: Fraction for validation
            use_lightgbm: Use LightGBM (faster) vs XGBoost
            
        Returns:
            Training metrics dict
        """
        from sklearn.model_selection import train_test_split
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Compute feature statistics from training data
        self.feature_stats = {}
        names = RegimeFeatures.feature_names()
        for i, name in enumerate(names):
            self.feature_stats[name] = {
                'mean': float(np.mean(X_train[:, i])),
                'std': float(np.std(X_train[:, i])),
                'min': float(np.min(X_train[:, i])),
                'max': float(np.max(X_train[:, i])),
            }
        
        # Normalize
        X_train_norm = self._normalize_batch(X_train)
        X_val_norm = self._normalize_batch(X_val)
        
        # Train model
        if use_lightgbm:
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    verbose=-1,
                )
            except ImportError:
                use_lightgbm = False
        
        if not use_lightgbm:
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                )
        
        # Fit
        self.model.fit(X_train_norm, y_train)
        
        # Calibrate probabilities
        self.calibrator = CalibratedClassifierCV(
            self.model, method='isotonic', cv='prefit'
        )
        self.calibrator.fit(X_val_norm, y_val)
        
        # Evaluate
        y_pred = self.model.predict(X_val_norm)
        y_pred_cal = self.calibrator.predict(X_val_norm)
        
        accuracy = accuracy_score(y_val, y_pred)
        accuracy_cal = accuracy_score(y_val, y_pred_cal)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            self.feature_importance = {
                name: float(imp)
                for name, imp in zip(names, importance)
            }
        
        # Training metadata
        self.training_metadata = {
            'version': self.MODEL_VERSION,
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'accuracy': accuracy,
            'accuracy_calibrated': accuracy_cal,
            'f1_weighted': f1,
            'class_distribution': {
                RegimeLabel(i).name: int(np.sum(y == i))
                for i in range(len(RegimeLabel))
            },
        }
        
        self.is_loaded = True
        
        logger.info(
            "regime_model_trained",
            accuracy=f"{accuracy:.3f}",
            f1=f"{f1:.3f}",
            n_samples=len(X),
        )
        
        return self.training_metadata
    
    def _normalize_batch(self, X: np.ndarray) -> np.ndarray:
        """Normalize a batch of features"""
        X_norm = np.zeros_like(X)
        names = RegimeFeatures.feature_names()
        
        for i, name in enumerate(names):
            if name in self.feature_stats:
                mean = self.feature_stats[name]['mean']
                std = self.feature_stats[name]['std']
                if std > 0:
                    X_norm[:, i] = (X[:, i] - mean) / std
                else:
                    X_norm[:, i] = X[:, i] - mean
            else:
                X_norm[:, i] = X[:, i]
        
        return X_norm
    
    def save(self, path: Path) -> None:
        """Save model to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_data = {
            'model': self.model,
            'calibrator': self.calibrator,
            'feature_stats': self.feature_stats,
            'feature_importance': self.feature_importance,
            'training_metadata': self.training_metadata,
            'version': self.MODEL_VERSION,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metadata as JSON
        meta_path = path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'feature_stats': self.feature_stats,
                'feature_importance': self.feature_importance,
                'training_metadata': self.training_metadata,
                'version': self.MODEL_VERSION,
            }, f, indent=2)
        
        logger.info("regime_model_saved", path=str(path))
    
    def load(self, path: Path) -> bool:
        """Load model from disk"""
        path = Path(path)
        
        if not path.exists():
            logger.warning("regime_model_not_found", path=str(path))
            return False
        
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.calibrator = model_data.get('calibrator')
            self.feature_stats = model_data.get('feature_stats', {})
            self.feature_importance = model_data.get('feature_importance', {})
            self.training_metadata = model_data.get('training_metadata', {})
            self.is_loaded = True
            
            logger.info(
                "regime_model_loaded",
                path=str(path),
                version=model_data.get('version'),
            )
            return True
            
        except Exception as e:
            logger.error("regime_model_load_failed", path=str(path), error=str(e))
            return False


def predict_regime_from_state(state: Any) -> RegimePrediction:
    """
    Convenience function to predict regime directly from MarketState.
    Uses rule-based classification with existing MarketState variables.
    
    Args:
        state: MarketState object from stage3_v3.models
        
    Returns:
        RegimePrediction with regime, confidence, and probabilities
    """
    # Extract features from MarketState
    features = RegimeFeatures(
        return_1h=state.price_change_1h,
        return_4h=state.price_change_4h,
        return_24h=state.price_change_24h,
        return_48h=state.price_change_48h,
        atr_14=state.atr_14 / state.current_price if state.current_price > 0 else 0,
        atr_expansion=state.vol_expansion_ratio,
        range_24h_pct=(state.high_24h - state.low_24h) / state.current_price if state.current_price > 0 else 0,
        range_vs_atr=state.range_vs_atr,
        price_position_in_range=(state.current_price - state.low_24h) / (state.high_24h - state.low_24h) if state.high_24h > state.low_24h else 0.5,
        price_vs_ema_20=state.trend.price_vs_ema20 if state.trend else 0,
        price_vs_ema_50=state.trend.price_vs_ema50 if state.trend else 0,
        rsi_14=state.trend.rsi_14 if state.trend else 50,
        volume_sma_ratio=state.volume_ratio,
        oi_change_1h=state.oi_delta_1h,
        oi_change_4h=state.oi_delta_4h,
        oi_change_24h=state.oi_delta_24h,
        funding_rate=state.funding_rate,
        funding_z=state.funding_z,
        liq_imbalance_1h=state.liq_imbalance_1h,
        liq_imbalance_4h=state.liq_imbalance_4h,
        cascade_active=state.cascade_active,
    )
    
    # Use rule-based prediction (no ML model needed)
    model = RegimeMLModel()
    return model._rule_based_predict(features)
