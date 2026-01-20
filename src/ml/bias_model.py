"""
ML-Based Bias Detection Model
=============================

Predicts directional bias (bullish/bearish/neutral) using:
- Funding rate dynamics (z-score, cumulative, trend)
- Open interest patterns (build/unwind, divergences)
- Liquidation imbalances (forced selling/buying pressure)
- Price momentum across timeframes
- Cross-asset signals (BTC dominance, correlation)

Output:
- Bias direction: BULLISH, BEARISH, NEUTRAL
- Confidence score: 0-1
- Contributing factors breakdown

Model: Gradient Boosting with temporal features and calibrated probabilities
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


class BiasLabel(Enum):
    """Market bias labels"""
    BULLISH = 0
    BEARISH = 1
    NEUTRAL = 2


@dataclass
class BiasFeatures:
    """Feature vector for bias classification"""
    
    # ===== Funding Rate Features =====
    funding_rate: float = 0.0           # Current funding rate
    funding_z: float = 0.0              # Z-score vs 30-day mean
    funding_percentile: float = 0.5     # Percentile vs 30-day range
    funding_trend_4h: float = 0.0       # Funding change over 4h
    funding_trend_24h: float = 0.0      # Funding change over 24h
    cumulative_funding_24h: float = 0.0 # Sum of funding over 24h
    cumulative_funding_7d: float = 0.0  # Sum of funding over 7d
    funding_mean_reversion: float = 0.0 # Distance from mean (decay signal)
    
    # ===== Open Interest Features =====
    oi_current: float = 0.0             # Current OI (normalized)
    oi_change_1h: float = 0.0           # OI change 1h
    oi_change_4h: float = 0.0           # OI change 4h
    oi_change_24h: float = 0.0          # OI change 24h
    oi_change_7d: float = 0.0           # OI change 7d
    oi_percentile: float = 0.5          # OI percentile vs 30-day range
    oi_price_divergence: float = 0.0    # Price up + OI down = bearish, etc.
    oi_velocity: float = 0.0            # Rate of OI change
    oi_acceleration: float = 0.0        # Change in OI velocity
    
    # ===== Liquidation Features =====
    liq_long_1h: float = 0.0            # Long liquidations 1h (USD)
    liq_short_1h: float = 0.0           # Short liquidations 1h (USD)
    liq_imbalance_1h: float = 0.0       # (long - short) / total
    liq_long_4h: float = 0.0
    liq_short_4h: float = 0.0
    liq_imbalance_4h: float = 0.0
    liq_long_24h: float = 0.0
    liq_short_24h: float = 0.0
    liq_imbalance_24h: float = 0.0
    cascade_active: bool = False        # Active liquidation cascade
    cascade_direction: float = 0.0      # +1 = long cascade, -1 = short cascade
    liq_exhaustion: bool = False        # Liquidations declining after spike
    
    # ===== Price Momentum Features =====
    return_1h: float = 0.0
    return_4h: float = 0.0
    return_24h: float = 0.0
    return_48h: float = 0.0
    return_7d: float = 0.0
    price_vs_vwap_24h: float = 0.0      # Price relative to 24h VWAP
    price_vs_high_24h: float = 0.0      # Distance from 24h high
    price_vs_low_24h: float = 0.0       # Distance from 24h low
    momentum_divergence: float = 0.0    # Short-term vs long-term momentum
    
    # ===== Trend Features =====
    trend_strength: float = 0.0         # 0-1 trend strength
    trend_direction: float = 0.0        # +1 = up, -1 = down, 0 = neutral
    ema_20_slope: float = 0.0
    ema_50_slope: float = 0.0
    ema_20_50_cross: float = 0.0        # Recent crossover signal
    price_vs_ema_20: float = 0.0
    price_vs_ema_50: float = 0.0
    
    # ===== Volume Features =====
    volume_ratio: float = 1.0           # Current vs average volume
    buy_volume_ratio: float = 0.5       # Buy volume / total volume
    cvd_1h: float = 0.0                 # Cumulative volume delta 1h
    cvd_4h: float = 0.0                 # Cumulative volume delta 4h
    cvd_24h: float = 0.0                # Cumulative volume delta 24h
    
    # ===== Volatility Features =====
    volatility_1h: float = 0.0
    volatility_24h: float = 0.0
    atr_14: float = 0.0
    atr_expansion: float = 1.0          # Short ATR / Long ATR
    bollinger_position: float = 0.5     # 0 = lower band, 1 = upper band
    
    # ===== Cross-Asset Features =====
    btc_correlation_24h: float = 0.0    # Correlation with BTC
    btc_return_24h: float = 0.0         # BTC return (for altcoin bias)
    btc_dominance_change: float = 0.0   # BTC dominance change
    
    # ===== Sentiment/Market Features =====
    long_short_ratio: float = 1.0       # Long/short account ratio
    top_trader_sentiment: float = 0.0   # Top trader positioning
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([
            # Funding
            self.funding_rate,
            self.funding_z,
            self.funding_percentile,
            self.funding_trend_4h,
            self.funding_trend_24h,
            self.cumulative_funding_24h,
            self.cumulative_funding_7d,
            self.funding_mean_reversion,
            # OI
            self.oi_current,
            self.oi_change_1h,
            self.oi_change_4h,
            self.oi_change_24h,
            self.oi_change_7d,
            self.oi_percentile,
            self.oi_price_divergence,
            self.oi_velocity,
            self.oi_acceleration,
            # Liquidations
            self.liq_long_1h,
            self.liq_short_1h,
            self.liq_imbalance_1h,
            self.liq_long_4h,
            self.liq_short_4h,
            self.liq_imbalance_4h,
            self.liq_long_24h,
            self.liq_short_24h,
            self.liq_imbalance_24h,
            1.0 if self.cascade_active else 0.0,
            self.cascade_direction,
            1.0 if self.liq_exhaustion else 0.0,
            # Price momentum
            self.return_1h,
            self.return_4h,
            self.return_24h,
            self.return_48h,
            self.return_7d,
            self.price_vs_vwap_24h,
            self.price_vs_high_24h,
            self.price_vs_low_24h,
            self.momentum_divergence,
            # Trend
            self.trend_strength,
            self.trend_direction,
            self.ema_20_slope,
            self.ema_50_slope,
            self.ema_20_50_cross,
            self.price_vs_ema_20,
            self.price_vs_ema_50,
            # Volume
            self.volume_ratio,
            self.buy_volume_ratio,
            self.cvd_1h,
            self.cvd_4h,
            self.cvd_24h,
            # Volatility
            self.volatility_1h,
            self.volatility_24h,
            self.atr_14,
            self.atr_expansion,
            self.bollinger_position,
            # Cross-asset
            self.btc_correlation_24h,
            self.btc_return_24h,
            self.btc_dominance_change,
            # Sentiment
            self.long_short_ratio,
            self.top_trader_sentiment,
        ])
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for model"""
        return [
            # Funding
            "funding_rate", "funding_z", "funding_percentile",
            "funding_trend_4h", "funding_trend_24h",
            "cumulative_funding_24h", "cumulative_funding_7d",
            "funding_mean_reversion",
            # OI
            "oi_current", "oi_change_1h", "oi_change_4h", "oi_change_24h",
            "oi_change_7d", "oi_percentile", "oi_price_divergence",
            "oi_velocity", "oi_acceleration",
            # Liquidations
            "liq_long_1h", "liq_short_1h", "liq_imbalance_1h",
            "liq_long_4h", "liq_short_4h", "liq_imbalance_4h",
            "liq_long_24h", "liq_short_24h", "liq_imbalance_24h",
            "cascade_active", "cascade_direction", "liq_exhaustion",
            # Price momentum
            "return_1h", "return_4h", "return_24h", "return_48h", "return_7d",
            "price_vs_vwap_24h", "price_vs_high_24h", "price_vs_low_24h",
            "momentum_divergence",
            # Trend
            "trend_strength", "trend_direction",
            "ema_20_slope", "ema_50_slope", "ema_20_50_cross",
            "price_vs_ema_20", "price_vs_ema_50",
            # Volume
            "volume_ratio", "buy_volume_ratio",
            "cvd_1h", "cvd_4h", "cvd_24h",
            # Volatility
            "volatility_1h", "volatility_24h",
            "atr_14", "atr_expansion", "bollinger_position",
            # Cross-asset
            "btc_correlation_24h", "btc_return_24h", "btc_dominance_change",
            # Sentiment
            "long_short_ratio", "top_trader_sentiment",
        ]


@dataclass
class BiasPrediction:
    """Bias prediction result"""
    bias: BiasLabel
    confidence: float  # 0-1
    probabilities: Dict[str, float]  # Probability per bias
    contributing_factors: Dict[str, float]  # Factor contributions
    features_used: Optional[Dict[str, float]] = None


@dataclass
class BiasFactorWeights:
    """Weights for different bias factors in rule-based fallback"""
    funding_weight: float = 0.25
    oi_weight: float = 0.20
    liquidation_weight: float = 0.20
    momentum_weight: float = 0.20
    trend_weight: float = 0.15


class BiasMLModel:
    """
    ML-based market bias classifier.
    
    Economic Thesis:
    - Extreme funding rates indicate crowded positioning → mean reversion likely
    - OI building with price = genuine interest; OI declining = profit taking
    - Liquidation imbalances show forced positioning
    - Momentum divergences signal exhaustion
    
    Training Pipeline:
    1. Collect historical data with labeled forward returns
    2. Label bias: BULLISH if 4h return > +1%, BEARISH if < -1%, else NEUTRAL
    3. Extract features using BiasFeatures
    4. Train gradient boosting with class weights
    5. Calibrate and save
    
    Inference Pipeline:
    1. Extract current features
    2. Predict bias with confidence
    3. Return contributing factors for explainability
    """
    
    MODEL_VERSION = "1.0.0"
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.calibrator = None
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.model_path = model_path
        self.is_loaded = False
        
        # Feature importance
        self.feature_importance: Dict[str, float] = {}
        
        # Training metadata
        self.training_metadata: Dict[str, Any] = {}
        
        # Factor weights for rule-based
        self.factor_weights = BiasFactorWeights()
        
        if model_path and model_path.exists():
            self.load(model_path)
    
    def extract_features(
        self,
        bars_1h: List[Dict],
        current_price: float,
        funding_history: Optional[List[Dict]] = None,
        oi_history: Optional[List[Dict]] = None,
        liquidation_data: Optional[Dict] = None,
        btc_data: Optional[Dict] = None,
    ) -> BiasFeatures:
        """
        Extract features from raw market data.
        
        Args:
            bars_1h: List of hourly OHLCV bars
            current_price: Current market price
            funding_history: List of funding rate snapshots
            oi_history: List of OI snapshots
            liquidation_data: Liquidation aggregates
            btc_data: BTC reference data for correlation
            
        Returns:
            BiasFeatures object
        """
        if len(bars_1h) < 50:
            logger.warning("insufficient_bars_for_bias_features", count=len(bars_1h))
            return BiasFeatures()
        
        # Convert to numpy
        closes = np.array([b['close'] for b in bars_1h])
        highs = np.array([b['high'] for b in bars_1h])
        lows = np.array([b['low'] for b in bars_1h])
        volumes = np.array([b.get('volume', 0) for b in bars_1h])
        
        features = BiasFeatures()
        
        # ===== Funding Features =====
        if funding_history and len(funding_history) > 0:
            funding_rates = [f.get('rate', 0) for f in funding_history]
            
            features.funding_rate = funding_rates[-1] if funding_rates else 0
            
            if len(funding_rates) > 1:
                mean_funding = np.mean(funding_rates)
                std_funding = np.std(funding_rates)
                if std_funding > 0:
                    features.funding_z = (features.funding_rate - mean_funding) / std_funding
                
                # Percentile
                features.funding_percentile = np.sum(
                    np.array(funding_rates) <= features.funding_rate
                ) / len(funding_rates)
                
                # Trends
                if len(funding_rates) >= 4:
                    features.funding_trend_4h = funding_rates[-1] - funding_rates[-4]
                if len(funding_rates) >= 24:
                    features.funding_trend_24h = funding_rates[-1] - funding_rates[-24]
                
                # Cumulative
                if len(funding_rates) >= 3:  # 3 funding periods = 24h
                    features.cumulative_funding_24h = sum(funding_rates[-3:])
                if len(funding_rates) >= 21:  # 21 funding periods = 7d
                    features.cumulative_funding_7d = sum(funding_rates[-21:])
                
                # Mean reversion signal
                features.funding_mean_reversion = features.funding_rate - mean_funding
        
        # ===== OI Features =====
        if oi_history and len(oi_history) > 0:
            oi_values = [o.get('oi', 0) for o in oi_history]
            
            if len(oi_values) > 0 and oi_values[-1] > 0:
                features.oi_current = oi_values[-1]
                
                if len(oi_values) >= 2:
                    features.oi_change_1h = (oi_values[-1] - oi_values[-2]) / oi_values[-2]
                if len(oi_values) >= 4:
                    features.oi_change_4h = (oi_values[-1] - oi_values[-4]) / oi_values[-4]
                if len(oi_values) >= 24:
                    features.oi_change_24h = (oi_values[-1] - oi_values[-24]) / oi_values[-24]
                if len(oi_values) >= 168:  # 7 days
                    features.oi_change_7d = (oi_values[-1] - oi_values[-168]) / oi_values[-168]
                
                # Percentile
                features.oi_percentile = np.sum(
                    np.array(oi_values) <= oi_values[-1]
                ) / len(oi_values)
                
                # OI velocity and acceleration
                if len(oi_values) >= 4:
                    oi_changes = np.diff(oi_values[-5:]) / np.array(oi_values[-5:-1])
                    features.oi_velocity = oi_changes[-1] if len(oi_changes) > 0 else 0
                    if len(oi_changes) >= 2:
                        features.oi_acceleration = oi_changes[-1] - oi_changes[-2]
                
                # Price-OI divergence
                if len(closes) >= 24 and len(oi_values) >= 24:
                    price_change = (closes[-1] - closes[-24]) / closes[-24]
                    oi_change = features.oi_change_24h
                    # Bullish: price up, OI up (new longs)
                    # Bearish: price up, OI down (short covering)
                    # Bearish: price down, OI up (new shorts)
                    # Bullish: price down, OI down (long capitulation ending)
                    if price_change > 0 and oi_change < 0:
                        features.oi_price_divergence = -1.0  # Bearish divergence
                    elif price_change < 0 and oi_change > 0:
                        features.oi_price_divergence = -1.0  # Bearish divergence
                    elif price_change > 0 and oi_change > 0:
                        features.oi_price_divergence = 1.0   # Bullish confirmation
                    elif price_change < 0 and oi_change < 0:
                        features.oi_price_divergence = 0.5   # Capitulation ending
        
        # ===== Liquidation Features =====
        if liquidation_data:
            features.liq_long_1h = liquidation_data.get('long_1h', 0)
            features.liq_short_1h = liquidation_data.get('short_1h', 0)
            total_1h = features.liq_long_1h + features.liq_short_1h
            if total_1h > 0:
                features.liq_imbalance_1h = (features.liq_long_1h - features.liq_short_1h) / total_1h
            
            features.liq_long_4h = liquidation_data.get('long_4h', 0)
            features.liq_short_4h = liquidation_data.get('short_4h', 0)
            total_4h = features.liq_long_4h + features.liq_short_4h
            if total_4h > 0:
                features.liq_imbalance_4h = (features.liq_long_4h - features.liq_short_4h) / total_4h
            
            features.liq_long_24h = liquidation_data.get('long_24h', 0)
            features.liq_short_24h = liquidation_data.get('short_24h', 0)
            total_24h = features.liq_long_24h + features.liq_short_24h
            if total_24h > 0:
                features.liq_imbalance_24h = (features.liq_long_24h - features.liq_short_24h) / total_24h
            
            features.cascade_active = liquidation_data.get('cascade_active', False)
            if features.cascade_active:
                features.cascade_direction = 1.0 if features.liq_imbalance_1h > 0 else -1.0
            
            features.liq_exhaustion = liquidation_data.get('exhaustion', False)
        
        # ===== Price Momentum Features =====
        if len(closes) > 1:
            features.return_1h = (closes[-1] - closes[-2]) / closes[-2]
        if len(closes) > 4:
            features.return_4h = (closes[-1] - closes[-5]) / closes[-5]
        if len(closes) > 24:
            features.return_24h = (closes[-1] - closes[-25]) / closes[-25]
        if len(closes) > 48:
            features.return_48h = (closes[-1] - closes[-49]) / closes[-49]
        if len(closes) > 168:
            features.return_7d = (closes[-1] - closes[-169]) / closes[-169]
        
        # VWAP approximation
        if len(closes) >= 24 and len(volumes) >= 24:
            typical_prices = (highs[-24:] + lows[-24:] + closes[-24:]) / 3
            vwap_24h = np.sum(typical_prices * volumes[-24:]) / np.sum(volumes[-24:]) if np.sum(volumes[-24:]) > 0 else closes[-1]
            features.price_vs_vwap_24h = (current_price - vwap_24h) / vwap_24h
        
        # Position in 24h range
        if len(highs) >= 24 and len(lows) >= 24:
            high_24h = np.max(highs[-24:])
            low_24h = np.min(lows[-24:])
            if high_24h > low_24h:
                features.price_vs_high_24h = (current_price - high_24h) / current_price
                features.price_vs_low_24h = (current_price - low_24h) / current_price
        
        # Momentum divergence
        short_momentum = features.return_4h
        long_momentum = features.return_24h / 6 if features.return_24h != 0 else 0  # Normalize
        features.momentum_divergence = short_momentum - long_momentum
        
        # ===== Trend Features =====
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
        features.price_vs_ema_20 = (current_price - ema_20[-1]) / current_price
        features.price_vs_ema_50 = (current_price - ema_50[-1]) / current_price
        
        # EMA crossover signal
        if len(ema_20) >= 5 and len(ema_50) >= 5:
            cross_now = ema_20[-1] - ema_50[-1]
            cross_prev = ema_20[-5] - ema_50[-5]
            if cross_now > 0 and cross_prev <= 0:
                features.ema_20_50_cross = 1.0  # Bullish cross
            elif cross_now < 0 and cross_prev >= 0:
                features.ema_20_50_cross = -1.0  # Bearish cross
        
        # Trend strength and direction
        trend_indicators = [
            1 if features.ema_20_slope > 0 else -1,
            1 if features.ema_50_slope > 0 else -1,
            1 if features.price_vs_ema_20 > 0 else -1,
            1 if features.price_vs_ema_50 > 0 else -1,
            1 if features.return_24h > 0 else -1,
        ]
        features.trend_direction = np.mean(trend_indicators)
        features.trend_strength = abs(features.trend_direction)
        
        # ===== Volume Features =====
        if len(volumes) >= 20 and np.mean(volumes[-20:]) > 0:
            features.volume_ratio = volumes[-1] / np.mean(volumes[-20:])
        
        # Buy volume ratio (simplified - using close position in bar)
        if len(closes) >= 24:
            buy_bars = sum(1 for i in range(-24, 0) if closes[i] > closes[i-1])
            features.buy_volume_ratio = buy_bars / 24
        
        # CVD approximation
        if len(closes) > 1 and len(volumes) > 0:
            price_changes = np.diff(closes)
            # Positive price change = buy volume, negative = sell volume
            cvd = np.cumsum(np.sign(price_changes) * volumes[1:len(price_changes)+1])
            if len(cvd) >= 1:
                features.cvd_1h = cvd[-1] / np.sum(volumes[-2:]) if np.sum(volumes[-2:]) > 0 else 0
            if len(cvd) >= 4:
                features.cvd_4h = (cvd[-1] - cvd[-4]) / np.sum(volumes[-5:]) if np.sum(volumes[-5:]) > 0 else 0
            if len(cvd) >= 24:
                features.cvd_24h = (cvd[-1] - cvd[-24]) / np.sum(volumes[-25:]) if np.sum(volumes[-25:]) > 0 else 0
        
        # ===== Volatility Features =====
        returns = np.diff(closes) / closes[:-1]
        if len(returns) >= 1:
            features.volatility_1h = np.std(returns[-1:])
        if len(returns) >= 24:
            features.volatility_24h = np.std(returns[-24:])
        
        # ATR
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        atr_14 = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
        features.atr_14 = atr_14 / current_price
        
        atr_short = np.mean(tr[-7:]) if len(tr) >= 7 else atr_14
        atr_long = np.mean(tr[-28:]) if len(tr) >= 28 else atr_14
        features.atr_expansion = atr_short / atr_long if atr_long > 0 else 1.0
        
        # Bollinger position
        if len(closes) >= 20:
            sma_20 = np.mean(closes[-20:])
            std_20 = np.std(closes[-20:])
            if std_20 > 0:
                upper_band = sma_20 + 2 * std_20
                lower_band = sma_20 - 2 * std_20
                features.bollinger_position = (current_price - lower_band) / (upper_band - lower_band)
        
        # ===== Cross-Asset Features =====
        if btc_data:
            features.btc_correlation_24h = btc_data.get('correlation_24h', 0)
            features.btc_return_24h = btc_data.get('return_24h', 0)
            features.btc_dominance_change = btc_data.get('dominance_change', 0)
        
        return features
    
    def normalize_features(self, features: BiasFeatures) -> np.ndarray:
        """Normalize features using training statistics"""
        raw = features.to_array()
        
        if not self.feature_stats:
            return raw
        
        normalized = np.zeros_like(raw)
        names = BiasFeatures.feature_names()
        
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
    
    def predict(self, features: BiasFeatures) -> BiasPrediction:
        """
        Predict market bias from features.
        
        Args:
            features: BiasFeatures object
            
        Returns:
            BiasPrediction with bias, confidence, probabilities, and factors
        """
        if self.model is None:
            return self._rule_based_predict(features)
        
        X = self.normalize_features(features).reshape(1, -1)
        
        try:
            proba = self.model.predict_proba(X)[0]
            
            if self.calibrator:
                proba = self.calibrator.predict_proba(proba.reshape(1, -1))[0]
            
            pred_idx = np.argmax(proba)
            bias = BiasLabel(pred_idx)
            confidence = float(proba[pred_idx])
            
            probabilities = {
                label.name: float(proba[label.value])
                for label in BiasLabel
            }
            
            # Calculate contributing factors
            contributing_factors = self._calculate_factors(features)
            
            return BiasPrediction(
                bias=bias,
                confidence=confidence,
                probabilities=probabilities,
                contributing_factors=contributing_factors,
                features_used={
                    name: float(val)
                    for name, val in zip(BiasFeatures.feature_names(), features.to_array())
                }
            )
            
        except Exception as e:
            logger.error("bias_prediction_failed", error=str(e))
            return self._rule_based_predict(features)
    
    def _calculate_factors(self, features: BiasFeatures) -> Dict[str, float]:
        """Calculate contributing factors for explainability"""
        factors = {}
        
        # Funding factor
        funding_signal = 0.0
        if abs(features.funding_z) > 1.5:
            # Extreme funding = mean reversion signal
            funding_signal = -np.sign(features.funding_z) * min(abs(features.funding_z) / 3, 1.0)
        else:
            funding_signal = features.funding_z / 3
        factors['funding'] = funding_signal
        
        # OI factor
        oi_signal = features.oi_price_divergence * 0.5
        if features.oi_change_24h > 0.1:
            oi_signal += 0.3  # OI building = bullish
        elif features.oi_change_24h < -0.1:
            oi_signal -= 0.3  # OI declining = bearish
        factors['open_interest'] = np.clip(oi_signal, -1, 1)
        
        # Liquidation factor
        liq_signal = 0.0
        if features.cascade_active:
            # Cascade = contrarian opportunity
            liq_signal = -features.cascade_direction * 0.5
        if features.liq_exhaustion:
            # Exhaustion = reversal
            liq_signal += -np.sign(features.liq_imbalance_4h) * 0.3
        else:
            # Ongoing liquidations = momentum continuation
            liq_signal += -features.liq_imbalance_4h * 0.3
        factors['liquidations'] = np.clip(liq_signal, -1, 1)
        
        # Momentum factor
        momentum_signal = (
            features.return_4h * 10 +
            features.return_24h * 5 +
            features.momentum_divergence * 5
        )
        factors['momentum'] = np.clip(momentum_signal, -1, 1)
        
        # Trend factor
        trend_signal = features.trend_direction * features.trend_strength
        factors['trend'] = np.clip(trend_signal, -1, 1)
        
        return factors
    
    def _rule_based_predict(self, features: BiasFeatures) -> BiasPrediction:
        """Fallback rule-based bias prediction"""
        factors = self._calculate_factors(features)
        
        # Weighted combination
        w = self.factor_weights
        total_signal = (
            factors['funding'] * w.funding_weight +
            factors['open_interest'] * w.oi_weight +
            factors['liquidations'] * w.liquidation_weight +
            factors['momentum'] * w.momentum_weight +
            factors['trend'] * w.trend_weight
        )
        
        # Convert to probabilities
        if total_signal > 0.2:
            bias = BiasLabel.BULLISH
            confidence = min(0.5 + total_signal, 0.95)
        elif total_signal < -0.2:
            bias = BiasLabel.BEARISH
            confidence = min(0.5 - total_signal, 0.95)
        else:
            bias = BiasLabel.NEUTRAL
            confidence = 0.5 + (0.2 - abs(total_signal)) * 2
        
        # Create probability distribution
        probs = {label.name: 0.1 for label in BiasLabel}
        probs[bias.name] = confidence
        remaining = 1.0 - confidence
        for label in BiasLabel:
            if label != bias:
                probs[label.name] = remaining / 2
        
        return BiasPrediction(
            bias=bias,
            confidence=confidence,
            probabilities=probs,
            contributing_factors=factors,
        )
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        class_weight: str = 'balanced',
    ) -> Dict[str, float]:
        """
        Train the bias classification model.
        
        Args:
            X: Feature matrix
            y: Labels (0=BULLISH, 1=BEARISH, 2=NEUTRAL)
            validation_split: Validation fraction
            class_weight: Class weighting strategy
            
        Returns:
            Training metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import accuracy_score, f1_score
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Feature statistics
        self.feature_stats = {}
        names = BiasFeatures.feature_names()
        for i, name in enumerate(names):
            self.feature_stats[name] = {
                'mean': float(np.mean(X_train[:, i])),
                'std': float(np.std(X_train[:, i])),
            }
        
        X_train_norm = self._normalize_batch(X_train)
        X_val_norm = self._normalize_batch(X_val)
        
        # Train model
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                class_weight=class_weight,
                random_state=42,
                verbose=-1,
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
            )
        
        self.model.fit(X_train_norm, y_train)
        
        # Calibrate
        self.calibrator = CalibratedClassifierCV(
            self.model, method='isotonic', cv='prefit'
        )
        self.calibrator.fit(X_val_norm, y_val)
        
        # Evaluate
        y_pred = self.model.predict(X_val_norm)
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = {
                name: float(imp)
                for name, imp in zip(names, self.model.feature_importances_)
            }
        
        self.training_metadata = {
            'version': self.MODEL_VERSION,
            'trained_at': datetime.now().isoformat(),
            'n_samples': len(X),
            'accuracy': accuracy,
            'f1_weighted': f1,
        }
        
        self.is_loaded = True
        
        logger.info("bias_model_trained", accuracy=f"{accuracy:.3f}", f1=f"{f1:.3f}")
        
        return self.training_metadata
    
    def _normalize_batch(self, X: np.ndarray) -> np.ndarray:
        """Normalize batch of features"""
        X_norm = np.zeros_like(X)
        names = BiasFeatures.feature_names()
        
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
        
        logger.info("bias_model_saved", path=str(path))
    
    def load(self, path: Path) -> bool:
        """Load model from disk"""
        path = Path(path)
        
        if not path.exists():
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
            
            logger.info("bias_model_loaded", path=str(path))
            return True
            
        except Exception as e:
            logger.error("bias_model_load_failed", error=str(e))
            return False


def predict_bias_from_state(state: Any) -> BiasPrediction:
    """
    Convenience function to predict bias directly from MarketState.
    Uses rule-based classification with existing MarketState variables.
    
    Args:
        state: MarketState object from stage3_v3.models
        
    Returns:
        BiasPrediction with bias, confidence, probabilities, and factors
    """
    # Extract features from MarketState
    features = BiasFeatures(
        # Funding
        funding_rate=state.funding_rate,
        funding_z=state.funding_z,
        cumulative_funding_24h=state.cumulative_funding_24h,
        
        # OI
        oi_change_1h=state.oi_delta_1h,
        oi_change_4h=state.oi_delta_4h,
        oi_change_24h=state.oi_delta_24h,
        
        # Liquidations
        liq_long_1h=state.liq_long_1h,
        liq_short_1h=state.liq_short_1h,
        liq_imbalance_1h=state.liq_imbalance_1h,
        liq_imbalance_4h=state.liq_imbalance_4h,
        cascade_active=state.cascade_active,
        cascade_direction=1.0 if state.liq_imbalance_1h > 0 else -1.0 if state.liq_imbalance_1h < 0 else 0,
        liq_exhaustion=state.liq_exhaustion,
        
        # Price momentum
        return_1h=state.price_change_1h,
        return_4h=state.price_change_4h,
        return_24h=state.price_change_24h,
        return_48h=state.price_change_48h,
        
        # Trend
        trend_strength=state.trend.strength if state.trend else 0.5,
        trend_direction=1.0 if state.trend and state.trend.direction.value == "LONG" else -1.0 if state.trend and state.trend.direction.value == "SHORT" else 0,
        price_vs_ema_20=state.trend.price_vs_ema20 if state.trend else 0,
        price_vs_ema_50=state.trend.price_vs_ema50 if state.trend else 0,
        
        # Volume
        volume_ratio=state.volume_ratio,
        
        # Volatility
        atr_14=state.atr_14 / state.current_price if state.current_price > 0 else 0,
        atr_expansion=state.vol_expansion_ratio,
    )
    
    # Use rule-based prediction (no ML model needed)
    model = BiasMLModel()
    return model._rule_based_predict(features)
