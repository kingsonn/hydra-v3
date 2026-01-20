"""
Stage 3 V3 Models
=================

Core data structures for the hybrid alpha system.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict
import time


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class MarketRegime(Enum):
    """
    Simplified regime types - only 4 actionable states:
    - TRENDING_UP: Clear uptrend, trade with trend (longs)
    - TRENDING_DOWN: Clear downtrend, trade with trend (shorts)
    - RANGING: Bounded price action, wait or fade extremes
    - CHOPPY: No clear structure, NO TRADE
    """
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    CHOPPY = "CHOPPY"  # NO TRADE regime


class SignalType(Enum):
    # Positional signals (background alpha)
    FUNDING_TREND = "FUNDING_TREND"
    TREND_PULLBACK = "TREND_PULLBACK"
    LIQUIDATION_FOLLOW = "LIQUIDATION_FOLLOW"
    RANGE_BREAKOUT = "RANGE_BREAKOUT"
    EXHAUSTION_REVERSAL = "EXHAUSTION_REVERSAL"
    # Entry-first signals (frequency alpha)
    EMA_CONTINUATION = "EMA_CONTINUATION"
    ADX_EXPANSION = "ADX_EXPANSION"
    STRUCTURE_BREAK = "STRUCTURE_BREAK"
    COMPRESSION_BREAK = "COMPRESSION_BREAK"
    SMA_CROSSOVER = "SMA_CROSSOVER"


@dataclass
class Bias:
    """Directional bias from positioning data"""
    direction: Direction = Direction.NEUTRAL
    strength: float = 0.0  # 0.0 to 1.0
    
    # Component scores
    funding_score: float = 0.0  # -1 to +1
    liquidation_score: float = 0.0  # -1 to +1
    oi_score: float = 0.0  # -1 to +1
    trend_score: float = 0.0  # -1 to +1
    
    reason: str = ""
    
    def is_bullish(self) -> bool:
        return self.direction == Direction.LONG and self.strength > 0.3
    
    def is_bearish(self) -> bool:
        return self.direction == Direction.SHORT and self.strength > 0.3
    
    def is_neutral(self) -> bool:
        return self.direction == Direction.NEUTRAL or self.strength < 0.3


@dataclass
class TrendState:
    """Trend analysis result"""
    direction: Direction = Direction.NEUTRAL
    strength: float = 0.0  # 0.0 to 1.0
    
    # MA values
    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    
    # Structure
    higher_high: bool = False
    higher_low: bool = False
    lower_high: bool = False
    lower_low: bool = False
    
    # Price position
    price_vs_ema20: float = 0.0  # % distance
    price_vs_ema50: float = 0.0
    
    # RSI
    rsi_14: float = 50.0
    
    def is_uptrend(self) -> bool:
        return self.direction == Direction.LONG and self.strength > 0.5
    
    def is_downtrend(self) -> bool:
        return self.direction == Direction.SHORT and self.strength > 0.5
    
    def is_pullback_to_ema(self, threshold_pct: float = 0.3) -> bool:
        """Check if price has pulled back to EMA20"""
        return abs(self.price_vs_ema20) < threshold_pct


@dataclass
class MarketState:
    """Complete market state for signal evaluation"""
    symbol: str = ""
    timestamp_ms: int = 0
    current_price: float = 0.0
    
    # Bias (from BiasCalculator)
    bias: Bias = field(default_factory=Bias)
    
    # Regime (from RegimeClassifier)
    regime: MarketRegime = MarketRegime.CHOPPY
    regime_confidence: float = 0.5
    
    # Trend (from TrendAnalyzer)
    trend: TrendState = field(default_factory=TrendState)
    
    # Positioning data
    funding_z: float = 0.0
    funding_rate: float = 0.0
    cumulative_funding_24h: float = 0.0  # Sum of last 3 funding rates (24h)
    oi_delta_1h: float = 0.0
    oi_delta_4h: float = 0.0
    oi_delta_24h: float = 0.0
    liq_imbalance_1h: float = 0.0
    liq_imbalance_4h: float = 0.0
    liq_total_1h: float = 0.0
    liq_long_1h: float = 0.0   # Long liquidations USD
    liq_short_1h: float = 0.0  # Short liquidations USD
    cascade_active: bool = False
    liq_exhaustion: bool = False
    
    # Volatility
    atr_14: float = 0.0
    vol_expansion_ratio: float = 1.0
    range_4h: float = 0.0
    range_vs_atr: float = 1.0
    
    # Price history (for structure detection)
    high_4h: float = 0.0
    low_4h: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    
    # Price changes
    price_change_1h: float = 0.0
    price_change_4h: float = 0.0
    price_change_24h: float = 0.0
    price_change_48h: float = 0.0  # For exhaustion detection
    
    # Volume
    volume_ratio: float = 1.0  # Current vs average
    
    def is_tradeable(self) -> bool:
        """Basic check if market is in tradeable state"""
        return self.regime not in [MarketRegime.CHOPPY]


@dataclass
class HybridSignal:
    """Output signal from hybrid system"""
    direction: Direction
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    
    # Trade parameters
    entry_price: float = 0.0
    stop_price: float = 0.0
    stop_pct: float = 0.015  # 1.5% default
    target_price: float = 0.0
    target_pct: float = 0.03  # 3% default (2R)
    
    # Sizing
    size_multiplier: float = 1.0
    
    # Context
    reason: str = ""
    bias_strength: float = 0.0
    regime: MarketRegime = MarketRegime.RANGING
    
    # Timing
    generated_at: float = field(default_factory=time.time)
    valid_for_seconds: int = 300  # 5 minutes
    
    def is_valid(self) -> bool:
        """Check if signal is still valid"""
        return (time.time() - self.generated_at) < self.valid_for_seconds
    
    def risk_reward_ratio(self) -> float:
        """Calculate R:R"""
        if self.stop_pct == 0:
            return 0
        return self.target_pct / self.stop_pct
    
    def to_dict(self) -> Dict:
        return {
            "direction": self.direction.value,
            "signal_type": self.signal_type.value,
            "confidence": round(self.confidence, 3),
            "entry_price": self.entry_price,
            "stop_pct": round(self.stop_pct, 4),
            "target_pct": round(self.target_pct, 4),
            "risk_reward": round(self.risk_reward_ratio(), 2),
            "reason": self.reason,
        }
