"""
Stage 3 V2 Models: Long-Horizon Positioning Alpha
=================================================

Core data models for the positioning-based signal system.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple
import time


class Direction(Enum):
    """Trade direction"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class SignalType(Enum):
    """Types of positioning signals"""
    POSITIONING_DIVERGENCE = "POSITIONING_DIVERGENCE"
    TRAPPED_REGIME = "TRAPPED_REGIME"
    LIQUIDATION_REGIME = "LIQUIDATION_REGIME"
    POST_LIQ_CONTINUATION = "POST_LIQ_CONTINUATION"
    POST_LIQ_REVERSAL = "POST_LIQ_REVERSAL"


class PositioningRegime(Enum):
    """Market positioning regime (AI-classified)"""
    NEUTRAL = "NEUTRAL"
    CROWDED_LONG = "CROWDED_LONG"
    CROWDED_SHORT = "CROWDED_SHORT"
    UNWINDING = "UNWINDING"
    POST_UNWIND = "POST_UNWIND"


@dataclass
class LongHorizonState:
    """
    Complete market state for long-horizon signal evaluation.
    Built from Stage 2 processors with extended time windows.
    
    All features needed for positioning-based signals in one place.
    """
    # Identifiers
    symbol: str = ""
    timestamp_ms: int = 0
    current_price: float = 0.0
    
    # === FUNDING FEATURES ===
    funding_rate: float = 0.0           # Raw funding rate
    funding_z: float = 0.0              # Z-score vs history
    funding_z_8h_avg: float = 0.0       # 8h rolling average of funding_z
    funding_z_change_8h: float = 0.0    # Trend in funding_z
    cumulative_funding_24h: float = 0.0 # Sum of funding over 24h
    
    # === OI FEATURES ===
    oi: float = 0.0                     # Current open interest
    oi_delta_1m: float = 0.0            # OI change in last 1 min (%)
    oi_delta_5m: float = 0.0            # OI change in last 5 min (%)
    oi_delta_1h: float = 0.0            # OI change in last 1h (%)
    oi_delta_4h: float = 0.0            # OI change in last 4h (%)
    oi_delta_8h: float = 0.0            # OI change in last 8h (%)
    oi_delta_24h: float = 0.0           # OI change in last 24h (%)
    oi_price_correlation_24h: float = 0.0  # Correlation of OI and price changes
    
    # === OI ENTRY ESTIMATION ===
    oi_avg_entry_price: float = 0.0     # Estimated avg entry of recent OI
    oi_entry_displacement_pct: float = 0.0  # (avg_entry - current) / current
    oi_concentration_above_pct: float = 0.0  # % of recent OI built above current
    
    # === PRICE FEATURES ===
    price_change_1h: float = 0.0        # Price change in last 1h (%)
    price_change_4h: float = 0.0        # Price change in last 4h (%)
    price_change_8h: float = 0.0        # Price change in last 8h (%)
    price_change_24h: float = 0.0       # Price change in last 24h (%)
    price_change_5m: float = 0.0        # For compatibility with Stage 2
    
    # === LIQUIDATION FEATURES ===
    liq_imbalance_30s: float = 0.0      # Imbalance in last 30s
    liq_imbalance_2m: float = 0.0       # Imbalance in last 2m
    liq_imbalance_5m: float = 0.0       # Imbalance in last 5m
    liq_imbalance_1h: float = 0.0       # Imbalance in last 1h
    liq_imbalance_4h: float = 0.0       # Imbalance in last 4h
    liq_imbalance_8h: float = 0.0       # Imbalance in last 8h
    liq_total_usd_1h: float = 0.0       # Total liquidation $ in 1h
    liq_total_usd_4h: float = 0.0       # Total liquidation $ in 4h
    liq_total_usd_24h: float = 0.0      # Total liquidation $ in 24h
    cascade_active: bool = False        # Is liquidation cascade happening?
    liq_exhaustion: bool = False        # Has cascade exhausted?
    liq_asymmetry_persistence_h: float = 0.0  # How long has imbalance persisted?
    
    # === VOLATILITY FEATURES ===
    vol_expansion_ratio: float = 1.0    # ATR_5m / ATR_1h
    vol_percentile_24h: float = 50.0    # Current vol vs 24h distribution
    vol_compression_duration_h: float = 0.0  # Hours since last expansion
    atr_5m: float = 0.0
    atr_1h: float = 0.0
    
    # === ORDER BOOK FEATURES ===
    depth_imbalance: float = 0.0        # (bid - ask) / (bid + ask)
    bid_depth_usd: float = 0.0          # Total bid depth
    ask_depth_usd: float = 0.0          # Total ask depth
    
    # === ABSORPTION FEATURES ===
    absorption_z: float = 0.0           # Current absorption z-score
    absorption_z_1h_avg: float = 0.0    # 1h rolling average
    absorption_z_24h_avg: float = 0.0   # 24h rolling average
    refill_rate: float = 0.0            # Liquidity refill rate
    liquidity_sweep: bool = False       # Recent sweep detected?
    
    # === ORDER FLOW FEATURES ===
    moi_1s: float = 0.0                 # Market order imbalance
    moi_std: float = 0.0                # MOI standard deviation
    moi_flip_rate: float = 0.0          # Flips per minute
    aggression_persistence: float = 0.0 # Sustained aggression
    delta_velocity: float = 0.0         # Rate of delta change
    
    # === REGIME (from Stage 2) ===
    regime: str = "COMPRESSION"         # CHOP, COMPRESSION, EXPANSION
    
    # === AI-COMPUTED (filled by AI modules) ===
    ai_positioning_regime: PositioningRegime = PositioningRegime.NEUTRAL
    ai_instability_score: float = 0.0   # 0-1, higher = more unstable
    ai_instability_direction_bias: float = 0.0  # -1 to +1
    ai_anomaly_score: float = 0.0       # 0-1, higher = more unusual
    ai_historical_win_rate: float = 0.5 # From similar context lookup
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/storage"""
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "current_price": self.current_price,
            "funding_z": self.funding_z,
            "oi_delta_24h": self.oi_delta_24h,
            "liq_imbalance_4h": self.liq_imbalance_4h,
            "vol_expansion_ratio": self.vol_expansion_ratio,
            "regime": self.regime,
            "ai_positioning_regime": self.ai_positioning_regime.value,
            "ai_instability_score": self.ai_instability_score,
        }


@dataclass
class PositioningSignal:
    """
    Long-horizon positioning signal output.
    Includes trade parameters and context.
    """
    direction: Direction
    confidence: float                   # 0-1
    signal_type: SignalType
    reason: str
    
    # Trade parameters
    expected_holding_hours: float = 24.0
    stop_pct: float = 0.03              # 3% default
    target_pct: float = 0.05            # 5% default
    size_multiplier: float = 1.0        # Adjusted by volatility/instability
    
    # Context for tracking
    regime_at_signal: str = ""
    positioning_regime: PositioningRegime = PositioningRegime.NEUTRAL
    
    # Timestamps
    generated_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "direction": self.direction.value,
            "confidence": self.confidence,
            "signal_type": self.signal_type.value,
            "reason": self.reason,
            "expected_holding_hours": self.expected_holding_hours,
            "stop_pct": self.stop_pct,
            "target_pct": self.target_pct,
            "size_multiplier": self.size_multiplier,
            "generated_at": self.generated_at,
        }


@dataclass
class SignalMemory:
    """
    Persistent memory for a signal type.
    Tracks regime state across evaluation cycles.
    """
    signal_type: str
    
    # Regime tracking
    regime_active: bool = False
    regime_start_time: Optional[float] = None
    regime_direction: Optional[Direction] = None
    confirmation_count: int = 0
    
    # Signal cooldown
    last_signal_time: Optional[float] = None
    
    # Position tracking (if we have an open position from this signal)
    has_open_position: bool = False
    position_entry_time: Optional[float] = None
    position_entry_price: Optional[float] = None
    position_direction: Optional[Direction] = None
    
    def reset(self):
        """Reset regime state (not position state)"""
        self.regime_active = False
        self.regime_start_time = None
        self.regime_direction = None
        self.confirmation_count = 0
    
    def regime_duration_hours(self) -> float:
        """Get current regime duration in hours"""
        if not self.regime_active or self.regime_start_time is None:
            return 0.0
        return (time.time() - self.regime_start_time) / 3600
    
    def hours_since_last_signal(self) -> float:
        """Hours since last signal was generated"""
        if self.last_signal_time is None:
            return float('inf')
        return (time.time() - self.last_signal_time) / 3600


@dataclass
class OIPriceSnapshot:
    """Snapshot for OI-weighted entry estimation"""
    timestamp_ms: int
    price: float
    oi: float
    oi_delta: float = 0.0  # Change from previous snapshot


@dataclass
class SignalEvaluation:
    """Result of signal evaluation (may or may not produce a signal)"""
    signal: Optional[PositioningSignal] = None
    vetoed: bool = False
    veto_reason: str = ""
    regime_status: str = ""
    
    @property
    def triggered(self) -> bool:
        return self.signal is not None and not self.vetoed


# ============================================================
# COOLDOWN CONFIGURATION
# ============================================================

SIGNAL_COOLDOWNS_HOURS: Dict[str, float] = {
    "POSITIONING_DIVERGENCE": 8.0,
    "TRAPPED_REGIME": 12.0,
    "LIQUIDATION_REGIME": 4.0,
    "POST_LIQ_CONTINUATION": 6.0,
    "POST_LIQ_REVERSAL": 6.0,
}

# ============================================================
# SIGNAL THRESHOLDS
# ============================================================

@dataclass
class PositioningThresholds:
    """Thresholds for positioning signals"""
    # Funding divergence
    funding_z_extreme: float = 1.5
    funding_z_very_extreme: float = 2.0
    
    # OI thresholds
    oi_expansion_significant: float = 0.03  # 3% OI change
    oi_expansion_large: float = 0.08        # 8% OI change
    oi_drop_significant: float = -0.03      # 3% OI drop
    
    # Trapped regime
    trapped_displacement_min: float = 0.025  # 2.5% underwater
    trapped_concentration_min: float = 0.55  # 55% OI on wrong side
    
    # Liquidation
    liq_imbalance_significant: float = 0.5  # 50% one-sided
    liq_imbalance_extreme: float = 0.7      # 70% one-sided
    liq_total_significant_usd: float = 100_000  # $100k+ in liquidations
    
    # Regime persistence (hours)
    regime_min_persistence_h: float = 2.0
    trapped_min_persistence_h: float = 4.0
    post_liq_wait_min_m: float = 30.0
    post_liq_wait_max_m: float = 180.0
    
    # Volatility
    vol_expansion_high: float = 1.5
    vol_compression_low: float = 0.7


# Default thresholds
DEFAULT_THRESHOLDS = PositioningThresholds()
