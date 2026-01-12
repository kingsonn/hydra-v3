"""
Stage 2 Data Models - MarketState and Feature Objects
All models optimized for speed with slots and minimal overhead
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import time


class Regime(Enum):
    """Market regime classification"""
    CHOP = "CHOP"              # Noisy, directionless - DO NOTHING
    COMPRESSION = "COMPRESSION" # Coiling, preparing to move
    EXPANSION = "EXPANSION"     # Active price discovery


class CrowdSide(Enum):
    """Crowd positioning bias from funding"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class ParticipationType(Enum):
    """OI-based participation classification"""
    NEW_LONGS = "NEW_LONGS"         # Price up + OI up
    NEW_SHORTS = "NEW_SHORTS"       # Price down + OI up
    LONG_COVERING = "LONG_COVERING" # Price down + OI down
    SHORT_COVERING = "SHORT_COVERING" # Price up + OI down
    NEUTRAL = "NEUTRAL"


@dataclass(slots=True)
class OrderFlowFeatures:
    """Order flow features from trades/bars"""
    moi_250ms: float = 0.0      # Market Order Imbalance (3 bars of 250ms)
    moi_1s: float = 0.0         # Market Order Imbalance (10 bars of 250ms)
    delta_velocity: float = 0.0  # MOI_1s(t) - MOI_1s(t-1)
    aggression_persistence: float = 0.0  # mean(|MOI|) / std(|MOI|)
    moi_std: float = 0.0        # Rolling std of MOI
    moi_flip_rate: float = 0.0  # Sign flips per minute
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "moi_250ms": self.moi_250ms,
            "moi_1s": self.moi_1s,
            "delta_velocity": self.delta_velocity,
            "aggression_persistence": self.aggression_persistence,
            "moi_std": self.moi_std,
            "moi_flip_rate": self.moi_flip_rate,
        }


@dataclass(slots=True)
class AbsorptionFeatures:
    """Order book absorption features"""
    absorption_z: float = 0.0       # Z-scored absorption ratio
    refill_rate: float = 0.0        # New resting size after trade / time
    liquidity_sweep: bool = False   # Multiple levels cleared rapidly
    bid_depth_usd: float = 0.0      # Total bid depth
    ask_depth_usd: float = 0.0      # Total ask depth
    depth_imbalance: float = 0.0    # (bid - ask) / total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "absorption_z": self.absorption_z,
            "refill_rate": self.refill_rate,
            "liquidity_sweep": self.liquidity_sweep,
            "bid_depth_usd": self.bid_depth_usd,
            "ask_depth_usd": self.ask_depth_usd,
            "depth_imbalance": self.depth_imbalance,
        }


@dataclass(slots=True)
class VolatilityFeatures:
    """Volatility features from bars"""
    vol_5m: float = 0.0          # Realized volatility (5m)
    vol_1h: float = 0.0          # Realized volatility (1h)
    vol_rank: float = 0.0        # Percentile rank vs history (0-100)
    vol_regime: str = "MID"      # LOW (<30%), MID (30-70%), HIGH (>70%)
    atr_5m: float = 0.0          # ATR over 5 minutes
    atr_1h: float = 0.0          # ATR over 1 hour
    vol_expansion_ratio: float = 0.0  # ATR_5m / ATR_1h
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vol_5m": self.vol_5m,
            "vol_1h": self.vol_1h,
            "vol_rank": self.vol_rank,
            "vol_regime": self.vol_regime,
            "atr_5m": self.atr_5m,
            "atr_1h": self.atr_1h,
            "vol_expansion_ratio": self.vol_expansion_ratio,
        }


@dataclass(slots=True)
class StructureFeatures:
    """Volume profile structural features"""
    poc: float = 0.0             # Point of Control price
    vah: float = 0.0             # Value Area High
    val: float = 0.0             # Value Area Low
    lvns: List[float] = field(default_factory=list)  # Low Volume Nodes
    dist_poc: float = 0.0        # Distance to POC (normalized by ATR)
    dist_lvn: float = 0.0        # Distance to nearest LVN (normalized)
    value_area_width: float = 0.0     # VAH - VAL
    value_width_ratio: float = 0.0    # width / median_width
    time_inside_value_pct: float = 0.0  # % time inside value area
    acceptance_outside_value: bool = False  # Accepted outside VAH/VAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "poc": self.poc,
            "vah": self.vah,
            "val": self.val,
            "lvns": self.lvns,
            "dist_poc": self.dist_poc,
            "dist_lvn": self.dist_lvn,
            "value_area_width": self.value_area_width,
            "value_width_ratio": self.value_width_ratio,
            "time_inside_value_pct": self.time_inside_value_pct,
            "acceptance_outside_value": self.acceptance_outside_value,
        }


@dataclass(slots=True)
class FundingFeatures:
    """Funding rate features"""
    rate: float = 0.0            # Current funding rate
    funding_z: float = 0.0       # Z-scored funding
    crowd_side: CrowdSide = CrowdSide.NEUTRAL
    annualized_pct: float = 0.0  # Annualized rate %
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rate": self.rate,
            "funding_z": self.funding_z,
            "crowd_side": self.crowd_side.value,
            "annualized_pct": self.annualized_pct,
        }


@dataclass(slots=True)
class OIFeatures:
    """Open interest features"""
    oi: float = 0.0              # Current OI
    oi_delta: float = 0.0        # OI change rate (vs prev reading)
    oi_delta_1m: float = 0.0     # 1 minute OI delta %
    oi_delta_5m: float = 0.0     # 5 minute OI delta %
    participation_type: ParticipationType = ParticipationType.NEUTRAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "oi": self.oi,
            "oi_delta": self.oi_delta,
            "oi_delta_1m": self.oi_delta_1m,
            "oi_delta_5m": self.oi_delta_5m,
            "participation_type": self.participation_type.value,
        }


@dataclass(slots=True)
class LiquidationFeatures:
    """Liquidation features with rolling windows"""
    # 30 second window
    long_usd_30s: float = 0.0
    short_usd_30s: float = 0.0
    imbalance_30s: float = 0.0
    
    # 2 minute window
    long_usd_2m: float = 0.0
    short_usd_2m: float = 0.0
    imbalance_2m: float = 0.0
    
    # 5 minute window
    long_usd_5m: float = 0.0
    short_usd_5m: float = 0.0
    imbalance_5m: float = 0.0
    
    # Derived states - THE EDGE
    cascade_active: bool = False   # Active liquidation cascade
    exhaustion: bool = False       # Liquidation exhaustion (reversal signal)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "long_usd_30s": self.long_usd_30s,
            "short_usd_30s": self.short_usd_30s,
            "imbalance_30s": self.imbalance_30s,
            "long_usd_2m": self.long_usd_2m,
            "short_usd_2m": self.short_usd_2m,
            "imbalance_2m": self.imbalance_2m,
            "long_usd_5m": self.long_usd_5m,
            "short_usd_5m": self.short_usd_5m,
            "imbalance_5m": self.imbalance_5m,
            "cascade_active": self.cascade_active,
            "exhaustion": self.exhaustion,
        }


@dataclass
class MarketState:
    """
    Complete market state output from Stage 2
    Updated every 250ms-1s per symbol
    """
    timestamp_ms: int
    symbol: str
    price: float
    regime: Regime
    
    # Feature blocks
    order_flow: OrderFlowFeatures
    absorption: AbsorptionFeatures
    structure: StructureFeatures
    volatility: VolatilityFeatures
    funding: FundingFeatures
    oi: OIFeatures
    liquidations: LiquidationFeatures
    
    # Additional computed fields
    price_change_5m: float = 0.0      # Price change over 5 min (%)
    time_in_regime: float = 0.0       # Seconds in current regime
    
    # Regime inputs (for debugging/transparency)
    regime_inputs: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "symbol": self.symbol,
            "price": self.price,
            "regime": self.regime.value,
            "order_flow": self.order_flow.to_dict(),
            "absorption": self.absorption.to_dict(),
            "structure": self.structure.to_dict(),
            "volatility": self.volatility.to_dict(),
            "funding": self.funding.to_dict(),
            "oi": self.oi.to_dict(),
            "liquidations": self.liquidations.to_dict(),
            "regime_inputs": self.regime_inputs,
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattened dict for UI display"""
        flat = {
            "timestamp_ms": self.timestamp_ms,
            "symbol": self.symbol,
            "price": self.price,
            "regime": self.regime.value,
            "price_change_5m": self.price_change_5m,
            "time_in_regime": self.time_in_regime,
        }
        # Flatten all feature dicts with prefixes
        for name, feat in [
            ("of", self.order_flow),
            ("abs", self.absorption),
            ("str", self.structure),
            ("vol", self.volatility),
            ("fund", self.funding),
            ("oi", self.oi),
            ("liq", self.liquidations),
        ]:
            for k, v in feat.to_dict().items():
                flat[f"{name}_{k}"] = v
        
        # Add regime inputs with prefix
        for k, v in self.regime_inputs.items():
            flat[f"regime_{k}"] = v
        
        return flat


@dataclass(slots=True)
class RegimeClassification:
    """Regime classification result with confidence"""
    regime: Regime
    confidence: float = 0.0      # 0-1 confidence score
    compression_score: float = 0.0
    expansion_score: float = 0.0
    chop_score: float = 0.0
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "compression_score": self.compression_score,
            "expansion_score": self.expansion_score,
            "chop_score": self.chop_score,
            "reason": self.reason,
        }
