"""
Stage 3 Data Models - Signal and Thesis Objects
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class Direction(Enum):
    """Trading direction"""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass(slots=True)
class Signal:
    """
    Individual thesis signal
    
    Represents a single detected market pressure or condition
    that suggests directional bias
    """
    direction: Direction
    confidence: float  # 0.0 - 1.0
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction.value,
            "confidence": self.confidence,
            "name": self.name,
        }


@dataclass
class Thesis:
    """
    Stage 3 output - Trading thesis
    
    Aggregated view of all active signals with directional bias
    Everything after Stage 3 must respect this thesis
    """
    allowed: bool                           # Whether trading is allowed
    direction: Direction                    # LONG, SHORT, or NONE
    strength: float                         # 0.0 - 1.0 combined strength
    reasons: List[Signal] = field(default_factory=list)  # Active signals
    veto_reason: Optional[str] = None       # If allowed=False, why
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "direction": self.direction.value,
            "strength": self.strength,
            "reasons": [s.to_dict() for s in self.reasons],
            "veto_reason": self.veto_reason,
            "signal_count": len(self.reasons),
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattened dict for UI display"""
        return {
            "thesis_allowed": self.allowed,
            "thesis_direction": self.direction.value,
            "thesis_strength": self.strength,
            "thesis_veto_reason": self.veto_reason or "",
            "thesis_signal_count": len(self.reasons),
            "thesis_signals": ", ".join(s.name for s in self.reasons),
            "thesis_long_signals": ", ".join(
                s.name for s in self.reasons if s.direction == Direction.LONG
            ),
            "thesis_short_signals": ", ".join(
                s.name for s in self.reasons if s.direction == Direction.SHORT
            ),
        }


@dataclass
class ThesisState:
    """
    Complete state for Stage 3 processing
    Combines MarketState inputs with computed values
    """
    # From MarketState
    symbol: str
    price: float
    regime: str  # CHOP, COMPRESSION, EXPANSION
    
    # Funding
    funding_z: float = 0.0
    
    # OI
    oi_delta_5m: float = 0.0
    oi_delta_15m: float = 0.0
    
    # Liquidations
    liq_imbalance: float = 0.0
    
    # Price changes
    price_change_5m: float = 0.0
    price_change_15m: float = 0.0
    
    # Volatility
    vol_regime: str = "MID"
    vol_rank: float = 0.0
    
    # Time tracking
    time_in_regime: float = 0.0  # seconds
    
    # Absorption (for liquidation exhaustion)
    absorption_z: float = 0.0
    
    # Structure (for Stage 4 filtering)
    dist_lvn: float = 0.0        # Distance to LVN in ATR units
    vah: float = 0.0             # Value Area High (30m)
    val: float = 0.0             # Value Area Low (30m)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "regime": self.regime,
            "funding_z": self.funding_z,
            "oi_delta_5m": self.oi_delta_5m,
            "oi_delta_15m": self.oi_delta_15m,
            "liq_imbalance": self.liq_imbalance,
            "price_change_5m": self.price_change_5m,
            "price_change_15m": self.price_change_15m,
            "vol_regime": self.vol_regime,
            "vol_rank": self.vol_rank,
            "time_in_regime": self.time_in_regime,
            "absorption_z": self.absorption_z,
            "dist_lvn": self.dist_lvn,
            "vah": self.vah,
            "val": self.val,
        }
