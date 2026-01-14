"""
Stage 6 Data Models - Position Sizing Objects
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


# Step sizes for each symbol (precision for position size)
SYMBOL_STEP_SIZES: Dict[str, float] = {
    "BTCUSDT": 0.001,
    "ETHUSDT": 0.001,
    "SOLUSDT": 0.1,
    "BNBUSDT": 0.01,
    "XRPUSDT": 0.1,
    "DOGEUSDT": 1.0,
    "ADAUSDT": 0.1,
    "LTCUSDT": 0.001,
}


def get_step_size(symbol: str) -> float:
    """Get step size for a symbol, default to 0.001"""
    return SYMBOL_STEP_SIZES.get(symbol, 0.001)


def round_to_step(value: float, step_size: float) -> float:
    """Round value to nearest step size with proper precision handling"""
    # Calculate number of decimal places from step_size
    if step_size >= 1:
        decimals = 0
    else:
        decimals = len(str(step_size).rstrip('0').split('.')[-1])
    
    # Round to step size and then round to proper decimals to avoid floating point errors
    result = round(value / step_size) * step_size
    return round(result, decimals)


@dataclass
class Position:
    """
    Single tranche position with entry, stop, and TP levels.
    """
    tranche: str                    # "A" or "B"
    side: str                       # "LONG" or "SHORT"
    symbol: str                     # e.g., "BTCUSDT"
    entry: float                    # Entry price
    stop: float                     # Stop loss price
    size: float                     # Position size (in base asset)
    breakeven: float                # Breakeven price (entry + fees)
    notional: float                 # Position notional value in USD
    risk: float                     # Risk in USD for this tranche
    raw_stop_pct: float             # Raw stop distance as percentage
    
    # Take profit levels
    tp_a: float                     # Tranche A TP (1R)
    tp_b_partial: float             # Tranche B partial TP (2R)
    tp_b_runner: float              # Tranche B runner TP (3R)
    
    # Tracking
    closed_pct: float = 0.0         # Percentage of position closed
    signal_name: str = ""           # Signal that generated this position
    timestamp: str = ""             # Creation time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tranche": self.tranche,
            "side": self.side,
            "symbol": self.symbol,
            "entry": self.entry,
            "stop": self.stop,
            "size": self.size,
            "breakeven": self.breakeven,
            "notional": self.notional,
            "risk": self.risk,
            "raw_stop_pct": self.raw_stop_pct,
            "tp_a": self.tp_a,
            "tp_b_partial": self.tp_b_partial,
            "tp_b_runner": self.tp_b_runner,
            "closed_pct": self.closed_pct,
            "signal_name": self.signal_name,
            "timestamp": self.timestamp,
        }


@dataclass
class PositionResult:
    """
    Result from Stage 6 position sizing.
    Contains two tranche positions or a rejection.
    """
    symbol: str
    allowed: bool                               # True if position sizing succeeded
    positions: List[Position] = field(default_factory=list)  # Tranche A and B
    rejection_reason: Optional[str] = None      # If not allowed, why
    
    # Inputs used
    side: str = ""
    entry_price: float = 0.0
    atr_5m_pct: float = 0.0
    atr_1h_pct: float = 0.0
    total_risk: float = 0.0
    signal_name: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "allowed": self.allowed,
            "positions": [p.to_dict() for p in self.positions],
            "rejection_reason": self.rejection_reason,
            "side": self.side,
            "entry_price": self.entry_price,
            "atr_5m_pct": self.atr_5m_pct,
            "atr_1h_pct": self.atr_1h_pct,
            "total_risk": self.total_risk,
            "signal_name": self.signal_name,
            "timestamp": self.timestamp,
        }
    
    def get_total_size(self) -> float:
        """Get total position size across all tranches"""
        return sum(p.size for p in self.positions)
    
    def get_total_notional(self) -> float:
        """Get total notional value across all tranches"""
        return sum(p.notional for p in self.positions)


@dataclass
class RejectionResult:
    """
    Result when Stage 6 rejects a trade due to insufficient ATR.
    """
    symbol: str
    reason: str
    atr_1h_pct: float
    min_required_pct: float
    signal_name: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "reason": self.reason,
            "atr_1h_pct": self.atr_1h_pct,
            "min_required_pct": self.min_required_pct,
            "signal_name": self.signal_name,
            "timestamp": self.timestamp,
        }
