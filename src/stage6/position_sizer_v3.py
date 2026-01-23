"""
Stage 6: Position Sizer V3 (Simplified for Hybrid Alpha)
=========================================================

Simplified position sizing for Stage 3 V3 hybrid signals:
- Signal provides stop_pct and target_pct
- Risk = 2% of equity per trade
- Single position (no tranches)
- Notional = Risk / stop_pct
- Margin = Notional / 20 (20x leverage)

No ATR calculation here - the signal already determined stop/target.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


# Constants
FEE_PCT = 0.0008              # 0.08% per side (0.16% round-trip)
RISK_PCT = 0.02               # 2% risk per trade
LEVERAGE = 20                 # 20x leverage
MIN_NOTIONAL_USD = 10.0       # Minimum notional
MAX_NOTIONAL_USD = 50000.0    # Maximum notional per trade


@dataclass
class PositionV3:
    """Single position from V3 signal"""
    symbol: str
    side: str  # "LONG" or "SHORT"
    entry_price: float
    size: float  # In base asset
    notional: float  # In USD
    margin: float  # Required margin
    
    stop_price: float
    stop_pct: float
    target_price: float
    target_pct: float
    
    breakeven_price: float  # Entry + 0.8% for trail
    trail_1r_price: float   # 1R profit level for trail
    
    risk_amount: float  # USD at risk
    signal_type: str
    signal_name: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "size": self.size,
            "notional": round(self.notional, 2),
            "margin": round(self.margin, 2),
            "stop_price": self.stop_price,
            "stop_pct": round(self.stop_pct * 100, 3),
            "target_price": self.target_price,
            "target_pct": round(self.target_pct * 100, 3),
            "breakeven_price": self.breakeven_price,
            "trail_1r_price": self.trail_1r_price,
            "risk_amount": round(self.risk_amount, 2),
            "signal_type": self.signal_type,
            "signal_name": self.signal_name,
            "timestamp": self.timestamp,
        }


@dataclass
class PositionResultV3:
    """Result of position sizing"""
    symbol: str
    allowed: bool
    rejection_reason: str = ""
    position: Optional[PositionV3] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "allowed": self.allowed,
            "rejection_reason": self.rejection_reason,
            "position": self.position.to_dict() if self.position else None,
        }


# Step sizes for rounding (from Binance)
STEP_SIZES = {
    "BTCUSDT": 0.001,
    "ETHUSDT": 0.001,
    "BNBUSDT": 0.1,
    "SOLUSDT": 0.1,
    "XRPUSDT": 0.1,
    "DOGEUSDT": 100.0,
    "ADAUSDT": 1.0,
    "LTCUSDT": 0.1,
}


def get_step_size(symbol: str) -> float:
    """Get step size for symbol"""
    return STEP_SIZES.get(symbol, 0.001)


def round_to_step(value: float, step: float) -> float:
    """Round value to step size"""
    if step <= 0:
        return value
    return round(value / step) * step


class PositionSizerV3:
    """
    Simplified position sizer for Stage 3 V3 hybrid signals.
    
    Key changes from V2:
    - Signal provides stop_pct and target_pct (no ATR calculation)
    - Single position (no tranches A/B)
    - Fixed 2% risk per trade
    - Simple trail logic: breakeven at +0.8%, trail at 1R
    """
    
    def __init__(
        self,
        risk_pct: float = RISK_PCT,
        fee_pct: float = FEE_PCT,
        leverage: int = LEVERAGE,
    ):
        self.risk_pct = risk_pct
        self.fee_pct = fee_pct
        self.leverage = leverage
        
        # Current equity (updated by trade manager)
        self._current_equity: float = 1000.0
        
        # Active positions per symbol
        self._active_positions: Dict[str, PositionV3] = {}
        
        # Holding state - no new trades if position exists
        self._holding: Dict[str, bool] = {}
        
        logger.info(
            "position_sizer_v3_initialized",
            risk_pct=f"{risk_pct*100:.1f}%",
            fee_pct=f"{fee_pct*100:.2f}%",
            leverage=leverage,
        )
    
    def set_equity(self, equity: float):
        """Update current equity"""
        self._current_equity = equity
    
    def calculate_position(
        self,
        symbol: str,
        side: str,  # "LONG" or "SHORT"
        entry_price: float,
        stop_pct: float,  # From signal (e.g., 0.02 for 2%)
        target_pct: float,  # From signal (e.g., 0.04 for 4%)
        signal_type: str,
        signal_name: str,
    ) -> PositionResultV3:
        """
        Calculate position size from signal parameters.
        
        Args:
            symbol: Trading pair
            side: "LONG" or "SHORT"
            entry_price: Entry price
            stop_pct: Stop loss percentage (from signal)
            target_pct: Target percentage (from signal)
            signal_type: Type of signal (e.g., "FUNDING_TREND")
            signal_name: Full signal name/reason
        
        Returns:
            PositionResultV3 with position or rejection
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Check if already holding
        if self._holding.get(symbol, False):
            return PositionResultV3(
                symbol=symbol,
                allowed=False,
                rejection_reason="HOLDING - position already active",
            )
        
        # Validate stop_pct
        min_stop = 2 * self.fee_pct  # Minimum stop > round-trip fees
        if stop_pct < min_stop:
            return PositionResultV3(
                symbol=symbol,
                allowed=False,
                rejection_reason=f"Stop too tight: {stop_pct*100:.2f}% < min {min_stop*100:.2f}%",
            )
        
        # Calculate risk amount (2% of equity)
        risk_amount = self._current_equity * self.risk_pct
        
        # Calculate position size
        # Risk = Notional * (stop_pct + fees)
        # Notional = Risk / (stop_pct + fees)
        effective_stop = stop_pct + 2 * self.fee_pct  # Include round-trip fees
        notional = risk_amount / effective_stop
        
        # Apply limits
        if notional < MIN_NOTIONAL_USD:
            return PositionResultV3(
                symbol=symbol,
                allowed=False,
                rejection_reason=f"Notional too small: ${notional:.2f} < ${MIN_NOTIONAL_USD}",
            )
        
        notional = min(notional, MAX_NOTIONAL_USD)
        
        # Calculate size in base asset
        raw_size = notional / entry_price
        step_size = get_step_size(symbol)
        size = round_to_step(raw_size, step_size)
        
        # Recalculate notional with rounded size
        actual_notional = size * entry_price
        
        # Calculate margin
        margin = actual_notional / self.leverage
        
        # Calculate stop and target prices
        if side == "LONG":
            stop_price = entry_price * (1 - stop_pct)
            target_price = entry_price * (1 + target_pct)
            breakeven_price = entry_price * (1 + 0.008)  # +0.8% for breakeven trail
            trail_1r_price = entry_price * (1 + stop_pct)  # 1R profit
        else:  # SHORT
            stop_price = entry_price * (1 + stop_pct)
            target_price = entry_price * (1 - target_pct)
            breakeven_price = entry_price * (1 - 0.008)  # -0.8% for breakeven trail
            trail_1r_price = entry_price * (1 - stop_pct)  # 1R profit
        
        position = PositionV3(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            size=size,
            notional=actual_notional,
            margin=margin,
            stop_price=stop_price,
            stop_pct=stop_pct,
            target_price=target_price,
            target_pct=target_pct,
            breakeven_price=breakeven_price,
            trail_1r_price=trail_1r_price,
            risk_amount=risk_amount,
            signal_type=signal_type,
            signal_name=signal_name,
            timestamp=timestamp,
        )
        
        # Store position and set holding
        self._active_positions[symbol] = position
        self._holding[symbol] = True
        
        logger.info(
            "position_calculated_v3",
            symbol=symbol,
            side=side,
            entry=f"${entry_price:.4f}",
            size=size,
            notional=f"${actual_notional:.2f}",
            stop_pct=f"{stop_pct*100:.2f}%",
            target_pct=f"{target_pct*100:.2f}%",
            risk=f"${risk_amount:.2f}",
            signal=signal_type,
        )
        
        return PositionResultV3(
            symbol=symbol,
            allowed=True,
            position=position,
        )
    
    def get_position(self, symbol: str) -> Optional[PositionV3]:
        """Get active position for symbol"""
        return self._active_positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PositionV3]:
        """Get all active positions"""
        return self._active_positions.copy()
    
    def is_holding(self, symbol: str) -> bool:
        """Check if symbol has active position"""
        return self._holding.get(symbol, False)
    
    def add_hold(self, symbol: str):
        """Mark symbol as holding (position opened)"""
        self._holding[symbol] = True
        logger.debug("position_hold_added", symbol=symbol)
    
    def release_hold(self, symbol: str):
        """Release hold on symbol (position closed)"""
        if symbol in self._holding:
            del self._holding[symbol]
        logger.debug("position_hold_released", symbol=symbol)
    
    def clear_position(self, symbol: str):
        """Clear position when closed"""
        if symbol in self._active_positions:
            del self._active_positions[symbol]
        if symbol in self._holding:
            del self._holding[symbol]
        logger.info("position_cleared_v3", symbol=symbol)
    
    def get_holding_symbols(self) -> List[str]:
        """Get list of symbols with active positions"""
        return [s for s, h in self._holding.items() if h]
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics"""
        return {
            "current_equity": self._current_equity,
            "risk_pct": self.risk_pct,
            "active_positions": len(self._active_positions),
            "holding_symbols": self.get_holding_symbols(),
            "total_notional": sum(p.notional for p in self._active_positions.values()),
            "total_margin_used": sum(p.margin for p in self._active_positions.values()),
        }
