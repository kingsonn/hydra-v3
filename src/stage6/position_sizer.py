"""
Stage 6: Position Sizer

Calculates position sizes with two tranches (A and B) for risk management.
- Tranche A: Tight stop (0.6x ATR_5m), single TP at 1R
- Tranche B: Wide stop (1.2x ATR_5m), partial TP at 2R (40%), runner TP at 3R

Rejects trades if ATR_1h_pct < min_stop_pct (2 * fee_pct = 0.32%)
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
import structlog

from src.stage6.models import (
    Position,
    PositionResult,
    RejectionResult,
    get_step_size,
    round_to_step,
)

logger = structlog.get_logger(__name__)


# -------- CONSTANTS --------
FEE_PCT = 0.0008                  # Futures trading fee (0.08% round-trip)
MIN_STOP_PCT = 2 * FEE_PCT        # 0.16% - minimum stop distance

TIGHT_MULT = 1.0                  # Tranche A multiplier
WIDE_MULT = 1.8                   # Tranche B multiplier

TP_A_R = 1.0                      # Tranche A full TP (1R profit after fees)
TP_B_PARTIAL_R = 2.0              # Tranche B partial TP (2R)
TP_B_PARTIAL_PCT = 0.4            # % of Tranche B size to close at partial TP
TP_B_RUNNER_R = 3.0               # Tranche B runner TP (3R)

DEFAULT_TOTAL_RISK = 12.0         # Default fallback risk in USD

# Dynamic risk based on percentile thresholds
RISK_NORMAL_PCT = 0.015           # 1.5% at percentile >= 85
RISK_HIGH_EDGE_PCT = 0.025        # 2.5% at percentile >= 92
RISK_MAX_PCT = 0.030              # 3.0% at percentile >= 96

PERCENTILE_NORMAL = 85.0
PERCENTILE_HIGH_EDGE = 92.0
PERCENTILE_MAX = 96.0


class PositionSizer:
    """
    Stage 6 Position Sizer
    
    Takes Stage 5 approved signals and calculates:
    - Two tranche positions (A and B)
    - Stop losses based on ATR_5m
    - Take profit levels
    - Position sizes based on risk allocation
    """
    
    def __init__(
        self,
        total_risk_dollars: float = DEFAULT_TOTAL_RISK,
        fee_pct: float = FEE_PCT,
    ):
        self.total_risk_dollars = total_risk_dollars
        self.fee_pct = fee_pct
        self.min_stop_pct = 2 * fee_pct
        
        # Track active positions per symbol
        self._active_positions: Dict[str, PositionResult] = {}
        
        # Track rejections per symbol
        self._rejections: Dict[str, RejectionResult] = {}
        
        # Track holding state - symbols with open positions (no new trades allowed)
        self._holding: Dict[str, bool] = {}
        
        # Current equity for dynamic risk (updated by trade manager)
        self._current_equity: float = 1000.0
        
        logger.info(
            "position_sizer_initialized",
            total_risk=total_risk_dollars,
            fee_pct=f"{fee_pct:.4f}",
            min_stop_pct=f"{self.min_stop_pct:.4f}",
        )
    
    def set_equity(self, equity: float) -> None:
        """Update current equity for dynamic risk calculation"""
        self._current_equity = equity
    
    def calculate_dynamic_risk(self, percentile_300: float) -> float:
        """
        Calculate risk amount based on equity and ML percentile
        
        Args:
            percentile_300: ML model's 300s percentile score
            
        Returns:
            Dollar amount to risk on this trade
        """
        # Determine risk percentage based on percentile
        if percentile_300 >= PERCENTILE_MAX:
            risk_pct = RISK_MAX_PCT
        elif percentile_300 >= PERCENTILE_HIGH_EDGE:
            risk_pct = RISK_HIGH_EDGE_PCT
        elif percentile_300 >= PERCENTILE_NORMAL:
            risk_pct = RISK_NORMAL_PCT
        else:
            # Below threshold, use minimum
            risk_pct = RISK_NORMAL_PCT
        
        risk_amount = self._current_equity * risk_pct
        
        logger.debug(
            "dynamic_risk_calculated",
            equity=f"${self._current_equity:.2f}",
            percentile=f"{percentile_300:.1f}",
            risk_pct=f"{risk_pct*100:.1f}%",
            risk_amount=f"${risk_amount:.2f}",
        )
        
        return risk_amount
    
    def calculate_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        atr_5m_pct: float,
        atr_1h_pct: float,
        signal_name: str,
        current_price: Optional[float] = None,
        total_risk_dollars: Optional[float] = None,
        percentile_300: Optional[float] = None,
    ) -> PositionResult:
        """
        Calculate position sizes for a Stage 5 approved signal.
        
        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: "LONG" or "SHORT"
            entry_price: Entry price for the position
            atr_5m_pct: 5-minute ATR as percentage (e.g., 0.005 for 0.5%)
            atr_1h_pct: 1-hour ATR as percentage (e.g., 0.01 for 1%)
            signal_name: Name of the signal that generated this trade
            current_price: Current market price (optional, for reference)
            total_risk_dollars: Override default total risk (takes priority)
            percentile_300: ML model's 300s percentile for dynamic risk sizing
        
        Returns:
            PositionResult with two tranche positions or rejection
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Determine risk: explicit override > dynamic (percentile) > default
        if total_risk_dollars is not None:
            risk = total_risk_dollars
        elif percentile_300 is not None:
            risk = self.calculate_dynamic_risk(percentile_300)
        else:
            risk = self.total_risk_dollars
        
        # -------- PERMISSION CHECK --------
        if atr_1h_pct < self.min_stop_pct:
            rejection = RejectionResult(
                symbol=symbol,
                reason=f"ATR_1h ({atr_1h_pct*100:.3f}%) < min_stop ({self.min_stop_pct*100:.3f}%)",
                atr_1h_pct=atr_1h_pct,
                min_required_pct=self.min_stop_pct,
                signal_name=signal_name,
                timestamp=timestamp,
            )
            self._rejections[symbol] = rejection
            
            logger.warning(
                "position_rejected_low_atr",
                symbol=symbol,
                atr_1h_pct=f"{atr_1h_pct*100:.3f}%",
                min_required=f"{self.min_stop_pct*100:.3f}%",
            )
            
            return PositionResult(
                symbol=symbol,
                allowed=False,
                rejection_reason=rejection.reason,
                side=side,
                entry_price=entry_price,
                atr_5m_pct=atr_5m_pct,
                atr_1h_pct=atr_1h_pct,
                total_risk=risk,
                signal_name=signal_name,
                timestamp=timestamp,
            )
        
        # -------- CALCULATE TRANCHES --------
        tranches = [
            {"name": "A", "mult": TIGHT_MULT, "risk": risk * 0.40},  # 40% risk
            {"name": "B", "mult": WIDE_MULT, "risk": risk * 0.60},   # 60% risk (runner)
        ]
        
        positions: List[Position] = []
        step_size = get_step_size(symbol)
        
        for t in tranches:
            raw_stop_pct = t["mult"] * atr_5m_pct
            tranche_risk = t["risk"]  # This is 1R for this tranche
            
            # Calculate position size based on risk
            # risk_pct = stop_loss_pct + fee_pct
            effective_r_pct = raw_stop_pct + self.fee_pct
            position_notional = tranche_risk / effective_r_pct
            
            # Calculate size in base asset and round to step size
            raw_size = position_notional / entry_price
            size = round_to_step(raw_size, step_size)
            
            # Recalculate notional with rounded size
            actual_notional = size * entry_price
            
            # Calculate fees (entry + exit)
            total_fees = actual_notional * self.fee_pct
            
            # Calculate TP prices based on R profit in dollars
            # For n*R profit: raw_pnl_needed = n * risk + total_fees
            # price_diff = raw_pnl_needed / size
            def calc_tp_price(r_mult: float) -> float:
                raw_pnl_needed = r_mult * tranche_risk + total_fees
                price_diff = raw_pnl_needed / size if size > 0 else 0
                if side == "LONG":
                    return entry_price + price_diff
                else:  # SHORT
                    return entry_price - price_diff
            
            tp_a_price = calc_tp_price(TP_A_R)
            tp_b_partial_price = calc_tp_price(TP_B_PARTIAL_R)
            tp_b_runner_price = calc_tp_price(TP_B_RUNNER_R)
            
            # Stop loss and breakeven
            if side == "LONG":
                stop_price = entry_price * (1 - raw_stop_pct)
                breakeven_price = entry_price * (1 + self.fee_pct)  # Cover round-trip fees
            else:  # SHORT
                stop_price = entry_price * (1 + raw_stop_pct)
                breakeven_price = entry_price * (1 - self.fee_pct)  # Cover round-trip fees
            
            position = Position(
                tranche=t["name"],
                side=side,
                symbol=symbol,
                entry=entry_price,
                stop=stop_price,
                size=size,
                breakeven=breakeven_price,
                notional=actual_notional,
                risk=tranche_risk,
                raw_stop_pct=raw_stop_pct,
                tp_a=tp_a_price,
                tp_b_partial=tp_b_partial_price,
                tp_b_runner=tp_b_runner_price,
                closed_pct=0.0,
                signal_name=signal_name,
                timestamp=timestamp,
            )
            positions.append(position)
        
        result = PositionResult(
            symbol=symbol,
            allowed=True,
            positions=positions,
            side=side,
            entry_price=entry_price,
            atr_5m_pct=atr_5m_pct,
            atr_1h_pct=atr_1h_pct,
            total_risk=risk,
            signal_name=signal_name,
            timestamp=timestamp,
        )
        
        # Store active position
        self._active_positions[symbol] = result
        
        # Set holding state - no new trades for this symbol
        self._holding[symbol] = True
        
        # Clear any previous rejection
        if symbol in self._rejections:
            del self._rejections[symbol]
        
        logger.info(
            "position_calculated",
            symbol=symbol,
            side=side,
            entry=f"${entry_price:.4f}",
            signal=signal_name,
            tranche_a_size=positions[0].size,
            tranche_b_size=positions[1].size,
            total_notional=f"${result.get_total_notional():.2f}",
        )
        
        return result
    
    def get_active_position(self, symbol: str) -> Optional[PositionResult]:
        """Get active position for a symbol"""
        return self._active_positions.get(symbol)
    
    def get_all_active_positions(self) -> Dict[str, PositionResult]:
        """Get all active positions"""
        return self._active_positions.copy()
    
    def get_rejection(self, symbol: str) -> Optional[RejectionResult]:
        """Get rejection for a symbol"""
        return self._rejections.get(symbol)
    
    def get_all_rejections(self) -> Dict[str, RejectionResult]:
        """Get all rejections"""
        return self._rejections.copy()
    
    def clear_position(self, symbol: str) -> None:
        """Clear position for a symbol (when trade is closed)"""
        if symbol in self._active_positions:
            del self._active_positions[symbol]
        # Also clear holding state
        if symbol in self._holding:
            del self._holding[symbol]
    
    def clear_rejection(self, symbol: str) -> None:
        """Clear rejection for a symbol"""
        if symbol in self._rejections:
            del self._rejections[symbol]
    
    def is_holding(self, symbol: str) -> bool:
        """Check if symbol is in holding state (has active position)"""
        return self._holding.get(symbol, False)
    
    def get_holding_symbols(self) -> List[str]:
        """Get list of symbols currently in holding state"""
        return [s for s, h in self._holding.items() if h]
    
    def release_holding(self, symbol: str) -> None:
        """Release holding state for a symbol (manual override)"""
        if symbol in self._holding:
            del self._holding[symbol]
        if symbol in self._active_positions:
            del self._active_positions[symbol]
        logger.info("holding_released", symbol=symbol)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get position sizer health metrics"""
        return {
            "active_positions": len(self._active_positions),
            "holding_count": len(self._holding),
            "rejections": len(self._rejections),
            "total_risk_per_trade": self.total_risk_dollars,
            "fee_pct": self.fee_pct,
            "min_stop_pct": self.min_stop_pct,
            "symbols_with_positions": list(self._active_positions.keys()),
            "symbols_holding": self.get_holding_symbols(),
            "symbols_rejected": list(self._rejections.keys()),
        }
