"""
Risk Manager Module

Handles:
1. Dynamic risk sizing based on equity and ML percentile
2. Drawdown limits (-6% pause 4h, -7% flatten + disable 8h)
3. Daily profit protection (+12% pause 8h)
4. Trade history tracking
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field, asdict
from collections import deque
import structlog

logger = structlog.get_logger(__name__)

# Risk sizing based on percentile
RISK_NORMAL_PCT = 0.015      # 1.5% at percentile >= 85
RISK_HIGH_EDGE_PCT = 0.025   # 2.5% at percentile >= 92
RISK_MAX_PCT = 0.030         # 3.0% at percentile >= 96

PERCENTILE_NORMAL = 85.0
PERCENTILE_HIGH_EDGE = 92.0
PERCENTILE_MAX = 96.0

# Drawdown limits
DRAWDOWN_PAUSE_PCT = -0.06   # -6% daily drawdown = pause 4 hours
DRAWDOWN_HARD_STOP_PCT = -0.07  # -7% = flatten + disable 8 hours
PAUSE_DURATION_HOURS = 4
HARD_STOP_DURATION_HOURS = 8

# Profit protection
PROFIT_PROTECTION_PCT = 0.12  # +12% = pause 8 hours
PROFIT_PAUSE_HOURS = 8


@dataclass
class ClosedTrade:
    """Record of a closed trade"""
    order_id: str
    symbol: str
    side: str
    tranche: str
    entry_price: float
    close_price: float
    size: float
    realized_pnl: float
    r_multiple: float
    close_reason: str
    signal_name: str
    opened_at: str
    closed_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: str
    starting_equity: float
    current_equity: float
    high_water_mark: float
    realized_pnl: float
    unrealized_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    @property
    def daily_return_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return (self.current_equity - self.starting_equity) / self.starting_equity
    
    @property
    def drawdown_pct(self) -> float:
        if self.high_water_mark <= 0:
            return 0.0
        return (self.current_equity - self.high_water_mark) / self.high_water_mark


class RiskManager:
    """
    Manages risk sizing and trading limits
    """
    
    def __init__(self, initial_equity: float = 1000.0):
        self._initial_equity = initial_equity
        self._current_equity = initial_equity
        self._high_water_mark = initial_equity
        
        # Daily tracking
        self._daily_start_equity = initial_equity
        self._daily_start_date = datetime.now().date().isoformat()
        
        # Trading state
        self._trading_paused = False
        self._pause_until: Optional[float] = None
        self._pause_reason: Optional[str] = None
        self._hard_stopped = False
        
        # Closed trades history (last 50)
        self._closed_trades: deque = deque(maxlen=50)
        
        # Daily stats
        self._daily_stats = DailyStats(
            date=self._daily_start_date,
            starting_equity=initial_equity,
            current_equity=initial_equity,
            high_water_mark=initial_equity,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )
        
        logger.info(
            "risk_manager_initialized",
            initial_equity=initial_equity,
        )
    
    def update_equity(self, current_equity: float, unrealized_pnl: float = 0.0) -> None:
        """Update current equity and check limits"""
        self._current_equity = current_equity
        self._daily_stats.current_equity = current_equity
        self._daily_stats.unrealized_pnl = unrealized_pnl
        
        # Update high water mark
        if current_equity > self._high_water_mark:
            self._high_water_mark = current_equity
            self._daily_stats.high_water_mark = current_equity
        
        # Check for new day
        today = datetime.now().date().isoformat()
        if today != self._daily_start_date:
            self._reset_daily_stats(current_equity)
        
        # Check pause expiry
        if self._pause_until and time.time() >= self._pause_until:
            self._trading_paused = False
            self._pause_until = None
            self._pause_reason = None
            self._hard_stopped = False
            logger.info("trading_pause_expired")
        
        # Check drawdown limits
        self._check_drawdown_limits()
        
        # Check profit protection
        self._check_profit_protection()
    
    def _reset_daily_stats(self, current_equity: float) -> None:
        """Reset daily stats for new day"""
        self._daily_start_date = datetime.now().date().isoformat()
        self._daily_start_equity = current_equity
        self._daily_stats = DailyStats(
            date=self._daily_start_date,
            starting_equity=current_equity,
            current_equity=current_equity,
            high_water_mark=current_equity,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
        )
        logger.info("daily_stats_reset", date=self._daily_start_date)
    
    def _check_drawdown_limits(self) -> None:
        """Check and enforce drawdown limits"""
        if self._trading_paused:
            return
        
        daily_return = self._daily_stats.daily_return_pct
        
        # Hard stop at -7%
        if daily_return <= DRAWDOWN_HARD_STOP_PCT:
            self._trading_paused = True
            self._hard_stopped = True
            self._pause_until = time.time() + (HARD_STOP_DURATION_HOURS * 3600)
            self._pause_reason = f"HARD_STOP: {daily_return*100:.1f}% drawdown"
            logger.warning(
                "hard_stop_triggered",
                drawdown_pct=f"{daily_return*100:.1f}%",
                pause_hours=HARD_STOP_DURATION_HOURS,
            )
            return
        
        # Pause at -6%
        if daily_return <= DRAWDOWN_PAUSE_PCT:
            self._trading_paused = True
            self._pause_until = time.time() + (PAUSE_DURATION_HOURS * 3600)
            self._pause_reason = f"DRAWDOWN_PAUSE: {daily_return*100:.1f}% drawdown"
            logger.warning(
                "drawdown_pause_triggered",
                drawdown_pct=f"{daily_return*100:.1f}%",
                pause_hours=PAUSE_DURATION_HOURS,
            )
    
    def _check_profit_protection(self) -> None:
        """Check and enforce profit protection"""
        if self._trading_paused:
            return
        
        daily_return = self._daily_stats.daily_return_pct
        
        # Pause at +12%
        if daily_return >= PROFIT_PROTECTION_PCT:
            self._trading_paused = True
            self._pause_until = time.time() + (PROFIT_PAUSE_HOURS * 3600)
            self._pause_reason = f"PROFIT_PROTECTION: +{daily_return*100:.1f}% profit"
            logger.info(
                "profit_protection_triggered",
                profit_pct=f"+{daily_return*100:.1f}%",
                pause_hours=PROFIT_PAUSE_HOURS,
            )
    
    def calculate_risk_amount(self, percentile_300: float) -> float:
        """
        Calculate risk amount based on current equity and ML percentile
        
        Returns dollar amount to risk on this trade
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
            "risk_calculated",
            equity=f"${self._current_equity:.2f}",
            percentile=f"{percentile_300:.1f}",
            risk_pct=f"{risk_pct*100:.1f}%",
            risk_amount=f"${risk_amount:.2f}",
        )
        
        return risk_amount
    
    def can_trade(self) -> tuple:
        """
        Check if trading is allowed
        
        Returns (can_trade: bool, reason: Optional[str])
        """
        if self._trading_paused:
            remaining = ""
            if self._pause_until:
                remaining_s = self._pause_until - time.time()
                if remaining_s > 0:
                    remaining = f" ({remaining_s/3600:.1f}h remaining)"
            return False, f"{self._pause_reason}{remaining}"
        
        return True, None
    
    def should_flatten_all(self) -> bool:
        """Check if we should flatten all positions (hard stop triggered)"""
        return self._hard_stopped
    
    def record_closed_trade(
        self,
        order_id: str,
        symbol: str,
        side: str,
        tranche: str,
        entry_price: float,
        close_price: float,
        size: float,
        realized_pnl: float,
        r_multiple: float,
        close_reason: str,
        signal_name: str,
        opened_at: str,
    ) -> None:
        """Record a closed trade"""
        closed_trade = ClosedTrade(
            order_id=order_id,
            symbol=symbol,
            side=side,
            tranche=tranche,
            entry_price=entry_price,
            close_price=close_price,
            size=size,
            realized_pnl=realized_pnl,
            r_multiple=r_multiple,
            close_reason=close_reason,
            signal_name=signal_name,
            opened_at=opened_at,
            closed_at=datetime.now().isoformat(),
        )
        self._closed_trades.append(closed_trade)
        
        # Update daily stats
        self._daily_stats.total_trades += 1
        self._daily_stats.realized_pnl += realized_pnl
        if realized_pnl > 0:
            self._daily_stats.winning_trades += 1
        else:
            self._daily_stats.losing_trades += 1
        
        logger.info(
            "trade_recorded",
            symbol=symbol,
            tranche=tranche,
            pnl=f"${realized_pnl:.2f}",
            r=f"{r_multiple:.2f}R",
            reason=close_reason,
        )
    
    def get_closed_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent closed trades"""
        trades = list(self._closed_trades)[-limit:]
        return [t.to_dict() for t in reversed(trades)]  # Most recent first
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get current daily stats"""
        return {
            "date": self._daily_stats.date,
            "starting_equity": self._daily_stats.starting_equity,
            "current_equity": self._daily_stats.current_equity,
            "high_water_mark": self._daily_stats.high_water_mark,
            "realized_pnl": self._daily_stats.realized_pnl,
            "unrealized_pnl": self._daily_stats.unrealized_pnl,
            "daily_return_pct": self._daily_stats.daily_return_pct * 100,
            "drawdown_pct": self._daily_stats.drawdown_pct * 100,
            "total_trades": self._daily_stats.total_trades,
            "winning_trades": self._daily_stats.winning_trades,
            "losing_trades": self._daily_stats.losing_trades,
            "trading_paused": self._trading_paused,
            "pause_reason": self._pause_reason,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get risk manager status"""
        can_trade, reason = self.can_trade()
        return {
            "current_equity": self._current_equity,
            "high_water_mark": self._high_water_mark,
            "daily_return_pct": self._daily_stats.daily_return_pct * 100,
            "can_trade": can_trade,
            "pause_reason": reason,
            "hard_stopped": self._hard_stopped,
        }
