"""
Trade Manager V3 (Simplified for Hybrid Alpha)
==============================================

Simplified trade management for Stage 3 V3 signals:
- Single position per symbol (no tranches)
- Exit on TP or SL hit only (no complex exit strategies)
- Trail stop to breakeven at +0.8%
- Trail stop to 1R profit level after 1R reached
- Single order placement (no batch)

Disabled features:
- Stage 5 ML gating
- Complex exit strategies (CHOP, POC, LVN, etc.)
- Tranche A/B splitting
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import structlog

from src.stage6.position_sizer_v3 import PositionV3, PositionResultV3

logger = structlog.get_logger(__name__)

# Constants
FEE_PCT = 0.0008
MARGIN_REQUIREMENT = 0.05  # 5% margin (20x leverage)
BREAKEVEN_TRIGGER_PCT = 0.008  # +0.8% to trigger breakeven trail


@dataclass
class TradeRecordV3:
    """Record of a trade for V3 system"""
    order_id: str
    symbol: str
    side: str
    entry_price: float
    size: float
    notional: float
    margin: float  # Margin used for this position
    risk_amount: float  # Actual $ risk (for correct R calculation)
    
    stop_price: float
    stop_pct: float
    target_price: float
    target_pct: float
    
    current_price: float = 0.0
    current_stop: float = 0.0  # Trailing stop
    
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    r_multiple: float = 0.0
    
    status: str = "open"  # open, closed_tp, closed_sl, closed_trail
    close_reason: str = ""
    close_price: float = 0.0
    
    signal_type: str = ""
    signal_name: str = ""
    created_at: str = ""
    closed_at: str = ""
    
    # Trail state
    breakeven_triggered: bool = False
    trail_1r_triggered: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "size": self.size,
            "notional": round(self.notional, 2),
            "margin": round(self.margin, 2),
            "risk_amount": round(self.risk_amount, 2),
            "stop_price": round(self.stop_price, 6),
            "stop_pct": round(self.stop_pct * 100, 2),
            "target_price": round(self.target_price, 6),
            "target_pct": round(self.target_pct * 100, 2),
            "current_price": round(self.current_price, 6),
            "current_stop": round(self.current_stop, 6),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "r_multiple": round(self.r_multiple, 2),
            "status": self.status,
            "close_reason": self.close_reason,
            "signal_type": self.signal_type,
            "signal_name": self.signal_name,
            "created_at": self.created_at,
            "closed_at": self.closed_at,
            "breakeven_triggered": self.breakeven_triggered,
            "trail_1r_triggered": self.trail_1r_triggered,
        }


@dataclass
class AccountStateV3:
    """Account state for V3 system"""
    initial_equity: float = 1000.0
    current_equity: float = 1000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 1000.0
    total_r: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_equity": self.initial_equity,
            "current_equity": round(self.current_equity, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "margin_used": round(self.margin_used, 2),
            "margin_available": round(self.margin_available, 2),
            "total_r": round(self.total_r, 2),
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "win_rate": round(self.win_rate() * 100, 1),
        }


class TradeManagerV3:
    """
    Simplified trade manager for V3 hybrid alpha system.
    
    Features:
    - Single position per symbol
    - TP/SL exit only
    - Trail to breakeven at +0.8%
    - Trail to 1R after 1R profit reached
    """
    
    def __init__(
        self,
        initial_equity: float = 1000.0,
        reset_on_start: bool = True,
    ):
        self.initial_equity = initial_equity
        
        # Account state
        self._account = AccountStateV3(
            initial_equity=initial_equity,
            current_equity=initial_equity,
            margin_available=initial_equity,
        )
        
        # Open trades per symbol
        self._open_trades: Dict[str, TradeRecordV3] = {}
        
        # Closed trades history
        self._closed_trades: List[TradeRecordV3] = []
        
        # Order ID counter
        self._order_counter = 0
        
        # Callbacks
        self._on_trade_closed: Optional[Callable] = None
        
        if reset_on_start:
            self._reset()
        
        logger.info(
            "trade_manager_v3_initialized",
            initial_equity=initial_equity,
        )
    
    def _reset(self):
        """Reset to initial state"""
        self._account = AccountStateV3(
            initial_equity=self.initial_equity,
            current_equity=self.initial_equity,
            margin_available=self.initial_equity,
        )
        self._open_trades.clear()
        self._closed_trades.clear()
        self._order_counter = 0
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self._order_counter += 1
        return f"V3_{int(time.time())}_{self._order_counter}"
    
    # ========== ORDER PLACEMENT ==========
    
    async def place_position(self, position: PositionV3) -> Optional[str]:
        """
        Place a new position from V3 signal.
        
        Returns:
            Order ID if successful, None if failed
        """
        symbol = position.symbol
        
        # Check if already holding
        if symbol in self._open_trades:
            logger.warning("position_already_exists", symbol=symbol)
            return None
        
        # Check margin
        required_margin = position.margin
        if required_margin > self._account.margin_available:
            logger.warning(
                "insufficient_margin",
                symbol=symbol,
                required=required_margin,
                available=self._account.margin_available,
            )
            return None
        
        # Create trade record
        order_id = self._generate_order_id()
        trade = TradeRecordV3(
            order_id=order_id,
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            size=position.size,
            notional=position.notional,
            margin=position.margin,  # Margin used for this position
            risk_amount=position.risk_amount,  # Actual $ risk for R calculation
            stop_price=position.stop_price,
            stop_pct=position.stop_pct,
            target_price=position.target_price,
            target_pct=position.target_pct,
            current_price=position.entry_price,
            current_stop=position.stop_price,
            signal_type=position.signal_type,
            signal_name=position.signal_name,
            created_at=datetime.now().isoformat(),
        )
        
        # Update account
        self._account.margin_used += required_margin
        self._account.margin_available -= required_margin
        
        # Store trade
        self._open_trades[symbol] = trade
        
        logger.info(
            "position_opened_v3",
            order_id=order_id,
            symbol=symbol,
            side=position.side,
            entry=f"${position.entry_price:.4f}",
            size=position.size,
            notional=f"${position.notional:.2f}",
            stop=f"${position.stop_price:.4f}",
            target=f"${position.target_price:.4f}",
        )
        
        return order_id
    
    # ========== PRICE UPDATES ==========
    
    async def update_price(self, symbol: str, current_price: float) -> Optional[Dict[str, Any]]:
        """
        Update price and check for exits.
        
        Returns:
            Trade closure info if trade closed, None otherwise
        """
        if symbol not in self._open_trades:
            return None
        
        trade = self._open_trades[symbol]
        trade.current_price = current_price
        
        # Calculate unrealized PnL
        if trade.side == "LONG":
            price_diff = current_price - trade.entry_price
        else:
            price_diff = trade.entry_price - current_price
        
        raw_pnl = price_diff * trade.size
        fees = trade.notional * FEE_PCT * 2  # Round-trip fees
        trade.unrealized_pnl = raw_pnl - fees
        
        # Calculate R-multiple
        risk_per_r = trade.notional * trade.stop_pct
        trade.r_multiple = trade.unrealized_pnl / risk_per_r if risk_per_r > 0 else 0
        
        # Update account unrealized PnL
        self._update_unrealized_pnl()
        
        # Check for trail triggers
        self._check_trail_triggers(trade, current_price)
        
        # Check for exits
        exit_result = self._check_exits(trade, current_price)
        
        if exit_result:
            return await self._close_trade(trade, exit_result["reason"], current_price)
        
        return None
    
    def _check_trail_triggers(self, trade: TradeRecordV3, current_price: float):
        """Check and update trailing stop"""
        is_long = trade.side == "LONG"
        
        if is_long:
            # Price distance from entry
            price_pct = (current_price - trade.entry_price) / trade.entry_price
            
            # Breakeven trail: trigger at +0.8%
            if not trade.breakeven_triggered and price_pct >= BREAKEVEN_TRIGGER_PCT:
                trade.breakeven_triggered = True
                # Move stop to entry + small buffer
                new_stop = trade.entry_price * 1.001  # Tiny profit on trail
                trade.current_stop = max(trade.current_stop, new_stop)
                logger.info(
                    "trail_breakeven_triggered",
                    symbol=trade.symbol,
                    new_stop=f"${new_stop:.4f}",
                )
            
            # 1R trail: trigger at 1R profit
            if not trade.trail_1r_triggered and trade.r_multiple >= 1.0:
                trade.trail_1r_triggered = True
                # Move stop to 0.5R profit level
                half_r_price = trade.entry_price * (1 + trade.stop_pct * 0.5)
                trade.current_stop = max(trade.current_stop, half_r_price)
                logger.info(
                    "trail_1r_triggered",
                    symbol=trade.symbol,
                    new_stop=f"${trade.current_stop:.4f}",
                    r_multiple=f"{trade.r_multiple:.2f}",
                )
        else:  # SHORT
            price_pct = (trade.entry_price - current_price) / trade.entry_price
            
            # Breakeven trail
            if not trade.breakeven_triggered and price_pct >= BREAKEVEN_TRIGGER_PCT:
                trade.breakeven_triggered = True
                new_stop = trade.entry_price * 0.999  # Tiny profit on trail
                trade.current_stop = min(trade.current_stop, new_stop)
                logger.info(
                    "trail_breakeven_triggered",
                    symbol=trade.symbol,
                    new_stop=f"${new_stop:.4f}",
                )
            
            # 1R trail
            if not trade.trail_1r_triggered and trade.r_multiple >= 1.0:
                trade.trail_1r_triggered = True
                half_r_price = trade.entry_price * (1 - trade.stop_pct * 0.5)
                trade.current_stop = min(trade.current_stop, half_r_price)
                logger.info(
                    "trail_1r_triggered",
                    symbol=trade.symbol,
                    new_stop=f"${trade.current_stop:.4f}",
                    r_multiple=f"{trade.r_multiple:.2f}",
                )
    
    def _check_exits(self, trade: TradeRecordV3, current_price: float) -> Optional[Dict[str, str]]:
        """Check for TP/SL exits"""
        is_long = trade.side == "LONG"
        
        if is_long:
            # Check TP
            if current_price >= trade.target_price:
                return {"reason": "TP_HIT"}
            
            # Check trailing stop (or original stop if no trail)
            if current_price <= trade.current_stop:
                if trade.breakeven_triggered or trade.trail_1r_triggered:
                    return {"reason": "TRAIL_STOP"}
                else:
                    return {"reason": "SL_HIT"}
        else:  # SHORT
            # Check TP
            if current_price <= trade.target_price:
                return {"reason": "TP_HIT"}
            
            # Check trailing stop
            if current_price >= trade.current_stop:
                if trade.breakeven_triggered or trade.trail_1r_triggered:
                    return {"reason": "TRAIL_STOP"}
                else:
                    return {"reason": "SL_HIT"}
        
        return None
    
    async def _close_trade(
        self, 
        trade: TradeRecordV3, 
        reason: str, 
        close_price: float
    ) -> Dict[str, Any]:
        """Close a trade and update account"""
        symbol = trade.symbol
        
        # Calculate final PnL
        if trade.side == "LONG":
            price_diff = close_price - trade.entry_price
        else:
            price_diff = trade.entry_price - close_price
        
        raw_pnl = price_diff * trade.size
        fees = trade.notional * FEE_PCT * 2
        realized_pnl = raw_pnl - fees
        
        # Calculate final R using actual risk amount (includes fees in sizing)
        final_r = realized_pnl / trade.risk_amount if trade.risk_amount > 0 else 0
        
        # Update trade record
        trade.status = f"closed_{reason.lower()}"
        trade.close_reason = reason
        trade.close_price = close_price
        trade.realized_pnl = realized_pnl
        trade.r_multiple = final_r
        trade.unrealized_pnl = 0
        trade.closed_at = datetime.now().isoformat()
        
        # Update account
        margin_to_release = trade.notional * MARGIN_REQUIREMENT
        self._account.margin_used -= margin_to_release
        self._account.margin_available += margin_to_release
        self._account.realized_pnl += realized_pnl
        self._account.current_equity += realized_pnl
        self._account.total_r += final_r
        self._account.trade_count += 1
        
        if realized_pnl > 0:
            self._account.win_count += 1
        else:
            self._account.loss_count += 1
        
        # Move to closed trades
        self._closed_trades.append(trade)
        del self._open_trades[symbol]
        
        logger.info(
            "trade_closed_v3",
            symbol=symbol,
            reason=reason,
            entry=f"${trade.entry_price:.4f}",
            exit=f"${close_price:.4f}",
            pnl=f"${realized_pnl:.2f}",
            r_multiple=f"{final_r:.2f}R",
            equity=f"${self._account.current_equity:.2f}",
        )
        
        # Callback
        if self._on_trade_closed:
            try:
                result = self._on_trade_closed(trade)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("trade_closed_callback_error", error=str(e)[:50])
        
        return trade.to_dict()
    
    def _update_unrealized_pnl(self):
        """Update total unrealized PnL"""
        total = sum(t.unrealized_pnl for t in self._open_trades.values())
        self._account.unrealized_pnl = total
    
    # ========== GETTERS ==========
    
    def get_open_trades(self) -> List[TradeRecordV3]:
        """Get all open trades"""
        return list(self._open_trades.values())
    
    def get_open_trade(self, symbol: str) -> Optional[TradeRecordV3]:
        """Get open trade for symbol"""
        return self._open_trades.get(symbol)
    
    def get_closed_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent closed trades"""
        trades = self._closed_trades[-limit:]
        return [t.to_dict() for t in trades]
    
    def get_account_state(self) -> AccountStateV3:
        """Get current account state"""
        return self._account
    
    def has_margin_available(self, required: float) -> bool:
        """Check if margin is available"""
        return self._account.margin_available >= required
    
    def set_trade_closed_callback(self, callback: Callable):
        """Set callback for trade closure"""
        self._on_trade_closed = callback
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics"""
        return {
            "open_trades": len(self._open_trades),
            "closed_trades": len(self._closed_trades),
            "account": self._account.to_dict(),
            "open_symbols": list(self._open_trades.keys()),
        }
    
    # ========== MANUAL OPERATIONS ==========
    
    async def flatten_all(self, reason: str = "MANUAL_FLATTEN"):
        """Close all open positions"""
        for symbol in list(self._open_trades.keys()):
            trade = self._open_trades[symbol]
            await self._close_trade(trade, reason, trade.current_price)
        
        logger.info("all_positions_flattened", reason=reason)
    
    async def close_position(self, symbol: str, reason: str = "MANUAL_CLOSE") -> Optional[Dict[str, Any]]:
        """Manually close a position"""
        if symbol not in self._open_trades:
            return None
        
        trade = self._open_trades[symbol]
        return await self._close_trade(trade, reason, trade.current_price)
