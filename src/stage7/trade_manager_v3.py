"""
Trade Manager V3 (Simplified for Hybrid Alpha)
==============================================

Simplified trade management for Stage 3 V3 signals:
- Single position per symbol (no tranches)
- Exit on TP or SL hit only (no complex exit strategies)
- Trail stop to breakeven at +0.8%
- Trail stop to 1R profit level after 1R reached
- Single order placement (no batch)
- WEEX API integration for real order execution

Disabled features:
- Stage 5 ML gating
- Complex exit strategies (CHOP, POC, LVN, etc.)
- Tranche A/B splitting
"""
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
import structlog

from src.stage6.position_sizer_v3 import PositionV3, PositionResultV3
from src.execution.weex import WeexClient, OrderResult
from src.execution.ai_log_generator import generate_ai_log, generate_close_ai_log

# File paths for persistence
AI_LOGS_PATH = Path("ai_logs.json")
META_PATH = Path("meta_v6.json")
ERRORS_PATH = Path("errors.json")

logger = structlog.get_logger(__name__)

# Constants
FEE_PCT = 0.0008
MARGIN_REQUIREMENT = 0.05  # 5% margin (20x leverage)
BREAKEVEN_TRIGGER_PCT = 0.008  # +0.8% to trigger breakeven trail


# ========== PERSISTENCE HELPERS ==========

def _save_ai_log(ai_log_data: Dict[str, Any]) -> None:
    """Append AI log to ai_logs.json"""
    try:
        logs = []
        if AI_LOGS_PATH.exists():
            with open(AI_LOGS_PATH, 'r') as f:
                content = f.read().strip()
                if content:
                    logs = json.loads(content)
        logs.append(ai_log_data)
        with open(AI_LOGS_PATH, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logger.warning("ai_log_save_failed", error=str(e)[:100])


def _save_position_meta(symbol: str, data: Dict[str, Any]) -> None:
    """Save position to meta_v6.json"""
    try:
        meta = {}
        if META_PATH.exists():
            with open(META_PATH, 'r') as f:
                content = f.read().strip()
                if content:
                    meta = json.loads(content)
        meta[symbol] = data
        with open(META_PATH, 'w') as f:
            json.dump(meta, f, indent=2)
        logger.info("position_meta_saved", symbol=symbol)
    except Exception as e:
        logger.warning("position_meta_save_failed", error=str(e)[:100])


def _delete_position_meta(symbol: str) -> None:
    """Delete position from meta_v6.json"""
    try:
        if not META_PATH.exists():
            return
        with open(META_PATH, 'r') as f:
            content = f.read().strip()
            if not content:
                return
            meta = json.loads(content)
        if symbol in meta:
            del meta[symbol]
            with open(META_PATH, 'w') as f:
                json.dump(meta, f, indent=2)
            logger.info("position_meta_deleted", symbol=symbol)
    except Exception as e:
        logger.warning("position_meta_delete_failed", error=str(e)[:100])


def _save_api_error(action: str, symbol: str, error_code: str, error_message: str, raw_response: dict = None) -> None:
    """Save API error to errors.json"""
    try:
        errors = []
        if ERRORS_PATH.exists():
            with open(ERRORS_PATH, 'r') as f:
                content = f.read().strip()
                if content:
                    errors = json.loads(content)
        errors.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "symbol": symbol,
            "error_code": error_code,
            "error_message": error_message,
            "raw_response": raw_response,
        })
        # Keep only last 100 errors
        if len(errors) > 100:
            errors = errors[-100:]
        with open(ERRORS_PATH, 'w') as f:
            json.dump(errors, f, indent=2)
    except Exception as e:
        logger.warning("error_save_failed", error=str(e)[:100])


def _load_positions_meta() -> Dict[str, Dict[str, Any]]:
    """Load all positions from meta_v6.json"""
    try:
        if not META_PATH.exists():
            return {}
        with open(META_PATH, 'r') as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except Exception as e:
        logger.warning("position_meta_load_failed", error=str(e)[:100])
        return {}


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
    - WEEX API integration for real execution
    """
    
    def __init__(
        self,
        initial_equity: float = 1000.0,
        reset_on_start: bool = True,
        live_trading: bool = False,
        dry_run: bool = True,
    ):
        self.initial_equity = initial_equity
        self.live_trading = live_trading
        
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
        
        # WEEX execution client
        self._weex_client: Optional[WeexClient] = None
        if live_trading:
            self._weex_client = WeexClient(
                equity=initial_equity,
                dry_run=dry_run,
            )
            logger.info(
                "weex_client_initialized",
                dry_run=dry_run,
                live_trading=live_trading,
            )
        
        if reset_on_start:
            self._reset()
        
        # Load existing positions from meta_v6.json on startup
        self._load_existing_positions()
        
        logger.info(
            "trade_manager_v3_initialized",
            initial_equity=initial_equity,
            live_trading=live_trading,
            dry_run=dry_run,
            loaded_positions=len(self._open_trades),
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
    
    def _load_existing_positions(self):
        """Load positions from meta_v6.json on startup"""
        positions = _load_positions_meta()
        if not positions:
            return
        
        for symbol, data in positions.items():
            try:
                trade = TradeRecordV3(
                    order_id=data.get("order_id", f"RESTORED_{symbol}"),
                    symbol=symbol,
                    side=data.get("side", "LONG"),
                    entry_price=data.get("entry_price", 0),
                    size=data.get("size", 0),
                    notional=data.get("notional", 0),
                    margin=data.get("notional", 0) * MARGIN_REQUIREMENT,
                    risk_amount=data.get("notional", 0) * data.get("stop_pct", 0.015),
                    stop_price=data.get("stop_price", 0),
                    stop_pct=data.get("stop_pct", 0.015),
                    target_price=data.get("target_price", 0),
                    target_pct=data.get("target_pct", 0.03),
                    current_price=data.get("entry_price", 0),
                    current_stop=data.get("current_stop", data.get("stop_price", 0)),
                    signal_type=data.get("signal_type", ""),
                    signal_name=data.get("signal_name", ""),
                    created_at=data.get("created_at", datetime.now().isoformat()),
                    breakeven_triggered=data.get("breakeven_triggered", False),
                    trail_1r_triggered=data.get("trail_1r_triggered", False),
                )
                self._open_trades[symbol] = trade
                
                # Reserve margin
                self._account.margin_used += trade.margin
                self._account.margin_available -= trade.margin
                
                logger.info(
                    "position_restored_from_meta",
                    symbol=symbol,
                    side=trade.side,
                    entry=f"${trade.entry_price:.4f}",
                    notional=f"${trade.notional:.2f}",
                )
            except Exception as e:
                logger.warning("position_restore_failed", symbol=symbol, error=str(e)[:100])
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self._order_counter += 1
        return f"V3_{int(time.time())}_{self._order_counter}"
    
    # ========== ORDER PLACEMENT ==========
    
    async def place_position(
        self, 
        position: PositionV3,
        market_state: Any = None,
        signal: Any = None,
    ) -> Optional[str]:
        """
        Place a new position from V3 signal.
        
        Args:
            position: Position sizing result
            market_state: MarketState for AI log generation (optional)
            signal: HybridSignal for AI log generation (optional)
        
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
        
        # Generate order ID
        order_id = self._generate_order_id()
        
        # Execute on WEEX if live trading enabled
        weex_order_id = None
        if self._weex_client and self.live_trading:
            result = await self._weex_client.place_order(
                symbol=symbol,
                direction=position.side,  # "LONG" or "SHORT"
                size_usd=position.notional,
                entry_price=position.entry_price,
                stop_price=position.stop_price,
                target_price=position.target_price,
                client_oid=order_id,
            )
            
            if not result.success:
                logger.error(
                    "weex_order_failed",
                    symbol=symbol,
                    error=result.error_message,
                )
                _save_api_error("PLACE_ORDER", symbol, result.error_code or "", result.error_message or "", result.raw_response)
                return None
            
            weex_order_id = result.order_id
            logger.info(
                "weex_order_placed",
                symbol=symbol,
                weex_order_id=weex_order_id,
                client_oid=order_id,
            )
            
            # Upload AI decision log for competition and save locally
            if weex_order_id and market_state and signal:
                try:
                    logger.info("ai_log_generation_start", symbol=symbol, order_id=weex_order_id)
                    ai_log = generate_ai_log(weex_order_id, market_state, signal)
                    logger.info("ai_log_generated", stage=ai_log.stage, model=ai_log.model)
                    
                    # Upload to WEEX
                    logger.info("ai_log_upload_start", order_id=ai_log.order_id)
                    await self._weex_client.upload_ai_log(
                        order_id=ai_log.order_id,
                        stage=ai_log.stage,
                        model=ai_log.model,
                        input_data=ai_log.input_data,
                        output_data=ai_log.output_data,
                        explanation=ai_log.explanation,
                    )
                    logger.info("ai_log_uploaded", order_id=ai_log.order_id)
                    
                    # Save locally to ai_logs.json
                    _save_ai_log({
                        "order_id": ai_log.order_id,
                        "timestamp": datetime.now().isoformat(),
                        "action": "OPEN",
                        "symbol": symbol,
                        "stage": ai_log.stage,
                        "model": ai_log.model,
                        "input": ai_log.input_data,
                        "output": ai_log.output_data,
                        "explanation": ai_log.explanation,
                    })
                    logger.info("ai_log_saved_locally", order_id=ai_log.order_id)
                except Exception as e:
                    logger.error("ai_log_upload_error", symbol=symbol, error=str(e)[:200], exc_info=True)
        
        # Create trade record
        logger.info("trade_record_creation_start", symbol=symbol)
        trade = TradeRecordV3(
            order_id=weex_order_id or order_id,
            symbol=symbol,
            side=position.side,
            entry_price=position.entry_price,
            size=position.size,
            notional=position.notional,
            margin=position.margin,
            risk_amount=position.risk_amount,
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
        logger.info("trade_record_created", symbol=symbol, order_id=trade.order_id)
        
        # Update account
        self._account.margin_used += required_margin
        self._account.margin_available -= required_margin
        logger.info("account_updated", symbol=symbol, margin_used=trade.margin, margin_available=self._account.margin_available)
        
        # Store trade
        self._open_trades[symbol] = trade
        logger.info("trade_stored", symbol=symbol, total_open_trades=len(self._open_trades))
        
        # Save position metadata to meta_v6.json
        _save_position_meta(symbol, {
            "order_id": trade.order_id,
            "symbol": symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "size": position.size,
            "notional": position.notional,
            "stop_price": position.stop_price,
            "stop_pct": position.stop_pct,
            "target_price": position.target_price,
            "target_pct": position.target_pct,
            "current_stop": position.stop_price,
            "signal_type": position.signal_type,
            "signal_name": position.signal_name,
            "created_at": trade.created_at,
            "breakeven_triggered": False,
            "trail_1r_triggered": False,
        })
        
        logger.info(
            "position_opened_v3",
            order_id=trade.order_id,
            symbol=symbol,
            side=position.side,
            entry=f"${position.entry_price:.4f}",
            size=position.size,
            notional=f"${position.notional:.2f}",
            stop=f"${position.stop_price:.4f}",
            target=f"${position.target_price:.4f}",
            live_trading=self.live_trading,
        )
        
        return trade.order_id
    
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
        close_order_id = None
        
        # Close position on WEEX if live trading enabled (TRAIL_STOP or BIAS_REVERSAL)
        # Note: TP and SL are handled by WEEX exchange directly via the order
        if self._weex_client and self.live_trading and reason in ("TRAIL_STOP", "BIAS_REVERSAL"):
            result = await self._weex_client.close_position(symbol)
            if result.success:
                # Extract successOrderId from response
                # Response format: [{"positionId": ..., "successOrderId": 123, ...}]
                raw = result.raw_response
                if raw and raw.get("data"):
                    data = raw["data"]
                    if isinstance(data, list) and len(data) > 0:
                        close_order_id = str(data[0].get("successOrderId", ""))
                        if close_order_id == "0":
                            close_order_id = None
                
                logger.info(
                    "weex_position_closed",
                    symbol=symbol,
                    reason=reason,
                    close_order_id=close_order_id,
                )
                
                ai_reason = "Bias Reversal Detected" if reason == "BIAS_REVERSAL" else "Trail Stop by System"
                if close_order_id:
                    try:
                        ai_log = generate_close_ai_log(
                            order_id=close_order_id,
                            symbol=symbol,
                            reason=ai_reason,
                            entry_price=trade.entry_price,
                            close_price=close_price,
                            side=trade.side,
                            pnl=(close_price - trade.entry_price) * trade.size if trade.side == "LONG" else (trade.entry_price - close_price) * trade.size,
                        )
                        await self._weex_client.upload_ai_log(
                            order_id=ai_log.order_id,
                            stage=ai_log.stage,
                            model=ai_log.model,
                            input_data=ai_log.input_data,
                            output_data=ai_log.output_data,
                            explanation=ai_log.explanation,
                        )
                        # Save locally
                        _save_ai_log({
                            "order_id": close_order_id,
                            "timestamp": datetime.now().isoformat(),
                            "action": "CLOSE",
                            "symbol": symbol,
                            "reason": "Trail Stop by System",
                            "stage": ai_log.stage,
                            "model": ai_log.model,
                            "input": ai_log.input_data,
                            "output": ai_log.output_data,
                            "explanation": ai_log.explanation,
                        })
                    except Exception as e:
                        logger.warning("close_ai_log_error", error=str(e)[:100])
            else:
                logger.error(
                    "weex_close_failed",
                    symbol=symbol,
                    error=result.error_message,
                )
                _save_api_error("CLOSE_POSITION", symbol, result.error_code or "", result.error_message or "", result.raw_response)
                # Continue with local close even if WEEX fails
        
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
        
        # Delete from meta_v6.json
        _delete_position_meta(symbol)
        
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
    
    async def manual_close_position(self, symbol: str, reason: str = "BIAS_REVERSAL") -> Dict[str, Any]:
        """Manually close a position (called from dashboard Signal Status button)"""
        trade = self._open_trades.get(symbol)
        if not trade:
            return {"success": False, "error": f"No open trade for {symbol}"}
        
        result = await self._close_trade(trade, reason, trade.current_price)
        return {"success": True, "data": result}
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics"""
        metrics = {
            "open_trades": len(self._open_trades),
            "closed_trades": len(self._closed_trades),
            "account": self._account.to_dict(),
            "open_symbols": list(self._open_trades.keys()),
            "live_trading": self.live_trading,
        }
        
        # Add WEEX stats if available
        if self._weex_client:
            metrics["weex"] = self._weex_client.get_stats()
        
        return metrics
    
    async def check_weex_connectivity(self) -> Tuple[bool, str]:
        """
        Check WEEX API connectivity on startup.
        
        Returns:
            Tuple of (success, message)
        """
        if not self._weex_client:
            return True, "WEEX client not enabled (paper trading mode)"
        
        return await self._weex_client.check_connectivity()
    
    async def shutdown(self):
        """Cleanup resources"""
        if self._weex_client:
            await self._weex_client.close()
            logger.info("weex_client_closed")
    
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
