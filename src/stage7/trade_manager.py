"""
Trade Manager Module

High-level trade management system that:
- Places orders via exchange API
- Tracks positions in real-time
- Monitors SL/TP hits
- Calculates PnL and R multiples
- Manages account equity and margin
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import structlog

from src.stage7.exchange_api import DummyExchangeAPI, BatchOrderResponse
from src.stage7.database import TradeDatabase, TrancheRecord, AccountState, FEE_PCT
from src.stage7.exit_strategies import ExitStrategyManager
from src.stage7.risk_manager import RiskManager

logger = structlog.get_logger(__name__)

# Margin requirement (percentage of notional)
MARGIN_REQUIREMENT = 0.05  # 5% margin (20x leverage)


@dataclass
class PositionSummary:
    """Summary of a position (both tranches)"""
    symbol: str
    side: str
    entry_price: float
    current_price: float
    total_size: float
    total_notional: float
    total_risk: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    r_multiple: float
    tranche_a_status: str
    tranche_b_status: str
    signal_name: str
    created_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TradeManager:
    """
    Trade Management System
    
    Manages the full lifecycle of trades:
    1. Place orders when position is confirmed (Stage 6)
    2. Track positions in real-time
    3. Monitor SL/TP levels
    4. Calculate PnL with fees
    5. Update account equity
    """
    
    def __init__(
        self,
        exchange_api: Optional[DummyExchangeAPI] = None,
        database: Optional[TradeDatabase] = None,
        initial_equity: float = 1000.0,
        reset_on_start: bool = True,
    ):
        self.exchange = exchange_api or DummyExchangeAPI()
        self.db = database or TradeDatabase(reset=reset_on_start)
        
        # Reset account if requested
        if reset_on_start:
            self.db.reset_account(initial_equity)
            self.db.clear_all_tranches()
        
        # Callbacks for events
        self._on_sl_hit: Optional[Callable] = None
        self._on_tp_hit: Optional[Callable] = None
        self._on_position_closed: Optional[Callable] = None
        
        # Cache for current prices
        self._current_prices: Dict[str, float] = {}
        
        # Exit strategy manager
        self.exit_manager = ExitStrategyManager()
        
        # Risk manager for dynamic sizing and limits
        self.risk_manager = RiskManager(initial_equity=initial_equity)
        
        logger.info(
            "trade_manager_initialized",
            initial_equity=initial_equity,
            reset=reset_on_start,
        )
    
    # ========== ORDER PLACEMENT ==========
    
    async def place_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        tranche_a_size: float,
        tranche_a_stop: float,
        tranche_a_tp: float,
        tranche_a_breakeven: float,
        tranche_a_risk: float,
        tranche_b_size: float,
        tranche_b_stop: float,
        tranche_b_tp_partial: float,
        tranche_b_tp_runner: float,
        tranche_b_breakeven: float,
        tranche_b_risk: float,
        signal_name: str,
        mode: str = "test",
    ) -> Optional[Dict[str, str]]:
        """
        Place a new position with 2 tranches
        
        Returns:
            Dict with order IDs for both tranches, or None if failed
        """
        # Check margin availability
        account = self.db.get_account()
        total_notional = (tranche_a_size + tranche_b_size) * entry_price
        required_margin = total_notional * MARGIN_REQUIREMENT
        
        if required_margin > account.margin_available:
            logger.warning(
                "insufficient_margin",
                symbol=symbol,
                required=required_margin,
                available=account.margin_available,
            )
            return None
        
        # Place batch order via exchange API
        response = await self.exchange.place_batch_order(
            symbol=symbol,
            side=side,
            tranche_a_size=tranche_a_size,
            tranche_a_price=entry_price,
            tranche_a_sl=tranche_a_stop,
            tranche_a_tp=tranche_a_tp,
            tranche_b_size=tranche_b_size,
            tranche_b_price=entry_price,
            tranche_b_sl=tranche_b_stop,
            tranche_b_tp=tranche_b_tp_runner,
        )
        
        if not response.success:
            logger.error("batch_order_failed", symbol=symbol, error=response.error)
            return None
        
        order_a = response.orders[0]
        order_b = response.orders[1]
        timestamp = datetime.now().isoformat()
        
        # Create tranche A record
        tranche_a = TrancheRecord(
            order_id=order_a.order_id,
            symbol=symbol,
            side=side,
            tranche="A",
            mode=mode,
            entry_price=entry_price,
            size=tranche_a_size,
            notional=tranche_a_size * entry_price,
            stop_loss=tranche_a_stop,
            take_profit=tranche_a_tp,
            breakeven=tranche_a_breakeven,
            tp_partial=0.0,
            tp_runner=0.0,
            partial_size=0.0,
            current_price=entry_price,
            risk_amount=tranche_a_risk,
            signal_name=signal_name,
            created_at=timestamp,
            status="open",
        )
        
        # Create tranche B record
        # For tranche B, partial TP is at 2R (40% of size), runner TP is at 3R
        partial_size = tranche_b_size * 0.4
        tranche_b = TrancheRecord(
            order_id=order_b.order_id,
            symbol=symbol,
            side=side,
            tranche="B",
            mode=mode,
            entry_price=entry_price,
            size=tranche_b_size,
            notional=tranche_b_size * entry_price,
            stop_loss=tranche_b_stop,
            take_profit=tranche_b_tp_runner,  # Runner TP (3R)
            breakeven=tranche_b_breakeven,
            tp_partial=tranche_b_tp_partial,  # Partial TP (2R)
            tp_runner=tranche_b_tp_runner,
            partial_size=partial_size,
            current_price=entry_price,
            risk_amount=tranche_b_risk,
            signal_name=signal_name,
            created_at=timestamp,
            status="open",
        )
        
        # Insert into database
        self.db.insert_tranche(tranche_a)
        self.db.insert_tranche(tranche_b)
        
        # Update account margin
        self.db.update_account({
            "margin_used": account.margin_used + required_margin,
            "margin_available": account.margin_available - required_margin,
            "total_trades": account.total_trades + 1,
        })
        
        logger.info(
            "position_placed",
            symbol=symbol,
            side=side,
            order_id_a=order_a.order_id,
            order_id_b=order_b.order_id,
            total_notional=f"${total_notional:.2f}",
            margin_used=f"${required_margin:.2f}",
        )
        
        return {
            "order_id_a": order_a.order_id,
            "order_id_b": order_b.order_id,
        }
    
    # ========== REAL-TIME TRACKING ==========
    
    async def update_price(
        self,
        symbol: str,
        current_price: float,
        market_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Update current price for a symbol and check SL/TP/Exit hits
        
        Args:
            symbol: Trading pair
            current_price: Current market price
            market_state: Optional dict with keys for exit strategy:
                regime, absorption_z, dist_poc, dist_lvn, atr_5m, atr_1h,
                vah, val, acceptance_outside_value, moi_z, delta_vel_z,
                liq_long_usd, liq_short_usd
        
        Returns:
            List of events (SL hit, TP hit, hard exit, etc.)
        """
        self._current_prices[symbol] = current_price
        events = []
        
        # Update exit strategy manager with market state
        if market_state:
            self.exit_manager.update_market_state(
                symbol=symbol,
                current_price=current_price,
                regime=market_state.get("regime", "MID"),
                absorption_z=market_state.get("absorption_z", 0.0),
                dist_poc=market_state.get("dist_poc", 0.0),
                dist_lvn=market_state.get("dist_lvn", 0.0),
                atr_5m=market_state.get("atr_5m", 0.0),
                atr_1h=market_state.get("atr_1h", 0.0),
                vah=market_state.get("vah", 0.0),
                val=market_state.get("val", 0.0),
                acceptance_outside_value=market_state.get("acceptance_outside_value", False),
                moi_z=market_state.get("moi_z", 0.0),
                delta_vel_z=market_state.get("delta_vel_z", 0.0),
                liq_long_usd=market_state.get("liq_long_usd", 0.0),
                liq_short_usd=market_state.get("liq_short_usd", 0.0),
            )
        
        # Get open tranches for this symbol
        tranches = self.db.get_tranches_by_symbol(symbol)
        
        for tranche in tranches:
            if tranche.status == "closed":
                continue
            
            # Calculate unrealized PnL
            pnl_data = self._calculate_pnl(tranche, current_price)
            
            # Update tranche with current values
            updates = {
                "current_price": current_price,
                "unrealized_pnl": pnl_data["unrealized_pnl"],
                "r_multiple": pnl_data["r_multiple"],
            }
            
            # Check SL hit (we track SL in DB and close ourselves)
            if self._check_sl_hit(tranche, current_price):
                event = await self._handle_sl_hit(tranche, current_price)
                if event:
                    events.append(event)
                continue  # Tranche closed, skip further checks
            
            # Check TP hit
            if tranche.tranche == "A":
                # Tranche A: single TP - API handles but dummy needs us to close
                if self._check_tp_hit(tranche, current_price, tranche.take_profit):
                    event = await self._handle_tp_hit_a(tranche, current_price)
                    if event:
                        events.append(event)
                    continue
            else:
                # Tranche B: 
                # 1. At 0.5R: Move SL to breakeven + fee
                # 2. At 2R: Close 40%, move SL to 1R
                # 3. Runner TP: API handles (dummy closes)
                
                # Check 0.5R - move SL to breakeven + fee
                if not tranche.sl_moved_be and pnl_data["r_multiple"] >= 0.5:
                    event = await self._handle_sl_move_be(tranche, current_price)
                    if event:
                        events.append(event)
                        updates["sl_moved_be"] = 1
                
                # Check 2R - close 40%, move SL to 1R
                if not tranche.tp_partial_hit and pnl_data["r_multiple"] >= 2.0:
                    event = await self._handle_tp_partial_b(tranche, current_price)
                    if event:
                        events.append(event)
                        updates["tp_partial_hit"] = 1
                        updates["sl_moved_1r"] = 1
                        updates["status"] = "partial"
                
                # Check runner TP hit - dummy needs to close
                if tranche.tp_partial_hit and self._check_tp_hit(tranche, current_price, tranche.tp_runner):
                    event = await self._handle_tp_runner_b(tranche, current_price)
                    if event:
                        events.append(event)
                    continue
                
                # Check hard exit conditions for tranche B
                if market_state:
                    # Calculate 2R price for exit manager
                    r_per_unit = tranche.risk_amount / tranche.size if tranche.size > 0 else 0
                    if tranche.side == "LONG":
                        tp_2r_price = tranche.entry_price + (2 * r_per_unit)
                    else:
                        tp_2r_price = tranche.entry_price - (2 * r_per_unit)
                    
                    exit_result = self.exit_manager.check_exit(
                        symbol=symbol,
                        side=tranche.side,
                        entry_price=tranche.entry_price,
                        breakeven_price=tranche.breakeven,
                        current_r=pnl_data["r_multiple"],
                        tp_2r_price=tp_2r_price,
                    )
                    
                    if exit_result:
                        event = await self._handle_hard_exit(tranche, current_price, exit_result)
                        if event:
                            events.append(event)
                        continue
            
            # Update tranche in database
            self.db.update_tranche(tranche.order_id, updates)
        
        # Update account unrealized PnL
        self._update_account_unrealized_pnl()
        
        return events
    
    def _calculate_pnl(self, tranche: TrancheRecord, current_price: float) -> Dict[str, float]:
        """Calculate PnL for a tranche"""
        if tranche.side == "LONG":
            price_diff = current_price - tranche.entry_price
        else:  # SHORT
            price_diff = tranche.entry_price - current_price
        
        # Calculate raw PnL
        raw_pnl = price_diff * tranche.size
        
        # Subtract entry fee (already paid)
        entry_fee = tranche.notional * FEE_PCT
        
        # Exit fee (would be paid if we close now)
        exit_notional = current_price * tranche.size
        exit_fee = exit_notional * FEE_PCT
        
        # Unrealized PnL after fees
        unrealized_pnl = raw_pnl - entry_fee - exit_fee
        
        # R multiple (based on risk amount)
        r_multiple = unrealized_pnl / tranche.risk_amount if tranche.risk_amount > 0 else 0
        
        return {
            "raw_pnl": raw_pnl,
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "unrealized_pnl": unrealized_pnl,
            "r_multiple": r_multiple,
        }
    
    def _check_sl_hit(self, tranche: TrancheRecord, current_price: float) -> bool:
        """Check if stop loss is hit"""
        if tranche.side == "LONG":
            return current_price <= tranche.stop_loss
        else:  # SHORT
            return current_price >= tranche.stop_loss
    
    def _check_tp_hit(self, tranche: TrancheRecord, current_price: float, tp_price: float) -> bool:
        """Check if take profit is hit"""
        if tranche.side == "LONG":
            return current_price >= tp_price
        else:  # SHORT
            return current_price <= tp_price
    
    async def _handle_sl_hit(self, tranche: TrancheRecord, current_price: float) -> Dict[str, Any]:
        """Handle stop loss hit"""
        # Close via API
        await self.exchange.close_order(tranche.symbol, tranche.size, current_price)
        
        # Calculate realized PnL
        pnl_data = self._calculate_pnl(tranche, current_price)
        realized_pnl = pnl_data["unrealized_pnl"]
        r_multiple = pnl_data["r_multiple"]
        
        # Update tranche
        self.db.close_tranche(
            order_id=tranche.order_id,
            close_price=current_price,
            close_reason="SL_HIT",
            realized_pnl=realized_pnl,
            r_multiple=r_multiple,
        )
        self.db.update_tranche(tranche.order_id, {"sl_hit": 1})
        
        # Update account
        self._update_account_on_close(tranche, realized_pnl, r_multiple, is_win=False)
        
        # Release margin
        self._release_margin(tranche)
        
        # Record with risk manager
        self.risk_manager.record_closed_trade(
            order_id=tranche.order_id,
            symbol=tranche.symbol,
            side=tranche.side,
            tranche=tranche.tranche,
            entry_price=tranche.entry_price,
            close_price=current_price,
            size=tranche.size,
            realized_pnl=realized_pnl,
            r_multiple=r_multiple,
            close_reason="SL_HIT",
            signal_name=tranche.signal_name,
            opened_at=tranche.created_at,
        )
        
        logger.info(
            "sl_hit",
            symbol=tranche.symbol,
            tranche=tranche.tranche,
            entry=tranche.entry_price,
            exit=current_price,
            pnl=f"${realized_pnl:.2f}",
            r=f"{r_multiple:.2f}R",
        )
        
        return {
            "type": "SL_HIT",
            "symbol": tranche.symbol,
            "tranche": tranche.tranche,
            "order_id": tranche.order_id,
            "entry_price": tranche.entry_price,
            "close_price": current_price,
            "realized_pnl": realized_pnl,
            "r_multiple": r_multiple,
        }
    
    async def _handle_tp_hit_a(self, tranche: TrancheRecord, current_price: float) -> Dict[str, Any]:
        """Handle take profit hit for tranche A"""
        # Close via API
        await self.exchange.close_order(tranche.symbol, tranche.size, current_price)
        
        # Calculate realized PnL
        pnl_data = self._calculate_pnl(tranche, current_price)
        realized_pnl = pnl_data["unrealized_pnl"]
        r_multiple = pnl_data["r_multiple"]
        
        # Update tranche
        self.db.close_tranche(
            order_id=tranche.order_id,
            close_price=current_price,
            close_reason="TP_HIT",
            realized_pnl=realized_pnl,
            r_multiple=r_multiple,
        )
        self.db.update_tranche(tranche.order_id, {"tp_hit": 1})
        
        # Update account
        self._update_account_on_close(tranche, realized_pnl, r_multiple, is_win=True)
        
        # Release margin
        self._release_margin(tranche)
        
        # Record with risk manager
        self.risk_manager.record_closed_trade(
            order_id=tranche.order_id,
            symbol=tranche.symbol,
            side=tranche.side,
            tranche=tranche.tranche,
            entry_price=tranche.entry_price,
            close_price=current_price,
            size=tranche.size,
            realized_pnl=realized_pnl,
            r_multiple=r_multiple,
            close_reason="TP_HIT",
            signal_name=tranche.signal_name,
            opened_at=tranche.created_at,
        )
        
        logger.info(
            "tp_hit_a",
            symbol=tranche.symbol,
            entry=tranche.entry_price,
            exit=current_price,
            pnl=f"${realized_pnl:.2f}",
            r=f"{r_multiple:.2f}R",
        )
        
        return {
            "type": "TP_HIT_A",
            "symbol": tranche.symbol,
            "tranche": "A",
            "order_id": tranche.order_id,
            "entry_price": tranche.entry_price,
            "close_price": current_price,
            "realized_pnl": realized_pnl,
            "r_multiple": r_multiple,
        }
    
    async def _handle_sl_move_be(self, tranche: TrancheRecord, current_price: float) -> Dict[str, Any]:
        """Handle SL move to breakeven + fee at 0.5R for tranche B"""
        # Calculate breakeven + fee price
        # Breakeven already includes fee from position_sizer
        new_sl = tranche.breakeven
        
        # Update SL in DB (we track and close ourselves)
        self.db.update_tranche(tranche.order_id, {
            "stop_loss": new_sl,
            "sl_moved_be": 1,
        })
        
        logger.info(
            "sl_moved_be",
            symbol=tranche.symbol,
            tranche="B",
            old_sl=tranche.stop_loss,
            new_sl=new_sl,
            r_multiple=0.5,
        )
        
        return {
            "type": "SL_MOVED_BE",
            "symbol": tranche.symbol,
            "tranche": "B",
            "order_id": tranche.order_id,
            "old_sl": tranche.stop_loss,
            "new_sl": new_sl,
            "trigger": "0.5R",
        }
    
    async def _handle_tp_partial_b(self, tranche: TrancheRecord, current_price: float) -> Dict[str, Any]:
        """Handle partial take profit for tranche B at 2R - close 40%, move SL to 1R"""
        # Close 40% of position via API
        close_size = tranche.partial_size
        await self.exchange.close_order(tranche.symbol, close_size, current_price)
        
        # Calculate partial realized PnL
        if tranche.side == "LONG":
            price_diff = current_price - tranche.entry_price
        else:
            price_diff = tranche.entry_price - current_price
        
        partial_notional = close_size * tranche.entry_price
        exit_notional = close_size * current_price
        
        raw_pnl = price_diff * close_size
        entry_fee = partial_notional * FEE_PCT
        exit_fee = exit_notional * FEE_PCT
        realized_pnl = raw_pnl - entry_fee - exit_fee
        
        # Calculate 1R price for new SL
        # 1R = entry + (risk_amount / size) for LONG, entry - (risk_amount / size) for SHORT
        r_per_unit = tranche.risk_amount / tranche.size if tranche.size > 0 else 0
        if tranche.side == "LONG":
            new_sl = tranche.entry_price + r_per_unit  # 1R above entry
        else:
            new_sl = tranche.entry_price - r_per_unit  # 1R below entry
        
        # Update tranche - reduce size, mark partial hit, move SL to 1R
        remaining_size = tranche.size - close_size
        self.db.update_tranche(tranche.order_id, {
            "size": remaining_size,
            "notional": remaining_size * tranche.entry_price,
            "tp_partial_hit": 1,
            "sl_moved_1r": 1,
            "status": "partial",
            "realized_pnl": tranche.realized_pnl + realized_pnl,
            "stop_loss": new_sl,
        })
        
        logger.info(
            "tp_partial_b_2r",
            symbol=tranche.symbol,
            closed_size=close_size,
            remaining_size=remaining_size,
            pnl=f"${realized_pnl:.2f}",
            new_sl=f"{new_sl:.4f} (1R)",
        )
        
        return {
            "type": "TP_PARTIAL_B",
            "symbol": tranche.symbol,
            "tranche": "B",
            "order_id": tranche.order_id,
            "close_size": close_size,
            "remaining_size": remaining_size,
            "close_price": current_price,
            "realized_pnl": realized_pnl,
            "new_sl": new_sl,
            "trigger": "2R",
        }
    
    async def _handle_tp_runner_b(self, tranche: TrancheRecord, current_price: float) -> Dict[str, Any]:
        """Handle runner take profit for tranche B (3R)"""
        # Close remaining position via API
        await self.exchange.close_order(tranche.symbol, tranche.size, current_price)
        
        # Calculate realized PnL for remaining size
        pnl_data = self._calculate_pnl(tranche, current_price)
        realized_pnl = tranche.realized_pnl + pnl_data["unrealized_pnl"]
        
        # R multiple based on original risk
        r_multiple = realized_pnl / tranche.risk_amount if tranche.risk_amount > 0 else 0
        
        # Update tranche
        self.db.close_tranche(
            order_id=tranche.order_id,
            close_price=current_price,
            close_reason="TP_RUNNER",
            realized_pnl=realized_pnl,
            r_multiple=r_multiple,
        )
        self.db.update_tranche(tranche.order_id, {"tp_runner_hit": 1})
        
        # Update account
        self._update_account_on_close(tranche, pnl_data["unrealized_pnl"], r_multiple, is_win=True)
        
        # Release margin
        self._release_margin(tranche)
        
        logger.info(
            "tp_runner_b",
            symbol=tranche.symbol,
            entry=tranche.entry_price,
            exit=current_price,
            total_pnl=f"${realized_pnl:.2f}",
            r=f"{r_multiple:.2f}R",
        )
        
        return {
            "type": "TP_RUNNER_B",
            "symbol": tranche.symbol,
            "tranche": "B",
            "order_id": tranche.order_id,
            "entry_price": tranche.entry_price,
            "close_price": current_price,
            "realized_pnl": realized_pnl,
            "r_multiple": r_multiple,
        }
    
    async def _handle_hard_exit(
        self,
        tranche: TrancheRecord,
        current_price: float,
        exit_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle hard exit from exit strategy manager"""
        # Close position via API
        await self.exchange.close_order(tranche.symbol, tranche.size, current_price)
        
        # Calculate realized PnL
        pnl_data = self._calculate_pnl(tranche, current_price)
        realized_pnl = tranche.realized_pnl + pnl_data["unrealized_pnl"]
        r_multiple = pnl_data["r_multiple"]
        
        # Determine if win
        is_win = realized_pnl > 0
        
        # Update tranche
        close_reason = f"HARD_EXIT_{exit_result['reason']}"
        self.db.close_tranche(
            order_id=tranche.order_id,
            close_price=current_price,
            close_reason=close_reason,
            realized_pnl=realized_pnl,
            r_multiple=r_multiple,
        )
        
        # Update account
        self._update_account_on_close(tranche, pnl_data["unrealized_pnl"], r_multiple, is_win=is_win)
        
        # Release margin
        self._release_margin(tranche)
        
        # Record with risk manager
        self.risk_manager.record_closed_trade(
            order_id=tranche.order_id,
            symbol=tranche.symbol,
            side=tranche.side,
            tranche=tranche.tranche,
            entry_price=tranche.entry_price,
            close_price=current_price,
            size=tranche.size,
            realized_pnl=realized_pnl,
            r_multiple=r_multiple,
            close_reason=close_reason,
            signal_name=tranche.signal_name,
            opened_at=tranche.created_at,
        )
        
        # Clear exit manager state for this symbol
        self.exit_manager.clear_symbol(tranche.symbol)
        
        logger.info(
            "hard_exit",
            symbol=tranche.symbol,
            tranche=tranche.tranche,
            reason=exit_result["reason"],
            details=exit_result.get("details", ""),
            entry=tranche.entry_price,
            exit=current_price,
            pnl=f"${realized_pnl:.2f}",
            r=f"{r_multiple:.2f}R",
        )
        
        return {
            "type": "HARD_EXIT",
            "symbol": tranche.symbol,
            "tranche": tranche.tranche,
            "order_id": tranche.order_id,
            "reason": exit_result["reason"],
            "details": exit_result.get("details", ""),
            "entry_price": tranche.entry_price,
            "close_price": current_price,
            "realized_pnl": realized_pnl,
            "r_multiple": r_multiple,
        }
    
    # ========== ACCOUNT MANAGEMENT ==========
    
    def _update_account_unrealized_pnl(self):
        """Update account with total unrealized PnL"""
        open_tranches = self.db.get_open_tranches()
        total_unrealized = sum(t.unrealized_pnl for t in open_tranches)
        
        account = self.db.get_account()
        current_equity = account.initial_equity + account.realized_pnl + total_unrealized
        
        self.db.update_account({
            "unrealized_pnl": total_unrealized,
            "current_equity": current_equity,
        })
    
    def _update_account_on_close(self, tranche: TrancheRecord, realized_pnl: float, r_multiple: float, is_win: bool):
        """Update account after closing a tranche"""
        account = self.db.get_account()
        
        updates = {
            "realized_pnl": account.realized_pnl + realized_pnl,
            "total_r": account.total_r + r_multiple,
        }
        
        if is_win:
            updates["winning_trades"] = account.winning_trades + 1
        else:
            updates["losing_trades"] = account.losing_trades + 1
        
        self.db.update_account(updates)
        self._update_account_unrealized_pnl()
    
    def _release_margin(self, tranche: TrancheRecord):
        """Release margin when position is closed"""
        margin_to_release = tranche.notional * MARGIN_REQUIREMENT
        account = self.db.get_account()
        
        self.db.update_account({
            "margin_used": max(0, account.margin_used - margin_to_release),
            "margin_available": account.margin_available + margin_to_release,
        })
    
    # ========== MANUAL CONTROLS ==========
    
    async def close_position(self, symbol: str, current_price: float, reason: str = "MANUAL") -> List[Dict[str, Any]]:
        """
        Close entire position for a symbol (both tranches)
        
        Returns list of close events
        """
        tranches = self.db.get_tranches_by_symbol(symbol)
        events = []
        
        if not tranches:
            return events
        
        # Close via API
        await self.exchange.close_position(symbol, current_price)
        
        for tranche in tranches:
            if tranche.status == "closed":
                continue
            
            pnl_data = self._calculate_pnl(tranche, current_price)
            realized_pnl = tranche.realized_pnl + pnl_data["unrealized_pnl"]
            r_multiple = realized_pnl / tranche.risk_amount if tranche.risk_amount > 0 else 0
            
            self.db.close_tranche(
                order_id=tranche.order_id,
                close_price=current_price,
                close_reason=reason,
                realized_pnl=realized_pnl,
                r_multiple=r_multiple,
            )
            
            is_win = realized_pnl > 0
            self._update_account_on_close(tranche, pnl_data["unrealized_pnl"], r_multiple, is_win)
            self._release_margin(tranche)
            
            # Record with risk manager
            self.risk_manager.record_closed_trade(
                order_id=tranche.order_id,
                symbol=tranche.symbol,
                side=tranche.side,
                tranche=tranche.tranche,
                entry_price=tranche.entry_price,
                close_price=current_price,
                size=tranche.size,
                realized_pnl=realized_pnl,
                r_multiple=r_multiple,
                close_reason=reason,
                signal_name=tranche.signal_name,
                opened_at=tranche.created_at,
            )
            
            events.append({
                "type": "POSITION_CLOSED",
                "symbol": symbol,
                "tranche": tranche.tranche,
                "reason": reason,
                "realized_pnl": realized_pnl,
                "r_multiple": r_multiple,
            })
        
        # Clear exit manager state
        self.exit_manager.clear_symbol(symbol)
        
        # Update risk manager with new equity
        account = self.db.get_account()
        self.risk_manager.update_equity(account.current_equity, account.unrealized_pnl)
        
        logger.info("position_closed", symbol=symbol, reason=reason, tranches=len(events))
        return events
    
    async def flatten_all(self, reason: str = "FLATTEN") -> List[Dict[str, Any]]:
        """Close all open positions (for hard stop)"""
        all_events = []
        open_tranches = self.db.get_open_tranches()
        
        # Get unique symbols
        symbols = list(set(t.symbol for t in open_tranches))
        
        for symbol in symbols:
            price = self._current_prices.get(symbol, 0)
            if price > 0:
                events = await self.close_position(symbol, price, reason)
                all_events.extend(events)
        
        logger.warning("all_positions_flattened", reason=reason, count=len(all_events))
        return all_events
    
    async def modify_stop_loss(self, order_id: str, new_sl: float) -> bool:
        """Modify stop loss for a tranche"""
        tranche = self.db.get_tranche(order_id)
        if not tranche:
            return False
        
        # Modify via API
        response = await self.exchange.change_tp_sl(order_id, "SL", new_sl)
        
        if response.success:
            self.db.update_tranche(order_id, {"stop_loss": new_sl})
            logger.info("sl_modified", order_id=order_id, new_sl=new_sl)
            return True
        
        return False
    
    # ========== GETTERS ==========
    
    def get_account_state(self) -> AccountState:
        """Get current account state"""
        return self.db.get_account()
    
    def get_open_positions(self) -> List[PositionSummary]:
        """Get summary of all open positions"""
        open_tranches = self.db.get_open_tranches()
        
        # Group by symbol
        positions_by_symbol: Dict[str, List[TrancheRecord]] = {}
        for t in open_tranches:
            if t.symbol not in positions_by_symbol:
                positions_by_symbol[t.symbol] = []
            positions_by_symbol[t.symbol].append(t)
        
        summaries = []
        for symbol, tranches in positions_by_symbol.items():
            tranche_a = next((t for t in tranches if t.tranche == "A"), None)
            tranche_b = next((t for t in tranches if t.tranche == "B"), None)
            
            total_size = sum(t.size for t in tranches)
            total_notional = sum(t.notional for t in tranches)
            total_risk = sum(t.risk_amount for t in tranches)
            total_unrealized = sum(t.unrealized_pnl for t in tranches)
            total_realized = sum(t.realized_pnl for t in tranches)
            
            current_price = tranches[0].current_price if tranches else 0
            entry_price = tranches[0].entry_price if tranches else 0
            
            total_pnl = total_unrealized + total_realized
            r_multiple = total_pnl / total_risk if total_risk > 0 else 0
            
            summaries.append(PositionSummary(
                symbol=symbol,
                side=tranches[0].side,
                entry_price=entry_price,
                current_price=current_price,
                total_size=total_size,
                total_notional=total_notional,
                total_risk=total_risk,
                unrealized_pnl=total_unrealized,
                realized_pnl=total_realized,
                total_pnl=total_pnl,
                r_multiple=r_multiple,
                tranche_a_status=tranche_a.status if tranche_a else "closed",
                tranche_b_status=tranche_b.status if tranche_b else "closed",
                signal_name=tranches[0].signal_name,
                created_at=tranches[0].created_at,
            ))
        
        return summaries
    
    def get_all_tranches(self) -> List[TrancheRecord]:
        """Get all tranches (open and closed)"""
        return self.db.get_all_tranches()
    
    def get_open_tranches(self) -> List[TrancheRecord]:
        """Get all open tranches"""
        return self.db.get_open_tranches()
    
    def has_margin_available(self, required_notional: float) -> bool:
        """Check if margin is available for a new position"""
        account = self.db.get_account()
        required_margin = required_notional * MARGIN_REQUIREMENT
        return required_margin <= account.margin_available
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get trade manager health metrics"""
        account = self.db.get_account()
        open_positions = self.get_open_positions()
        
        return {
            "equity": account.current_equity,
            "margin_available": account.margin_available,
            "margin_used": account.margin_used,
            "realized_pnl": account.realized_pnl,
            "unrealized_pnl": account.unrealized_pnl,
            "total_r": account.total_r,
            "total_trades": account.total_trades,
            "winning_trades": account.winning_trades,
            "losing_trades": account.losing_trades,
            "win_rate": account.winning_trades / account.total_trades if account.total_trades > 0 else 0,
            "open_positions": len(open_positions),
            "symbols_holding": [p.symbol for p in open_positions],
        }
