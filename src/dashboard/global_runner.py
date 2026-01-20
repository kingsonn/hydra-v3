"""
Global Pipeline Runner - Runs all stages with Global Dashboard
Connects Stage 1 → Stage 2 → Stage 3 → Stage 5 → Stage 6

Stage 4/4.5 REMOVED per audit - filters destroyed edge via over-filtering.

Emits real-time pipeline state for each symbol to the global dashboard.

Startup sequence:
1. Bootstrap ATR and volatility from historical klines (no cold start)
2. Initialize Stage 1 data collection
3. Start Stage 2 feature computation
4. Start Stage 3 thesis engine with Stage 5 ML gate
5. Launch global dashboard
"""
import asyncio
import signal
import sys
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass
import structlog

from config import settings
from src.stage1 import Stage1Orchestrator
from src.stage2.orchestrator import Stage2Orchestrator
from src.stage2.models import MarketState, Regime
from src.stage3.thesis_engine import ThesisEngine, PROB_MIN_THRESHOLD
from src.stage3.models import Thesis, Direction
# Stage 4/4.5 REMOVED - filters destroyed edge via over-filtering
# FilterResult import removed
from src.stage5.predictor_v3 import PredictionResultV3
from src.stage6.position_sizer import PositionSizer
from src.stage6.models import PositionResult
from src.stage7.trade_manager import TradeManager

from src.dashboard.global_dashboard import (
    broadcast_pipeline_state,
    broadcast_trade,
    broadcast_position,
    broadcast_rejection,
    broadcast_open_positions,
    broadcast_closed_trades,
)

logger = structlog.get_logger(__name__)


@dataclass
class SymbolTracker:
    """Track data freshness per symbol"""
    symbol: str
    trade_count: int = 0
    last_trade_ms: int = 0
    last_book_ms: int = 0
    last_funding_ms: int = 0
    last_oi_ms: int = 0
    start_time_ms: int = 0
    
    def trades_per_sec(self) -> float:
        if self.start_time_ms == 0:
            return 0.0
        elapsed_s = (time.time() * 1000 - self.start_time_ms) / 1000.0
        if elapsed_s <= 0:
            return 0.0
        return self.trade_count / elapsed_s


class GlobalPipelineRunner:
    """
    Runs the complete pipeline for all symbols with real-time dashboard.
    
    Pipeline Flow:
    1. Stage 1: Data Ingestion (WebSocket trades, orderbook, REST derivatives)
    2. Stage 2: Feature Computation + Regime Classification
    3. Stage 3: Signal Detection (thesis generation)
    4. Stage 5: ML Prediction + Probability Gate
    5. Stage 6: Position Sizing (2 tranches with SL/TP)
    
    Stage 4/4.5 REMOVED - see audit notes.
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        dashboard_port: int = 8888,
        enable_dashboard: bool = True,
    ):
        self.symbols = symbols or settings.SYMBOLS
        self.dashboard_port = dashboard_port
        self.enable_dashboard = enable_dashboard
        
        # Startup delay - disable trading for 6 minutes to let data stabilize
        self._start_time = time.time()
        self._trading_enabled_after = 6 * 60  # 6 minutes in seconds
        
        # Per-symbol tracking
        self._trackers: Dict[str, SymbolTracker] = {
            s: SymbolTracker(symbol=s, start_time_ms=int(time.time() * 1000))
            for s in self.symbols
        }
        
        # Pipeline state per symbol (dict format for dashboard)
        self._pipeline_states: Dict[str, Dict[str, Any]] = {
            s: {"symbol": s} for s in self.symbols
        }
        
        # Stage 3 thesis engine (includes Stage 4, 4.5, 5 internally)
        self.thesis_engine = ThesisEngine(
            symbols=self.symbols,
            on_thesis=self._on_thesis,
        )
        
        # Stage 6 position sizer
        self.position_sizer = PositionSizer(total_risk_dollars=12.0)
        
        # Stage 7 trade manager (reset on start for fresh equity)
        self.trade_manager = TradeManager(
            initial_equity=1000.0,
            reset_on_start=True,
        )
        
        # Stage 2 orchestrator
        self.stage2 = Stage2Orchestrator(
            symbols=self.symbols,
            on_market_state=self._on_market_state,
        )
        
        # Stage 1 orchestrator with callbacks
        self.stage1 = Stage1Orchestrator(
            symbols=self.symbols,
            on_trade=self._on_trade_wrapper,
            on_bar=self.stage2.on_bar,
            on_book_snapshot=self._on_book_wrapper,
            on_volume_profile=self.stage2.on_volume_profile,
            on_funding=self._on_funding_wrapper,
            on_oi=self._on_oi_wrapper,
            on_liquidation=self.stage2.on_liquidation,
        )
        
        self._running = False
        
        logger.info(
            "global_pipeline_runner_initialized",
            symbols=self.symbols,
            dashboard_port=dashboard_port,
        )
    
    # ========== STAGE 1 WRAPPERS (for tracking) ==========
    
    def _on_trade_wrapper(self, trade) -> None:
        """Wrap trade callback to track freshness"""
        symbol = trade.symbol
        if symbol in self._trackers:
            self._trackers[symbol].trade_count += 1
            self._trackers[symbol].last_trade_ms = trade.timestamp_ms
        self.stage2.on_trade(trade)
    
    def _on_book_wrapper(self, snapshot) -> None:
        """Wrap book callback to track freshness"""
        symbol = snapshot.symbol
        if symbol in self._trackers:
            self._trackers[symbol].last_book_ms = snapshot.timestamp_ms
        self.stage2.on_book_snapshot(snapshot)
    
    def _on_funding_wrapper(self, funding) -> None:
        """Wrap funding callback to track freshness"""
        symbol = funding.symbol
        if symbol in self._trackers:
            self._trackers[symbol].last_funding_ms = int(time.time() * 1000)
        self.stage2.on_funding(funding)
    
    def _on_oi_wrapper(self, oi) -> None:
        """Wrap OI callback to track freshness"""
        symbol = oi.symbol
        if symbol in self._trackers:
            self._trackers[symbol].last_oi_ms = int(time.time() * 1000)
        self.stage2.on_oi(oi)
    
    # ========== STAGE 2 CALLBACK ==========
    
    async def _on_market_state(self, state: MarketState) -> None:
        """
        Callback when Stage 2 emits a MarketState.
        Process through Stage 3-5 and update dashboard with ALL Stage 2 values.
        """
        try:
            await self._process_market_state(state)
        except Exception as e:
            logger.error("market_state_processing_error", symbol=state.symbol, error=str(e))
    
    async def _process_market_state(self, state: MarketState) -> None:
        """Internal processing of market state - wrapped with error handling"""
        symbol = state.symbol
        now_ms = int(time.time() * 1000)
        
        # Get tracker data
        tracker = self._trackers.get(symbol, SymbolTracker(symbol=symbol))
        
        # Process through Stage 3 (which internally calls Stage 4, 4.5, 5)
        thesis = self.thesis_engine.process(state)
        
        # Get ML results (Stage 4/4.5 filters REMOVED)
        ml_prediction = self.thesis_engine.get_ml_prediction(symbol)
        
        # Build complete pipeline state dict with ALL Stage 2 values
        data_fresh = (now_ms - tracker.last_trade_ms) < 5000 if tracker.last_trade_ms > 0 else False
        
        # Determine stage statuses based on veto_reason prefix
        # Stage 3: Signal fires if direction is set
        signal_fired = thesis.direction != Direction.NONE
        
        # Stage 4/4.5 REMOVED - filters destroyed edge via over-filtering
        # Signals now go directly from Stage 3 to Stage 5
        
        # Stage 5 status - only pass if both probabilities meet minimum threshold
        stage5_pass = False
        if ml_prediction and signal_fired and thesis.allowed:
            prob_60_ok = ml_prediction.prob_60 >= PROB_MIN_THRESHOLD
            prob_300_ok = ml_prediction.prob_300 >= PROB_MIN_THRESHOLD
            stage5_pass = prob_60_ok and prob_300_ok
        
        # Final trade status
        is_trade = thesis.allowed and stage5_pass
        
        # Check if symbol is in HOLDING state (already has active position)
        is_holding = self.position_sizer.is_holding(symbol)
        
        # Stage 6: Position Sizing (only when trade passes Stage 5 and NOT holding)
        stage6_pass = False
        stage6_rejection = ""
        position_result = None
        
        if is_trade and is_holding:
            # Symbol is holding - don't allow new trades
            is_trade = False
            stage6_rejection = "HOLDING - position already active"
        
        # Check startup delay - disable trading for first 6 minutes
        time_since_start = time.time() - self._start_time
        trading_enabled = time_since_start >= self._trading_enabled_after
        
        if is_trade and not trading_enabled:
            is_trade = False
            remaining = int(self._trading_enabled_after - time_since_start)
            stage6_rejection = f"WARMUP - {remaining}s remaining"
        
        # Check risk manager limits (drawdown/profit protection)
        if is_trade:
            can_trade, pause_reason = self.trade_manager.risk_manager.can_trade()
            if not can_trade:
                is_trade = False
                stage6_rejection = pause_reason or "RISK_LIMIT"
        
        # Check if we need to flatten all positions (hard stop triggered)
        if self.trade_manager.risk_manager.should_flatten_all():
            await self.trade_manager.flatten_all("HARD_STOP_-7%")
        
        if is_trade:
            # Get ATR as percentage
            atr_5m_pct = state.volatility.atr_5m / state.price if state.price > 0 else 0
            atr_1h_pct = state.volatility.atr_1h / state.price if state.price > 0 else 0
            
            # Get signal name from reasons
            signal_name = ", ".join([s.name for s in thesis.reasons]) if thesis.reasons else "unknown"
            
            # Update position sizer with current equity
            account = self.trade_manager.get_account_state()
            self.position_sizer.set_equity(account.current_equity)
            
            # Get probabilities from prediction for dynamic risk sizing
            prob_60 = ml_prediction.prob_60 if ml_prediction else None
            prob_300 = ml_prediction.prob_300 if ml_prediction else None
            
            # Calculate position with dynamic risk
            position_result = self.position_sizer.calculate_position(
                symbol=symbol,
                side=thesis.direction.value,
                entry_price=state.price,
                atr_5m_pct=atr_5m_pct,
                atr_1h_pct=atr_1h_pct,
                signal_name=signal_name,
                current_price=state.price,
                prob_60=prob_60,
                prob_300=prob_300,
            )
            
            stage6_pass = position_result.allowed
            if not position_result.allowed:
                stage6_rejection = position_result.rejection_reason or "unknown"
                is_trade = False  # Reject the trade if Stage 6 fails
            else:
                # Stage 7: Place order via TradeManager
                positions = position_result.positions
                if len(positions) >= 2:
                    tranche_a = positions[0]
                    tranche_b = positions[1]
                    
                    # Check margin availability
                    total_notional = tranche_a.notional + tranche_b.notional
                    if not self.trade_manager.has_margin_available(total_notional):
                        stage6_rejection = "INSUFFICIENT_MARGIN"
                        is_trade = False
                    else:
                        # Place order via trade manager
                        order_ids = await self.trade_manager.place_position(
                            symbol=symbol,
                            side=thesis.direction.value,
                            entry_price=state.price,
                            tranche_a_size=tranche_a.size,
                            tranche_a_stop=tranche_a.stop,
                            tranche_a_tp=tranche_a.tp_a,
                            tranche_a_breakeven=tranche_a.breakeven,
                            tranche_a_risk=tranche_a.risk,
                            tranche_b_size=tranche_b.size,
                            tranche_b_stop=tranche_b.stop,
                            tranche_b_tp_partial=tranche_b.tp_b_partial,
                            tranche_b_tp_runner=tranche_b.tp_b_runner,
                            tranche_b_breakeven=tranche_b.breakeven,
                            tranche_b_risk=tranche_b.risk,
                            signal_name=signal_name,
                            mode="test",
                        )
                        
                        if not order_ids:
                            stage6_rejection = "ORDER_PLACEMENT_FAILED"
                            is_trade = False
        
        # Build market state dict for exit strategy evaluation
        exit_market_state = {
            "regime": state.regime.value if hasattr(state.regime, 'value') else str(state.regime),
            "absorption_z": state.absorption.absorption_z,
            "dist_poc": state.structure.dist_poc,
            "dist_lvn": state.structure.dist_lvn,
            "atr_5m": state.volatility.atr_5m,
            "atr_1h": state.volatility.atr_1h,
            "vah": state.structure.vah,
            "val": state.structure.val,
            "acceptance_outside_value": state.structure.acceptance_outside_value,
            "moi_z": state.order_flow.moi_z if hasattr(state.order_flow, 'moi_z') else 0.0,
            "delta_vel_z": state.order_flow.delta_velocity_z if hasattr(state.order_flow, 'delta_velocity_z') else 0.0,
            "liq_long_usd": state.liquidations.long_usd_30s,
            "liq_short_usd": state.liquidations.short_usd_30s,
        }
        
        # Update trade manager with current price and market state for exit evaluation
        trade_events = await self.trade_manager.update_price(symbol, state.price, exit_market_state)
        
        # Check if all tranches for this symbol are now closed - if so, clear holding state
        symbol_tranches = [t for t in self.trade_manager.get_open_tranches() if t.symbol == symbol]
        if not symbol_tranches and self.position_sizer.is_holding(symbol):
            self.position_sizer.clear_position(symbol)
            logger.info("holding_state_cleared", symbol=symbol, reason="all_tranches_closed")
        
        # Get account state for dashboard
        account_state = self.trade_manager.get_account_state()
        
        # Update risk manager with current equity for limit checks
        self.trade_manager.risk_manager.update_equity(
            account_state.current_equity,
            account_state.unrealized_pnl,
        )
        
        # Broadcast open positions with live PnL data (always broadcast to clear when empty)
        open_tranches = self.trade_manager.get_open_tranches()
        open_positions_data = [
            {
                "order_id": t.order_id,
                "symbol": t.symbol,
                "side": t.side,
                "tranche": t.tranche,
                "entry_price": t.entry_price,
                "current_price": t.current_price,
                "size": t.size,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "tp_partial": t.tp_partial,
                "tp_runner": t.tp_runner,
                "unrealized_pnl": t.unrealized_pnl,
                "realized_pnl": t.realized_pnl,
                "r_multiple": t.r_multiple,
                "status": t.status,
                "signal_name": t.signal_name,
                "created_at": t.created_at,
            }
            for t in open_tranches
        ]
        await broadcast_open_positions(open_positions_data)
        
        # Broadcast closed trades from risk manager
        closed_trades = self.trade_manager.risk_manager.get_closed_trades(limit=20)
        if closed_trades:
            await broadcast_closed_trades(closed_trades)
        
        # Build complete state dict
        ps = {
            "symbol": symbol,
            "timestamp_ms": now_ms,
            "price": state.price,
            
            # ===== STAGE 1: Data Ingestion =====
            "stage1_ok": data_fresh,
            "trades_per_sec": tracker.trades_per_sec(),
            "last_trade_ms": tracker.last_trade_ms,
            "last_book_ms": tracker.last_book_ms,
            "last_funding_ms": tracker.last_funding_ms,
            
            # ===== STAGE 2: ALL VALUES =====
            # Regime
            "regime": state.regime.value,
            "regime_rejected": state.regime == Regime.CHOP,
            "regime_confidence": state.regime_inputs.get("confidence", 0.0) if state.regime_inputs else 0.0,
            "regime_expansion_score": state.regime_inputs.get("expansion_score", 0) if state.regime_inputs else 0,
            "regime_compression_score": state.regime_inputs.get("compression_score", 0) if state.regime_inputs else 0,
            "regime_chop_score": state.regime_inputs.get("chop_score", 0) if state.regime_inputs else 0,
            "time_in_regime": state.time_in_regime,
            "price_change_5m": state.price_change_5m,
            
            # Order Flow
            "of_moi_250ms": state.order_flow.moi_250ms,
            "of_moi_1s": state.order_flow.moi_1s,
            "of_delta_velocity": state.order_flow.delta_velocity,
            "of_aggression_persistence": state.order_flow.aggression_persistence,
            "of_moi_std": state.order_flow.moi_std,
            "of_moi_flip_rate": state.order_flow.moi_flip_rate,
            
            # Absorption
            "abs_absorption_z": state.absorption.absorption_z,
            "abs_refill_rate": state.absorption.refill_rate,
            "abs_liquidity_sweep": state.absorption.liquidity_sweep,
            "abs_depth_imbalance": state.absorption.depth_imbalance,
            "abs_bid_depth_usd": state.absorption.bid_depth_usd,
            "abs_ask_depth_usd": state.absorption.ask_depth_usd,
            
            # Volatility
            "vol_atr_5m": state.volatility.atr_5m,
            "vol_atr_1h": state.volatility.atr_1h,
            "vol_vol_expansion_ratio": state.volatility.vol_expansion_ratio,
            "vol_vol_rank": state.volatility.vol_rank,
            "vol_vol_5m": state.volatility.vol_5m,
            "vol_vol_regime": state.volatility.vol_regime,
            
            # Structure
            "str_poc": state.structure.poc,
            "str_vah": state.structure.vah,
            "str_val": state.structure.val,
            "str_lvns": state.structure.lvns,
            "str_value_area_width": state.structure.value_area_width,
            "str_dist_poc": state.structure.dist_poc,
            "str_dist_lvn": state.structure.dist_lvn,
            "str_value_width_ratio": state.structure.value_width_ratio,
            "str_time_inside_value_pct": state.structure.time_inside_value_pct,
            "str_acceptance_outside_value": state.structure.acceptance_outside_value,
            
            # Liquidations
            "liq_long_usd_30s": state.liquidations.long_usd_30s,
            "liq_short_usd_30s": state.liquidations.short_usd_30s,
            "liq_imbalance_30s": state.liquidations.imbalance_30s,
            "liq_long_usd_2m": state.liquidations.long_usd_2m,
            "liq_short_usd_2m": state.liquidations.short_usd_2m,
            "liq_imbalance_2m": state.liquidations.imbalance_2m,
            "liq_long_usd_5m": state.liquidations.long_usd_5m,
            "liq_short_usd_5m": state.liquidations.short_usd_5m,
            "liq_imbalance_5m": state.liquidations.imbalance_5m,
            "liq_cascade_active": state.liquidations.cascade_active,
            "liq_exhaustion": state.liquidations.exhaustion,
            
            # Funding & OI (Derivatives)
            "fund_rate": state.funding.rate,
            "fund_funding_z": state.funding.funding_z,
            "fund_crowd_side": state.funding.crowd_side.value,
            "fund_annualized_pct": state.funding.annualized_pct,
            "oi_oi": state.oi.oi,
            "oi_oi_delta_1m": state.oi.oi_delta_1m,
            "oi_oi_delta_5m": state.oi.oi_delta_5m,
            "oi_participation_type": state.oi.participation_type.value,
            
            # ===== STAGE 3: Signal Detection =====
            "signal_fired": signal_fired or (thesis.direction != Direction.NONE),
            "signal_direction": thesis.direction.value,
            "signal_strength": thesis.strength,
            "signal_reasons": [s.name for s in thesis.reasons] if thesis.reasons else [],
            "stage3_veto": thesis.veto_reason or "",
            
            # Stage 4/4.5 REMOVED - filters destroyed edge via over-filtering
            # Signals now go directly from Stage 3 to Stage 5
            
            # ===== STAGE 5: ML Prediction V3 (probability-based) =====
            "stage5_pass": stage5_pass,
            "prob_60": ml_prediction.prob_60 if (ml_prediction and signal_fired) else 0.0,
            "prob_300": ml_prediction.prob_300 if (ml_prediction and signal_fired) else 0.0,
            "percentile_60": ml_prediction.percentile_60 if (ml_prediction and signal_fired) else 0.0,
            "percentile_300": ml_prediction.percentile_300 if (ml_prediction and signal_fired) else 0.0,
            "model_used": ml_prediction.model_300 if (ml_prediction and signal_fired) else "",
            
            # ===== STAGE 6: Position Sizing =====
            "stage6_pass": stage6_pass,
            "stage6_rejection": stage6_rejection,
            "is_holding": is_holding,
            "position_result": position_result.to_dict() if position_result else None,
            
            # ===== TRADE STATUS =====
            "is_trade": is_trade,
            "trade_direction": thesis.direction.value if is_trade else "",
            "trade_price": state.price if is_trade else 0.0,
            "trade_time": datetime.now().strftime("%H:%M:%S") if is_trade else "",
            
            # ===== ACCOUNT STATE (Stage 7) =====
            "account_equity": account_state.current_equity,
            "account_margin_available": account_state.margin_available,
            "account_margin_used": account_state.margin_used,
            "account_realized_pnl": account_state.realized_pnl,
            "account_unrealized_pnl": account_state.unrealized_pnl,
            "account_total_r": account_state.total_r,
        }
        
        # Store state
        self._pipeline_states[symbol] = ps
        
        # Broadcast trade and position if new
        if is_trade and position_result and position_result.allowed:
            trade_data = {
                "symbol": symbol,
                "direction": thesis.direction.value,
                "price": state.price,
                "time": ps["trade_time"],
                "prob_60": ps["prob_60"],
                "prob_300": ps["prob_300"],
                "strength": thesis.strength,
                "signals": ps["signal_reasons"],
            }
            await broadcast_trade(trade_data)
            
            # Broadcast position with all tranche details
            position_data = {
                "symbol": symbol,
                "side": position_result.side,
                "entry_price": position_result.entry_price,
                "signal_name": position_result.signal_name,
                "time": position_result.timestamp,
                "total_risk": position_result.total_risk,
                "positions": [p.to_dict() for p in position_result.positions],
            }
            await broadcast_position(position_data)
            
            logger.info(
                "trade_signal_with_position",
                symbol=symbol,
                direction=thesis.direction.value,
                price=state.price,
                prob_300=f"{ps['prob_300']:.1%}",
                signal=position_result.signal_name,
                tranche_a_size=position_result.positions[0].size if position_result.positions else 0,
                tranche_b_size=position_result.positions[1].size if len(position_result.positions) > 1 else 0,
            )
        
        # Broadcast rejection if Stage 6 failed
        if position_result and not position_result.allowed:
            rejection_data = {
                "symbol": symbol,
                "reason": stage6_rejection,
                "atr_1h_pct": position_result.atr_1h_pct,
                "signal_name": position_result.signal_name,
                "time": position_result.timestamp,
            }
            await broadcast_rejection(rejection_data)
        
        # Broadcast to dashboard
        if self.enable_dashboard:
            await broadcast_pipeline_state(symbol, ps)
    
    def _on_thesis(self, symbol: str, thesis: Thesis) -> None:
        """Callback when Stage 3 emits a Thesis (for logging)"""
        if thesis.allowed:
            logger.info(
                "thesis_allowed",
                symbol=symbol,
                direction=thesis.direction.value,
                strength=f"{thesis.strength:.2f}",
            )
    
    # ========== LIFECYCLE ==========
    
    async def start(self) -> None:
        """Start all pipeline components"""
        self._running = True
        
        logger.info(
            "global_pipeline_starting",
            symbols=self.symbols,
            dashboard_port=self.dashboard_port if self.enable_dashboard else "disabled",
        )
        
        # NOTE: ATR bootstrap removed - use GlobalPipelineRunnerV3 for new pipeline
        # Old volatility bootstrap skipped, Stage 2 stub processor handles this
        logger.info("skipping_old_bootstrap", reason="Use GlobalPipelineRunnerV3 for V3 pipeline")
        
        # Step 1: Initialize Stage 1
        await self.stage1.initialize()
        
        # Step 3: Start all components
        tasks = [
            asyncio.create_task(self.stage1.start()),
            asyncio.create_task(self.stage2.start_update_loop()),
        ]
        
        if self.enable_dashboard:
            from src.dashboard.global_dashboard import start_dashboard_async
            tasks.append(asyncio.create_task(
                start_dashboard_async("0.0.0.0", self.dashboard_port)
            ))
        
        try:
            # return_exceptions=True prevents one task failure from crashing others
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error("task_failed", task_index=i, error=str(result))
        except asyncio.CancelledError:
            logger.info("global_pipeline_cancelled")
    
    async def stop(self) -> None:
        """Stop all components gracefully"""
        if not self._running:
            return
        
        self._running = False
        logger.info("global_pipeline_stopping")
        
        try:
            # Stop stages with timeout
            await asyncio.wait_for(self.stage1.stop(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("stage1_stop_timeout")
        except Exception as e:
            logger.warning("stage1_stop_error", error=str(e))
        
        try:
            await asyncio.wait_for(self.stage2.stop(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("stage2_stop_timeout")
        except Exception as e:
            logger.warning("stage2_stop_error", error=str(e))
        
        logger.info("global_pipeline_stopped")
    
    # ========== PUBLIC API ==========
    
    def get_pipeline_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current pipeline state for symbol"""
        return self._pipeline_states.get(symbol)
    
    def get_all_pipeline_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all pipeline states"""
        return self._pipeline_states.copy()
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get all symbols currently showing trade signals"""
        trades = []
        for symbol, ps in self._pipeline_states.items():
            if ps.get("is_trade"):
                trades.append({
                    "symbol": symbol,
                    "direction": ps.get("trade_direction", ""),
                    "price": ps.get("trade_price", 0.0),
                    "time": ps.get("trade_time", ""),
                    "prob_300": ps.get("prob_300", 0.0),
                })
        return trades
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get overall pipeline health"""
        stage1_ok = sum(1 for ps in self._pipeline_states.values() if ps.get("stage1_ok"))
        non_chop = sum(1 for ps in self._pipeline_states.values() if ps.get("regime") != "CHOP")
        signals = sum(1 for ps in self._pipeline_states.values() if ps.get("signal_fired"))
        s4_pass = sum(1 for ps in self._pipeline_states.values() if ps.get("stage4_pass"))
        s45_pass = sum(1 for ps in self._pipeline_states.values() if ps.get("stage45_pass"))
        s5_pass = sum(1 for ps in self._pipeline_states.values() if ps.get("stage5_pass"))
        s6_pass = sum(1 for ps in self._pipeline_states.values() if ps.get("stage6_pass"))
        trades = sum(1 for ps in self._pipeline_states.values() if ps.get("is_trade"))
        
        return {
            "total_symbols": len(self.symbols),
            "stage1_ok": stage1_ok,
            "non_chop_regimes": non_chop,
            "signals_fired": signals,
            "stage4_passed": s4_pass,
            "stage45_passed": s45_pass,
            "stage5_passed": s5_pass,
            "stage6_passed": s6_pass,
            "active_trades": trades,
            "thesis_engine_stats": self.thesis_engine.get_health_metrics(),
            "position_sizer_stats": self.position_sizer.get_health_metrics(),
        }


async def run_global_pipeline(
    symbols: Optional[List[str]] = None,
    duration_seconds: Optional[int] = None,
    dashboard_port: int = 8888,
) -> None:
    """
    Run the global pipeline with all 6 stages.
    
    Args:
        symbols: List of symbols to track (default: all from settings)
        duration_seconds: Run for this many seconds then stop (None = forever)
        dashboard_port: Port for global dashboard UI
    """
    runner = GlobalPipelineRunner(
        symbols=symbols,
        dashboard_port=dashboard_port,
        enable_dashboard=True,
    )
    
    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("shutdown_signal_received")
        asyncio.create_task(runner.stop())
    
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
    
    print(f"\n{'='*70}")
    print("HYDRA Global Pipeline - All 6 Stages")
    print(f"{'='*70}")
    print(f"Symbols: {runner.symbols}")
    print(f"Dashboard: http://localhost:{dashboard_port}")
    print(f"")
    print("Pipeline Flow:")
    print("  Stage 1: Data Ingestion (WebSocket + REST)")
    print("  Stage 2: Feature Computation + Regime Classification")
    print("  Stage 3: Signal Detection (Thesis Generation)")
    print("  Stage 4: Structural Location Filter")
    print("  Stage 4.5: Orderflow Confirmation")
    print("  Stage 5: ML Prediction V3 (prob_60 >= 50% AND prob_300 >= 50%)")
    print("  Stage 6: Position Sizing (2 tranches with SL/TP)")
    print(f"")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*70}\n")
    
    task = asyncio.create_task(runner.start())
    
    try:
        if duration_seconds:
            await asyncio.sleep(duration_seconds)
            await runner.stop()
        else:
            await task
    except KeyboardInterrupt:
        print("\nShutting down...")
        await runner.stop()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("fatal_pipeline_error", error=str(e))
        print(f"\n[FATAL ERROR] {e}")
        await runner.stop()


def global_exception_handler(loop, context):
    """Global exception handler for uncaught exceptions in async tasks"""
    exception = context.get("exception")
    message = context.get("message", "Unknown error")
    logger.error(
        "uncaught_async_exception",
        message=message,
        exception=str(exception) if exception else "None",
    )
    # Don't crash - just log and continue


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(global_exception_handler)
    try:
        loop.run_until_complete(run_global_pipeline())
    except KeyboardInterrupt:
        pass
    finally:
        # Cancel all pending tasks before closing
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        # Wait for cancellation with timeout
        if pending:
            loop.run_until_complete(
                asyncio.wait(pending, timeout=5.0)
            )
        
        loop.close()
