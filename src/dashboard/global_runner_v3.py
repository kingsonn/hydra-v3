"""
Global Pipeline Runner V3 - Hybrid Alpha System
================================================

Simplified pipeline for Stage 3 V3 hybrid signals:
- Stage 1: Data Ingestion (trades, orderbook, derivatives)
- Stage 2: Feature Computation + Alpha State
- Stage 3 V3: Hybrid Signal Detection (FUNDING_TREND, LIQUIDATION_FOLLOW, OI_DIVERGENCE)
- Stage 6 V3: Position Sizing (2% risk, signal stop/target)
- Stage 7 V3: Trade Management (TP/SL/trail only)

DISABLED:
- Stage 5 ML gating (not needed for structural alpha)
- Complex exit strategies
- Tranche splitting

Startup sequence:
1. Bootstrap historical data (OI, funding, price, liquidations)
2. Initialize Stage 1 data collection
3. Start alpha state processor
4. Start Stage 3 V3 signals
5. Launch dashboard
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
from src.collectors.bootstrap import AlphaDataBootstrap
from src.stage2.processors.alpha_state import AlphaStateProcessor
from src.stage3_v3.models import MarketState, HybridSignal, Direction, MarketRegime
from src.stage3_v3.bias import BiasCalculator
from src.stage3_v3.regime import RegimeClassifier, get_regime_classifier
from src.stage3_v3.signals import (
    # Positional signals (background alpha)
    FundingPressureContinuation,
    TrendPullbackSignal,
    LiquidationCascadeSignal,
    CompressedRangeBreakout,
    TrendExhaustionReversal,
    # Entry-first signals (frequency alpha)
    EMATrendContinuation,
    ADXExpansionMomentum,
    StructureBreakRetest,
    CompressionBreakout,
    SMACrossover,
)
from src.stage6.position_sizer_v3 import PositionSizerV3, PositionResultV3
from src.stage7.trade_manager_v3 import TradeManagerV3
from src.dashboard.global_dashboard_v3 import broadcast_closed_trades, set_close_position_callback, set_manual_signal_callback

logger = structlog.get_logger(__name__)


@dataclass
class SymbolTrackerV3:
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
        return self.trade_count / elapsed_s if elapsed_s > 0 else 0.0


@dataclass
class HourlyBarBuilder:
    """Builds 1-hour bars from trades for bootstrap price history"""
    symbol: str
    bar_start_ms: int = 0
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = float('inf')
    close_price: float = 0.0
    trade_count: int = 0
    
    def reset(self, start_ms: int, price: float):
        """Reset for new bar"""
        self.bar_start_ms = start_ms
        self.open_price = price
        self.high_price = price
        self.low_price = price
        self.close_price = price
        self.trade_count = 1
    
    def update(self, price: float):
        """Update with new trade price"""
        if self.trade_count == 0:
            self.open_price = price
            self.low_price = price
        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.close_price = price
        self.trade_count += 1
    
    def is_complete(self, current_ms: int) -> bool:
        """Check if 1 hour has elapsed"""
        if self.bar_start_ms == 0:
            return False
        return current_ms >= self.bar_start_ms + 3600_000  # 1 hour in ms
    
    def get_bar_data(self) -> tuple:
        """Return (open, high, low, close)"""
        return (self.open_price, self.high_price, self.low_price, self.close_price)


class GlobalPipelineRunnerV3:
    """
    Runs the V3 hybrid alpha pipeline.
    
    Pipeline Flow:
    1. Stage 1: Data Ingestion
    2. Stage 2: Alpha State Computation  
    3. Stage 3 V3: Hybrid Signal Detection
    4. Stage 6 V3: Position Sizing
    5. Stage 7 V3: Trade Management
    
    NO Stage 5 ML gating - structural alpha doesn't need it.
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        dashboard_port: int = 8889,  # Different port from old runner
        enable_dashboard: bool = True,
        initial_equity: float = 1000.0,
        live_trading: bool = False,  # Enable real order execution via WEEX
        dry_run: bool = True,  # If live_trading=True, dry_run=True simulates without real orders
    ):
        self.symbols = symbols or settings.SYMBOLS
        self.dashboard_port = dashboard_port
        self.enable_dashboard = enable_dashboard
        self.initial_equity = initial_equity
        self.live_trading = live_trading
        self.dry_run = dry_run
        
        # Warmup delay - 5 minutes to let data stabilize
        self._start_time = time.time()
        self._warmup_seconds = 5 * 60
        
        # Per-symbol tracking
        self._trackers: Dict[str, SymbolTrackerV3] = {
            s: SymbolTrackerV3(symbol=s, start_time_ms=int(time.time() * 1000))
            for s in self.symbols
        }
        
        # Pipeline state for dashboard
        self._pipeline_states: Dict[str, Dict[str, Any]] = {
            s: {"symbol": s} for s in self.symbols
        }
        
        # Bootstrap for historical data
        self.bootstrap = AlphaDataBootstrap(self.symbols)
        
        # Alpha state processor
        self.alpha_processor = AlphaStateProcessor(self.symbols)
        
        # Bias and regime calculators
        self.bias_calculator = BiasCalculator()
        self.regime_classifier = get_regime_classifier()  # Singleton with persistence
        
        # Stage 3 V3 signals - All 10 hybrid signals active
        self._signals = {
            # Positional signals (background alpha - 5 signals)
            "FUNDING_PRESSURE": FundingPressureContinuation(),
            "TREND_PULLBACK": TrendPullbackSignal(),
            "LIQUIDATION_CASCADE": LiquidationCascadeSignal(),
            "RANGE_BREAKOUT": CompressedRangeBreakout(),
            "TREND_EXHAUSTION": TrendExhaustionReversal(),
            # Entry-first signals (frequency alpha - 5 signals)
            "EMA_CONTINUATION": EMATrendContinuation(),
            "ADX_EXPANSION": ADXExpansionMomentum(),
            "STRUCTURE_BREAK": StructureBreakRetest(),
            "COMPRESSION_BREAK": CompressionBreakout(),
            "SMA_CROSSOVER": SMACrossover(),
        }
        
        # Stage 6 V3 position sizer
        self.position_sizer = PositionSizerV3(risk_pct=0.02)
        
        # Stage 7 V3 trade manager (with WEEX execution if live_trading enabled)
        self.trade_manager = TradeManagerV3(
            initial_equity=initial_equity,
            reset_on_start=True,
            live_trading=live_trading,
            dry_run=dry_run,
        )
        
        # Set up callback for closed trades broadcast
        self.trade_manager.set_trade_closed_callback(self._on_trade_closed)
        
        # Register manual close callback for dashboard Signal Status button
        set_close_position_callback(self._manual_close_position)
        
        # Register manual signal callback for dashboard Long/Short buttons
        set_manual_signal_callback(self._manual_signal_trigger)
        
        # Stage 1 orchestrator (will be initialized in start)
        self.stage1: Optional[Stage1Orchestrator] = None
        
        self._running = False
        self._last_bar_update: Dict[str, int] = {s: 0 for s in self.symbols}
        self._last_health_log: float = 0
        self._health_log_interval: int = 3600  # Log health every hour
        self._error_counts: Dict[str, int] = {}  # Track errors per symbol
        self._max_errors_before_reset: int = 100  # Reset error count threshold
        
        # Hourly bar builders for live price integration with bootstrap
        self._hourly_bars: Dict[str, HourlyBarBuilder] = {
            s: HourlyBarBuilder(symbol=s) for s in self.symbols
        }
        
        logger.info(
            "global_pipeline_v3_initialized",
            symbols=self.symbols,
            dashboard_port=dashboard_port,
            signals=list(self._signals.keys()),
        )
    
    # ========== DATA CALLBACKS ==========
    
    def _on_trade(self, trade) -> None:
        """Handle trade data and build hourly bars for bootstrap"""
        symbol = trade.symbol
        now_ms = trade.timestamp_ms
        price = trade.price
        
        if symbol in self._trackers:
            self._trackers[symbol].trade_count += 1
            self._trackers[symbol].last_trade_ms = now_ms
        
        # Update alpha state with price
        self.alpha_processor.update_price(symbol, price, now_ms)
        
        # Build hourly bars for live bootstrap price history
        if symbol in self._hourly_bars:
            bar_builder = self._hourly_bars[symbol]
            
            # Check if current bar is complete
            if bar_builder.is_complete(now_ms) and bar_builder.trade_count > 0:
                # Finalize the completed bar and update bootstrap
                open_, high, low, close = bar_builder.get_bar_data()
                self.bootstrap.update_price_bar(
                    symbol, bar_builder.bar_start_ms, open_, high, low, close
                )
                
                # Update alpha state with completed bar
                self.alpha_processor.update_price_bar(symbol, high, low, close, now_ms)
                
                # Update ATR from bootstrap
                self.alpha_processor.update_atr(
                    symbol,
                    self.bootstrap.get_atr_short(symbol),
                    self.bootstrap.get_atr_long(symbol),
                )
                
                # Update price changes from bootstrap
                # For 1h: use live price for rolling calculation
                # For 4h/24h: use bar closes (less frequent updates needed)
                self.alpha_processor.update_price_changes(
                    symbol,
                    self.bootstrap.get_price_change(symbol, 1, current_price=price),
                    self.bootstrap.get_price_change(symbol, 4),
                    self.bootstrap.get_price_change(symbol, 24),
                )
                
                # Update price ranges from bootstrap
                h4, l4 = self.bootstrap.get_high_low_4h(symbol)
                h24, l24 = self.bootstrap.get_high_low_24h(symbol)
                self.alpha_processor.update_price_ranges(symbol, h4, l4, h24, l24)
                
                logger.info(
                    "hourly_bar_completed",
                    symbol=symbol,
                    open=f"{open_:.2f}",
                    high=f"{high:.2f}",
                    low=f"{low:.2f}",
                    close=f"{close:.2f}",
                )
                
                # Start new bar aligned to hour boundary
                hour_start = (now_ms // 3600_000) * 3600_000
                bar_builder.reset(hour_start, price)
            
            elif bar_builder.bar_start_ms == 0:
                # First trade - initialize bar aligned to current hour
                hour_start = (now_ms // 3600_000) * 3600_000
                bar_builder.reset(hour_start, price)
            else:
                # Update current bar
                bar_builder.update(price)
    
    def _on_bar(self, bar) -> None:
        """Handle bar completion - update alpha state"""
        symbol = bar.symbol
        now_ms = int(time.time() * 1000)
        
        # Only process hourly bars for alpha state
        # Check if it's been at least 55 minutes since last update
        last_update = self._last_bar_update.get(symbol, 0)
        if now_ms - last_update < 55 * 60 * 1000:
            return
        
        self._last_bar_update[symbol] = now_ms
        
        # Update bootstrap with new bar
        self.bootstrap.update_price_bar(
            symbol, now_ms, bar.open, bar.high, bar.low, bar.close
        )
        
        # Update alpha state
        self.alpha_processor.update_price_bar(
            symbol, bar.high, bar.low, bar.close, now_ms
        )
        
        # Update ATR from bootstrap
        self.alpha_processor.update_atr(
            symbol,
            self.bootstrap.get_atr_short(symbol),
            self.bootstrap.get_atr_long(symbol),
        )
    
    def _on_funding(self, funding) -> None:
        """Handle funding rate update"""
        symbol = funding.symbol
        if symbol in self._trackers:
            self._trackers[symbol].last_funding_ms = int(time.time() * 1000)
        
        # Update bootstrap
        self.bootstrap.update_funding(
            symbol,
            funding.timestamp_ms,
            funding.funding_rate,
        )
        
        # Update alpha state
        funding_z = self.bootstrap.get_funding_z(symbol)
        self.alpha_processor.update_funding(
            symbol,
            funding.funding_rate,
            funding_z,
            funding.timestamp_ms,
        )
    
    def _on_oi(self, oi) -> None:
        """Handle OI update"""
        symbol = oi.symbol
        now_ms = int(time.time() * 1000)
        
        if symbol in self._trackers:
            self._trackers[symbol].last_oi_ms = now_ms
        
        # Update bootstrap (every 5 min)
        self.bootstrap.update_oi(symbol, now_ms, oi.open_interest)
        
        # Update alpha state
        self.alpha_processor.update_oi(
            symbol,
            self.bootstrap.get_oi_change(symbol, 60),
            self.bootstrap.get_oi_change(symbol, 240),
            self.bootstrap.get_oi_change(symbol, 1440),
        )
        
        # Update price changes
        # Get current price from alpha state for live 1h calculation
        alpha_state = self.alpha_processor.get_state(symbol)
        current_price = alpha_state.current_price if alpha_state else 0.0
        
        self.alpha_processor.update_price_changes(
            symbol,
            self.bootstrap.get_price_change(symbol, 1, current_price=current_price),
            self.bootstrap.get_price_change(symbol, 4),
            self.bootstrap.get_price_change(symbol, 24),
        )
        
        # Update price ranges
        h4, l4 = self.bootstrap.get_high_low_4h(symbol)
        h24, l24 = self.bootstrap.get_high_low_24h(symbol)
        self.alpha_processor.update_price_ranges(symbol, h4, l4, h24, l24)
    
    def _on_liquidation(self, liq) -> None:
        """Handle liquidation event"""
        symbol = liq.symbol
        now_ms = int(time.time() * 1000)
        
        # Determine if long or short liquidation
        is_long_liq = liq.side.value == "BUY"  # Buyer liquidated = long position closed
        long_usd = liq.notional if is_long_liq else 0
        short_usd = liq.notional if not is_long_liq else 0
        
        # Update bootstrap
        self.bootstrap.update_liquidation(symbol, now_ms, long_usd, short_usd)
        
        # Get totals from bootstrap
        l1h, s1h = self.bootstrap.get_liq_totals(symbol, 60)
        l4h, s4h = self.bootstrap.get_liq_totals(symbol, 240)
        l8h, s8h = self.bootstrap.get_liq_totals(symbol, 480)
        l24h, s24h = self.bootstrap.get_liq_totals(symbol, 1440)
        
        # Update alpha state
        self.alpha_processor.update_liquidations(
            symbol, l1h, s1h, l4h, s4h, l8h, s8h, l24h, s24h
        )
    
    # ========== SIGNAL PROCESSING ==========
    
    async def _process_symbol(self, symbol: str) -> None:
        """Process a symbol through the signal pipeline"""
        try:
            # Get market state from alpha processor
            market_state = self.alpha_processor.get_market_state(symbol)
            if market_state is None:
                return
            
            # Add bar history from bootstrap (for signals that need SMA etc.)
            market_state.bar_closes_1h = self.bootstrap.get_bar_closes_1h(symbol)
            
            # Calculate bias
            alpha_state = self.alpha_processor.get_state(symbol)
            if alpha_state is None:
                return
            
            bias = self.bias_calculator.calculate(
                funding_z=alpha_state.funding_z,
                liq_imbalance_4h=alpha_state.liq_imbalance_4h,
                oi_delta_24h=alpha_state.oi_change_24h,
                price_change_4h=alpha_state.price_change_4h,
                price_change_24h=alpha_state.price_change_24h,
            )
            market_state.bias = bias
            
            # Update regime based on trend - using new RegimeClassifier API
            trend = market_state.trend
            regime, confidence = self.regime_classifier.classify(
                symbol=symbol,
                higher_high=trend.higher_high,
                higher_low=trend.higher_low,
                lower_high=trend.lower_high,
                lower_low=trend.lower_low,
                price_change_1h=alpha_state.price_change_1h,
                price_change_4h=alpha_state.price_change_4h,
                vol_expansion_ratio=alpha_state.vol_expansion_ratio,
                range_vs_atr=market_state.range_vs_atr,
                ema_20=trend.ema_20,
                ema_50=trend.ema_50,
                current_price=market_state.current_price,
                moi_flip_rate=alpha_state.moi_flip_rate,
            )
            market_state.regime = regime
            market_state.regime_confidence = confidence
            
            # Check warmup
            time_since_start = time.time() - self._start_time
            in_warmup = time_since_start < self._warmup_seconds
            
            # Check for signals (only if not in warmup)
            signal_fired: Optional[HybridSignal] = None
            signal_name = ""
            
            if not in_warmup and not self.position_sizer.is_holding(symbol):
                for name, signal_detector in self._signals.items():
                    result = signal_detector.evaluate(market_state)
                    if result is not None:
                        signal_fired = result
                        signal_name = name
                        break
            
            # Process signal through Stage 6 and 7
            position_result: Optional[PositionResultV3] = None
            trade_opened = False
            
            if signal_fired is not None:
                # Update position sizer with current equity
                account = self.trade_manager.get_account_state()
                self.position_sizer.set_equity(account.current_equity)
                
                # Calculate position
                position_result = self.position_sizer.calculate_position(
                    symbol=symbol,
                    side=signal_fired.direction.value,
                    entry_price=market_state.current_price,
                    stop_pct=signal_fired.stop_pct,
                    target_pct=signal_fired.target_pct,
                    signal_type=signal_fired.signal_type.value,
                    signal_name=signal_fired.reason,
                )
                
                # Place order if allowed
                if position_result.allowed and position_result.position:
                    order_id = await self.trade_manager.place_position(
                        position_result.position,
                        market_state=market_state,
                        signal=signal_fired,
                    )
                    trade_opened = order_id is not None
                    
                    # If trade failed to open (e.g., margin issue), clear holding state
                    if not trade_opened:
                        self.position_sizer.clear_position(symbol)
                        logger.warning("trade_failed_clearing_holding", symbol=symbol)
            
            # Update price in trade manager for open positions
            trade_closed = await self.trade_manager.update_price(symbol, market_state.current_price)
            
            # Clear position sizer if trade closed
            if trade_closed:
                self.position_sizer.clear_position(symbol)
            
            # Build pipeline state for dashboard
            tracker = self._trackers.get(symbol, SymbolTrackerV3(symbol=symbol))
            now_ms = int(time.time() * 1000)
            data_fresh = (now_ms - tracker.last_trade_ms) < 5000 if tracker.last_trade_ms > 0 else False
            
            account = self.trade_manager.get_account_state()
            open_trade = self.trade_manager.get_open_trade(symbol)
            
            ps = {
                "symbol": symbol,
                "timestamp_ms": now_ms,
                "price": market_state.current_price,
                
                # Stage 1: Data freshness
                "stage1_ok": data_fresh,
                "trades_per_sec": tracker.trades_per_sec(),
                
                # Stage 2: Alpha State
                "funding_z": alpha_state.funding_z,
                "funding_rate": alpha_state.funding_rate,
                "oi_change_1h": alpha_state.oi_change_1h,
                "oi_change_4h": alpha_state.oi_change_4h,
                "oi_change_24h": alpha_state.oi_change_24h,
                "price_change_1h": alpha_state.price_change_1h,
                "price_change_4h": alpha_state.price_change_4h,
                "price_change_24h": alpha_state.price_change_24h,
                "atr_short": alpha_state.atr_short,
                "atr_long": alpha_state.atr_long,
                "vol_expansion_ratio": alpha_state.vol_expansion_ratio,
                "liq_imbalance_1h": alpha_state.liq_imbalance_1h,
                "liq_imbalance_4h": alpha_state.liq_imbalance_4h,
                "liq_total_1h": alpha_state.liq_long_usd_1h + alpha_state.liq_short_usd_1h,
                "cascade_active": alpha_state.cascade_active,
                "liq_exhaustion": alpha_state.liq_exhaustion,
                
                # Trend
                "trend_direction": alpha_state.trend_direction_1h.value,
                "trend_strength": alpha_state.trend_strength_1h,
                "ema_20": alpha_state.ema_20_1h,
                "ema_50": alpha_state.ema_50_1h,
                "rsi_14": alpha_state.rsi_14,
                "price_vs_ema20": ((alpha_state.current_price - alpha_state.ema_20_1h) / alpha_state.ema_20_1h * 100) if alpha_state.ema_20_1h > 0 else 0,
                "entry_state": "PULLBACK_READY" if alpha_state.ema_20_1h > 0 and abs((alpha_state.current_price - alpha_state.ema_20_1h) / alpha_state.ema_20_1h) < 0.008 else ("EXTENDED" if alpha_state.ema_20_1h > 0 and abs((alpha_state.current_price - alpha_state.ema_20_1h) / alpha_state.ema_20_1h) > 0.02 else "WAITING"),
                
                # Bias
                "bias_direction": bias.direction.value,
                "bias_strength": bias.strength,
                "bias_reason": bias.reason,
                
                # Regime
                "regime": regime.value,
                "regime_confidence": confidence,
                "is_tradeable": regime not in [MarketRegime.CHOPPY],
                
                # Signal
                "signal_fired": signal_fired is not None,
                "signal_type": signal_name if signal_fired else "",
                "signal_direction": signal_fired.direction.value if signal_fired else "",
                "signal_confidence": signal_fired.confidence if signal_fired else 0,
                "signal_stop_pct": signal_fired.stop_pct * 100 if signal_fired else 0,
                "signal_target_pct": signal_fired.target_pct * 100 if signal_fired else 0,
                
                # Position
                "is_holding": self.position_sizer.is_holding(symbol),
                "position_allowed": position_result.allowed if position_result else False,
                "position_rejection": position_result.rejection_reason if position_result else "",
                "trade_opened": trade_opened,
                
                # Open trade info
                "has_open_trade": open_trade is not None,
                "open_trade_side": open_trade.side if open_trade else "",
                "open_trade_entry": open_trade.entry_price if open_trade else 0,
                "open_trade_pnl": open_trade.unrealized_pnl if open_trade else 0,
                "open_trade_signal": open_trade.signal_type if open_trade else "",
                "open_trade_r": open_trade.r_multiple if open_trade else 0,
                "open_trade_stop": open_trade.current_stop if open_trade else 0,
                "open_trade_target": open_trade.target_price if open_trade else 0,
                "open_trade_notional": open_trade.notional if open_trade else 0,
                "open_trade_margin": open_trade.margin if open_trade else 0,
                "open_trade_size": open_trade.size if open_trade else 0,
                "breakeven_triggered": open_trade.breakeven_triggered if open_trade else False,
                "trail_1r_triggered": open_trade.trail_1r_triggered if open_trade else False,
                
                # Account
                "account_equity": account.current_equity,
                "account_realized_pnl": account.realized_pnl,
                "account_unrealized_pnl": account.unrealized_pnl,
                "account_total_r": account.total_r,
                "account_win_rate": account.win_rate() * 100,
                "account_trade_count": account.trade_count,
                "account_margin_available": account.margin_available,
                
                # Status
                "in_warmup": in_warmup,
                "warmup_remaining": max(0, int(self._warmup_seconds - time_since_start)),
            }
            
            self._pipeline_states[symbol] = ps
            
            # Broadcast to dashboard
            if self.enable_dashboard:
                await self._broadcast_state(symbol, ps)
                
        except Exception as e:
            logger.error("symbol_processing_error", symbol=symbol, error=str(e)[:100])
    
    async def _broadcast_state(self, symbol: str, state: Dict[str, Any]):
        """Broadcast state to V3 dashboard"""
        try:
            from src.dashboard.global_dashboard_v3 import broadcast_pipeline_state
            await broadcast_pipeline_state(symbol, state)
        except Exception as e:
            logger.debug("broadcast_error", error=str(e)[:50])
    
    # ========== MAIN LOOP ==========
    
    async def _update_loop(self):
        """Main update loop - process all symbols periodically"""
        update_interval = settings.UPDATE_INTERVAL_S if settings.VM_MODE else 1.0
        while self._running:
            try:
                for symbol in self.symbols:
                    await self._process_symbol(symbol)
                    if settings.VM_MODE:
                        await asyncio.sleep(0.05)
                
                await self._log_health_if_needed()
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("update_loop_error", error=str(e)[:100])
                await asyncio.sleep(5.0)
    
    async def _log_health_if_needed(self):
        """Log health metrics periodically for long-running monitoring"""
        now = time.time()
        if now - self._last_health_log < self._health_log_interval:
            return
        
        self._last_health_log = now
        uptime_hours = (now - self._start_time) / 3600
        
        # Get account state
        account = self.trade_manager.get_account_state()
        
        # Get Stage1 health if available
        stage1_healthy = False
        if self.stage1:
            try:
                stage1_healthy = self.stage1.is_healthy()
            except Exception:
                pass
        
        # Count active signals
        active_positions = len(self.trade_manager.get_open_trades())
        closed_trades = account.trade_count
        
        logger.info(
            "v3_health_heartbeat",
            uptime_hours=f"{uptime_hours:.1f}",
            equity=f"${account.current_equity:.2f}",
            realized_pnl=f"${account.realized_pnl:.2f}",
            total_r=f"{account.total_r:.2f}R",
            win_rate=f"{account.win_rate()*100:.1f}%",
            trade_count=closed_trades,
            active_positions=active_positions,
            stage1_healthy=stage1_healthy,
            warmup_complete=(now - self._start_time) >= self._warmup_seconds,
        )
        
        # Memory cleanup: trim signal history if needed
        for signal_name, signal_obj in self._signals.items():
            if hasattr(signal_obj, '_last_signal') and len(getattr(signal_obj, '_last_signal', {})) > 50:
                # Trim old entries
                signal_obj._last_signal = dict(list(signal_obj._last_signal.items())[-20:])
    
    # ========== LIFECYCLE ==========
    
    async def start(self) -> None:
        """Start the V3 pipeline"""
        self._running = True
        
        logger.info(
            "global_pipeline_v3_starting",
            symbols=self.symbols,
            dashboard_port=self.dashboard_port,
            live_trading=self.live_trading,
            dry_run=self.dry_run,
        )
        
        # Step 0: Check WEEX connectivity if live trading enabled
        if self.live_trading:
            connected, message = await self.trade_manager.check_weex_connectivity()
            if connected:
                logger.info("weex_connectivity_ok", message=message)
            else:
                logger.error("weex_connectivity_failed", message=message)
                if not self.dry_run:
                    # In real trading mode, abort if WEEX is not connected
                    raise RuntimeError(f"WEEX connection required for live trading: {message}")
        
        # Step 1: Bootstrap historical data
        logger.info("bootstrapping_alpha_data", symbols=len(self.symbols))
        try:
            await self.bootstrap.bootstrap_all()
            
            # Initialize alpha processor from bootstrap
            self.alpha_processor.initialize_from_bootstrap(self.bootstrap)
            
            logger.info("bootstrap_complete")
        except Exception as e:
            logger.error("bootstrap_failed", error=str(e)[:100])
        
        # Step 2: Initialize Stage 1
        self.stage1 = Stage1Orchestrator(
            symbols=self.symbols,
            on_trade=self._on_trade,
            on_bar=self._on_bar,
            on_funding=self._on_funding,
            on_oi=self._on_oi,
            on_liquidation=self._on_liquidation,
        )
        
        await self.stage1.initialize()
        
        # Step 3: Start all tasks
        tasks = [
            asyncio.create_task(self.stage1.start()),
            asyncio.create_task(self._update_loop()),
        ]
        
        if self.enable_dashboard:
            try:
                from src.dashboard.global_dashboard_v3 import start_dashboard_async
                tasks.append(asyncio.create_task(
                    start_dashboard_async("0.0.0.0", self.dashboard_port)
                ))
            except Exception as e:
                logger.warning("dashboard_start_failed", error=str(e)[:50])
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error("task_failed", task_index=i, error=str(result)[:100])
        except asyncio.CancelledError:
            logger.info("pipeline_v3_cancelled")
    
    async def stop(self) -> None:
        """Stop the pipeline"""
        if not self._running:
            return
        
        self._running = False
        logger.info("global_pipeline_v3_stopping")
        
        if self.stage1:
            try:
                await asyncio.wait_for(self.stage1.stop(), timeout=10.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning("stage1_stop_error", error=str(e)[:50])
        
        logger.info("global_pipeline_v3_stopped")
    
    # ========== PUBLIC API ==========
    
    def get_pipeline_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current pipeline state for symbol"""
        return self._pipeline_states.get(symbol)
    
    def get_all_pipeline_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all pipeline states"""
        return self._pipeline_states.copy()
    
    async def _on_trade_closed(self, trade) -> None:
        """Callback when a trade is closed - broadcast to dashboard"""
        if self.enable_dashboard:
            closed_trades = self.trade_manager.get_closed_trades(limit=50)
            await broadcast_closed_trades(closed_trades)
    
    async def _manual_close_position(self, symbol: str, reason: str) -> Dict[str, Any]:
        """Manual close position from dashboard Signal Status button"""
        logger.info("manual_close_requested", symbol=symbol, reason=reason)
        result = await self.trade_manager.manual_close_position(symbol, reason)
        if result.get("success"):
            self.position_sizer.release_hold(symbol)
            closed_trades = self.trade_manager.get_closed_trades(limit=50)
            await broadcast_closed_trades(closed_trades)
        return result
    
    async def _manual_signal_trigger(self, symbol: str, direction: str) -> Dict[str, Any]:
        """Manual signal trigger from dashboard Long/Short buttons - mimics FUNDING_PRESSURE signal"""
        logger.info("manual_signal_requested", symbol=symbol, direction=direction)
        
        # Check if already holding this symbol
        if self.position_sizer.is_holding(symbol):
            return {"success": False, "error": f"Already holding {symbol}"}
        
        # Get current market state
        alpha_state = self.alpha_processor.get_state(symbol)
        if alpha_state is None:
            return {"success": False, "error": f"No alpha state for {symbol}"}
        
        current_price = alpha_state.current_price
        if current_price <= 0:
            return {"success": False, "error": f"Invalid price for {symbol}"}
        
        # Create a fake FUNDING_PRESSURE signal
        from src.stage3_v3.models import HybridSignal, Direction, SignalType, MarketRegime
        
        # Calculate stop/target based on ATR
        atr_pct = alpha_state.atr_short / current_price if current_price > 0 else 0.015
        stop_pct = max(0.012, min(0.020, atr_pct * 1.2))
        target_pct = stop_pct * 2.5
        
        signal = HybridSignal(
            direction=Direction.LONG if direction == "LONG" else Direction.SHORT,
            signal_type=SignalType.FUNDING_TREND,
            confidence=0.75,
            entry_price=current_price,
            stop_pct=stop_pct,
            target_pct=target_pct,
            reason=f"{direction} signal triggered via funding_trend signal",
            bias_strength=alpha_state.trend_strength_1h,
            regime=MarketRegime.TRENDING_UP if direction == "LONG" else MarketRegime.TRENDING_DOWN,
        )
        
        # Run through position sizer
        position_result = self.position_sizer.calculate_position(
            symbol=symbol,
            side=direction,
            entry_price=current_price,
            stop_pct=stop_pct,
            target_pct=target_pct,
            signal_name="FUNDING_PRESSURE",
        )
        
        if not position_result.allowed:
            return {"success": False, "error": f"Position rejected: {position_result.rejection_reason}"}
        
        # Place the trade
        trade_opened = await self.trade_manager.place_position(
            signal=signal,
            position=position_result.position,
        )
        
        if trade_opened:
            self.position_sizer.add_hold(symbol)
            logger.info(
                "manual_signal_trade_opened",
                symbol=symbol,
                direction=direction,
                entry=current_price,
                size=position_result.position.size,
                notional=position_result.position.notional,
            )
            return {"success": True, "data": {"symbol": symbol, "direction": direction, "entry": current_price}}
        else:
            return {"success": False, "error": "Trade placement failed"}
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get overall health metrics"""
        return {
            "total_symbols": len(self.symbols),
            "running": self._running,
            "warmup_complete": (time.time() - self._start_time) >= self._warmup_seconds,
            "position_sizer": self.position_sizer.get_health_metrics(),
            "trade_manager": self.trade_manager.get_health_metrics(),
            "alpha_processor": self.alpha_processor.get_health_metrics(),
        }


async def run_global_pipeline_v3(
    symbols: Optional[List[str]] = None,
    duration_seconds: Optional[int] = None,
    dashboard_port: int = 8889,
    live_trading: bool = False,
    dry_run: bool = True,
) -> None:
    """
    Run the V3 global pipeline.
    
    Args:
        symbols: List of symbols (default: from settings)
        duration_seconds: Run for N seconds then stop (None = forever)
        dashboard_port: Port for dashboard UI
        live_trading: Enable real order execution via WEEX API
        dry_run: If live_trading=True, dry_run=True simulates without real orders
    """
    runner = GlobalPipelineRunnerV3(
        symbols=symbols,
        dashboard_port=dashboard_port,
        enable_dashboard=True,
        live_trading=live_trading,
        dry_run=dry_run,
    )
    
    # Handle shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("shutdown_signal_received")
        asyncio.create_task(runner.stop())
    
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
    
    try:
        if duration_seconds:
            async def run_with_timeout():
                start_task = asyncio.create_task(runner.start())
                await asyncio.sleep(duration_seconds)
                await runner.stop()
                start_task.cancel()
            
            await run_with_timeout()
        else:
            await runner.start()
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt")
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(run_global_pipeline_v3())
