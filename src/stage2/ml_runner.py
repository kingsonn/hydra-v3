"""
ML Model Testing Runner - Integrated Stage 1 + Stage 2 + ML Dashboard

Runs all components together:
1. Stage 1: Data collection (trades, bars, order book, funding, OI, liquidations)
2. Stage 2: Feature computation (order flow, absorption, volatility, structure)
3. ML Dashboard: Real-time model predictions display
"""
import asyncio
import signal
import sys
from typing import Optional
import structlog

from config import settings
from src.stage1 import Stage1Orchestrator
from src.stage2.orchestrator import Stage2Orchestrator
from src.stage2.models import MarketState
from src.stage2.ml_dashboard import (
    ml_broadcast_state, ml_app, start_ml_dashboard_async, 
    load_ml_models, ML_MODELS
)
from src.collectors.klines import bootstrap_all_symbols

logger = structlog.get_logger(__name__)


class MLTestRunner:
    """
    Runs Stage 1 + Stage 2 + ML Model Testing Dashboard
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        dashboard_port: int = 8081,
    ):
        self.symbols = symbols or settings.SYMBOLS
        self.dashboard_port = dashboard_port
        
        # Stage 2 orchestrator with ML callback
        self.stage2 = Stage2Orchestrator(
            symbols=self.symbols,
            on_market_state=self._on_market_state,
        )
        
        # Stage 1 orchestrator with Stage 2 callbacks
        self.stage1 = Stage1Orchestrator(
            symbols=self.symbols,
            on_trade=self.stage2.on_trade,
            on_bar=self.stage2.on_bar,
            on_book_snapshot=self.stage2.on_book_snapshot,
            on_volume_profile=self.stage2.on_volume_profile,
            on_funding=self.stage2.on_funding,
            on_oi=self.stage2.on_oi,
            on_liquidation=self.stage2.on_liquidation,
        )
        
        self._running = False
    
    async def _on_market_state(self, state: MarketState) -> None:
        """Callback when Stage 2 emits a MarketState - broadcast to ML dashboard"""
        await ml_broadcast_state(state)
    
    async def start(self) -> None:
        """Start all components"""
        global ML_MODELS
        self._running = True
        
        logger.info(
            "ml_test_runner_starting",
            symbols=self.symbols,
            dashboard_port=self.dashboard_port,
        )
        
        # Step 1: Load ML models
        logger.info("loading_ml_models")
        ML_MODELS.update(load_ml_models())
        logger.info("ml_models_ready", count=len(ML_MODELS))
        
        # Step 2: Bootstrap ATR from historical klines
        logger.info("bootstrapping_atr_data", symbols=len(self.symbols))
        try:
            atr_data, _ = await bootstrap_all_symbols(self.symbols)
            
            for symbol in self.symbols:
                if symbol in atr_data:
                    self.stage2.bootstrap_volatility(
                        symbol=symbol,
                        tr_5m_values=atr_data[symbol].tr_5m_deque,
                        tr_1h_values=atr_data[symbol].tr_1h_deque,
                        atr_5m=atr_data[symbol].atr_5m,
                        atr_1h=atr_data[symbol].atr_1h,
                        last_close_5m=atr_data[symbol].last_close_5m,
                        last_close_1h=atr_data[symbol].last_close_1h,
                    )
                    logger.info(
                        "symbol_atr_bootstrapped",
                        symbol=symbol,
                        atr_5m=f"{atr_data[symbol].atr_5m:.4f}",
                        atr_1h=f"{atr_data[symbol].atr_1h:.4f}",
                    )
        except Exception as e:
            logger.error("bootstrap_failed", error=str(e))
        
        # Step 3: Initialize Stage 1
        await self.stage1.initialize()
        
        # Step 4: Start all components
        tasks = [
            asyncio.create_task(self.stage1.start()),
            asyncio.create_task(self.stage2.start_update_loop()),
            asyncio.create_task(start_ml_dashboard_async("0.0.0.0", self.dashboard_port)),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("ml_test_runner_cancelled")
    
    async def stop(self) -> None:
        """Stop all components"""
        self._running = False
        await self.stage1.stop()
        await self.stage2.stop()
        logger.info("ml_test_runner_stopped")


async def run_ml_test(
    symbols: Optional[list] = None,
    duration_seconds: Optional[int] = None,
    dashboard_port: int = 8081,
) -> None:
    """
    Run ML model testing with live data
    
    Args:
        symbols: List of symbols to track (default: all from settings)
        duration_seconds: Run for this many seconds then stop (None = forever)
        dashboard_port: Port for ML dashboard UI
    """
    runner = MLTestRunner(
        symbols=symbols,
        dashboard_port=dashboard_port,
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
    print("HYDRA ML Model Testing Dashboard")
    print(f"{'='*70}")
    print(f"Symbols: {runner.symbols}")
    print(f"ML Dashboard: http://localhost:{dashboard_port}")
    print(f"")
    print(f"Models loaded from: ml_models/")
    print(f"  - 2 directions (up/down) × 3 regimes (high/mid/low) × 2 times (60/300)")
    print(f"  - Total: 12 models")
    print(f"")
    print(f"Features (15 total):")
    print(f"  - 7 computed: MOI_250ms, MOI_1s, delta_velocity, AggressionPersistence,")
    print(f"                absorption_z, dist_lvn, vol_5m")
    print(f"  - 8 one-hot pairs: ADA, BNB, BTC, DOGE, ETH, LTC, SOL, XRP")
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


if __name__ == "__main__":
    duration = None
    port = 8081
    
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--duration="):
                duration = int(arg.split("=")[1])
            elif arg.startswith("--port="):
                port = int(arg.split("=")[1])
    
    asyncio.run(run_ml_test(duration_seconds=duration, dashboard_port=port))
