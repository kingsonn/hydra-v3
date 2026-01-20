"""
Stage 2 Runner - Integrated Stage 1 + Stage 2 with Dashboard

Startup sequence:
1. Bootstrap ATR and volatility from historical klines (no cold start)
2. Initialize Stage 1 data collection
3. Start Stage 2 feature computation
4. Launch dashboard
"""
import asyncio
import signal
import sys
from typing import Optional
import structlog

from config import settings
from src.stage1 import Stage1Orchestrator
from src.stage2.orchestrator import Stage2Orchestrator
from src.stage2.dashboard import broadcast_state, app, start_dashboard_async
from src.stage2.models import MarketState

logger = structlog.get_logger(__name__)


class IntegratedRunner:
    """
    Runs Stage 1 (data collection) + Stage 2 (feature computation) + Dashboard
    """
    
    def __init__(
        self,
        symbols: Optional[list] = None,
        dashboard_port: int = 8080,
        enable_dashboard: bool = True,
    ):
        self.symbols = symbols or settings.SYMBOLS
        self.dashboard_port = dashboard_port
        self.enable_dashboard = enable_dashboard
        
        # Stage 2 orchestrator
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
        """Callback when Stage 2 emits a MarketState"""
        if self.enable_dashboard:
            await broadcast_state(state)
    
    async def start(self) -> None:
        """Start all components"""
        self._running = True
        
        logger.info(
            "integrated_runner_starting",
            symbols=self.symbols,
            dashboard_port=self.dashboard_port if self.enable_dashboard else "disabled",
        )
        
        # NOTE: ATR bootstrap removed - use GlobalPipelineRunnerV3 for new pipeline
        logger.info("skipping_old_bootstrap", reason="Use GlobalPipelineRunnerV3 for V3 pipeline")
        
        # Step 1: Initialize Stage 1
        await self.stage1.initialize()
        
        # Step 3: Start all components
        tasks = [
            asyncio.create_task(self.stage1.start()),
            asyncio.create_task(self.stage2.start_update_loop()),
        ]
        
        if self.enable_dashboard:
            tasks.append(asyncio.create_task(
                start_dashboard_async("0.0.0.0", self.dashboard_port)
            ))
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("integrated_runner_cancelled")
    
    async def stop(self) -> None:
        """Stop all components"""
        self._running = False
        await self.stage1.stop()
        await self.stage2.stop()
        logger.info("integrated_runner_stopped")


async def run_stage2_live(
    symbols: Optional[list] = None,
    duration_seconds: Optional[int] = None,
    dashboard_port: int = 8080,
) -> None:
    """
    Run Stage 2 with live data from Stage 1
    
    Args:
        symbols: List of symbols to track (default: all from settings)
        duration_seconds: Run for this many seconds then stop (None = forever)
        dashboard_port: Port for dashboard UI
    """
    runner = IntegratedRunner(
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
    
    print(f"\n{'='*60}")
    print("HYDRA Stage 2 - Market State Engine")
    print(f"{'='*60}")
    print(f"Symbols: {runner.symbols}")
    print(f"Dashboard: http://localhost:{dashboard_port}")
    print(f"Press Ctrl+C to stop")
    print(f"{'='*60}\n")
    
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


async def test_stage2_standalone(duration_seconds: int = 60) -> None:
    """
    Test Stage 2 with simulated/live data
    Prints MarketState updates to console
    """
    states_received = []
    
    def on_state(state: MarketState):
        states_received.append(state)
        print(f"\n[{state.symbol}] ${state.price:.2f} | Regime: {state.regime.value}")
        print(f"  Order Flow: MOI={state.order_flow.moi_1s:.3f}, "
              f"DeltaVel={state.order_flow.delta_velocity:.3f}, "
              f"Aggression={state.order_flow.aggression_persistence:.2f}")
        print(f"  Volatility: Rank={state.volatility.vol_rank:.0f}%, "
              f"Expansion={state.volatility.vol_expansion_ratio:.2f}x")
        print(f"  Liquidations: 30s=${state.liquidations.long_usd_30s + state.liquidations.short_usd_30s:.0f}, "
              f"Cascade={state.liquidations.cascade_active}, "
              f"Exhaustion={state.liquidations.exhaustion}")
    
    runner = IntegratedRunner(
        symbols=["BTCUSDT", "ETHUSDT"],
        enable_dashboard=False,
    )
    runner.stage2.on_market_state = on_state
    
    print(f"Starting Stage 2 test for {duration_seconds}s...")
    
    task = asyncio.create_task(runner.start())
    
    try:
        await asyncio.sleep(duration_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        await runner.stop()
        task.cancel()
    
    print(f"\n{'='*60}")
    print(f"Test Complete: {len(states_received)} MarketState updates received")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        asyncio.run(test_stage2_standalone(duration))
    else:
        asyncio.run(run_stage2_live())
