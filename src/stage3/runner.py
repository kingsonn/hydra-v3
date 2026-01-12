"""
Stage 3 Runner - Integrated Stage 1 + Stage 2 + Stage 3 with Dashboard
"""
import asyncio
import signal
import sys
from typing import Optional, Dict, Any
import structlog

from config import settings
from src.stage1 import Stage1Orchestrator
from src.stage2.orchestrator import Stage2Orchestrator
from src.stage2.models import MarketState
from src.stage3.thesis_engine import ThesisEngine
from src.stage3.models import Thesis

logger = structlog.get_logger(__name__)


class Stage3Runner:
    """
    Runs Stage 1 (data collection) + Stage 2 (features) + Stage 3 (thesis) + Dashboard
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
        
        # Stage 3 thesis engine
        self.thesis_engine = ThesisEngine(
            symbols=self.symbols,
            on_thesis=self._on_thesis,
        )
        
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
        
        # Store latest combined state for dashboard
        self._latest_combined: Dict[str, Dict[str, Any]] = {}
    
    async def _on_market_state(self, state: MarketState) -> None:
        """Callback when Stage 2 emits a MarketState"""
        # Process through Stage 3
        thesis = self.thesis_engine.process(state)
        
        # Build combined state for dashboard
        combined = state.to_flat_dict()
        combined.update(thesis.to_flat_dict())
        
        # Add thesis state inputs for debugging
        price_tracker = self.thesis_engine._price_trackers.get(state.symbol)
        if price_tracker:
            combined["thesis_price_change_5m"] = price_tracker.get_price_change(300)
            combined["thesis_price_change_15m"] = price_tracker.get_price_change(900)
        
        regime_tracker = self.thesis_engine._regime_trackers.get(state.symbol)
        if regime_tracker:
            combined["thesis_time_in_regime"] = regime_tracker.update(state.regime)
        
        self._latest_combined[state.symbol] = combined
        
        # Broadcast to dashboard
        if self.enable_dashboard:
            from src.stage3.dashboard import broadcast_combined_state
            await broadcast_combined_state(state.symbol, combined)
    
    def _on_thesis(self, symbol: str, thesis: Thesis) -> None:
        """Callback when Stage 3 emits a Thesis"""
        if thesis.allowed:
            logger.info(
                "thesis_generated",
                symbol=symbol,
                direction=thesis.direction.value,
                strength=f"{thesis.strength:.2f}",
                signals=[s.name for s in thesis.reasons],
            )
    
    async def start(self) -> None:
        """Start all components"""
        self._running = True
        
        logger.info(
            "stage3_runner_starting",
            symbols=self.symbols,
            dashboard_port=self.dashboard_port if self.enable_dashboard else "disabled",
        )
        
        # Initialize Stage 1 first
        await self.stage1.initialize()
        
        tasks = [
            asyncio.create_task(self.stage1.start()),
            asyncio.create_task(self.stage2.start_update_loop()),
        ]
        
        if self.enable_dashboard:
            from src.stage3.dashboard import start_dashboard_async
            tasks.append(asyncio.create_task(
                start_dashboard_async("0.0.0.0", self.dashboard_port)
            ))
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("stage3_runner_cancelled")
    
    async def stop(self) -> None:
        """Stop all components"""
        self._running = False
        await self.stage1.stop()
        await self.stage2.stop()
        logger.info("stage3_runner_stopped")
    
    def get_thesis(self, symbol: str) -> Optional[Thesis]:
        """Get current thesis for symbol"""
        return self.thesis_engine.get_thesis(symbol)
    
    def get_all_theses(self) -> Dict[str, Thesis]:
        """Get all current theses"""
        return self.thesis_engine.get_all_theses()


async def run_stage3_live(
    symbols: Optional[list] = None,
    duration_seconds: Optional[int] = None,
    dashboard_port: int = 8080,
) -> None:
    """
    Run Stage 3 with live data from Stage 1 + Stage 2
    
    Args:
        symbols: List of symbols to track (default: all from settings)
        duration_seconds: Run for this many seconds then stop (None = forever)
        dashboard_port: Port for dashboard UI
    """
    runner = Stage3Runner(
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
    print("HYDRA Stage 3 - Thesis Engine")
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


async def test_stage3_standalone(duration_seconds: int = 60) -> None:
    """
    Test Stage 3 with live data
    Prints thesis updates to console
    """
    theses_received = []
    
    def on_thesis(symbol: str, thesis: Thesis):
        theses_received.append((symbol, thesis))
        status = "✓ ALLOWED" if thesis.allowed else "✗ BLOCKED"
        direction = thesis.direction.value
        print(f"\n[{symbol}] {status} | {direction} | Strength: {thesis.strength:.2f}")
        
        if thesis.reasons:
            print(f"  Signals: {', '.join(s.name for s in thesis.reasons)}")
        if thesis.veto_reason:
            print(f"  Veto: {thesis.veto_reason}")
    
    runner = Stage3Runner(
        symbols=["BTCUSDT", "ETHUSDT"],
        enable_dashboard=False,
    )
    runner.thesis_engine.on_thesis = on_thesis
    
    print(f"Starting Stage 3 test for {duration_seconds}s...")
    
    task = asyncio.create_task(runner.start())
    
    try:
        await asyncio.sleep(duration_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        await runner.stop()
        task.cancel()
    
    # Summary
    allowed = sum(1 for _, t in theses_received if t.allowed)
    print(f"\n{'='*60}")
    print(f"Test Complete: {len(theses_received)} theses generated, {allowed} allowed")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        asyncio.run(test_stage3_standalone(duration))
    else:
        asyncio.run(run_stage3_live())
