"""
STAGE 1 DATA INGESTION ORCHESTRATOR
Coordinates all data collection, processing, and storage
"""
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
import structlog

from config import settings
from src.core.models import Trade, Bar, OrderBookSnapshot, FundingRate, OpenInterest, Liquidation
from src.core.storage import StorageManager, get_storage
from src.collectors.trades import TradesCollector
from src.collectors.orderbook import OrderBookCollector
from src.collectors.derivatives import DerivativesCollector
from src.processors.bars import BarAggregator
from src.processors.volume_profile import RollingVolumeProfiler
from src.health.monitor import HealthMonitor, HealthStatus

logger = structlog.get_logger(__name__)


class Stage1Orchestrator:
    """
    Main orchestrator for Stage 1 data ingestion
    
    Coordinates:
    - Trade collection (WebSocket)
    - Order book collection (WebSocket)
    - Derivatives data (REST polling)
    - Bar aggregation
    - Volume profile computation
    - Health monitoring
    - Data persistence
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        enable_storage: bool = True,
        storage_batch_size: int = 1000,
        profile_update_interval_s: int = 30,
        # External callbacks for Stage 2 integration
        on_trade: Optional[Callable[[Trade], Any]] = None,
        on_bar: Optional[Callable[[Bar], Any]] = None,
        on_book_snapshot: Optional[Callable[[OrderBookSnapshot], Any]] = None,
        on_volume_profile: Optional[Callable[[Any], Any]] = None,
        on_funding: Optional[Callable[[FundingRate], Any]] = None,
        on_oi: Optional[Callable[[OpenInterest], Any]] = None,
        on_liquidation: Optional[Callable[[Liquidation], Any]] = None,
    ):
        self.symbols = symbols or settings.SYMBOLS
        self.enable_storage = enable_storage
        self.storage_batch_size = storage_batch_size
        self.profile_update_interval_s = profile_update_interval_s
        
        # External callbacks
        self._ext_on_trade = on_trade
        self._ext_on_bar = on_bar
        self._ext_on_book_snapshot = on_book_snapshot
        self._ext_on_volume_profile = on_volume_profile
        self._ext_on_funding = on_funding
        self._ext_on_oi = on_oi
        self._ext_on_liquidation = on_liquidation
        
        # Components
        self._trades_collector: Optional[TradesCollector] = None
        self._orderbook_collector: Optional[OrderBookCollector] = None
        self._derivatives_collector: Optional[DerivativesCollector] = None
        self._bar_aggregator: Optional[BarAggregator] = None
        self._profiler: Optional[RollingVolumeProfiler] = None
        self._health_monitor: Optional[HealthMonitor] = None
        self._storage: Optional[StorageManager] = None
        
        # Buffers for batch storage
        self._trade_buffer: List[Trade] = []
        self._bar_buffer: List[Bar] = []
        
        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Stats
        self._start_time: Optional[float] = None
        self._trades_processed = 0
        self._bars_completed = 0
    
    async def initialize(self) -> None:
        """Initialize all components"""
        logger.info("stage1_initializing", symbols=self.symbols)
        
        # Storage
        if self.enable_storage:
            self._storage = await get_storage()
        
        # Health monitor
        self._health_monitor = HealthMonitor(
            symbols=self.symbols,
            on_alert=self._on_health_alert,
        )
        
        # Bar aggregator
        self._bar_aggregator = BarAggregator(
            symbols=self.symbols,
            on_bar=self._on_bar_completed,
        )
        
        # Volume profiler
        self._profiler = RollingVolumeProfiler(
            symbols=self.symbols,
            update_interval_s=self.profile_update_interval_s,
        )
        
        # Trades collector
        self._trades_collector = TradesCollector(
            symbols=self.symbols,
            on_trade=self._on_trade,
        )
        
        # Order book collector
        self._orderbook_collector = OrderBookCollector(
            symbols=self.symbols,
            on_snapshot=self._on_orderbook_snapshot,
        )
        
        # Derivatives collector
        self._derivatives_collector = DerivativesCollector(
            symbols=self.symbols,
            on_funding=self._on_funding,
            on_oi=self._on_oi,
            on_liquidation=self._on_liquidation,
        )
        
        logger.info("stage1_initialized")
    
    async def start(self) -> None:
        """Start all data collection"""
        if self._running:
            logger.warning("stage1_already_running")
            return
        
        self._running = True
        self._start_time = time.time()
        
        logger.info("stage1_starting")
        
        # Start all collectors as background tasks
        self._tasks = [
            asyncio.create_task(self._trades_collector.start()),
            asyncio.create_task(self._orderbook_collector.start()),
            asyncio.create_task(self._derivatives_collector.start()),
            asyncio.create_task(self._health_monitor.start()),
            asyncio.create_task(self._storage_loop()),
            asyncio.create_task(self._profile_loop()),
        ]
        
        logger.info("stage1_started", task_count=len(self._tasks))
    
    async def stop(self) -> None:
        """Stop all data collection"""
        if not self._running:
            return
        
        self._running = False
        logger.info("stage1_stopping")
        
        # Stop collectors
        await self._trades_collector.stop()
        await self._orderbook_collector.stop()
        await self._derivatives_collector.stop()
        await self._health_monitor.stop()
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Flush remaining data
        if self._bar_aggregator:
            remaining_bars = self._bar_aggregator.flush_all()
            self._bar_buffer.extend(remaining_bars)
        
        # Final storage flush
        if self._storage and self._trade_buffer:
            await self._storage.store_trades(self._trade_buffer)
        if self._storage and self._bar_buffer:
            await self._storage.store_bars(self._bar_buffer)
        
        logger.info("stage1_stopped")
    
    # ========== CALLBACKS ==========
    
    async def _on_trade(self, trade: Trade) -> None:
        """Handle incoming trade"""
        self._trades_processed += 1
        
        # Update health monitor
        self._health_monitor.record_trade(trade.symbol, trade.timestamp_ms)
        
        # Process through bar aggregator
        self._bar_aggregator.process_trade(trade)
        
        # Add to profiler
        self._profiler.add_trade(trade)
        
        # Buffer for storage
        if self.enable_storage:
            self._trade_buffer.append(trade)
        
        # External callback (Stage 2)
        if self._ext_on_trade:
            self._ext_on_trade(trade)
    
    async def _on_bar_completed(self, bar: Bar) -> None:
        """Handle completed bar"""
        self._bars_completed += 1
        
        if self.enable_storage:
            self._bar_buffer.append(bar)
        
        # External callback (Stage 2)
        if self._ext_on_bar:
            self._ext_on_bar(bar)
    
    async def _on_orderbook_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Handle order book snapshot"""
        self._health_monitor.record_book_update(
            snapshot.symbol, snapshot.timestamp_ms
        )
        
        # External callback (Stage 2)
        if self._ext_on_book_snapshot:
            self._ext_on_book_snapshot(snapshot)
    
    async def _on_funding(self, rate: FundingRate) -> None:
        """Handle funding rate update"""
        if self._storage:
            await self._storage.store_funding_rates([rate])
        
        # External callback (Stage 2)
        if self._ext_on_funding:
            self._ext_on_funding(rate)
    
    async def _on_oi(self, oi: OpenInterest) -> None:
        """Handle open interest update"""
        if self._storage:
            await self._storage.store_open_interest([oi])
        
        # External callback (Stage 2)
        if self._ext_on_oi:
            self._ext_on_oi(oi)
    
    async def _on_liquidation(self, liq: Liquidation) -> None:
        """Handle liquidation event"""
        if self._storage:
            await self._storage.store_liquidations([liq])
        
        # External callback (Stage 2)
        if self._ext_on_liquidation:
            self._ext_on_liquidation(liq)
    
    async def _on_health_alert(self, alert: str) -> None:
        """Handle health alert"""
        logger.warning("health_alert", alert=alert)
    
    # ========== BACKGROUND LOOPS ==========
    
    async def _storage_loop(self) -> None:
        """Periodically flush buffers to storage"""
        while self._running:
            try:
                await asyncio.sleep(5)  # Flush every 5 seconds
                
                if not self._storage:
                    continue
                
                # Flush trades
                if len(self._trade_buffer) >= self.storage_batch_size:
                    trades_to_store = self._trade_buffer[:self.storage_batch_size]
                    self._trade_buffer = self._trade_buffer[self.storage_batch_size:]
                    await self._storage.store_trades(trades_to_store)
                
                # Flush bars
                if self._bar_buffer:
                    bars_to_store = self._bar_buffer.copy()
                    self._bar_buffer.clear()
                    await self._storage.store_bars(bars_to_store)
            except Exception as e:
                logger.error("storage_loop_error", error=str(e))
    
    async def _profile_loop(self) -> None:
        """Periodically update volume profiles"""
        while self._running:
            try:
                await asyncio.sleep(self.profile_update_interval_s)
                
                for symbol in self.symbols:
                    if self._profiler.should_update(symbol):
                        profiles = self._profiler.update_profiles(symbol)
                        
                        # Store profiles and external callback
                        for profile in profiles.values():
                            if self._storage:
                                await self._storage.store_volume_profile(profile)
                            
                            # External callback (Stage 2)
                            if self._ext_on_volume_profile:
                                self._ext_on_volume_profile(profile)
            except Exception as e:
                logger.error("profile_loop_error", error=str(e))
    
    # ========== PUBLIC API ==========
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all components"""
        uptime = time.time() - self._start_time if self._start_time else 0
        
        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "trades_processed": self._trades_processed,
            "bars_completed": self._bars_completed,
            "trades_per_second": self._trades_processed / uptime if uptime > 0 else 0,
            "health": self._health_monitor.get_health().to_dict() if self._health_monitor else None,
            "buffer_sizes": {
                "trades": len(self._trade_buffer),
                "bars": len(self._bar_buffer),
            },
        }
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return self._health_monitor.is_healthy() if self._health_monitor else False
    
    def is_tradeable(self, symbol: str) -> bool:
        """Check if symbol has fresh data"""
        return self._health_monitor.is_tradeable(symbol) if self._health_monitor else False
    
    def get_latest_bar(self, symbol: str, interval: str) -> Optional[Bar]:
        """Get most recent bar"""
        return self._bar_aggregator.get_latest_bar(symbol, interval) if self._bar_aggregator else None
    
    def get_bars(self, symbol: str, interval: str, count: int = 100) -> List[Bar]:
        """Get recent bars"""
        return self._bar_aggregator.get_bars(symbol, interval, count) if self._bar_aggregator else []
    
    def get_orderbook(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get current order book"""
        return self._orderbook_collector.get_book(symbol) if self._orderbook_collector else None
    
    def get_structural_levels(self, symbol: str) -> Dict[str, Any]:
        """Get POC, VAH, VAL, LVNs"""
        return self._profiler.get_structural_levels(symbol) if self._profiler else {}
    
    def get_recent_trades(self, symbol: str, count: int = 100) -> List[Trade]:
        """Get recent trades"""
        return self._trades_collector.get_trades_for_symbol(symbol, count) if self._trades_collector else []


# ========== CONVENIENCE FUNCTIONS ==========

async def run_stage1(duration_seconds: Optional[int] = None) -> Stage1Orchestrator:
    """
    Run Stage 1 data ingestion
    
    Usage:
        from src.stage1 import run_stage1
        import asyncio
        
        # Run for 60 seconds
        orchestrator = asyncio.run(run_stage1(60))
        
        # Or run indefinitely (Ctrl+C to stop)
        orchestrator = asyncio.run(run_stage1())
    """
    orchestrator = Stage1Orchestrator()
    await orchestrator.initialize()
    await orchestrator.start()
    
    if duration_seconds:
        await asyncio.sleep(duration_seconds)
        await orchestrator.stop()
    else:
        # Run until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await orchestrator.stop()
    
    return orchestrator


async def test_stage1_full(duration_seconds: int = 30) -> Dict[str, Any]:
    """
    Full integration test of Stage 1
    
    Usage:
        from src.stage1 import test_stage1_full
        import asyncio
        result = asyncio.run(test_stage1_full(30))
    """
    print(f"=== STAGE 1 FULL INTEGRATION TEST ({duration_seconds}s) ===\n")
    
    orchestrator = Stage1Orchestrator(
        symbols=["BTCUSDT", "ETHUSDT"],
        enable_storage=False,  # Disable storage for quick test
    )
    
    await orchestrator.initialize()
    print("✓ Initialized")
    
    await orchestrator.start()
    print("✓ Started all collectors")
    
    # Monitor progress
    for i in range(duration_seconds // 5):
        await asyncio.sleep(5)
        status = orchestrator.get_status()
        print(f"\n[{(i+1)*5}s] Status:")
        print(f"  Trades: {status['trades_processed']}")
        print(f"  Bars: {status['bars_completed']}")
        print(f"  Rate: {status['trades_per_second']:.1f}/s")
        print(f"  Healthy: {orchestrator.is_healthy()}")
    
    await orchestrator.stop()
    print("\n✓ Stopped")
    
    final_status = orchestrator.get_status()
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Total trades: {final_status['trades_processed']}")
    print(f"Total bars: {final_status['bars_completed']}")
    print(f"Avg rate: {final_status['trades_per_second']:.1f} trades/sec")
    
    return final_status
