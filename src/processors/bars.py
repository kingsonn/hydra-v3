"""
C️⃣ DERIVED BARS ENGINE
Builds OHLCV bars with order flow metrics from raw trades
Resolutions: 250ms, 1s, 5m, 30m
"""
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass, field
import structlog

from config import settings
from src.core.models import Trade, Bar, Side

logger = structlog.get_logger(__name__)


@dataclass
class BarBuilder:
    """Accumulates trades into a single bar"""
    symbol: str
    interval: str
    interval_ms: int
    start_ms: int
    
    # OHLCV
    open: Optional[float] = None
    high: float = float('-inf')
    low: float = float('inf')
    close: float = 0.0
    volume: float = 0.0
    notional: float = 0.0
    trade_count: int = 0
    
    # Order flow
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_notional: float = 0.0
    sell_notional: float = 0.0
    buy_count: int = 0
    sell_count: int = 0
    
    def add_trade(self, trade: Trade) -> None:
        """Add a trade to this bar"""
        price = trade.price
        qty = trade.quantity
        notional = trade.notional
        
        # OHLC
        if self.open is None:
            self.open = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        
        # Volume
        self.volume += qty
        self.notional += notional
        self.trade_count += 1
        
        # Order flow
        if trade.side == Side.BUY:
            self.buy_volume += qty
            self.buy_notional += notional
            self.buy_count += 1
        else:
            self.sell_volume += qty
            self.sell_notional += notional
            self.sell_count += 1
    
    def to_bar(self) -> Optional[Bar]:
        """Convert to finalized Bar object"""
        if self.open is None:
            return None
        
        return Bar(
            symbol=self.symbol,
            interval=self.interval,
            timestamp_ms=self.start_ms,
            open=self.open,
            high=self.high if self.high != float('-inf') else self.open,
            low=self.low if self.low != float('inf') else self.open,
            close=self.close,
            volume=self.volume,
            notional=self.notional,
            trade_count=self.trade_count,
            buy_volume=self.buy_volume,
            sell_volume=self.sell_volume,
            buy_notional=self.buy_notional,
            sell_notional=self.sell_notional,
            buy_count=self.buy_count,
            sell_count=self.sell_count,
        )
    
    def is_empty(self) -> bool:
        return self.trade_count == 0


class BarAggregator:
    """
    Aggregates trades into multi-resolution bars
    
    Features:
    - Multiple intervals: 250ms, 1s, 5m, 30m
    - Built from raw trades only (never from candles)
    - Real-time bar completion callbacks
    - Bar history buffer per symbol/interval
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        intervals: Optional[Dict[str, int]] = None,
        on_bar: Optional[Callable[[Bar], Any]] = None,
        buffer_size: int = 1000,
    ):
        self.symbols = symbols or settings.SYMBOLS
        self.intervals = intervals or settings.BAR_INTERVALS_MS
        self.on_bar = on_bar
        self.buffer_size = buffer_size
        
        # Current building bars: {symbol: {interval: BarBuilder}}
        self._builders: Dict[str, Dict[str, BarBuilder]] = {}
        
        # Completed bars buffer: {symbol: {interval: deque[Bar]}}
        self._bars: Dict[str, Dict[str, deque]] = {}
        
        # Initialize structures
        for symbol in self.symbols:
            self._builders[symbol] = {}
            self._bars[symbol] = {}
            for interval in self.intervals:
                self._bars[symbol][interval] = deque(maxlen=buffer_size)
        
        # Stats
        self._trades_processed = 0
        self._bars_completed = 0
    
    def _get_bar_start(self, timestamp_ms: int, interval_ms: int) -> int:
        """Get the start time of the bar containing this timestamp"""
        return (timestamp_ms // interval_ms) * interval_ms
    
    def _get_or_create_builder(
        self,
        symbol: str,
        interval: str,
        interval_ms: int,
        timestamp_ms: int
    ) -> BarBuilder:
        """Get existing builder or create new one"""
        start_ms = self._get_bar_start(timestamp_ms, interval_ms)
        
        if symbol not in self._builders:
            self._builders[symbol] = {}
        
        builder = self._builders[symbol].get(interval)
        
        # Need new builder if none exists or bar period changed
        if builder is None or builder.start_ms != start_ms:
            # Finalize old builder if exists
            if builder is not None:
                self._finalize_bar(builder)
            
            # Create new builder
            builder = BarBuilder(
                symbol=symbol,
                interval=interval,
                interval_ms=interval_ms,
                start_ms=start_ms,
            )
            self._builders[symbol][interval] = builder
        
        return builder
    
    def _finalize_bar(self, builder: BarBuilder) -> Optional[Bar]:
        """Finalize a bar builder and emit the bar"""
        bar = builder.to_bar()
        
        if bar is not None:
            # Store in buffer
            symbol = builder.symbol
            interval = builder.interval
            
            if symbol not in self._bars:
                self._bars[symbol] = {}
            if interval not in self._bars[symbol]:
                self._bars[symbol][interval] = deque(maxlen=self.buffer_size)
            
            self._bars[symbol][interval].append(bar)
            self._bars_completed += 1
            
            # Callback
            if self.on_bar:
                try:
                    result = self.on_bar(bar)
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception as e:
                    logger.error("bar_callback_error", error=str(e))
        
        return bar
    
    def process_trade(self, trade: Trade) -> List[Bar]:
        """
        Process a single trade, returns any completed bars
        
        This is the main entry point - call this for each incoming trade
        """
        self._trades_processed += 1
        completed_bars = []
        symbol = trade.symbol
        
        for interval, interval_ms in self.intervals.items():
            # Check if we need to close the current bar
            current_builder = self._builders.get(symbol, {}).get(interval)
            
            if current_builder is not None:
                current_bar_end = current_builder.start_ms + current_builder.interval_ms
                
                # Trade belongs to a new bar period
                if trade.timestamp_ms >= current_bar_end:
                    bar = self._finalize_bar(current_builder)
                    if bar:
                        completed_bars.append(bar)
            
            # Get/create builder and add trade
            builder = self._get_or_create_builder(
                symbol, interval, interval_ms, trade.timestamp_ms
            )
            builder.add_trade(trade)
        
        return completed_bars
    
    def process_trades(self, trades: List[Trade]) -> List[Bar]:
        """Process multiple trades, returns all completed bars"""
        completed = []
        for trade in trades:
            completed.extend(self.process_trade(trade))
        return completed
    
    def flush_all(self) -> List[Bar]:
        """Force-complete all current bars (call at shutdown)"""
        completed = []
        for symbol, intervals in self._builders.items():
            for interval, builder in intervals.items():
                if not builder.is_empty():
                    bar = self._finalize_bar(builder)
                    if bar:
                        completed.append(bar)
        self._builders.clear()
        return completed
    
    # ========== PUBLIC API FOR TESTING ==========
    
    def get_bars(
        self,
        symbol: str,
        interval: str,
        count: Optional[int] = None
    ) -> List[Bar]:
        """Get completed bars for symbol/interval"""
        bars = self._bars.get(symbol, {}).get(interval, [])
        bar_list = list(bars)
        if count:
            return bar_list[-count:]
        return bar_list
    
    def get_latest_bar(self, symbol: str, interval: str) -> Optional[Bar]:
        """Get the most recent completed bar"""
        bars = self._bars.get(symbol, {}).get(interval, [])
        return bars[-1] if bars else None
    
    def get_current_builder(self, symbol: str, interval: str) -> Optional[BarBuilder]:
        """Get the current (incomplete) bar builder"""
        return self._builders.get(symbol, {}).get(interval)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics"""
        bar_counts = {}
        for symbol in self._bars:
            bar_counts[symbol] = {
                interval: len(bars)
                for interval, bars in self._bars[symbol].items()
            }
        
        return {
            "trades_processed": self._trades_processed,
            "bars_completed": self._bars_completed,
            "bar_counts": bar_counts,
        }


# ========== STANDALONE TEST FUNCTIONS ==========

def test_bar_builder() -> Bar:
    """
    Test BarBuilder with synthetic trades
    
    Usage:
        from src.processors.bars import test_bar_builder
        bar = test_bar_builder()
    """
    builder = BarBuilder(
        symbol="BTCUSDT",
        interval="1s",
        interval_ms=1000,
        start_ms=1700000000000,
    )
    
    # Add synthetic trades
    trades = [
        Trade("BTCUSDT", 1700000000100, 50000.0, 1.0, False, 1, 1, 1),  # Buy
        Trade("BTCUSDT", 1700000000200, 50010.0, 0.5, True, 2, 2, 2),   # Sell
        Trade("BTCUSDT", 1700000000300, 50020.0, 2.0, False, 3, 3, 3),  # Buy
        Trade("BTCUSDT", 1700000000400, 49990.0, 0.3, True, 4, 4, 4),   # Sell
        Trade("BTCUSDT", 1700000000500, 50005.0, 1.5, False, 5, 5, 5),  # Buy
    ]
    
    for trade in trades:
        builder.add_trade(trade)
    
    bar = builder.to_bar()
    
    print("=== BAR BUILDER TEST ===")
    print(f"Symbol: {bar.symbol}")
    print(f"OHLC: {bar.open:.2f} / {bar.high:.2f} / {bar.low:.2f} / {bar.close:.2f}")
    print(f"Volume: {bar.volume:.4f}")
    print(f"Trades: {bar.trade_count}")
    print(f"Buy Volume: {bar.buy_volume:.4f} ({bar.buy_count} trades)")
    print(f"Sell Volume: {bar.sell_volume:.4f} ({bar.sell_count} trades)")
    print(f"Delta: {bar.delta:.4f} ({bar.delta_pct:.1%})")
    print(f"VWAP: {bar.vwap:.2f}")
    
    return bar


def test_bar_aggregator_sync() -> Dict[str, Any]:
    """
    Test BarAggregator with synthetic trades (synchronous)
    
    Usage:
        from src.processors.bars import test_bar_aggregator_sync
        result = test_bar_aggregator_sync()
    """
    completed_bars = []
    
    def on_bar(bar: Bar):
        completed_bars.append(bar)
        print(f"  Bar completed: {bar.symbol} {bar.interval} O={bar.open:.2f} C={bar.close:.2f} V={bar.volume:.4f}")
    
    aggregator = BarAggregator(
        symbols=["BTCUSDT"],
        intervals={"250ms": 250, "1s": 1000},
        on_bar=on_bar,
    )
    
    print("=== BAR AGGREGATOR TEST ===")
    print("Generating synthetic trades spanning multiple bar periods...")
    
    # Generate trades spanning 3 seconds (multiple 250ms and 1s bars)
    base_time = 1700000000000
    price = 50000.0
    
    for i in range(30):  # 30 trades over ~3 seconds
        timestamp = base_time + (i * 100)  # 100ms apart
        price += (i % 2) * 10 - 5  # Small price movement
        qty = 0.1 + (i % 5) * 0.05
        is_buyer_maker = i % 3 == 0
        
        trade = Trade(
            symbol="BTCUSDT",
            timestamp_ms=timestamp,
            price=price,
            quantity=qty,
            is_buyer_maker=is_buyer_maker,
            trade_id=i,
            first_trade_id=i,
            last_trade_id=i,
        )
        
        aggregator.process_trade(trade)
    
    # Flush remaining
    remaining = aggregator.flush_all()
    
    stats = aggregator.get_stats()
    
    print(f"\nCompleted bars via callback: {len(completed_bars)}")
    print(f"Flushed bars: {len(remaining)}")
    print(f"Total trades processed: {stats['trades_processed']}")
    
    return {
        "completed_bars": len(completed_bars),
        "flushed_bars": len(remaining),
        "stats": stats,
    }


async def test_bar_aggregator_live(duration_seconds: int = 10) -> Dict[str, Any]:
    """
    Test BarAggregator with live trades from WebSocket
    
    Usage:
        from src.processors.bars import test_bar_aggregator_live
        import asyncio
        result = asyncio.run(test_bar_aggregator_live(10))
    """
    from src.collectors.trades import TradesCollector
    
    completed_bars: List[Bar] = []
    
    def on_bar(bar: Bar):
        completed_bars.append(bar)
        if bar.interval == "1s":  # Only print 1s bars to reduce noise
            print(f"  {bar.symbol} {bar.interval}: O={bar.open:.2f} C={bar.close:.2f} Δ={bar.delta:.4f}")
    
    aggregator = BarAggregator(
        symbols=["BTCUSDT", "ETHUSDT"],
        intervals={"250ms": 250, "1s": 1000},
        on_bar=on_bar,
    )
    
    def on_trade(trade: Trade):
        aggregator.process_trade(trade)
    
    collector = TradesCollector(
        symbols=["BTCUSDT", "ETHUSDT"],
        on_trade=on_trade,
    )
    
    print(f"Starting live bar aggregation test for {duration_seconds}s...")
    
    task = asyncio.create_task(collector.start())
    await asyncio.sleep(duration_seconds)
    await collector.stop()
    task.cancel()
    
    # Flush remaining bars
    remaining = aggregator.flush_all()
    
    stats = aggregator.get_stats()
    
    result = {
        "completed_bars": len(completed_bars),
        "flushed_bars": len(remaining),
        "stats": stats,
        "bars_by_interval": {},
    }
    
    for bar in completed_bars:
        key = f"{bar.symbol}_{bar.interval}"
        if key not in result["bars_by_interval"]:
            result["bars_by_interval"][key] = 0
        result["bars_by_interval"][key] += 1
    
    print(f"\n=== BAR AGGREGATOR LIVE TEST RESULTS ===")
    print(f"Total completed bars: {len(completed_bars)}")
    print(f"Trades processed: {stats['trades_processed']}")
    for key, count in result["bars_by_interval"].items():
        print(f"  {key}: {count} bars")
    
    return result
