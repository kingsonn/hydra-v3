"""
A️⃣ TRADES COLLECTOR
WebSocket aggTrade stream from Binance Futures
This is the most critical data source for order flow analysis
"""
import asyncio
import time
import random
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
import orjson
import structlog

from config import settings
from src.core.models import Trade

logger = structlog.get_logger(__name__)


class TradesCollector:
    """
    Collects aggregated trades from Binance Futures WebSocket
    
    Features:
    - Multi-symbol subscription via combined stream
    - Automatic reconnection
    - Trade buffering for batch processing
    - Callback support for real-time processing
    - Health metrics tracking
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        on_trade: Optional[Callable[[Trade], Any]] = None,
        on_batch: Optional[Callable[[List[Trade]], Any]] = None,
        batch_size: int = 100,
        buffer_max_size: int = 10000,
    ):
        self.symbols = [s.lower() for s in (symbols or settings.SYMBOLS)]
        self.on_trade = on_trade  # Callback for each trade
        self.on_batch = on_batch  # Callback for trade batches
        self.batch_size = batch_size
        
        # Trade buffer (thread-safe deque)
        self._buffer: deque[Trade] = deque(maxlen=buffer_max_size)
        self._batch_buffer: List[Trade] = []
        
        # Message queue for decoupling WS receive from processing
        self._msg_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=10000)
        
        # Connection state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0
        self._consecutive_errors = 0
        
        # Health metrics
        self._last_trade_time: Dict[str, int] = {}
        self._trade_count: Dict[str, int] = {s.upper(): 0 for s in self.symbols}
        self._connection_time: Optional[float] = None
        self._message_count = 0
        self._error_count = 0
    
    @property
    def ws_url(self) -> str:
        """Build combined stream URL for all symbols"""
        streams = "/".join(f"{s}@aggTrade" for s in self.symbols)
        return f"{settings.BINANCE_WS_BASE}/stream?streams={streams}"
    
    async def start(self) -> None:
        """Start the trades collector with robust reconnection"""
        self._running = True
        logger.info("trades_collector_starting", symbols=self.symbols)
        
        while self._running:
            try:
                await self._connect_and_listen()
                # If we exit cleanly, reset error count
                self._consecutive_errors = 0
            except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as e:
                self._error_count += 1
                self._consecutive_errors += 1
                logger.warning("trades_ws_disconnected", reason=str(e), consecutive=self._consecutive_errors)
            except Exception as e:
                self._error_count += 1
                self._consecutive_errors += 1
                logger.error("trades_collector_error", error=str(e), consecutive=self._consecutive_errors)
            
            if self._running:
                # Exponential backoff with jitter for reconnection
                jitter = random.uniform(0.5, 1.5)
                delay = min(self._reconnect_delay * (2 ** min(self._consecutive_errors - 1, 5)) * jitter, self._max_reconnect_delay)
                logger.info("trades_reconnecting", delay_s=f"{delay:.1f}")
                await asyncio.sleep(delay)
    
    async def stop(self) -> None:
        """Stop the trades collector"""
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info("trades_collector_stopped")
    
    async def _connect_and_listen(self) -> None:
        """Connect to WebSocket and listen for trades"""
        async with websockets.connect(
            self.ws_url,
            ping_interval=20,      # Relaxed ping interval (Binance-safe)
            ping_timeout=20,       # Match ping interval
            close_timeout=5,       # Faster close
            max_size=10_000_000,   # 10MB max message size
            compression=None,      # Disable compression for speed
        ) as ws:
            self._ws = ws
            self._connection_time = time.time()
            self._consecutive_errors = 0  # Reset on successful connect
            
            logger.info("trades_ws_connected", url=self.ws_url)
            
            # Start worker task to process messages from queue
            worker_task = asyncio.create_task(self._message_worker())
            
            try:
                # WS loop ONLY queues messages - ultra lightweight
                async for message in ws:
                    if not self._running:
                        break
                    
                    self._message_count += 1
                    try:
                        self._msg_queue.put_nowait(message)
                    except asyncio.QueueFull:
                        # Drop message if queue full - better than blocking WS
                        pass
            finally:
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass
    
    async def _message_worker(self) -> None:
        """Worker that processes messages from queue - decoupled from WS receive"""
        while self._running:
            try:
                message = await asyncio.wait_for(self._msg_queue.get(), timeout=1.0)
                await self._process_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("trade_process_error", error=str(e))
    
    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message"""
        try:
            data = orjson.loads(message)
            
            # Combined stream format: {"stream": "...", "data": {...}}
            if "stream" in data:
                stream = data["stream"]
                trade_data = data["data"]
                symbol = stream.split("@")[0].upper()
            else:
                # Single stream format
                trade_data = data
                symbol = trade_data.get("s", "").upper()
            
            # Parse trade
            trade = Trade.from_binance(trade_data, symbol)
            
            # Update metrics
            self._last_trade_time[symbol] = trade.timestamp_ms
            self._trade_count[symbol] = self._trade_count.get(symbol, 0) + 1
            
            # Add to buffer
            self._buffer.append(trade)
            
            # Callback for individual trade
            if self.on_trade:
                await self._safe_callback(self.on_trade, trade)
            
            # Batch callback
            if self.on_batch:
                self._batch_buffer.append(trade)
                if len(self._batch_buffer) >= self.batch_size:
                    await self._safe_callback(self.on_batch, self._batch_buffer.copy())
                    self._batch_buffer.clear()
        
        except Exception as e:
            logger.error("trade_parse_error", error=str(e), message=message[:200])
    
    async def _safe_callback(self, callback: Callable, *args) -> None:
        """Safely execute callback (sync or async)"""
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error("callback_error", error=str(e))
    
    # ========== PUBLIC API FOR TESTING ==========
    
    def get_recent_trades(self, count: int = 100) -> List[Trade]:
        """Get most recent trades from buffer"""
        trades = list(self._buffer)
        return trades[-count:] if len(trades) > count else trades
    
    def get_trades_for_symbol(self, symbol: str, count: int = 100) -> List[Trade]:
        """Get recent trades for a specific symbol"""
        symbol = symbol.upper()
        trades = [t for t in self._buffer if t.symbol == symbol]
        return trades[-count:] if len(trades) > count else trades
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get collector health metrics"""
        now_ms = int(time.time() * 1000)
        
        # Check connection state safely
        is_connected = False
        if self._ws is not None:
            try:
                # Try to check connection state
                state = self._ws.state
                is_connected = state == 1  # OPEN state
            except AttributeError:
                # Fallback: assume connected if no error
                is_connected = True
        
        return {
            "connected": is_connected,
            "uptime_seconds": time.time() - self._connection_time if self._connection_time else 0,
            "message_count": self._message_count,
            "error_count": self._error_count,
            "trade_counts": self._trade_count.copy(),
            "last_trade_lag_ms": {
                symbol: now_ms - ts
                for symbol, ts in self._last_trade_time.items()
            },
            "buffer_size": len(self._buffer),
        }
    
    def clear_buffer(self) -> int:
        """Clear buffer and return count of cleared trades"""
        count = len(self._buffer)
        self._buffer.clear()
        return count


# ========== STANDALONE TEST FUNCTION ==========

async def test_trades_collector(duration_seconds: int = 10) -> Dict[str, Any]:
    """
    Test the trades collector standalone
    
    Usage:
        from src.collectors.trades import test_trades_collector
        import asyncio
        result = asyncio.run(test_trades_collector(10))
        print(result)
    """
    trades_received: List[Trade] = []
    
    def on_trade(trade: Trade):
        trades_received.append(trade)
        if len(trades_received) <= 5:
            print(f"  Trade: {trade.symbol} {trade.side.value} {trade.quantity:.4f} @ {trade.price:.2f}")
    
    collector = TradesCollector(
        symbols=["BTCUSDT", "ETHUSDT"],
        on_trade=on_trade,
    )
    
    print(f"Starting trades collector test for {duration_seconds}s...")
    
    # Start collector in background
    task = asyncio.create_task(collector.start())
    
    # Wait for duration
    await asyncio.sleep(duration_seconds)
    
    # Stop collector
    await collector.stop()
    task.cancel()
    
    # Get results
    metrics = collector.get_health_metrics()
    
    result = {
        "total_trades": len(trades_received),
        "trades_per_second": len(trades_received) / duration_seconds,
        "by_symbol": {},
        "sample_trade": trades_received[0].to_dict() if trades_received else None,
        "health_metrics": metrics,
    }
    
    # Count by symbol
    for trade in trades_received:
        sym = trade.symbol
        if sym not in result["by_symbol"]:
            result["by_symbol"][sym] = {"count": 0, "volume": 0.0, "notional": 0.0}
        result["by_symbol"][sym]["count"] += 1
        result["by_symbol"][sym]["volume"] += trade.quantity
        result["by_symbol"][sym]["notional"] += trade.notional
    
    print(f"\n=== TRADES COLLECTOR TEST RESULTS ===")
    print(f"Total trades: {result['total_trades']}")
    print(f"Trades/sec: {result['trades_per_second']:.1f}")
    for sym, data in result["by_symbol"].items():
        print(f"  {sym}: {data['count']} trades, ${data['notional']:,.0f} notional")
    
    return result
