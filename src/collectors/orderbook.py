"""
B️⃣ ORDER BOOK COLLECTOR
WebSocket depth@100ms stream from Binance Futures
Used for absorption analysis, liquidity detection, sweep flags
"""
import asyncio
import time
import random
import socket
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
import orjson
import structlog

from config import settings
from src.core.models import OrderBookSnapshot, OrderBookLevel
from src.core.resilience import (
    get_supervisor, AdaptiveBackoff, WebSocketConfig,
    CircuitState
)

logger = structlog.get_logger(__name__)


@dataclass
class OrderBookMetrics:
    """Derived metrics from order book updates"""
    timestamp_ms: int
    symbol: str
    bid_depth_5: float      # USD depth at top 5 bid levels
    ask_depth_5: float      # USD depth at top 5 ask levels
    bid_depth_10: float     # USD depth at top 10 bid levels
    ask_depth_10: float     # USD depth at top 10 ask levels
    depth_imbalance_5: float  # (bid-ask)/(bid+ask) for top 5
    depth_imbalance_10: float # (bid-ask)/(bid+ask) for top 10
    spread_bps: float       # Spread in basis points
    mid_price: float
    
    # Change metrics (vs previous snapshot)
    bid_refill_rate: float  # Change in bid depth
    ask_refill_rate: float  # Change in ask depth
    sweep_flag: bool        # Large sudden depth removal


class OrderBookCollector:
    """
    Collects order book depth from Binance Futures WebSocket
    
    Features:
    - 100ms depth updates (delta or full snapshot)
    - Top N levels only (configurable, default 20)
    - Real-time metrics computation
    - Periodic snapshot storage
    - Sweep detection
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        depth_levels: int = 20,
        on_snapshot: Optional[Callable[[OrderBookSnapshot], Any]] = None,
        on_metrics: Optional[Callable[[OrderBookMetrics], Any]] = None,
        snapshot_interval_s: int = 5,
    ):
        self.symbols = [s.lower() for s in (symbols or settings.SYMBOLS)]
        self.depth_levels = depth_levels
        self.on_snapshot = on_snapshot
        self.on_metrics = on_metrics
        self.snapshot_interval_s = snapshot_interval_s
        
        # Current order books per symbol
        self._books: Dict[str, OrderBookSnapshot] = {}
        self._prev_books: Dict[str, OrderBookSnapshot] = {}
        
        # Metrics buffer
        self._metrics_buffer: deque[OrderBookMetrics] = deque(maxlen=1000)
        
        # Snapshot buffer for storage
        self._snapshot_buffer: deque[OrderBookSnapshot] = deque(maxlen=100)
        self._last_snapshot_time: Dict[str, float] = {}
        
        # Message queue for decoupling WS receive from processing
        self._msg_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=5000)
        
        # Connection state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._consecutive_errors = 0
        
        # Resilience components - MORE TOLERANT for 24/7 operation
        self._backoff = AdaptiveBackoff(base_delay_s=2.0, max_delay_s=120.0)
        self._ws_config = WebSocketConfig(
            ping_interval=10.0,  # More frequent pings to keep NAT alive
            ping_timeout=30.0,
            close_timeout=10.0,
            connect_timeout=45.0,
        )
        self._supervisor = get_supervisor()
        self._conn_name = "orderbook_ws"
        self._force_reconnect = False  # Flag for graceful reconnect
        
        # Health metrics
        self._last_update_time: Dict[str, int] = {}
        self._update_count: Dict[str, int] = {s.upper(): 0 for s in self.symbols}
        self._connection_time: Optional[float] = None
        self._message_count = 0
        self._last_message_time: float = 0
        
        # Periodic refresh (reconnect every 12 hours for stability)
        self._max_connection_age_s = 12 * 3600  # 12 hours
    
    @property
    def ws_url(self) -> str:
        """Build combined stream URL for depth"""
        streams = "/".join(f"{s}@depth@100ms" for s in self.symbols)
        return f"{settings.BINANCE_WS_BASE}/stream?streams={streams}"
    
    async def start(self) -> None:
        """Start the order book collector with robust reconnection"""
        self._running = True
        logger.info("orderbook_collector_starting", symbols=self.symbols)
        
        # Register with supervisor
        self._supervisor.register_connection(self._conn_name)
        
        while self._running:
            # Check circuit breaker
            circuit = self._supervisor.get_circuit(self._conn_name)
            if circuit and circuit.state == CircuitState.OPEN:
                logger.debug("orderbook_circuit_open", waiting_s=30)
                await asyncio.sleep(30)
                continue
            
            try:
                await self._connect_and_listen()
                self._consecutive_errors = 0
                self._backoff.record_success()
            except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as e:
                if not self._running:
                    break
                self._consecutive_errors += 1
                self._backoff.record_failure()
                self._supervisor.record_error(self._conn_name)
                reason = str(e)[:100]  # Truncate for logging
                logger.warning("orderbook_ws_disconnected", reason=reason, consecutive=self._consecutive_errors)
            except asyncio.TimeoutError:
                if not self._running:
                    break
                self._consecutive_errors += 1
                self._backoff.record_failure()
                self._supervisor.record_error(self._conn_name)
                logger.warning("orderbook_connect_timeout", consecutive=self._consecutive_errors)
            except (GeneratorExit, asyncio.CancelledError):
                break
            except RuntimeError as e:
                if "no running event loop" in str(e) or "Event loop is closed" in str(e):
                    break
                self._consecutive_errors += 1
                self._backoff.record_failure()
                self._supervisor.record_error(self._conn_name)
                logger.error("orderbook_collector_error", error=str(e)[:100], consecutive=self._consecutive_errors)
            except Exception as e:
                if not self._running:
                    break
                self._consecutive_errors += 1
                self._backoff.record_failure()
                self._supervisor.record_error(self._conn_name)
                logger.error("orderbook_collector_error", error=str(e)[:100], consecutive=self._consecutive_errors)
            
            if self._running:
                # Check if supervisor allows reconnection
                if not self._supervisor.should_reconnect(self._conn_name):
                    logger.info("orderbook_reconnect_paused")
                    await asyncio.sleep(30)
                    continue
                
                delay = self._backoff.get_delay()
                logger.info("orderbook_reconnecting", delay_s=f"{delay:.1f}")
                try:
                    await asyncio.sleep(delay)
                except (asyncio.CancelledError, GeneratorExit):
                    break
    
    async def stop(self) -> None:
        """Stop the order book collector gracefully"""
        self._running = False
        if self._ws:
            try:
                await asyncio.wait_for(self._ws.close(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass  # Best effort close
        self._ws = None
        logger.info("orderbook_collector_stopped")
    
    async def _connect_and_listen(self) -> None:
        """Connect and listen for order book updates"""
        # Reset reconnect flag
        self._force_reconnect = False
        
        # Create socket with TCP keepalive to prevent NAT timeout
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # Set keepalive parameters (platform-specific, may not work on all OS)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
        except (AttributeError, OSError):
            pass  # Not available on all platforms
        
        # Use timeout for connection with extra headers
        async with asyncio.timeout(self._ws_config.connect_timeout):
            ws = await websockets.connect(
                self.ws_url,
                **self._ws_config.to_kwargs(),
                extra_headers={
                    "User-Agent": "HYDRA-Trading-Bot/3.0",
                    "Origin": "https://fapi.binance.com",
                },
            )
        
        async with ws:
            self._ws = ws
            self._connection_time = time.time()
            self._consecutive_errors = 0
            self._backoff.reset()
            
            # Notify supervisor
            self._supervisor.record_connect(self._conn_name)
            
            logger.info("orderbook_ws_connected")
            
            # Start worker task and health checker
            worker_task = asyncio.create_task(self._message_worker())
            health_task = asyncio.create_task(self._connection_health_loop())
            
            try:
                # WS loop ONLY queues messages - ultra lightweight
                async for message in ws:
                    if not self._running:
                        break
                    
                    self._message_count += 1
                    self._last_message_time = time.time()
                    self._supervisor.record_message(self._conn_name)
                    
                    try:
                        self._msg_queue.put_nowait(message)
                    except asyncio.QueueFull:
                        # Drop oldest and add new (prefer fresh data)
                        try:
                            self._msg_queue.get_nowait()
                            self._msg_queue.put_nowait(message)
                        except asyncio.QueueEmpty:
                            pass
                    
                    # Check if connection is too old (periodic refresh)
                    if time.time() - self._connection_time > self._max_connection_age_s:
                        logger.info("orderbook_connection_refresh", age_hours=self._max_connection_age_s/3600)
                        self._force_reconnect = True
                        break
                    
                    # Check if health loop requested reconnect
                    if self._force_reconnect:
                        logger.info("orderbook_force_reconnect_requested")
                        break
                        
            except GeneratorExit:
                pass
            except asyncio.CancelledError:
                pass
            finally:
                self._supervisor.record_disconnect(self._conn_name)
                worker_task.cancel()
                health_task.cancel()
                try:
                    await asyncio.gather(worker_task, health_task, return_exceptions=True)
                except Exception:
                    pass
    
    async def _message_worker(self) -> None:
        """Worker that processes messages from queue - decoupled from WS receive"""
        while self._running:
            try:
                message = await asyncio.wait_for(self._msg_queue.get(), timeout=1.0)
                await self._process_message(message)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("orderbook_process_error", error=str(e)[:100])
    
    async def _connection_health_loop(self) -> None:
        """Monitor connection health and signal reconnect if needed"""
        while self._running and not self._force_reconnect:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check for message silence (connection might be dead)
                silence = time.time() - self._last_message_time if self._last_message_time else 0
                if silence > 90:  # 90 seconds of silence = likely dead
                    logger.warning("orderbook_connection_silent", silence_s=f"{silence:.1f}")
                    # Signal reconnect instead of forcing close (avoids race condition)
                    self._force_reconnect = True
                    break
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("orderbook_health_check_error", error=str(e)[:50])
    
    async def _process_message(self, message: str) -> None:
        """Process order book update"""
        try:
            data = orjson.loads(message)
            
            if "stream" in data:
                stream = data["stream"]
                depth_data = data["data"]
                symbol = stream.split("@")[0].upper()
            else:
                depth_data = data
                symbol = depth_data.get("s", "").upper()
            
            # Parse order book
            timestamp_ms = depth_data.get("E", int(time.time() * 1000))
            
            bids = [
                OrderBookLevel(price=float(p), quantity=float(q))
                for p, q in depth_data.get("b", [])[:self.depth_levels]
            ]
            asks = [
                OrderBookLevel(price=float(p), quantity=float(q))
                for p, q in depth_data.get("a", [])[:self.depth_levels]
            ]
            
            snapshot = OrderBookSnapshot(
                symbol=symbol,
                timestamp_ms=timestamp_ms,
                bids=bids,
                asks=asks,
                last_update_id=depth_data.get("u", 0),
            )
            
            # Store previous for metrics
            if symbol in self._books:
                self._prev_books[symbol] = self._books[symbol]
            
            self._books[symbol] = snapshot
            self._last_update_time[symbol] = timestamp_ms
            self._update_count[symbol] = self._update_count.get(symbol, 0) + 1
            
            # Compute metrics
            metrics = self._compute_metrics(snapshot)
            if metrics:
                self._metrics_buffer.append(metrics)
                if self.on_metrics:
                    await self._safe_callback(self.on_metrics, metrics)
            
            # Periodic snapshot callback
            now = time.time()
            last_snap = self._last_snapshot_time.get(symbol, 0)
            if now - last_snap >= self.snapshot_interval_s:
                self._snapshot_buffer.append(snapshot)
                self._last_snapshot_time[symbol] = now
                if self.on_snapshot:
                    await self._safe_callback(self.on_snapshot, snapshot)
        
        except Exception as e:
            logger.error("orderbook_parse_error", error=str(e))
    
    def _compute_metrics(self, snapshot: OrderBookSnapshot) -> Optional[OrderBookMetrics]:
        """Compute derived metrics from order book"""
        if not snapshot.bids or not snapshot.asks:
            return None
        
        symbol = snapshot.symbol
        prev = self._prev_books.get(symbol)
        
        bid_depth_5 = snapshot.bid_depth_usd(5)
        ask_depth_5 = snapshot.ask_depth_usd(5)
        bid_depth_10 = snapshot.bid_depth_usd(10)
        ask_depth_10 = snapshot.ask_depth_usd(10)
        
        # Refill rates (change in depth)
        bid_refill = 0.0
        ask_refill = 0.0
        sweep_flag = False
        
        if prev:
            prev_bid_5 = prev.bid_depth_usd(5)
            prev_ask_5 = prev.ask_depth_usd(5)
            bid_refill = bid_depth_5 - prev_bid_5
            ask_refill = ask_depth_5 - prev_ask_5
            
            # Sweep detection: >30% depth removed in single update
            if prev_bid_5 > 0 and bid_refill < -0.3 * prev_bid_5:
                sweep_flag = True
            if prev_ask_5 > 0 and ask_refill < -0.3 * prev_ask_5:
                sweep_flag = True
        
        return OrderBookMetrics(
            timestamp_ms=snapshot.timestamp_ms,
            symbol=symbol,
            bid_depth_5=bid_depth_5,
            ask_depth_5=ask_depth_5,
            bid_depth_10=bid_depth_10,
            ask_depth_10=ask_depth_10,
            depth_imbalance_5=snapshot.depth_imbalance(5),
            depth_imbalance_10=snapshot.depth_imbalance(10),
            spread_bps=snapshot.spread_bps or 0.0,
            mid_price=snapshot.mid_price or 0.0,
            bid_refill_rate=bid_refill,
            ask_refill_rate=ask_refill,
            sweep_flag=sweep_flag,
        )
    
    async def _safe_callback(self, callback: Callable, *args) -> None:
        """Safely execute callback"""
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error("orderbook_callback_error", error=str(e))
    
    # ========== PUBLIC API FOR TESTING ==========
    
    def get_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get current order book for symbol"""
        return self._books.get(symbol.upper())
    
    def get_all_books(self) -> Dict[str, OrderBookSnapshot]:
        """Get all current order books"""
        return self._books.copy()
    
    def get_recent_metrics(self, count: int = 100) -> List[OrderBookMetrics]:
        """Get recent metrics"""
        metrics = list(self._metrics_buffer)
        return metrics[-count:]
    
    def get_recent_snapshots(self) -> List[OrderBookSnapshot]:
        """Get buffered snapshots"""
        return list(self._snapshot_buffer)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get collector health metrics"""
        now_ms = int(time.time() * 1000)
        
        # Check connection state safely
        is_connected = False
        if self._ws is not None:
            try:
                state = self._ws.state
                is_connected = state == 1  # OPEN state
            except AttributeError:
                is_connected = True
        
        return {
            "connected": is_connected,
            "uptime_seconds": time.time() - self._connection_time if self._connection_time else 0,
            "message_count": self._message_count,
            "update_counts": self._update_count.copy(),
            "last_update_lag_ms": {
                sym: now_ms - ts for sym, ts in self._last_update_time.items()
            },
            "books_tracked": len(self._books),
        }


# ========== STANDALONE TEST FUNCTION ==========

async def test_orderbook_collector(duration_seconds: int = 10) -> Dict[str, Any]:
    """
    Test the order book collector standalone
    
    Usage:
        from src.collectors.orderbook import test_orderbook_collector
        import asyncio
        result = asyncio.run(test_orderbook_collector(10))
    """
    snapshots: List[OrderBookSnapshot] = []
    metrics_list: List[OrderBookMetrics] = []
    
    def on_snapshot(snap: OrderBookSnapshot):
        snapshots.append(snap)
        print(f"  Snapshot: {snap.symbol} mid={snap.mid_price:.2f} spread={snap.spread_bps:.2f}bps")
    
    def on_metrics(m: OrderBookMetrics):
        metrics_list.append(m)
        if m.sweep_flag:
            print(f"  ⚠️ SWEEP DETECTED: {m.symbol}")
    
    collector = OrderBookCollector(
        symbols=["BTCUSDT", "ETHUSDT"],
        on_snapshot=on_snapshot,
        on_metrics=on_metrics,
        snapshot_interval_s=2,
    )
    
    print(f"Starting orderbook collector test for {duration_seconds}s...")
    
    task = asyncio.create_task(collector.start())
    await asyncio.sleep(duration_seconds)
    await collector.stop()
    task.cancel()
    
    health = collector.get_health_metrics()
    
    result = {
        "snapshots_captured": len(snapshots),
        "metrics_computed": len(metrics_list),
        "updates_per_second": health["message_count"] / duration_seconds,
        "sweeps_detected": sum(1 for m in metrics_list if m.sweep_flag),
        "final_books": {},
        "health": health,
    }
    
    for symbol, book in collector.get_all_books().items():
        result["final_books"][symbol] = {
            "mid_price": book.mid_price,
            "spread_bps": book.spread_bps,
            "bid_depth_5_usd": book.bid_depth_usd(5),
            "ask_depth_5_usd": book.ask_depth_usd(5),
            "imbalance_5": book.depth_imbalance(5),
        }
    
    print(f"\n=== ORDERBOOK COLLECTOR TEST RESULTS ===")
    print(f"Updates/sec: {result['updates_per_second']:.1f}")
    print(f"Snapshots: {result['snapshots_captured']}")
    print(f"Sweeps: {result['sweeps_detected']}")
    for sym, data in result["final_books"].items():
        print(f"  {sym}: mid=${data['mid_price']:.2f}, imbalance={data['imbalance_5']:.2%}")
    
    return result
