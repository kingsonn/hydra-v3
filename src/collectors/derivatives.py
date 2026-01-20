"""
E️⃣ DERIVATIVES CONTEXT COLLECTOR
REST API polling for funding rates, open interest
WebSocket stream for liquidations
"""
import asyncio
import time
import random
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass
import httpx
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
import orjson
import structlog

from config import settings
from src.core.models import FundingRate, OpenInterest, Liquidation, Side
from src.core.resilience import (
    get_supervisor, AdaptiveBackoff, WebSocketConfig,
    CircuitState
)

logger = structlog.get_logger(__name__)


@dataclass
class LiquidationBucket:
    """Rolling bucket for liquidation tracking"""
    window_ms: int
    long_usd: float = 0.0
    short_usd: float = 0.0
    long_count: int = 0
    short_count: int = 0
    
    @property
    def total_usd(self) -> float:
        return self.long_usd + self.short_usd
    
    @property
    def imbalance(self) -> float:
        """Positive = more longs liquidated, negative = more shorts"""
        total = self.total_usd
        if total == 0:
            return 0.0
        return (self.long_usd - self.short_usd) / total


@dataclass
class LiquidationStats:
    """Aggregated liquidation statistics with rolling buckets (30s, 2m, 5m, 1h)"""
    timestamp_ms: int
    symbol: str
    bucket_30s: LiquidationBucket
    bucket_2m: LiquidationBucket
    bucket_5m: LiquidationBucket
    bucket_1h: LiquidationBucket = None
    
    def __post_init__(self):
        if self.bucket_1h is None:
            self.bucket_1h = LiquidationBucket(window_ms=3600_000)
    
    @property
    def liq_usd_1h(self) -> float:
        """Total liquidation USD in last hour"""
        return self.bucket_1h.total_usd if self.bucket_1h else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "symbol": self.symbol,
            "liq_long_usd_30s": self.bucket_30s.long_usd,
            "liq_short_usd_30s": self.bucket_30s.short_usd,
            "imbalance_30s": self.bucket_30s.imbalance,
            "liq_long_usd_2m": self.bucket_2m.long_usd,
            "liq_short_usd_2m": self.bucket_2m.short_usd,
            "imbalance_2m": self.bucket_2m.imbalance,
            "liq_long_usd_5m": self.bucket_5m.long_usd,
            "liq_short_usd_5m": self.bucket_5m.short_usd,
            "imbalance_5m": self.bucket_5m.imbalance,
            "liq_long_usd_1h": self.bucket_1h.long_usd if self.bucket_1h else 0.0,
            "liq_short_usd_1h": self.bucket_1h.short_usd if self.bucket_1h else 0.0,
            "liq_usd_1h": self.liq_usd_1h,
            "imbalance_1h": self.bucket_1h.imbalance if self.bucket_1h else 0.0,
        }


class DerivativesCollector:
    """
    Collects derivatives data:
    - Funding rates (REST API polling)
    - Open interest (REST API polling)
    - Liquidations (WebSocket forceOrder stream)
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        on_funding: Optional[Callable[[FundingRate], Any]] = None,
        on_oi: Optional[Callable[[OpenInterest], Any]] = None,
        on_liquidation: Optional[Callable[[Liquidation], Any]] = None,
        on_liq_stats: Optional[Callable[[LiquidationStats], Any]] = None,
        funding_interval_s: int = 60,
        oi_interval_s: int = 60,
    ):
        self.symbols = symbols or settings.SYMBOLS
        self.on_funding = on_funding
        self.on_oi = on_oi
        self.on_liquidation = on_liquidation
        self.on_liq_stats = on_liq_stats
        
        self.funding_interval_s = funding_interval_s
        self.oi_interval_s = oi_interval_s
        
        # Data buffers
        self._funding_rates: Dict[str, deque[FundingRate]] = {
            s: deque(maxlen=100) for s in self.symbols
        }
        self._open_interest: Dict[str, deque[OpenInterest]] = {
            s: deque(maxlen=1000) for s in self.symbols
        }
        
        # Liquidation tracking with timestamps
        self._liquidations: deque[tuple[int, Liquidation]] = deque(maxlen=10000)
        
        # Message queue for decoupling WS receive from processing
        self._liq_msg_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=5000)
        
        # Rolling buckets per symbol (30s, 2m, 5m, 1h windows)
        self._liq_buckets: Dict[str, Dict[str, LiquidationBucket]] = {
            s: {
                "30s": LiquidationBucket(window_ms=30_000),
                "2m": LiquidationBucket(window_ms=120_000),
                "5m": LiquidationBucket(window_ms=300_000),
                "1h": LiquidationBucket(window_ms=3600_000),
            }
            for s in self.symbols
        }
        
        # 5-min interval snapshots for long-term rolling memory
        self._liq_5m_snapshots: Dict[str, deque] = {
            s: deque(maxlen=96) for s in self.symbols  # 8 hours of 5-min data
        }
        self._last_5m_snapshot_time: Dict[str, int] = {s: 0 for s in self.symbols}
        
        # State
        self._running = False
        self._client: Optional[httpx.AsyncClient] = None
        self._liq_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._last_funding_time: Dict[str, int] = {}
        
        # Resilience components - MORE TOLERANT for 24/7 operation
        self._backoff = AdaptiveBackoff(base_delay_s=2.0, max_delay_s=120.0)
        self._ws_config = WebSocketConfig(
            ping_interval=10.0,  # More frequent pings to keep NAT alive
            ping_timeout=30.0,
            close_timeout=10.0,
            connect_timeout=45.0,
        )
        self._supervisor = get_supervisor()
        self._conn_name = "liquidations_ws"
        self._rest_conn_name = "rest_api"
        self._force_reconnect = False  # Flag for graceful reconnect
        
        # Health
        self._request_count = 0
        self._error_count = 0
        self._liq_message_count = 0
        self._consecutive_errors = 0
        self._last_liq_message_time: float = 0
        
        # HTTP client refresh (recreate every 2 hours to avoid stale connections)
        self._client_created_time: float = 0
        self._max_client_age_s = 2 * 3600  # 2 hours
        
        # WS periodic refresh
        self._liq_connection_time: float = 0
        self._max_ws_age_s = 12 * 3600  # 12 hours
    
    async def start(self) -> None:
        """Start all data collection tasks"""
        self._running = True
        await self._ensure_http_client()
        
        # Register with supervisor
        self._supervisor.register_connection(self._conn_name)
        self._supervisor.register_connection(self._rest_conn_name)
        
        logger.info("derivatives_collector_starting", symbols=self.symbols)
        
        # Run all tasks concurrently - return_exceptions prevents one failure from crashing all
        results = await asyncio.gather(
            self._poll_funding_loop(),
            self._poll_oi_loop(),
            self._liquidations_ws_loop(),
            self._update_buckets_loop(),
            self._http_client_refresh_loop(),
            return_exceptions=True,
        )
        # Log any task failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("derivatives_task_failed", task_index=i, error=str(result)[:100])
    
    async def _ensure_http_client(self) -> None:
        """Ensure HTTP client exists and is fresh"""
        now = time.time()
        needs_refresh = (
            self._client is None or
            now - self._client_created_time > self._max_client_age_s
        )
        
        if needs_refresh:
            # Close old client
            if self._client:
                try:
                    await asyncio.wait_for(self._client.aclose(), timeout=5.0)
                except Exception:
                    pass
            
            # Create new client with LONGER timeouts for stability
            self._client = httpx.AsyncClient(
                base_url=settings.BINANCE_REST_BASE,
                timeout=httpx.Timeout(60.0, connect=30.0, pool=30.0),  # Longer timeouts
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                headers={"User-Agent": "HYDRA-Trading-Bot/1.0"},
            )
            self._client_created_time = now
            logger.debug("http_client_refreshed")
    
    async def _http_client_refresh_loop(self) -> None:
        """Periodically refresh HTTP client to avoid stale connections"""
        while self._running:
            try:
                await asyncio.sleep(self._max_client_age_s / 2)  # Check at half the max age
                await self._ensure_http_client()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("http_client_refresh_error", error=str(e)[:50])
    
    async def stop(self) -> None:
        """Stop all collection gracefully"""
        self._running = False
        
        # Close HTTP client
        if self._client:
            try:
                await asyncio.wait_for(self._client.aclose(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass
        self._client = None
        
        # Close WebSocket
        if self._liq_ws:
            try:
                await asyncio.wait_for(self._liq_ws.close(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass
        self._liq_ws = None
        
        logger.info("derivatives_collector_stopped")
    
    # ========== FUNDING RATES ==========
    
    async def _poll_funding_loop(self) -> None:
        """Poll funding rates periodically"""
        while self._running:
            try:
                await self._fetch_funding_rates()
            except Exception as e:
                self._error_count += 1
                logger.error("funding_poll_error", error=str(e))
            
            await asyncio.sleep(self.funding_interval_s)
    
    async def _fetch_funding_rates(self) -> List[FundingRate]:
        """Fetch current funding rates for all symbols"""
        rates = []
        
        for i, symbol in enumerate(self.symbols):
            # Small delay between requests to avoid rate limiting
            if i > 0:
                await asyncio.sleep(0.1)
            
            try:
                self._request_count += 1
                resp = await self._client.get(
                    "/fapi/v1/fundingRate",
                    params={"symbol": symbol, "limit": 1}
                )
                resp.raise_for_status()
                data = resp.json()
                
                if data:
                    item = data[0]
                    rate = FundingRate(
                        symbol=symbol,
                        timestamp_ms=item["fundingTime"],
                        funding_rate=float(item["fundingRate"]),
                        mark_price=float(item.get("markPrice", 0)),
                    )
                    
                    # Only emit if new
                    last_time = self._last_funding_time.get(symbol, 0)
                    if rate.timestamp_ms > last_time:
                        self._funding_rates[symbol].append(rate)
                        self._last_funding_time[symbol] = rate.timestamp_ms
                        rates.append(rate)
                        
                        if self.on_funding:
                            await self._safe_callback(self.on_funding, rate)
            
            except httpx.TimeoutException:
                logger.warning("funding_fetch_timeout", symbol=symbol)
            except httpx.HTTPStatusError as e:
                logger.warning("funding_fetch_http_error", symbol=symbol, status=e.response.status_code)
            except Exception as e:
                logger.warning("funding_fetch_error", symbol=symbol, error=f"{type(e).__name__}: {e}")
        
        return rates
    
    async def fetch_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        """Fetch single funding rate (for testing)"""
        try:
            self._request_count += 1
            resp = await self._client.get(
                "/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": 1}
            )
            resp.raise_for_status()
            data = resp.json()
            
            if data:
                item = data[0]
                return FundingRate(
                    symbol=symbol,
                    timestamp_ms=item["fundingTime"],
                    funding_rate=float(item["fundingRate"]),
                    mark_price=float(item.get("markPrice", 0)),
                )
        except Exception as e:
            logger.error("funding_fetch_error", symbol=symbol, error=str(e))
        return None
    
    # ========== OPEN INTEREST ==========
    
    async def _poll_oi_loop(self) -> None:
        """Poll open interest periodically"""
        while self._running:
            try:
                await self._fetch_open_interest()
            except Exception as e:
                self._error_count += 1
                logger.error("oi_poll_error", error=str(e))
            
            await asyncio.sleep(self.oi_interval_s)
    
    async def _fetch_open_interest(self) -> List[OpenInterest]:
        """Fetch current open interest for all symbols"""
        oi_list = []
        
        for i, symbol in enumerate(self.symbols):
            # Small delay between requests to avoid rate limiting
            if i > 0:
                await asyncio.sleep(0.1)
            
            try:
                self._request_count += 1
                resp = await self._client.get(
                    "/fapi/v1/openInterest",
                    params={"symbol": symbol}
                )
                resp.raise_for_status()
                data = resp.json()
                
                now_ms = int(time.time() * 1000)
                oi = OpenInterest(
                    symbol=symbol,
                    timestamp_ms=now_ms,
                    open_interest=float(data["openInterest"]),
                    open_interest_value=0.0,  # Need mark price
                )
                
                self._open_interest[symbol].append(oi)
                oi_list.append(oi)
                
                if self.on_oi:
                    await self._safe_callback(self.on_oi, oi)
            
            except httpx.TimeoutException:
                logger.warning("oi_fetch_timeout", symbol=symbol)
            except httpx.HTTPStatusError as e:
                logger.warning("oi_fetch_http_error", symbol=symbol, status=e.response.status_code)
            except Exception as e:
                logger.warning("oi_fetch_error", symbol=symbol, error=f"{type(e).__name__}: {e}")
        
        return oi_list
    
    async def fetch_open_interest(self, symbol: str) -> Optional[OpenInterest]:
        """Fetch single OI (for testing)"""
        try:
            self._request_count += 1
            resp = await self._client.get(
                "/fapi/v1/openInterest",
                params={"symbol": symbol}
            )
            resp.raise_for_status()
            data = resp.json()
            
            return OpenInterest(
                symbol=symbol,
                timestamp_ms=int(time.time() * 1000),
                open_interest=float(data["openInterest"]),
                open_interest_value=0.0,
            )
        except Exception as e:
            logger.error("oi_fetch_error", symbol=symbol, error=str(e))
        return None
    
    # ========== LIQUIDATIONS (WebSocket) ==========
    
    @property
    def liq_ws_url(self) -> str:
        """Build combined stream URL for liquidations"""
        streams = "/".join(f"{s.lower()}@forceOrder" for s in self.symbols)
        return f"{settings.BINANCE_WS_BASE}/stream?streams={streams}"
    
    async def _liquidations_ws_loop(self) -> None:
        """WebSocket loop for liquidation events with robust reconnection"""
        while self._running:
            # Check circuit breaker
            circuit = self._supervisor.get_circuit(self._conn_name)
            if circuit and circuit.state == CircuitState.OPEN:
                logger.debug("liquidations_circuit_open", waiting_s=30)
                await asyncio.sleep(30)
                continue
            
            try:
                await self._connect_and_listen_liquidations()
                self._consecutive_errors = 0
                self._backoff.record_success()
            except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as e:
                if not self._running:
                    break
                self._consecutive_errors += 1
                self._backoff.record_failure()
                self._supervisor.record_error(self._conn_name)
                reason = str(e)[:100]
                logger.warning("liquidations_ws_disconnected", reason=reason, consecutive=self._consecutive_errors)
            except asyncio.TimeoutError:
                if not self._running:
                    break
                self._consecutive_errors += 1
                self._backoff.record_failure()
                self._supervisor.record_error(self._conn_name)
                logger.warning("liquidations_connect_timeout", consecutive=self._consecutive_errors)
            except (GeneratorExit, asyncio.CancelledError):
                break
            except RuntimeError as e:
                if "no running event loop" in str(e) or "Event loop is closed" in str(e):
                    break
                self._error_count += 1
                self._consecutive_errors += 1
                self._backoff.record_failure()
                self._supervisor.record_error(self._conn_name)
                logger.error("liquidations_ws_error", error=str(e)[:100], consecutive=self._consecutive_errors)
            except Exception as e:
                if not self._running:
                    break
                self._error_count += 1
                self._consecutive_errors += 1
                self._backoff.record_failure()
                self._supervisor.record_error(self._conn_name)
                logger.error("liquidations_ws_error", error=str(e)[:100], consecutive=self._consecutive_errors)
            
            if self._running:
                # Check if supervisor allows reconnection
                if not self._supervisor.should_reconnect(self._conn_name):
                    logger.info("liquidations_reconnect_paused")
                    await asyncio.sleep(30)
                    continue
                
                delay = self._backoff.get_delay()
                logger.info("liquidations_reconnecting", delay_s=f"{delay:.1f}")
                try:
                    await asyncio.sleep(delay)
                except (asyncio.CancelledError, GeneratorExit):
                    break
    
    async def _connect_and_listen_liquidations(self) -> None:
        """Connect to liquidation WebSocket and listen"""
        # Reset reconnect flag
        self._force_reconnect = False
        
        # Use timeout for connection
        async with asyncio.timeout(self._ws_config.connect_timeout):
            ws = await websockets.connect(
                self.liq_ws_url,
                **self._ws_config.to_kwargs(),
            )
        
        async with ws:
            self._liq_ws = ws
            self._consecutive_errors = 0
            self._liq_connection_time = time.time()
            self._backoff.reset()
            
            # Notify supervisor
            self._supervisor.record_connect(self._conn_name)
            
            logger.info("liquidations_ws_connected")
            
            # Start worker task and health checker
            worker_task = asyncio.create_task(self._liq_message_worker())
            health_task = asyncio.create_task(self._liq_health_loop())
            
            try:
                # WS loop ONLY queues messages - ultra lightweight
                async for message in ws:
                    if not self._running:
                        break
                    
                    self._liq_message_count += 1
                    self._last_liq_message_time = time.time()
                    self._supervisor.record_message(self._conn_name)
                    
                    try:
                        self._liq_msg_queue.put_nowait(message)
                    except asyncio.QueueFull:
                        try:
                            self._liq_msg_queue.get_nowait()
                            self._liq_msg_queue.put_nowait(message)
                        except asyncio.QueueEmpty:
                            pass
                    
                    # Check if connection is too old
                    if time.time() - self._liq_connection_time > self._max_ws_age_s:
                        logger.info("liquidations_connection_refresh", age_hours=self._max_ws_age_s/3600)
                        self._force_reconnect = True
                        break
                    
                    # Check if health loop requested reconnect
                    if self._force_reconnect:
                        logger.info("liquidations_force_reconnect_requested")
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
    
    async def _liq_message_worker(self) -> None:
        """Worker that processes liquidation messages from queue"""
        while self._running:
            try:
                message = await asyncio.wait_for(self._liq_msg_queue.get(), timeout=1.0)
                await self._process_liquidation_message(message)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("liquidation_process_error", error=str(e)[:100])
    
    async def _liq_health_loop(self) -> None:
        """Monitor liquidation connection health - signal reconnect if needed"""
        while self._running and not self._force_reconnect:
            try:
                await asyncio.sleep(60)  # Check every 60 seconds (liquidations are sparse)
                
                # For liquidations, silence is normal - only check if WS seems dead
                # We use ping/pong for that, but also check extreme silence
                silence = time.time() - self._last_liq_message_time if self._last_liq_message_time else 0
                
                # If we had messages before but now 10+ minutes of silence, might be dead
                if self._last_liq_message_time > 0 and silence > 600:
                    # Only force reconnect if we're past connection refresh time anyway
                    if time.time() - self._liq_connection_time > self._max_ws_age_s / 2:
                        logger.warning("liquidations_long_silence", silence_s=f"{silence:.0f}")
                        # Signal reconnect instead of forcing close (avoids race condition)
                        self._force_reconnect = True
                        break
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("liquidations_health_check_error", error=str(e)[:50])
    
    async def _process_liquidation_message(self, message: str) -> None:
        """Process liquidation WebSocket message"""
        try:
            data = orjson.loads(message)
            
            # Combined stream format
            if "stream" in data:
                stream = data["stream"]
                liq_data = data["data"]
                symbol = stream.split("@")[0].upper()
            else:
                liq_data = data
                symbol = liq_data.get("o", {}).get("s", "").upper()
            
            # Parse liquidation from forceOrder format
            order = liq_data.get("o", liq_data)
            
            liq = Liquidation(
                symbol=symbol,
                timestamp_ms=order["T"],
                side=Side.BUY if order["S"] == "BUY" else Side.SELL,
                price=float(order["p"]),
                quantity=float(order["q"]),
            )
            
            # Store with timestamp
            now_ms = int(time.time() * 1000)
            self._liquidations.append((now_ms, liq))
            
            # Add to rolling buckets
            self._add_to_buckets(liq)
            
            # Callback
            if self.on_liquidation:
                await self._safe_callback(self.on_liquidation, liq)
        
        except Exception as e:
            logger.error("liquidation_parse_error", error=str(e), message=message[:200])
    
    def _add_to_buckets(self, liq: Liquidation) -> None:
        """Add liquidation to rolling buckets"""
        symbol = liq.symbol
        if symbol not in self._liq_buckets:
            return
        
        notional = liq.notional
        
        for bucket in self._liq_buckets[symbol].values():
            if liq.side == Side.BUY:
                # Long liquidation (buyer was liquidated)
                bucket.long_usd += notional
                bucket.long_count += 1
            else:
                # Short liquidation (seller was liquidated)
                bucket.short_usd += notional
                bucket.short_count += 1
    
    async def _update_buckets_loop(self) -> None:
        """Periodically update rolling buckets (prune old data)"""
        while self._running:
            await asyncio.sleep(1)  # Update every second
            
            try:
                now_ms = int(time.time() * 1000)
                
                for symbol in self.symbols:
                    # Reset buckets
                    for window_name, bucket in self._liq_buckets[symbol].items():
                        bucket.long_usd = 0.0
                        bucket.short_usd = 0.0
                        bucket.long_count = 0
                        bucket.short_count = 0
                    
                    # Recalculate from recent liquidations
                    cutoff_30s = now_ms - 30_000
                    cutoff_2m = now_ms - 120_000
                    cutoff_5m = now_ms - 300_000
                    cutoff_1h = now_ms - 3600_000
                    
                    for ts, liq in self._liquidations:
                        if liq.symbol != symbol:
                            continue
                        
                        notional = liq.notional
                        
                        # 30s bucket
                        if liq.timestamp_ms >= cutoff_30s:
                            bucket = self._liq_buckets[symbol]["30s"]
                            if liq.side == Side.BUY:
                                bucket.long_usd += notional
                                bucket.long_count += 1
                            else:
                                bucket.short_usd += notional
                                bucket.short_count += 1
                        
                        # 2m bucket
                        if liq.timestamp_ms >= cutoff_2m:
                            bucket = self._liq_buckets[symbol]["2m"]
                            if liq.side == Side.BUY:
                                bucket.long_usd += notional
                                bucket.long_count += 1
                            else:
                                bucket.short_usd += notional
                                bucket.short_count += 1
                        
                        # 5m bucket
                        if liq.timestamp_ms >= cutoff_5m:
                            bucket = self._liq_buckets[symbol]["5m"]
                            if liq.side == Side.BUY:
                                bucket.long_usd += notional
                                bucket.long_count += 1
                            else:
                                bucket.short_usd += notional
                                bucket.short_count += 1
                        
                        # 1h bucket
                        if liq.timestamp_ms >= cutoff_1h:
                            bucket = self._liq_buckets[symbol]["1h"]
                            if liq.side == Side.BUY:
                                bucket.long_usd += notional
                                bucket.long_count += 1
                            else:
                                bucket.short_usd += notional
                                bucket.short_count += 1
                    
                    # Store 5-min snapshot for long-term rolling memory
                    last_snapshot = self._last_5m_snapshot_time[symbol]
                    if now_ms - last_snapshot >= 300_000:  # 5 minutes
                        self._liq_5m_snapshots[symbol].append({
                            "timestamp_ms": now_ms,
                            "long_usd": self._liq_buckets[symbol]["5m"].long_usd,
                            "short_usd": self._liq_buckets[symbol]["5m"].short_usd,
                            "imbalance": self._liq_buckets[symbol]["5m"].imbalance,
                        })
                        self._last_5m_snapshot_time[symbol] = now_ms
                    
                    # Emit stats callback
                    if self.on_liq_stats:
                        stats = LiquidationStats(
                            timestamp_ms=now_ms,
                            symbol=symbol,
                            bucket_30s=self._liq_buckets[symbol]["30s"],
                            bucket_2m=self._liq_buckets[symbol]["2m"],
                            bucket_5m=self._liq_buckets[symbol]["5m"],
                            bucket_1h=self._liq_buckets[symbol]["1h"],
                        )
                        await self._safe_callback(self.on_liq_stats, stats)
            
            except Exception as e:
                logger.error("bucket_update_error", error=str(e))
    
    async def _safe_callback(self, callback: Callable, *args) -> None:
        """Safely execute callback"""
        try:
            result = callback(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error("derivatives_callback_error", error=str(e))
    
    # ========== PUBLIC API FOR TESTING ==========
    
    def get_funding_rates(self, symbol: str) -> List[FundingRate]:
        """Get cached funding rates for symbol"""
        return list(self._funding_rates.get(symbol, []))
    
    def get_latest_funding(self, symbol: str) -> Optional[FundingRate]:
        """Get latest funding rate"""
        rates = self._funding_rates.get(symbol, [])
        return rates[-1] if rates else None
    
    def get_open_interest(self, symbol: str) -> List[OpenInterest]:
        """Get cached OI for symbol"""
        return list(self._open_interest.get(symbol, []))
    
    def get_latest_oi(self, symbol: str) -> Optional[OpenInterest]:
        """Get latest OI"""
        oi = self._open_interest.get(symbol, [])
        return oi[-1] if oi else None
    
    def get_recent_liquidations(self, count: int = 100) -> List[Liquidation]:
        """Get recent liquidations"""
        liqs = [liq for _, liq in self._liquidations]
        return liqs[-count:]
    
    def get_liq_stats(self, symbol: str) -> Optional[LiquidationStats]:
        """Get current liquidation stats for symbol"""
        if symbol not in self._liq_buckets:
            return None
        
        now_ms = int(time.time() * 1000)
        return LiquidationStats(
            timestamp_ms=now_ms,
            symbol=symbol,
            bucket_30s=self._liq_buckets[symbol]["30s"],
            bucket_2m=self._liq_buckets[symbol]["2m"],
            bucket_5m=self._liq_buckets[symbol]["5m"],
            bucket_1h=self._liq_buckets[symbol]["1h"],
        )
    
    def get_liq_5m_snapshots(self, symbol: str) -> List[Dict]:
        """Get 5-min interval snapshots for long-term rolling memory"""
        return list(self._liq_5m_snapshots.get(symbol, []))
    
    def get_liq_usd_1h(self, symbol: str) -> float:
        """Get total liquidation USD in last hour"""
        if symbol not in self._liq_buckets:
            return 0.0
        return self._liq_buckets[symbol]["1h"].total_usd
    
    def get_liq_buckets(self, symbol: str) -> Optional[Dict[str, LiquidationBucket]]:
        """Get raw liquidation buckets for symbol"""
        return self._liq_buckets.get(symbol)
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get collector health metrics"""
        # Check WS connection
        is_liq_ws_connected = False
        if self._liq_ws is not None:
            try:
                state = self._liq_ws.state
                is_liq_ws_connected = state == 1
            except AttributeError:
                is_liq_ws_connected = True
        
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "liq_ws_connected": is_liq_ws_connected,
            "liq_message_count": self._liq_message_count,
            "funding_rates_cached": {s: len(q) for s, q in self._funding_rates.items()},
            "oi_cached": {s: len(q) for s, q in self._open_interest.items()},
            "liquidations_cached": len(self._liquidations),
        }


# ========== STANDALONE TEST FUNCTIONS ==========

async def test_funding_rate(symbol: str = "BTCUSDT") -> Optional[FundingRate]:
    """
    Test fetching a single funding rate
    
    Usage:
        from src.collectors.derivatives import test_funding_rate
        import asyncio
        rate = asyncio.run(test_funding_rate("BTCUSDT"))
    """
    async with httpx.AsyncClient(base_url=settings.BINANCE_REST_BASE) as client:
        resp = await client.get("/fapi/v1/fundingRate", params={"symbol": symbol, "limit": 1})
        resp.raise_for_status()
        data = resp.json()
        
        if data:
            item = data[0]
            rate = FundingRate(
                symbol=symbol,
                timestamp_ms=item["fundingTime"],
                funding_rate=float(item["fundingRate"]),
                mark_price=float(item.get("markPrice", 0)),
            )
            print(f"Funding Rate for {symbol}:")
            print(f"  Rate: {rate.funding_rate:.6f} ({rate.annualized_rate:.2f}% APR)")
            print(f"  Time: {rate.timestamp_ms}")
            return rate
    return None


async def test_open_interest(symbol: str = "BTCUSDT") -> Optional[OpenInterest]:
    """
    Test fetching open interest
    
    Usage:
        from src.collectors.derivatives import test_open_interest
        import asyncio
        oi = asyncio.run(test_open_interest("BTCUSDT"))
    """
    async with httpx.AsyncClient(base_url=settings.BINANCE_REST_BASE) as client:
        resp = await client.get("/fapi/v1/openInterest", params={"symbol": symbol})
        resp.raise_for_status()
        data = resp.json()
        
        oi = OpenInterest(
            symbol=symbol,
            timestamp_ms=int(time.time() * 1000),
            open_interest=float(data["openInterest"]),
            open_interest_value=0.0,
        )
        print(f"Open Interest for {symbol}:")
        print(f"  OI: {oi.open_interest:,.2f} contracts")
        return oi


async def test_liquidations_ws(duration_seconds: int = None) -> Dict[str, Any]:
    """
    Test liquidations WebSocket stream with rolling buckets
    
    Usage:
        from src.collectors.derivatives import test_liquidations_ws
        import asyncio
        result = asyncio.run(test_liquidations_ws(30))
    """
    liquidations = []
    stats_updates = []
    
    def on_liq(liq: Liquidation):
        liquidations.append(liq)
        print(f"  Liq: {liq.symbol} {liq.side.value} {liq.quantity:.4f} @ ${liq.price:.2f} (${liq.notional:,.0f})")
    
    def on_stats(stats: LiquidationStats):
        stats_updates.append(stats)
        if len(stats_updates) % 10 == 0:  # Print every 10th update
            print(f"  Stats {stats.symbol}: 30s=${stats.bucket_30s.total_usd:,.0f} (imb={stats.bucket_30s.imbalance:.2%}), "
                  f"2m=${stats.bucket_2m.total_usd:,.0f}, 5m=${stats.bucket_5m.total_usd:,.0f}")
    
    collector = DerivativesCollector(
        symbols=["BTCUSDT", "ETHUSDT"],
        on_liquidation=on_liq,
        on_liq_stats=on_stats,
        funding_interval_s=999999,  # Disable funding polling for this test
        oi_interval_s=999999,  # Disable OI polling for this test
    )
    
    if duration_seconds:
        print(f"Starting liquidations WebSocket test for {duration_seconds}s...")
    else:
        print("Starting liquidations WebSocket test (run until Ctrl+C)...")
    
    task = asyncio.create_task(collector.start())
    
    try:
        if duration_seconds:
            await asyncio.sleep(duration_seconds)
        else:
            # Run indefinitely until cancelled
            while True:
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("\nTest cancelled by user")
    finally:
        await collector.stop()
        task.cancel()
    
    result = {
        "total_liquidations": len(liquidations),
        "stats_updates": len(stats_updates),
        "by_symbol": {},
        "final_buckets": {},
    }
    
    for liq in liquidations:
        sym = liq.symbol
        if sym not in result["by_symbol"]:
            result["by_symbol"][sym] = {"count": 0, "long_usd": 0.0, "short_usd": 0.0}
        result["by_symbol"][sym]["count"] += 1
        if liq.side == Side.BUY:
            result["by_symbol"][sym]["long_usd"] += liq.notional
        else:
            result["by_symbol"][sym]["short_usd"] += liq.notional
    
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        stats = collector.get_liq_stats(symbol)
        if stats:
            result["final_buckets"][symbol] = stats.to_dict()
    
    print(f"\n=== LIQUIDATIONS WEBSOCKET TEST RESULTS ===")
    print(f"Total liquidations: {result['total_liquidations']}")
    print(f"Stats updates: {result['stats_updates']}")
    for sym, data in result["by_symbol"].items():
        print(f"  {sym}: {data['count']} liqs, Long=${data['long_usd']:,.0f}, Short=${data['short_usd']:,.0f}")
    
    return result


async def test_derivatives_collector(duration_seconds: int = 30) -> Dict[str, Any]:
    """
    Test full derivatives collector with WebSocket liquidations
    
    Usage:
        from src.collectors.derivatives import test_derivatives_collector
        import asyncio
        result = asyncio.run(test_derivatives_collector(30))
    """
    funding_events = []
    oi_events = []
    liq_events = []
    stats_events = []
    
    def on_funding(rate: FundingRate):
        funding_events.append(rate)
        print(f"  Funding: {rate.symbol} {rate.funding_rate:.6f}")
    
    def on_oi(oi: OpenInterest):
        oi_events.append(oi)
        print(f"  OI: {oi.symbol} {oi.open_interest:,.0f}")
    
    def on_liq(liq: Liquidation):
        liq_events.append(liq)
        if len(liq_events) <= 10:
            print(f"  Liq: {liq.symbol} {liq.side.value} ${liq.notional:,.0f}")
    
    def on_stats(stats: LiquidationStats):
        stats_events.append(stats)
    
    collector = DerivativesCollector(
        symbols=["BTCUSDT", "ETHUSDT"],
        on_funding=on_funding,
        on_oi=on_oi,
        on_liquidation=on_liq,
        on_liq_stats=on_stats,
        funding_interval_s=10,
        oi_interval_s=10,
    )
    
    print(f"Starting derivatives collector test for {duration_seconds}s...")
    
    task = asyncio.create_task(collector.start())
    await asyncio.sleep(duration_seconds)
    await collector.stop()
    task.cancel()
    
    result = {
        "funding_events": len(funding_events),
        "oi_events": len(oi_events),
        "liquidation_events": len(liq_events),
        "stats_events": len(stats_events),
        "health": collector.get_health_metrics(),
        "final_buckets": {},
    }
    
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        stats = collector.get_liq_stats(symbol)
        if stats:
            result["final_buckets"][symbol] = stats.to_dict()
    
    print(f"\n=== DERIVATIVES COLLECTOR TEST RESULTS ===")
    print(f"Funding events: {result['funding_events']}")
    print(f"OI events: {result['oi_events']}")
    print(f"Liquidation events: {result['liquidation_events']}")
    print(f"Stats updates: {result['stats_events']}")
    for sym, buckets in result["final_buckets"].items():
        print(f"  {sym}: 30s=${buckets['liq_long_usd_30s']+buckets['liq_short_usd_30s']:,.0f}, "
              f"2m=${buckets['liq_long_usd_2m']+buckets['liq_short_usd_2m']:,.0f}, "
              f"5m=${buckets['liq_long_usd_5m']+buckets['liq_short_usd_5m']:,.0f}")
    
    return result
