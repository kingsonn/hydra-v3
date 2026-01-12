"""
F️⃣ METADATA & HEALTH MONITORING
Tracks WebSocket lag, dropped messages, data integrity
"""
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import structlog

from config import settings

logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component"""
    name: str
    status: HealthStatus
    last_update_ms: int
    lag_ms: int = 0
    message_count: int = 0
    error_count: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


@dataclass
class SymbolHealth:
    """Health status per trading symbol"""
    symbol: str
    trade_lag_ms: int = 0
    book_lag_ms: int = 0
    last_trade_ms: int = 0
    last_book_ms: int = 0
    trade_rate_per_sec: float = 0.0
    book_update_rate_per_sec: float = 0.0
    
    def get_status(self, max_lag_ms: int = 5000) -> HealthStatus:
        if self.trade_lag_ms > max_lag_ms or self.book_lag_ms > max_lag_ms:
            return HealthStatus.UNHEALTHY
        if self.trade_lag_ms > max_lag_ms / 2 or self.book_lag_ms > max_lag_ms / 2:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


@dataclass
class SystemHealth:
    """Overall system health snapshot"""
    timestamp_ms: int
    status: HealthStatus
    components: Dict[str, ComponentHealth]
    symbols: Dict[str, SymbolHealth]
    alerts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "status": self.status.value,
            "components": {
                name: {
                    "status": c.status.value,
                    "lag_ms": c.lag_ms,
                    "message_count": c.message_count,
                    "error_count": c.error_count,
                }
                for name, c in self.components.items()
            },
            "symbols": {
                sym: {
                    "trade_lag_ms": s.trade_lag_ms,
                    "book_lag_ms": s.book_lag_ms,
                    "status": s.get_status().value,
                }
                for sym, s in self.symbols.items()
            },
            "alerts": self.alerts,
        }


class HealthMonitor:
    """
    Monitors health of all Stage 1 components
    
    Tracks:
    - WebSocket connection status
    - Message lag per symbol
    - Trade/book update rates
    - Error counts
    - Clock sync issues
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        max_lag_ms: int = 5000,
        check_interval_s: int = 5,
        on_alert: Optional[Callable[[str], Any]] = None,
    ):
        self.symbols = symbols or settings.SYMBOLS
        self.max_lag_ms = max_lag_ms
        self.check_interval_s = check_interval_s
        self.on_alert = on_alert
        
        # Component health trackers
        self._components: Dict[str, ComponentHealth] = {}
        self._symbols: Dict[str, SymbolHealth] = {
            s: SymbolHealth(symbol=s) for s in self.symbols
        }
        
        # Rate tracking
        self._trade_times: Dict[str, deque] = {
            s: deque(maxlen=100) for s in self.symbols
        }
        self._book_times: Dict[str, deque] = {
            s: deque(maxlen=100) for s in self.symbols
        }
        
        # Alert history
        self._alerts: deque = deque(maxlen=100)
        self._alert_cooldown: Dict[str, float] = {}
        self._alert_cooldown_seconds = 300  # 5 minutes between same alert type
        
        # State
        self._running = False
        self._last_check = 0
        
        # Per-pair stale thresholds (ms) - higher for low volume pairs
        self._stale_thresholds: Dict[str, int] = {
            "BTCUSDT": 10_000,   # 10s - high volume
            "ETHUSDT": 10_000,   # 10s - high volume
            "SOLUSDT": 15_000,   # 15s - medium volume
            "BNBUSDT": 20_000,   # 20s - medium volume
            "DOGEUSDT": 15_000,  # 15s - medium volume
            "ADAUSDT": 20_000,   # 20s - lower volume
            "XRPUSDT": 20_000,   # 20s - lower volume
            "LTCUSDT": 30_000,   # 30s - lower volume
        }
        self._default_stale_threshold = 15_000  # 15s default
    
    def _get_stale_threshold(self, symbol: str) -> int:
        """Get stale threshold for symbol based on expected volume"""
        return self._stale_thresholds.get(symbol, self._default_stale_threshold)
    
    async def start(self) -> None:
        """Start health monitoring loop"""
        self._running = True
        logger.info("health_monitor_starting")
        
        while self._running:
            await self._check_health()
            await asyncio.sleep(self.check_interval_s)
    
    async def stop(self) -> None:
        """Stop health monitoring"""
        self._running = False
        logger.info("health_monitor_stopped")
    
    # ========== UPDATE METHODS (called by collectors) ==========
    
    def update_component(
        self,
        name: str,
        status: HealthStatus,
        lag_ms: int = 0,
        message_count: int = 0,
        error_count: int = 0,
        details: Optional[Dict] = None,
    ) -> None:
        """Update component health status"""
        now_ms = int(time.time() * 1000)
        self._components[name] = ComponentHealth(
            name=name,
            status=status,
            last_update_ms=now_ms,
            lag_ms=lag_ms,
            message_count=message_count,
            error_count=error_count,
            details=details or {},
        )
    
    def record_trade(self, symbol: str, timestamp_ms: int) -> None:
        """Record trade receipt for latency tracking"""
        now_ms = int(time.time() * 1000)
        
        if symbol in self._symbols:
            self._symbols[symbol].last_trade_ms = timestamp_ms
            self._symbols[symbol].trade_lag_ms = now_ms - timestamp_ms
            self._trade_times[symbol].append(now_ms)
    
    def record_book_update(self, symbol: str, timestamp_ms: int) -> None:
        """Record order book update for latency tracking"""
        now_ms = int(time.time() * 1000)
        
        if symbol in self._symbols:
            self._symbols[symbol].last_book_ms = timestamp_ms
            self._symbols[symbol].book_lag_ms = now_ms - timestamp_ms
            self._book_times[symbol].append(now_ms)
    
    # ========== HEALTH CHECK ==========
    
    async def _check_health(self) -> SystemHealth:
        """Perform health check and emit alerts"""
        now_ms = int(time.time() * 1000)
        now = time.time()
        alerts = []
        
        # Update rates
        for symbol in self.symbols:
            # Trade rate
            trade_times = list(self._trade_times[symbol])
            if len(trade_times) >= 2:
                duration = (trade_times[-1] - trade_times[0]) / 1000
                if duration > 0:
                    self._symbols[symbol].trade_rate_per_sec = len(trade_times) / duration
            
            # Book rate
            book_times = list(self._book_times[symbol])
            if len(book_times) >= 2:
                duration = (book_times[-1] - book_times[0]) / 1000
                if duration > 0:
                    self._symbols[symbol].book_update_rate_per_sec = len(book_times) / duration
        
        # Check for issues
        for symbol, health in self._symbols.items():
            # High trade lag
            if health.trade_lag_ms > self.max_lag_ms:
                alert_key = f"HIGH_TRADE_LAG:{symbol}"
                alert = f"HIGH_TRADE_LAG: {symbol} lag={health.trade_lag_ms}ms"
                if await self._emit_alert(alert, alert_key):
                    alerts.append(alert)
            
            # High book lag
            if health.book_lag_ms > self.max_lag_ms:
                alert_key = f"HIGH_BOOK_LAG:{symbol}"
                alert = f"HIGH_BOOK_LAG: {symbol} lag={health.book_lag_ms}ms"
                if await self._emit_alert(alert, alert_key):
                    alerts.append(alert)
            
            # No recent trades (stale data) - per-pair thresholds
            if health.last_trade_ms > 0:
                trade_staleness = now_ms - health.last_trade_ms
                stale_threshold = self._get_stale_threshold(symbol)
                if trade_staleness > stale_threshold:
                    alert_key = f"STALE_TRADES:{symbol}"
                    alert = f"STALE_TRADES: {symbol} no trades for {trade_staleness}ms"
                    if await self._emit_alert(alert, alert_key):
                        alerts.append(alert)
        
        # Check components
        for name, comp in self._components.items():
            if comp.status == HealthStatus.UNHEALTHY:
                alert_key = f"UNHEALTHY_COMPONENT:{name}"
                alert = f"UNHEALTHY_COMPONENT: {name}"
                if await self._emit_alert(alert, alert_key):
                    alerts.append(alert)
            
            # Stale component
            staleness = now_ms - comp.last_update_ms
            if staleness > 30000:  # 30 seconds
                alert_key = f"STALE_COMPONENT:{name}"
                alert = f"STALE_COMPONENT: {name} no update for {staleness}ms"
                if await self._emit_alert(alert, alert_key):
                    alerts.append(alert)
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        for symbol_health in self._symbols.values():
            sym_status = symbol_health.get_status(self.max_lag_ms)
            if sym_status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
                break
            elif sym_status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        self._last_check = now
        
        return SystemHealth(
            timestamp_ms=now_ms,
            status=overall_status,
            components=self._components.copy(),
            symbols=self._symbols.copy(),
            alerts=alerts,
        )
    
    async def _emit_alert(self, alert: str, alert_key: Optional[str] = None) -> bool:
        """Emit alert with cooldown. Returns True if alert was emitted."""
        now = time.time()
        key = alert_key or alert
        
        # Check cooldown (don't spam same alert type)
        last_alert_time = self._alert_cooldown.get(key, 0)
        if now - last_alert_time < self._alert_cooldown_seconds:
            return False
        
        self._alert_cooldown[key] = now
        self._alerts.append((now, alert))
        
        logger.warning("health_alert", alert=alert)
        
        if self.on_alert:
            try:
                result = self.on_alert(alert)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("alert_callback_error", error=str(e))
        
        return True
    
    # ========== PUBLIC API ==========
    
    def get_health(self) -> SystemHealth:
        """Get current health snapshot (sync)"""
        now_ms = int(time.time() * 1000)
        
        # Determine status
        overall_status = HealthStatus.HEALTHY
        for symbol_health in self._symbols.values():
            sym_status = symbol_health.get_status(self.max_lag_ms)
            if sym_status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
                break
            elif sym_status == HealthStatus.DEGRADED:
                overall_status = HealthStatus.DEGRADED
        
        return SystemHealth(
            timestamp_ms=now_ms,
            status=overall_status,
            components=self._components.copy(),
            symbols=self._symbols.copy(),
            alerts=list(a[1] for a in self._alerts)[-10:],
        )
    
    def get_symbol_health(self, symbol: str) -> Optional[SymbolHealth]:
        """Get health for specific symbol"""
        return self._symbols.get(symbol)
    
    def is_healthy(self) -> bool:
        """Quick check if system is healthy"""
        health = self.get_health()
        return health.status == HealthStatus.HEALTHY
    
    def is_tradeable(self, symbol: str) -> bool:
        """Check if a symbol has fresh data (safe to trade)"""
        sym_health = self._symbols.get(symbol)
        if not sym_health:
            return False
        
        status = sym_health.get_status(self.max_lag_ms)
        return status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
    
    def get_recent_alerts(self, count: int = 10) -> List[str]:
        """Get recent alerts"""
        alerts = list(self._alerts)
        return [a[1] for a in alerts[-count:]]


# ========== STANDALONE TEST FUNCTION ==========

async def test_health_monitor(duration_seconds: int = 10) -> Dict[str, Any]:
    """
    Test health monitor standalone
    
    Usage:
        from src.health.monitor import test_health_monitor
        import asyncio
        result = asyncio.run(test_health_monitor(10))
    """
    alerts_received = []
    
    def on_alert(alert: str):
        alerts_received.append(alert)
        print(f"  ⚠️ ALERT: {alert}")
    
    monitor = HealthMonitor(
        symbols=["BTCUSDT", "ETHUSDT"],
        max_lag_ms=5000,
        check_interval_s=2,
        on_alert=on_alert,
    )
    
    print(f"Starting health monitor test for {duration_seconds}s...")
    
    # Start monitor
    task = asyncio.create_task(monitor.start())
    
    # Simulate some data
    now_ms = int(time.time() * 1000)
    
    # Simulate healthy trades
    for i in range(5):
        await asyncio.sleep(1)
        current_ms = int(time.time() * 1000)
        monitor.record_trade("BTCUSDT", current_ms - 100)  # 100ms lag
        monitor.record_trade("ETHUSDT", current_ms - 150)  # 150ms lag
        monitor.record_book_update("BTCUSDT", current_ms - 50)
        monitor.record_book_update("ETHUSDT", current_ms - 50)
        
        health = monitor.get_health()
        print(f"  Health: {health.status.value}")
    
    # Simulate high lag
    print("  Simulating high lag...")
    monitor.record_trade("BTCUSDT", now_ms - 10000)  # 10s lag
    
    await asyncio.sleep(3)
    
    await monitor.stop()
    task.cancel()
    
    final_health = monitor.get_health()
    
    result = {
        "final_status": final_health.status.value,
        "alerts_received": len(alerts_received),
        "health_snapshot": final_health.to_dict(),
    }
    
    print(f"\n=== HEALTH MONITOR TEST RESULTS ===")
    print(f"Final status: {result['final_status']}")
    print(f"Alerts: {result['alerts_received']}")
    
    return result
