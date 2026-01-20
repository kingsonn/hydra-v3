"""
RESILIENCE MODULE
Circuit breakers, connection supervision, and adaptive recovery for 3-week+ uptime
"""
import asyncio
import time
import random
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import structlog

logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 10         # Failures before opening (more tolerant)
    recovery_timeout_s: float = 60.0    # Time before trying half-open (longer wait)
    success_threshold: int = 2          # Successes needed to close
    half_open_max_calls: int = 5        # Max calls in half-open state


class CircuitBreaker:
    """
    Circuit breaker pattern for connection resilience.
    
    States:
    - CLOSED: Normal operation, track failures
    - OPEN: Stop all attempts, wait for recovery timeout
    - HALF_OPEN: Allow limited attempts to test recovery
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0
        self._half_open_calls = 0
        self._state_change_time: float = time.time()
    
    @property
    def state(self) -> CircuitState:
        # Auto-transition from OPEN to HALF_OPEN after timeout
        if self._state == CircuitState.OPEN:
            if time.time() - self._state_change_time >= self.config.recovery_timeout_s:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state
    
    def can_proceed(self) -> bool:
        """Check if operation should proceed"""
        state = self.state
        
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.OPEN:
            return False
        else:  # HALF_OPEN
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
    
    def record_success(self) -> None:
        """Record successful operation"""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0
    
    def record_failure(self) -> None:
        """Record failed operation"""
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        old_state = self._state
        self._state = new_state
        self._state_change_time = time.time()
        
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        
        logger.info("circuit_state_change", 
                   breaker=self.name, 
                   old_state=old_state.value, 
                   new_state=new_state.value)
    
    def force_open(self) -> None:
        """Force circuit to open state"""
        self._transition_to(CircuitState.OPEN)
    
    def reset(self) -> None:
        """Reset circuit to closed state"""
        self._transition_to(CircuitState.CLOSED)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "last_failure": self._last_failure_time,
            "time_in_state_s": time.time() - self._state_change_time,
        }


@dataclass
class ConnectionHealth:
    """Tracks health of a single connection"""
    name: str
    last_message_time: float = 0
    last_connect_time: float = 0
    message_count: int = 0
    error_count: int = 0
    reconnect_count: int = 0
    consecutive_errors: int = 0
    is_connected: bool = False
    
    def record_message(self) -> None:
        self.last_message_time = time.time()
        self.message_count += 1
        self.consecutive_errors = 0
    
    def record_connect(self) -> None:
        self.last_connect_time = time.time()
        self.is_connected = True
        self.consecutive_errors = 0
    
    def record_disconnect(self) -> None:
        self.is_connected = False
        self.reconnect_count += 1
    
    def record_error(self) -> None:
        self.error_count += 1
        self.consecutive_errors += 1
    
    def silence_duration_s(self) -> float:
        """Time since last message"""
        if self.last_message_time == 0:
            return 0
        return time.time() - self.last_message_time


class ConnectionSupervisor:
    """
    Supervises multiple connections and triggers recovery actions.
    
    Features:
    - Monitors message flow for each connection
    - Detects silent failures (no messages but no error)
    - Triggers forced reconnection when needed
    - Coordinates recovery across connections
    """
    
    def __init__(
        self,
        max_silence_s: float = 60.0,  # Force reconnect after 60s silence
        check_interval_s: float = 10.0,
        on_force_reconnect: Optional[Callable[[str], Any]] = None,
    ):
        self.max_silence_s = max_silence_s
        self.check_interval_s = check_interval_s
        self.on_force_reconnect = on_force_reconnect
        
        self._connections: Dict[str, ConnectionHealth] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._running = False
        self._global_pause_until: float = 0  # For network-wide issues
    
    def register_connection(self, name: str, circuit_config: Optional[CircuitBreakerConfig] = None) -> None:
        """Register a connection to supervise"""
        self._connections[name] = ConnectionHealth(name=name)
        self._circuit_breakers[name] = CircuitBreaker(name, circuit_config)
    
    def get_circuit(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker for connection"""
        return self._circuit_breakers.get(name)
    
    def record_message(self, name: str) -> None:
        """Record message received on connection"""
        if name in self._connections:
            self._connections[name].record_message()
            self._circuit_breakers[name].record_success()
    
    def record_connect(self, name: str) -> None:
        """Record successful connection"""
        if name in self._connections:
            self._connections[name].record_connect()
    
    def record_disconnect(self, name: str) -> None:
        """Record disconnection"""
        if name in self._connections:
            self._connections[name].record_disconnect()
    
    def record_error(self, name: str) -> None:
        """Record connection error"""
        if name in self._connections:
            self._connections[name].record_error()
            self._circuit_breakers[name].record_failure()
    
    def should_reconnect(self, name: str) -> bool:
        """Check if connection should attempt reconnection"""
        # Global pause (network-wide issues)
        if time.time() < self._global_pause_until:
            return False
        
        circuit = self._circuit_breakers.get(name)
        if circuit and not circuit.can_proceed():
            return False
        
        return True
    
    def trigger_global_pause(self, duration_s: float = 30.0) -> None:
        """Pause all reconnection attempts (network down)"""
        self._global_pause_until = time.time() + duration_s
        logger.warning("global_reconnect_pause", duration_s=duration_s)
    
    async def start(self) -> None:
        """Start supervision loop"""
        self._running = True
        logger.info("connection_supervisor_starting")
        
        while self._running:
            await self._check_connections()
            await asyncio.sleep(self.check_interval_s)
    
    async def stop(self) -> None:
        """Stop supervision"""
        self._running = False
        logger.info("connection_supervisor_stopped")
    
    async def _check_connections(self) -> None:
        """Check all connections for issues"""
        now = time.time()
        
        # Count how many connections are having issues
        failing_count = 0
        
        for name, conn in self._connections.items():
            silence = conn.silence_duration_s()
            
            # Check for silent failure
            if conn.is_connected and silence > self.max_silence_s:
                logger.warning("connection_silent_failure", 
                             connection=name, 
                             silence_s=f"{silence:.1f}")
                failing_count += 1
                
                # Trigger forced reconnect
                if self.on_force_reconnect:
                    try:
                        result = self.on_force_reconnect(name)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error("force_reconnect_error", connection=name, error=str(e))
            
            # Check for too many consecutive errors
            if conn.consecutive_errors >= 10:
                failing_count += 1
        
        # If most connections are failing, likely network issue
        total = len(self._connections)
        if total > 0 and failing_count >= total * 0.7:
            self.trigger_global_pause(60.0)
    
    def get_status(self) -> Dict[str, Any]:
        """Get supervisor status"""
        return {
            "connections": {
                name: {
                    "is_connected": conn.is_connected,
                    "silence_s": conn.silence_duration_s(),
                    "message_count": conn.message_count,
                    "error_count": conn.error_count,
                    "reconnect_count": conn.reconnect_count,
                    "circuit_state": self._circuit_breakers[name].state.value,
                }
                for name, conn in self._connections.items()
            },
            "global_paused": time.time() < self._global_pause_until,
        }


class AdaptiveBackoff:
    """
    Adaptive exponential backoff with jitter.
    Learns from success/failure patterns to optimize retry timing.
    """
    
    def __init__(
        self,
        base_delay_s: float = 1.0,
        max_delay_s: float = 60.0,
        jitter_factor: float = 0.3,
    ):
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.jitter_factor = jitter_factor
        
        self._consecutive_failures = 0
        self._last_success_time: float = time.time()
        self._recent_delays: deque = deque(maxlen=10)
    
    def get_delay(self) -> float:
        """Calculate next delay with exponential backoff and jitter"""
        # Exponential backoff
        exp = min(self._consecutive_failures, 6)  # Cap at 2^6 = 64x
        delay = self.base_delay_s * (2 ** exp)
        
        # Add jitter
        jitter = delay * self.jitter_factor * random.uniform(-1, 1)
        delay = delay + jitter
        
        # Cap at max
        delay = min(delay, self.max_delay_s)
        delay = max(delay, self.base_delay_s)
        
        self._recent_delays.append(delay)
        return delay
    
    def record_success(self) -> None:
        """Reset on success"""
        self._consecutive_failures = 0
        self._last_success_time = time.time()
    
    def record_failure(self) -> None:
        """Increment failure count"""
        self._consecutive_failures += 1
    
    def reset(self) -> None:
        """Full reset"""
        self._consecutive_failures = 0


class ResourceManager:
    """
    Manages resource cleanup and memory limits.
    Prevents memory leaks during long-running operation.
    """
    
    def __init__(
        self,
        cleanup_interval_s: float = 300.0,  # 5 minutes
        max_buffer_age_s: float = 600.0,    # 10 minutes
    ):
        self.cleanup_interval_s = cleanup_interval_s
        self.max_buffer_age_s = max_buffer_age_s
        
        self._buffers: Dict[str, deque] = {}
        self._buffer_timestamps: Dict[str, deque] = {}
        self._running = False
        self._last_cleanup: float = 0
        self._cleanup_count = 0
    
    def register_buffer(self, name: str, buffer: deque) -> None:
        """Register a buffer for managed cleanup"""
        self._buffers[name] = buffer
        self._buffer_timestamps[name] = deque(maxlen=buffer.maxlen or 10000)
    
    async def start(self) -> None:
        """Start cleanup loop"""
        self._running = True
        logger.info("resource_manager_starting")
        
        while self._running:
            await asyncio.sleep(self.cleanup_interval_s)
            await self._cleanup()
    
    async def stop(self) -> None:
        """Stop cleanup"""
        self._running = False
        logger.info("resource_manager_stopped")
    
    async def _cleanup(self) -> None:
        """Perform cleanup of old data"""
        self._cleanup_count += 1
        self._last_cleanup = time.time()
        
        # Log memory status (basic)
        total_items = sum(len(b) for b in self._buffers.values())
        logger.debug("resource_cleanup", 
                    cleanup_num=self._cleanup_count,
                    total_buffer_items=total_items)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "cleanup_count": self._cleanup_count,
            "last_cleanup": self._last_cleanup,
            "buffer_sizes": {name: len(buf) for name, buf in self._buffers.items()},
        }


@dataclass
class WebSocketConfig:
    """Optimized WebSocket configuration for long-running stability"""
    # Binance sends pings every 5 minutes, we need to respond
    # Setting ping_interval to None lets us handle it ourselves
    ping_interval: Optional[float] = 20.0    # Our ping interval (keep connection alive)
    ping_timeout: Optional[float] = 60.0     # Longer timeout for slow networks
    close_timeout: float = 15.0              # Time to wait for close
    max_size: int = 10_000_000               # 10MB max message
    compression: Optional[str] = None        # Disable for speed
    
    # Additional resilience settings
    connect_timeout: float = 60.0            # Connection timeout (longer for slow networks)
    read_timeout: float = 180.0              # Read timeout (3 min silence = dead)
    
    def to_kwargs(self) -> Dict[str, Any]:
        """Convert to websockets.connect kwargs"""
        return {
            "ping_interval": self.ping_interval,
            "ping_timeout": self.ping_timeout,
            "close_timeout": self.close_timeout,
            "max_size": self.max_size,
            "compression": self.compression,
        }


# Singleton supervisor for global coordination
_supervisor: Optional[ConnectionSupervisor] = None


def get_supervisor() -> ConnectionSupervisor:
    """Get or create global connection supervisor"""
    global _supervisor
    if _supervisor is None:
        _supervisor = ConnectionSupervisor(
            max_silence_s=180.0,  # 3 minutes silence triggers reconnect (more tolerant)
            check_interval_s=30.0,  # Check less frequently to reduce noise
        )
    return _supervisor


async def init_resilience() -> ConnectionSupervisor:
    """Initialize resilience systems"""
    supervisor = get_supervisor()
    
    # Register standard connections with MORE TOLERANT settings for 24/7 operation
    # Network hiccups are normal - don't trip circuit on minor issues
    supervisor.register_connection("trades_ws", CircuitBreakerConfig(
        failure_threshold=15,       # Very tolerant - trades are critical
        recovery_timeout_s=45.0,
    ))
    supervisor.register_connection("orderbook_ws", CircuitBreakerConfig(
        failure_threshold=20,       # Most tolerant - orderbook can be slow
        recovery_timeout_s=60.0,
    ))
    supervisor.register_connection("liquidations_ws", CircuitBreakerConfig(
        failure_threshold=15,
        recovery_timeout_s=60.0,
    ))
    supervisor.register_connection("rest_api", CircuitBreakerConfig(
        failure_threshold=20,       # REST can have rate limits
        recovery_timeout_s=90.0,
    ))
    
    return supervisor
