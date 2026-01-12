"""
Absorption Processor - Absorption ratio, refill rate, liquidity sweep detection
"""
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from src.core.models import OrderBookSnapshot, Trade
from src.stage2.models import AbsorptionFeatures


@dataclass
class DepthSnapshot:
    """Lightweight depth snapshot for tracking"""
    timestamp_ms: int
    bid_depth: float
    ask_depth: float
    best_bid: float
    best_ask: float
    bid_levels_qty: List[float]  # Quantities at each level
    ask_levels_qty: List[float]


class AbsorptionProcessor:
    """
    Computes absorption and liquidity features from order book
    """
    
    def __init__(self, symbol: str, history_size: int = 100):
        self.symbol = symbol
        
        # Rolling buffers
        self._depth_history: deque[DepthSnapshot] = deque(maxlen=history_size)
        self._absorption_history: deque[float] = deque(maxlen=history_size)
        self._price_history: deque[float] = deque(maxlen=history_size)
        self._volume_history: deque[float] = deque(maxlen=history_size)
        
        # For refill rate calculation
        self._last_depth: Optional[DepthSnapshot] = None
        self._refill_events: deque[Tuple[int, float]] = deque(maxlen=50)
        
        # For sweep detection
        self._level_clear_events: deque[int] = deque(maxlen=20)  # timestamps
        
        # Stats for z-scoring
        self._absorption_mean = 0.0
        self._absorption_std = 1.0
        
        # Cached features
        self._last_features = AbsorptionFeatures()
    
    def add_book_snapshot(
        self, 
        snapshot: OrderBookSnapshot, 
        traded_volume: float = 0.0,
        price_change: float = 0.0
    ) -> AbsorptionFeatures:
        """
        Add order book snapshot and compute features
        
        Args:
            snapshot: Order book snapshot
            traded_volume: Volume traded since last snapshot
            price_change: Price change since last snapshot
        """
        # Create lightweight snapshot
        depth = DepthSnapshot(
            timestamp_ms=snapshot.timestamp_ms,
            bid_depth=snapshot.bid_depth_usd(10),
            ask_depth=snapshot.ask_depth_usd(10),
            best_bid=snapshot.best_bid or 0.0,
            best_ask=snapshot.best_ask or 0.0,
            bid_levels_qty=[l.quantity for l in snapshot.bids[:10]],
            ask_levels_qty=[l.quantity for l in snapshot.asks[:10]],
        )
        
        self._depth_history.append(depth)
        
        # Compute absorption ratio
        absorption = self._compute_absorption(traded_volume, price_change)
        self._absorption_history.append(absorption)
        
        # Detect liquidity sweep
        sweep = self._detect_sweep(depth)
        
        # Compute refill rate
        refill = self._compute_refill_rate(depth)
        
        # Update stats for z-scoring
        self._update_stats()
        
        # Z-score absorption
        absorption_z = 0.0
        if self._absorption_std > 1e-9:
            absorption_z = (absorption - self._absorption_mean) / self._absorption_std
        
        self._last_features = AbsorptionFeatures(
            absorption_z=absorption_z,
            refill_rate=refill,
            liquidity_sweep=sweep,
            bid_depth_usd=depth.bid_depth,
            ask_depth_usd=depth.ask_depth,
            depth_imbalance=snapshot.depth_imbalance(10),
        )
        
        self._last_depth = depth
        return self._last_features
    
    def _compute_absorption(self, traded_volume: float, price_change: float) -> float:
        """
        Absorption = traded_volume / abs(price_change)
        High absorption = passive orders absorbing aggression
        """
        if abs(price_change) < 1e-9:
            return 0.0
        return traded_volume / abs(price_change)
    
    def _detect_sweep(self, current: DepthSnapshot) -> bool:
        """
        Detect liquidity sweep: multiple levels cleared rapidly
        """
        if self._last_depth is None:
            return False
        
        # Check if multiple bid or ask levels got significantly reduced
        bid_cleared = 0
        ask_cleared = 0
        
        for i in range(min(5, len(current.bid_levels_qty), len(self._last_depth.bid_levels_qty))):
            if current.bid_levels_qty[i] < self._last_depth.bid_levels_qty[i] * 0.3:
                bid_cleared += 1
        
        for i in range(min(5, len(current.ask_levels_qty), len(self._last_depth.ask_levels_qty))):
            if current.ask_levels_qty[i] < self._last_depth.ask_levels_qty[i] * 0.3:
                ask_cleared += 1
        
        # Sweep if 3+ levels cleared on either side
        if bid_cleared >= 3 or ask_cleared >= 3:
            self._level_clear_events.append(current.timestamp_ms)
            return True
        
        # Also check for rapid consecutive clears
        now = current.timestamp_ms
        recent_clears = sum(1 for t in self._level_clear_events if now - t < 5000)
        return recent_clears >= 3
    
    def _compute_refill_rate(self, current: DepthSnapshot) -> float:
        """
        Refill rate = new resting size after trade / time
        Detects iceberg orders and passive refills
        """
        if self._last_depth is None:
            return 0.0
        
        time_delta_s = (current.timestamp_ms - self._last_depth.timestamp_ms) / 1000.0
        if time_delta_s < 0.01:
            return 0.0
        
        # Calculate depth increase (refill)
        bid_increase = max(0, current.bid_depth - self._last_depth.bid_depth)
        ask_increase = max(0, current.ask_depth - self._last_depth.ask_depth)
        
        total_refill = bid_increase + ask_increase
        return total_refill / time_delta_s
    
    def _update_stats(self) -> None:
        """Update rolling mean/std for z-scoring"""
        if len(self._absorption_history) >= 20:
            arr = np.array(self._absorption_history, dtype=np.float64)
            self._absorption_mean = float(np.mean(arr))
            self._absorption_std = float(np.std(arr))
            if self._absorption_std < 1e-9:
                self._absorption_std = 1.0
    
    def get_features(self) -> AbsorptionFeatures:
        """Get current features"""
        return self._last_features


class MultiSymbolAbsorptionProcessor:
    """Manages absorption processors for multiple symbols"""
    
    def __init__(self, symbols: List[str]):
        self.processors: Dict[str, AbsorptionProcessor] = {
            s: AbsorptionProcessor(s) for s in symbols
        }
    
    def add_book_snapshot(
        self, 
        snapshot: OrderBookSnapshot,
        traded_volume: float = 0.0,
        price_change: float = 0.0
    ) -> AbsorptionFeatures:
        if snapshot.symbol in self.processors:
            return self.processors[snapshot.symbol].add_book_snapshot(
                snapshot, traded_volume, price_change
            )
        return AbsorptionFeatures()
    
    def get_features(self, symbol: str) -> AbsorptionFeatures:
        if symbol in self.processors:
            return self.processors[symbol].get_features()
        return AbsorptionFeatures()
