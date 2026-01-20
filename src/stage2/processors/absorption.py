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
        self._sweep_side: Optional[str] = None  # "ASK" or "BID"
        self._post_sweep_depth: Dict[str, float] = {"bid": 0.0, "ask": 0.0}
        self._sweep_timestamp_ms: Optional[int] = None
        self._sweep_expiry_ms: int = 10000  # Refill tracking expires after 10s
        
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
        Absorption = high volume with low price impact.
        Returns higher values when price is stable despite volume.
        Formula: volume / (price_impact_bps + floor)
        """
        if traded_volume < 1e-6:
            return 0.0
        # Convert price change to basis points for normalization
        price_impact_bps = abs(price_change) * 10000
        # Higher absorption = more volume absorbed per unit of price impact
        # Floor of 0.1 bps prevents division issues when price doesn't move
        return traded_volume / (price_impact_bps + 0.1)
    
    def _detect_sweep(self, current: DepthSnapshot) -> bool:
        """
        Detect liquidity sweep: multiple levels cleared rapidly.
        Hardened thresholds: require 90% cleared on 4+ levels to avoid noise.
        """
        if self._last_depth is None:
            return False
        
        # Check if multiple bid or ask levels got significantly reduced
        # Require 90% reduction (0.1 threshold) to count as "cleared"
        bid_cleared = 0
        ask_cleared = 0
        
        for i in range(min(5, len(current.bid_levels_qty), len(self._last_depth.bid_levels_qty))):
            if self._last_depth.bid_levels_qty[i] > 1e-9:  # Avoid div by zero
                if current.bid_levels_qty[i] < self._last_depth.bid_levels_qty[i] * 0.1:
                    bid_cleared += 1
        
        for i in range(min(5, len(current.ask_levels_qty), len(self._last_depth.ask_levels_qty))):
            if self._last_depth.ask_levels_qty[i] > 1e-9:  # Avoid div by zero
                if current.ask_levels_qty[i] < self._last_depth.ask_levels_qty[i] * 0.1:
                    ask_cleared += 1
        
        # Sweep if 4+ levels cleared on either side (hardened from 3)
        if bid_cleared >= 4 or ask_cleared >= 4:
            self._level_clear_events.append(current.timestamp_ms)
            # Store POST-sweep depth (current depleted depth) and side
            if ask_cleared >= 4:
                self._sweep_side = "ASK"
            else:
                self._sweep_side = "BID"
            # CRITICAL FIX: Store POST-sweep depth, not pre-sweep
            self._post_sweep_depth = {
                "bid": current.bid_depth,
                "ask": current.ask_depth,
            }
            self._sweep_timestamp_ms = current.timestamp_ms
            return True
        
        # Consecutive clears within 5s also counts
        now = current.timestamp_ms
        recent_clears = sum(1 for t in self._level_clear_events if now - t < 5000)
        return recent_clears >= 4
    
    def _compute_refill_rate(self, current: DepthSnapshot) -> float:
        """
        Refill rate = depth recovery after sweep, compared to POST-sweep depleted depth.
        Side-aware: tracks refill on the swept side only.
        Only valid 0.5s - 10s after sweep.
        Returns: refill amount per second, normalized by post-sweep depth.
        """
        if self._sweep_timestamp_ms is None:
            return 0.0
        
        time_delta_s = (current.timestamp_ms - self._sweep_timestamp_ms) / 1000.0
        
        # Only valid in 0.5s - 10s window after sweep
        if time_delta_s < 0.5 or time_delta_s > 10.0:
            # Expire the sweep tracking after 10s
            if time_delta_s > 10.0:
                self._sweep_timestamp_ms = None
            return 0.0
        
        # CRITICAL FIX: Compare current depth to POST-sweep depth (depleted state)
        # Refill = how much depth has recovered since the sweep
        if self._sweep_side == "ASK":
            post_sweep = self._post_sweep_depth.get("ask", 0.0)
            refill = current.ask_depth - post_sweep
        else:
            post_sweep = self._post_sweep_depth.get("bid", 0.0)
            refill = current.bid_depth - post_sweep
        
        if refill <= 0:
            return 0.0
        
        # Refill rate = depth recovered per second
        # Normalize by post-sweep depth to make it comparable across symbols
        baseline = max(post_sweep, 1000.0)  # Min baseline of $1000
        refill_rate = (refill / baseline) / time_delta_s
        return min(refill_rate, 10.0)  # Cap at 10 to avoid outliers
    
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
