"""
Liquidation Processor - Rolling windows, cascade/exhaustion detection
THE EDGE: Detecting liquidation states, not raw streams
"""
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import time

from src.core.models import Liquidation, Side
from src.stage2.models import LiquidationFeatures


class LiquidationProcessor:
    """
    Processes liquidation events into actionable features
    Maintains rolling windows: 30s, 2m, 5m
    Detects cascade and exhaustion states
    """
    
    def __init__(
        self, 
        symbol: str,
        cascade_multiplier: float = 2.0,      # k1: cascade threshold vs avg (lowered from 3.0)
        cascade_imbalance_thresh: float = 0.6, # Imbalance threshold for cascade
        exhaustion_threshold: float = 100.0,   # USD threshold for "no activity"
        history_size: int = 600,               # 10 min of events max
    ):
        self.symbol = symbol
        self.cascade_multiplier = cascade_multiplier
        self.cascade_imbalance_thresh = cascade_imbalance_thresh
        self.exhaustion_threshold = exhaustion_threshold
        
        # Liquidation events with timestamps: (timestamp_ms, side, notional_usd)
        self._events: deque[Tuple[int, Side, float]] = deque(maxlen=history_size)
        
        # Rolling avg for cascade detection
        self._avg_30s_history: deque[float] = deque(maxlen=120)  # 2 min of 30s avgs
        
        # Cached features
        self._last_features = LiquidationFeatures()
        
        # State tracking
        self._recent_high_activity = False
        self._high_activity_timestamp = 0
    
    def add_liquidation(self, liq: Liquidation) -> LiquidationFeatures:
        """Add liquidation event and update features"""
        self._events.append((liq.timestamp_ms, liq.side, liq.notional))
        return self._compute_features()
    
    def update(self) -> LiquidationFeatures:
        """Update features without new event (called periodically)"""
        return self._compute_features()
    
    def _compute_features(self) -> LiquidationFeatures:
        """Compute all liquidation features"""
        now_ms = int(time.time() * 1000)
        
        # Compute window aggregates
        stats_30s = self._compute_window_stats(now_ms, 30_000)
        stats_2m = self._compute_window_stats(now_ms, 120_000)
        stats_5m = self._compute_window_stats(now_ms, 300_000)
        
        # Track 30s average for cascade detection
        total_30s = stats_30s["long_usd"] + stats_30s["short_usd"]
        self._avg_30s_history.append(total_30s)
        
        # Compute cascade and exhaustion states
        cascade_active = self._detect_cascade(stats_30s)
        exhaustion = self._detect_exhaustion(stats_30s, stats_2m)
        
        self._last_features = LiquidationFeatures(
            long_usd_30s=stats_30s["long_usd"],
            short_usd_30s=stats_30s["short_usd"],
            imbalance_30s=stats_30s["imbalance"],
            long_usd_2m=stats_2m["long_usd"],
            short_usd_2m=stats_2m["short_usd"],
            imbalance_2m=stats_2m["imbalance"],
            long_usd_5m=stats_5m["long_usd"],
            short_usd_5m=stats_5m["short_usd"],
            imbalance_5m=stats_5m["imbalance"],
            cascade_active=cascade_active,
            exhaustion=exhaustion,
        )
        
        return self._last_features
    
    def _compute_window_stats(self, now_ms: int, window_ms: int) -> Dict[str, float]:
        """Compute liquidation stats for a time window"""
        cutoff = now_ms - window_ms
        
        long_usd = 0.0
        short_usd = 0.0
        
        for ts, side, notional in self._events:
            if ts >= cutoff:
                if side == Side.BUY:
                    long_usd += notional
                else:
                    short_usd += notional
        
        total = long_usd + short_usd
        
        # Compute imbalance [-1, +1]
        # +1 = only longs liquidated
        # -1 = only shorts liquidated
        imbalance = 0.0
        if total > 0:
            imbalance = (long_usd - short_usd) / total
        
        return {
            "long_usd": long_usd,
            "short_usd": short_usd,
            "total_usd": total,
            "imbalance": imbalance,
        }
    
    def _detect_cascade(self, stats_30s: Dict[str, float]) -> bool:
        """
        Detect active liquidation cascade
        Cascade = high activity + strong imbalance
        """
        total_30s = stats_30s["total_usd"]
        imbalance = abs(stats_30s["imbalance"])
        
        # Need history to compute average
        if len(self._avg_30s_history) < 10:
            return False
        
        avg_30s = float(np.mean(self._avg_30s_history))
        
        # Cascade conditions:
        # 1. Current activity significantly above average
        # 2. Strong directional imbalance
        is_cascade = (
            total_30s > self.cascade_multiplier * avg_30s
            and imbalance > self.cascade_imbalance_thresh
            and total_30s > 1000  # Minimum USD threshold
        )
        
        # Track high activity for exhaustion detection
        if is_cascade:
            self._recent_high_activity = True
            self._high_activity_timestamp = int(time.time() * 1000)
        
        return is_cascade
    
    def _detect_exhaustion(
        self, stats_30s: Dict[str, float], stats_2m: Dict[str, float]
    ) -> bool:
        """
        Detect liquidation exhaustion (reversal signal)
        Exhaustion = recent high activity now gone + strong prior imbalance
        """
        total_30s = stats_30s["total_usd"]
        total_2m = stats_2m["total_usd"]
        imbalance_2m = abs(stats_2m["imbalance"])
        
        now_ms = int(time.time() * 1000)
        
        # Exhaustion conditions:
        # 1. Current 30s activity is very low
        # 2. Recent 2m had significant activity
        # 3. Strong imbalance in 2m window
        # 4. High activity was recent (within last 3 min)
        time_since_high = now_ms - self._high_activity_timestamp
        
        is_exhaustion = (
            total_30s < self.exhaustion_threshold
            and total_2m > 5000  # Had meaningful activity
            and imbalance_2m > self.cascade_imbalance_thresh
            and self._recent_high_activity
            and time_since_high < 180_000  # Within 3 min
            and time_since_high > 10_000   # At least 10s passed
        )
        
        # Reset high activity flag if exhaustion detected
        if is_exhaustion:
            self._recent_high_activity = False
        
        return is_exhaustion
    
    def get_features(self) -> LiquidationFeatures:
        """Get current features"""
        return self._last_features
    
    def clear_old_events(self, max_age_ms: int = 600_000) -> None:
        """Clear events older than max_age"""
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - max_age_ms
        
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()


class MultiSymbolLiquidationProcessor:
    """Manages liquidation processors for multiple symbols"""
    
    def __init__(self, symbols: List[str]):
        self.processors: Dict[str, LiquidationProcessor] = {
            s: LiquidationProcessor(s) for s in symbols
        }
    
    def add_liquidation(self, liq: Liquidation) -> LiquidationFeatures:
        if liq.symbol in self.processors:
            return self.processors[liq.symbol].add_liquidation(liq)
        return LiquidationFeatures()
    
    def update(self, symbol: str) -> LiquidationFeatures:
        """Periodic update for a symbol"""
        if symbol in self.processors:
            return self.processors[symbol].update()
        return LiquidationFeatures()
    
    def update_all(self) -> Dict[str, LiquidationFeatures]:
        """Update all symbols"""
        return {s: p.update() for s, p in self.processors.items()}
    
    def get_features(self, symbol: str) -> LiquidationFeatures:
        if symbol in self.processors:
            return self.processors[symbol].get_features()
        return LiquidationFeatures()
