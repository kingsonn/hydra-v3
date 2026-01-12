"""
Structure Processor - Volume profile from rolling trade buffers
Computes POC, VAH, VAL, LVN, distance metrics

Key logic:
- Maintain rolling 5-min and 30-min trade buffers
- BIN_SIZE = 10 (required for ML model compatibility)
- LVN computed from 5-min buffer only
- POC/VAH/VAL from 5-min buffer until 30 mins pass
- After 30 mins, POC/VAH/VAL permanently switch to 30-min values
- Recompute every 1 second (not on every trade)
"""
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import time

from src.core.models import Trade
from src.stage2.models import StructureFeatures


# Rolling windows in milliseconds
WINDOW_5M_MS = 300_000    # 5 minutes
WINDOW_30M_MS = 1_800_000  # 30 minutes


def get_bin_size(price: float) -> float:
    """
    Dynamic bin size based on price level
    Ensures ~0.01% of price per bin for all pairs
    
    Examples:
    - BTC at $95000 → bin = $10
    - ETH at $3500 → bin = $0.35 → round to $0.5
    - SOL at $200 → bin = $0.02 → round to $0.02
    - DOGE at $0.30 → bin = $0.00003 → round to $0.0001
    """
    if price <= 0:
        return 10.0
    
    # Target ~0.01% of price, but with sensible minimums
    raw_bin = price * 0.0001
    
    # Round to nice numbers
    if raw_bin >= 10:
        return round(raw_bin / 10) * 10  # $10, $20, etc
    elif raw_bin >= 1:
        return round(raw_bin)  # $1, $2, etc
    elif raw_bin >= 0.1:
        return round(raw_bin, 1)  # $0.1, $0.2, etc
    elif raw_bin >= 0.01:
        return round(raw_bin, 2)  # $0.01, $0.02, etc
    elif raw_bin >= 0.001:
        return round(raw_bin, 3)  # $0.001, etc
    elif raw_bin >= 0.0001:
        return round(raw_bin, 4)
    else:
        return round(raw_bin, 5)


class StructureProcessor:
    """
    Computes structural features from rolling trade data
    
    Flow:
    1. Add trades to rolling buffers (5min, 30min)
    2. Every 1 sec, build volume profile from buffers
    3. First 30 mins: use 5-min values for all
    4. After 30 mins: POC/VAH/VAL permanently from 30-min buffer
    5. LVN always from 5-min buffer
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Rolling trade buffers: (price, qty, timestamp_ms)
        self._trades_5m: deque[Tuple[float, float, int]] = deque()
        self._trades_30m: deque[Tuple[float, float, int]] = deque()
        
        # Cached volume profiles for fast access
        self._profile_5m: Dict[float, float] = {}
        self._profile_30m: Dict[float, float] = {}
        
        # Computed values
        self._poc_5m = 0.0
        self._vah_5m = 0.0
        self._val_5m = 0.0
        self._lvn_5m = 0.0
        
        self._poc_30m = 0.0
        self._vah_30m = 0.0
        self._val_30m = 0.0
        
        # Global values (switch to 30m after 30 mins)
        self._poc = 0.0
        self._vah = 0.0
        self._val = 0.0
        self._lvn = 0.0
        
        # Time tracking
        self._start_time_ms: Optional[int] = None
        self._has_30m_data = False
        self._last_compute_ms = 0
        
        # Current price for distance calculations
        self._current_price = 0.0
        
        # Acceptance tracking
        self._outside_value_start: Optional[int] = None
        self._acceptance_threshold_ms = 180_000  # 3 min
        
        # Price history for time-inside-value (1 sec samples)
        self._price_history: deque[Tuple[int, float]] = deque(maxlen=600)
        
        # Value area widths for median (24h rolling)
        self._value_area_widths: deque[float] = deque(maxlen=2880)  # 24h at 30s
        
        # ATR for distance normalization
        self._atr_5m = 0.0
        
        # Cached features
        self._last_features = StructureFeatures()
    
    def set_atr(self, atr_5m: float) -> None:
        """Set ATR for distance normalization"""
        self._atr_5m = atr_5m
    
    def add_trade(self, trade: Trade) -> None:
        """
        Add trade to rolling buffers
        Called on every aggTrade - fast path, no recomputation
        """
        ts = trade.timestamp_ms
        price = trade.price
        qty = trade.quantity
        
        # Initialize start time
        if self._start_time_ms is None:
            self._start_time_ms = ts
        
        # Update current price
        self._current_price = price
        
        # Add to both buffers
        self._trades_5m.append((price, qty, ts))
        self._trades_30m.append((price, qty, ts))
        
        # Trim old trades from 5m buffer
        cutoff_5m = ts - WINDOW_5M_MS
        while self._trades_5m and self._trades_5m[0][2] < cutoff_5m:
            self._trades_5m.popleft()
        
        # Trim old trades from 30m buffer
        cutoff_30m = ts - WINDOW_30M_MS
        while self._trades_30m and self._trades_30m[0][2] < cutoff_30m:
            self._trades_30m.popleft()
    
    def compute(self) -> StructureFeatures:
        """
        Recompute volume profile and all features
        Call this every 1 second, not on every trade
        """
        now_ms = int(time.time() * 1000)
        self._last_compute_ms = now_ms
        
        # Check if we have 30m of data
        if self._start_time_ms is not None:
            elapsed = now_ms - self._start_time_ms
            if elapsed >= WINDOW_30M_MS and not self._has_30m_data:
                self._has_30m_data = True
        
        # Build volume profiles
        self._build_profile_5m()
        self._build_profile_30m()
        
        # Compute POC/VAH/VAL/LVN from profiles
        self._compute_5m_values()
        if self._has_30m_data:
            self._compute_30m_values()
        
        # Set global values based on whether we have 30m data
        if self._has_30m_data:
            # After 30 mins: use 30m for POC/VAH/VAL
            self._poc = self._poc_30m
            self._vah = self._vah_30m
            self._val = self._val_30m
        else:
            # Before 30 mins: use 5m for everything
            self._poc = self._poc_5m
            self._vah = self._vah_5m
            self._val = self._val_5m
        
        # LVN always from 5m
        self._lvn = self._lvn_5m
        
        # Track price for time-inside-value
        if self._current_price > 0:
            self._price_history.append((now_ms, self._current_price))
        
        # Track value area width
        width = self._vah - self._val
        if width > 0:
            self._value_area_widths.append(width)
        
        return self._compute_features()
    
    def _build_profile_5m(self) -> None:
        """Build volume profile from 5m trade buffer - optimized"""
        self._profile_5m.clear()
        
        # Get dynamic bin size based on current price
        bin_size = get_bin_size(self._current_price)
        if bin_size <= 0:
            bin_size = 1.0
        
        for price, qty, _ in self._trades_5m:
            price_bin = round(price / bin_size) * bin_size
            self._profile_5m[price_bin] = self._profile_5m.get(price_bin, 0.0) + qty
    
    def _build_profile_30m(self) -> None:
        """Build volume profile from 30m trade buffer - optimized"""
        self._profile_30m.clear()
        
        # Get dynamic bin size based on current price
        bin_size = get_bin_size(self._current_price)
        if bin_size <= 0:
            bin_size = 1.0
        
        for price, qty, _ in self._trades_30m:
            price_bin = round(price / bin_size) * bin_size
            self._profile_30m[price_bin] = self._profile_30m.get(price_bin, 0.0) + qty
    
    def _compute_5m_values(self) -> None:
        """Compute POC, VAH, VAL, LVN from 5m profile"""
        if not self._profile_5m:
            return
        
        # POC = price with max volume
        self._poc_5m = max(self._profile_5m.items(), key=lambda x: x[1])[0]
        
        # LVN = price with min volume
        self._lvn_5m = min(self._profile_5m.items(), key=lambda x: x[1])[0]
        
        # VAH/VAL = 70% value area around POC
        self._vah_5m, self._val_5m = self._compute_value_area(self._profile_5m, self._poc_5m)
    
    def _compute_30m_values(self) -> None:
        """Compute POC, VAH, VAL from 30m profile (no LVN for 30m)"""
        if not self._profile_30m:
            return
        
        # POC = price with max volume
        self._poc_30m = max(self._profile_30m.items(), key=lambda x: x[1])[0]
        
        # VAH/VAL = 70% value area around POC
        self._vah_30m, self._val_30m = self._compute_value_area(self._profile_30m, self._poc_30m)
    
    def _compute_value_area(self, profile: Dict[float, float], poc: float) -> Tuple[float, float]:
        """
        Compute VAH and VAL as smallest range around POC containing 70% of volume
        Exact AMT logic, not a heuristic
        """
        if not profile:
            return 0.0, 0.0
        
        total_volume = sum(profile.values())
        if total_volume <= 0:
            return poc, poc
        
        target_volume = 0.70 * total_volume
        
        # Sort bins by distance from POC
        bins = list(profile.items())
        bins_sorted = sorted(bins, key=lambda x: abs(x[0] - poc))
        
        # Expand from POC until 70% reached
        cum_volume = 0.0
        value_bins = []
        
        for price, vol in bins_sorted:
            value_bins.append(price)
            cum_volume += vol
            if cum_volume >= target_volume:
                break
        
        if not value_bins:
            return poc, poc
        
        vah = max(value_bins)
        val = min(value_bins)
        
        return vah, val
    
    def _compute_features(self) -> StructureFeatures:
        """Compute all structural features"""
        # Distance to POC (normalized by ATR)
        dist_poc = 0.0
        if self._atr_5m > 1e-9 and self._poc > 0:
            dist_poc = abs(self._current_price - self._poc) / self._atr_5m
        
        # Distance to LVN (normalized by ATR)
        dist_lvn = 0.0
        if self._atr_5m > 1e-9 and self._lvn > 0:
            dist_lvn = abs(self._current_price - self._lvn) / self._atr_5m
        
        # Value area width
        value_area_width = self._vah - self._val
        
        # Value width ratio vs median
        value_width_ratio = 0.0
        if len(self._value_area_widths) >= 20:
            median_width = float(np.median(self._value_area_widths))
            if median_width > 1e-9:
                value_width_ratio = value_area_width / median_width
        
        # Time inside value area
        time_inside_value_pct = self._compute_time_inside_value()
        
        # Acceptance outside value
        acceptance = self._check_acceptance()
        
        self._last_features = StructureFeatures(
            poc=self._poc,
            vah=self._vah,
            val=self._val,
            lvns=[self._lvn] if self._lvn > 0 else [],
            dist_poc=dist_poc,
            dist_lvn=dist_lvn,
            value_area_width=value_area_width,
            value_width_ratio=value_width_ratio,
            time_inside_value_pct=time_inside_value_pct,
            acceptance_outside_value=acceptance,
        )
        
        return self._last_features
    
    def _compute_time_inside_value(self) -> float:
        """Compute % time price spent inside VAH/VAL (last 5 min)"""
        if len(self._price_history) < 10 or self._vah <= 0 or self._val <= 0:
            return 100.0
        
        now_ms = self._price_history[-1][0]
        cutoff = now_ms - 300_000  # 5 min
        
        inside_count = 0
        total_count = 0
        
        for ts, price in self._price_history:
            if ts >= cutoff:
                total_count += 1
                if self._val <= price <= self._vah:
                    inside_count += 1
        
        if total_count == 0:
            return 100.0
        
        return (inside_count / total_count) * 100.0
    
    def _check_acceptance(self) -> bool:
        """Check if price accepted outside value area (3 min sustained)"""
        if self._vah <= 0 or self._val <= 0:
            return False
        
        now_ms = int(time.time() * 1000)
        is_outside = self._current_price > self._vah or self._current_price < self._val
        
        if is_outside:
            if self._outside_value_start is None:
                self._outside_value_start = now_ms
            elif now_ms - self._outside_value_start >= self._acceptance_threshold_ms:
                return True
        else:
            self._outside_value_start = None
        
        return False
    
    def get_features(self) -> StructureFeatures:
        """Get current features"""
        return self._last_features
    
    def has_data(self) -> bool:
        """Check if we have enough data to compute features"""
        return len(self._trades_5m) > 0


class MultiSymbolStructureProcessor:
    """Manages structure processors for multiple symbols"""
    
    def __init__(self, symbols: List[str]):
        self.processors: Dict[str, StructureProcessor] = {
            s: StructureProcessor(s) for s in symbols
        }
    
    def add_trade(self, trade: Trade) -> None:
        """Add trade to appropriate processor"""
        if trade.symbol in self.processors:
            self.processors[trade.symbol].add_trade(trade)
    
    def compute(self, symbol: str) -> StructureFeatures:
        """Compute features for symbol (call every 1 sec)"""
        if symbol in self.processors:
            return self.processors[symbol].compute()
        return StructureFeatures()
    
    def compute_all(self) -> Dict[str, StructureFeatures]:
        """Compute features for all symbols"""
        return {s: p.compute() for s, p in self.processors.items()}
    
    def set_atr(self, symbol: str, atr_5m: float) -> None:
        """Set ATR for distance normalization"""
        if symbol in self.processors:
            self.processors[symbol].set_atr(atr_5m)
    
    def get_features(self, symbol: str) -> StructureFeatures:
        """Get current features"""
        if symbol in self.processors:
            return self.processors[symbol].get_features()
        return StructureFeatures()
    
    def has_data(self, symbol: str) -> bool:
        """Check if processor has data"""
        if symbol in self.processors:
            return self.processors[symbol].has_data()
        return False
