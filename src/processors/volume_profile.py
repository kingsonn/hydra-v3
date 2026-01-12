"""
D️⃣ STRUCTURAL CONTEXT - VOLUME PROFILE
Computes POC, VAH, VAL, and LVNs from trades
Rolling windows: 5m, 30m
"""
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import structlog

from config import settings
from src.core.models import Trade, VolumeProfile, VolumeProfileBin, Side

logger = structlog.get_logger(__name__)


@dataclass
class ProfileLevel:
    """Single price level in profile computation"""
    price_low: float
    price_high: float
    volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trade_count: int = 0
    
    @property
    def price_mid(self) -> float:
        return (self.price_low + self.price_high) / 2
    
    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume
    
    def add_trade(self, trade: Trade) -> None:
        self.volume += trade.quantity
        self.trade_count += 1
        if trade.side == Side.BUY:
            self.buy_volume += trade.quantity
        else:
            self.sell_volume += trade.quantity


class VolumeProfileCalculator:
    """
    Calculates volume profiles from trades
    
    Features:
    - Configurable bin size (tick-based)
    - Rolling window computation (5m, 30m)
    - POC (Point of Control) detection
    - Value Area (VAH/VAL) at 70%
    - LVN (Low Volume Node) detection
    """
    
    def __init__(
        self,
        tick_size: float = 0.10,  # Price bin size
        value_area_pct: float = 0.70,  # 70% of volume
        lvn_threshold_pct: float = 0.20,  # LVN = bins with < 20% of POC volume
    ):
        self.tick_size = tick_size
        self.value_area_pct = value_area_pct
        self.lvn_threshold_pct = lvn_threshold_pct
    
    def _price_to_bin(self, price: float) -> int:
        """Convert price to bin index"""
        return int(price / self.tick_size)
    
    def _bin_to_price_range(self, bin_idx: int) -> Tuple[float, float]:
        """Convert bin index to price range"""
        low = bin_idx * self.tick_size
        high = (bin_idx + 1) * self.tick_size
        return low, high
    
    def compute_profile(
        self,
        trades: List[Trade],
        symbol: str,
        window: str,
    ) -> Optional[VolumeProfile]:
        """
        Compute volume profile from a list of trades
        
        Args:
            trades: List of trades to analyze
            symbol: Symbol name
            window: Window identifier ("5m", "30m")
        
        Returns:
            VolumeProfile with POC, VAH, VAL, LVNs
        """
        if not trades:
            return None
        
        # Build price bins
        bins: Dict[int, ProfileLevel] = defaultdict(
            lambda: ProfileLevel(0, 0)
        )
        
        for trade in trades:
            bin_idx = self._price_to_bin(trade.price)
            
            if bin_idx not in bins:
                low, high = self._bin_to_price_range(bin_idx)
                bins[bin_idx] = ProfileLevel(low, high)
            
            bins[bin_idx].add_trade(trade)
        
        if not bins:
            return None
        
        # Convert to sorted list
        sorted_bins = sorted(bins.items(), key=lambda x: x[0])
        levels = [lvl for _, lvl in sorted_bins]
        
        # Total volume
        total_volume = sum(lvl.volume for lvl in levels)
        if total_volume == 0:
            return None
        
        # Find POC (bin with highest volume)
        poc_level = max(levels, key=lambda x: x.volume)
        poc_price = poc_level.price_mid
        poc_volume = poc_level.volume
        
        # Find Value Area (70% of volume around POC)
        vah_price, val_price = self._compute_value_area(levels, poc_level, total_volume)
        
        # Find LVNs (low volume nodes)
        lvn_prices = self._find_lvns(levels, poc_volume)
        
        # Create profile bins
        profile_bins = [
            VolumeProfileBin(
                price_low=lvl.price_low,
                price_high=lvl.price_high,
                volume=lvl.volume,
                buy_volume=lvl.buy_volume,
                sell_volume=lvl.sell_volume,
                trade_count=lvl.trade_count,
            )
            for lvl in levels
        ]
        
        now_ms = int(time.time() * 1000)
        start_ms = min(t.timestamp_ms for t in trades)
        end_ms = max(t.timestamp_ms for t in trades)
        
        return VolumeProfile(
            symbol=symbol,
            window=window,
            timestamp_ms=now_ms,
            start_ms=start_ms,
            end_ms=end_ms,
            bins=profile_bins,
            poc_price=poc_price,
            vah_price=vah_price,
            val_price=val_price,
            lvn_prices=lvn_prices,
            total_volume=total_volume,
        )
    
    def _compute_value_area(
        self,
        levels: List[ProfileLevel],
        poc_level: ProfileLevel,
        total_volume: float,
    ) -> Tuple[float, float]:
        """
        Compute Value Area High and Low
        Value Area contains 70% of volume, expanding from POC
        """
        target_volume = total_volume * self.value_area_pct
        
        # Find POC index
        poc_idx = next(i for i, lvl in enumerate(levels) if lvl is poc_level)
        
        # Start with POC volume
        va_volume = poc_level.volume
        va_high_idx = poc_idx
        va_low_idx = poc_idx
        
        # Expand outward from POC
        while va_volume < target_volume:
            # Check volume above and below
            above_vol = 0.0
            below_vol = 0.0
            
            if va_high_idx + 1 < len(levels):
                above_vol = levels[va_high_idx + 1].volume
            if va_low_idx - 1 >= 0:
                below_vol = levels[va_low_idx - 1].volume
            
            if above_vol == 0 and below_vol == 0:
                break
            
            # Expand towards higher volume
            if above_vol >= below_vol and va_high_idx + 1 < len(levels):
                va_high_idx += 1
                va_volume += above_vol
            elif va_low_idx - 1 >= 0:
                va_low_idx -= 1
                va_volume += below_vol
            else:
                break
        
        vah_price = levels[va_high_idx].price_high
        val_price = levels[va_low_idx].price_low
        
        return vah_price, val_price
    
    def _find_lvns(
        self,
        levels: List[ProfileLevel],
        poc_volume: float,
    ) -> List[float]:
        """
        Find Low Volume Nodes (LVNs)
        LVNs are price levels with significantly lower volume than surrounding levels
        """
        lvn_threshold = poc_volume * self.lvn_threshold_pct
        lvns = []
        
        for i, level in enumerate(levels):
            if level.volume < lvn_threshold:
                # Check if it's a local minimum
                prev_vol = levels[i - 1].volume if i > 0 else float('inf')
                next_vol = levels[i + 1].volume if i < len(levels) - 1 else float('inf')
                
                if level.volume < prev_vol and level.volume < next_vol:
                    lvns.append(level.price_mid)
        
        return lvns
    
    def compute_profiles_from_trades(
        self,
        trades: List[Trade],
        symbol: str,
        windows_ms: Dict[str, int],
    ) -> Dict[str, VolumeProfile]:
        """
        Compute multiple rolling window profiles from trades
        
        Args:
            trades: All trades (should be sorted by time)
            symbol: Symbol name
            windows_ms: Dict of window name to milliseconds
        
        Returns:
            Dict of window name to VolumeProfile
        """
        if not trades:
            return {}
        
        now_ms = max(t.timestamp_ms for t in trades)
        profiles = {}
        
        for window_name, window_ms in windows_ms.items():
            cutoff_ms = now_ms - window_ms
            window_trades = [t for t in trades if t.timestamp_ms >= cutoff_ms]
            
            if window_trades:
                profile = self.compute_profile(window_trades, symbol, window_name)
                if profile:
                    profiles[window_name] = profile
        
        return profiles


class RollingVolumeProfiler:
    """
    Maintains rolling volume profiles updated periodically
    """
    
    def __init__(
        self,
        symbols: List[str],
        windows: Optional[Dict[str, int]] = None,
        tick_size: float = 0.10,
        update_interval_s: int = 30,
    ):
        self.symbols = symbols
        self.windows = windows or {
            "5m": 5 * 60 * 1000,
            "30m": 30 * 60 * 1000,
        }
        self.update_interval_s = update_interval_s
        
        self.calculator = VolumeProfileCalculator(tick_size=tick_size)
        
        # Trade buffers per symbol (keep last 30 minutes)
        self._trades: Dict[str, List[Trade]] = {s: [] for s in symbols}
        self._max_buffer_ms = max(self.windows.values()) + 60_000
        
        # Latest profiles
        self._profiles: Dict[str, Dict[str, VolumeProfile]] = {
            s: {} for s in symbols
        }
        
        self._last_update: Dict[str, float] = {}
    
    def add_trade(self, trade: Trade) -> None:
        """Add trade to buffer"""
        symbol = trade.symbol
        if symbol in self._trades:
            self._trades[symbol].append(trade)
            self._prune_buffer(symbol)
    
    def _prune_buffer(self, symbol: str) -> None:
        """Remove old trades from buffer"""
        if not self._trades[symbol]:
            return
        
        cutoff = self._trades[symbol][-1].timestamp_ms - self._max_buffer_ms
        self._trades[symbol] = [
            t for t in self._trades[symbol] if t.timestamp_ms >= cutoff
        ]
    
    def update_profiles(self, symbol: str) -> Dict[str, VolumeProfile]:
        """Recompute profiles for symbol"""
        trades = self._trades.get(symbol, [])
        profiles = self.calculator.compute_profiles_from_trades(
            trades, symbol, self.windows
        )
        self._profiles[symbol] = profiles
        self._last_update[symbol] = time.time()
        return profiles
    
    def should_update(self, symbol: str) -> bool:
        """Check if profile needs update"""
        last = self._last_update.get(symbol, 0)
        return time.time() - last >= self.update_interval_s
    
    def get_profile(self, symbol: str, window: str) -> Optional[VolumeProfile]:
        """Get latest profile for symbol/window"""
        return self._profiles.get(symbol, {}).get(window)
    
    def get_all_profiles(self, symbol: str) -> Dict[str, VolumeProfile]:
        """Get all profiles for symbol"""
        return self._profiles.get(symbol, {}).copy()
    
    def get_structural_levels(self, symbol: str) -> Dict[str, any]:
        """Get key structural levels for trading"""
        result = {
            "poc": {},
            "vah": {},
            "val": {},
            "lvns": {},
        }
        
        for window, profile in self._profiles.get(symbol, {}).items():
            result["poc"][window] = profile.poc_price
            result["vah"][window] = profile.vah_price
            result["val"][window] = profile.val_price
            result["lvns"][window] = profile.lvn_prices
        
        return result


# ========== STANDALONE TEST FUNCTIONS ==========

def test_volume_profile_sync() -> VolumeProfile:
    """
    Test VolumeProfileCalculator with synthetic trades
    
    Usage:
        from src.processors.volume_profile import test_volume_profile_sync
        profile = test_volume_profile_sync()
    """
    import random
    
    calculator = VolumeProfileCalculator(tick_size=1.0)
    
    # Generate synthetic trades around a price with volume concentration
    base_price = 50000.0
    trades = []
    
    # Most volume around POC (50000-50010)
    for i in range(100):
        price = base_price + random.gauss(5, 2)  # Concentrated around 50005
        qty = random.uniform(0.1, 0.5)
        is_buyer_maker = random.random() < 0.5
        
        trades.append(Trade(
            symbol="BTCUSDT",
            timestamp_ms=1700000000000 + i * 100,
            price=price,
            quantity=qty,
            is_buyer_maker=is_buyer_maker,
            trade_id=i,
            first_trade_id=i,
            last_trade_id=i,
        ))
    
    # Add some outlier trades for LVN detection
    for i in range(10):
        price = base_price + random.choice([-20, 20]) + random.uniform(-2, 2)
        trades.append(Trade(
            symbol="BTCUSDT",
            timestamp_ms=1700000010000 + i * 100,
            price=price,
            quantity=random.uniform(0.01, 0.05),
            is_buyer_maker=random.random() < 0.5,
            trade_id=100 + i,
            first_trade_id=100 + i,
            last_trade_id=100 + i,
        ))
    
    profile = calculator.compute_profile(trades, "BTCUSDT", "5m")
    
    print("=== VOLUME PROFILE TEST ===")
    print(f"Symbol: {profile.symbol}")
    print(f"Window: {profile.window}")
    print(f"Total Volume: {profile.total_volume:.4f}")
    print(f"POC: ${profile.poc_price:.2f}")
    print(f"VAH: ${profile.vah_price:.2f}")
    print(f"VAL: ${profile.val_price:.2f}")
    print(f"LVNs: {[f'${p:.2f}' for p in profile.lvn_prices]}")
    print(f"Bins: {len(profile.bins)}")
    
    return profile


async def test_rolling_profiler_live(duration_seconds: int = 30) -> Dict[str, any]:
    """
    Test RollingVolumeProfiler with live trades
    
    Usage:
        from src.processors.volume_profile import test_rolling_profiler_live
        import asyncio
        result = asyncio.run(test_rolling_profiler_live(30))
    """
    from src.collectors.trades import TradesCollector
    
    profiler = RollingVolumeProfiler(
        symbols=["BTCUSDT"],
        windows={"30s": 30_000, "60s": 60_000},  # Short windows for testing
        tick_size=1.0,
        update_interval_s=5,
    )
    
    trade_count = 0
    
    def on_trade(trade: Trade):
        nonlocal trade_count
        profiler.add_trade(trade)
        trade_count += 1
        
        # Update profile periodically
        if profiler.should_update(trade.symbol):
            profiles = profiler.update_profiles(trade.symbol)
            for window, profile in profiles.items():
                print(f"  Profile {window}: POC=${profile.poc_price:.2f}, "
                      f"VAH=${profile.vah_price:.2f}, VAL=${profile.val_price:.2f}")
    
    collector = TradesCollector(
        symbols=["BTCUSDT"],
        on_trade=on_trade,
    )
    
    print(f"Starting rolling profiler test for {duration_seconds}s...")
    
    task = asyncio.create_task(collector.start())
    await asyncio.sleep(duration_seconds)
    await collector.stop()
    task.cancel()
    
    # Final update
    profiles = profiler.update_profiles("BTCUSDT")
    levels = profiler.get_structural_levels("BTCUSDT")
    
    print(f"\n=== ROLLING PROFILER TEST RESULTS ===")
    print(f"Trades processed: {trade_count}")
    print(f"Structural levels: {levels}")
    
    return {
        "trades_processed": trade_count,
        "profiles": {w: p.to_dict() for w, p in profiles.items()},
        "levels": levels,
    }
