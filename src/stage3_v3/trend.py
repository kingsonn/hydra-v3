"""
Trend Analyzer
==============

Calculates trend indicators and structure for entry timing.
This is the TACTICAL layer that tells us WHEN to enter.

Components:
- EMAs (20, 50, 200)
- Price structure (higher highs/lows)
- RSI
- Pullback detection
"""
import numpy as np
from typing import List, Optional, Tuple
from collections import deque
from src.stage3_v3.models import TrendState, Direction


class TrendAnalyzer:
    """
    Analyze price trend and structure.
    
    Maintains price history for indicator calculation.
    """
    
    def __init__(self, history_size: int = 500):
        # Price history
        self._prices: deque[float] = deque(maxlen=history_size)
        self._highs: deque[float] = deque(maxlen=history_size)
        self._lows: deque[float] = deque(maxlen=history_size)
        self._timestamps: deque[int] = deque(maxlen=history_size)
        
        # Cached EMAs (updated incrementally)
        self._ema_20: float = 0.0
        self._ema_50: float = 0.0
        self._ema_200: float = 0.0
        
        # EMA multipliers
        self._ema_20_mult = 2 / (20 + 1)
        self._ema_50_mult = 2 / (50 + 1)
        self._ema_200_mult = 2 / (200 + 1)
        
        # RSI
        self._gains: deque[float] = deque(maxlen=14)
        self._losses: deque[float] = deque(maxlen=14)
        
        # Swing tracking - FIXED: Initialize to None to track if we have valid swings
        self._recent_swing_high: float = 0.0
        self._recent_swing_low: float = float('inf')
        self._prev_swing_high: float = 0.0
        self._prev_swing_low: float = float('inf')
        self._swing_high_count: int = 0  # Track how many swing highs we've seen
        self._swing_low_count: int = 0   # Track how many swing lows we've seen
    
    def update(self, price: float, high: float, low: float, timestamp_ms: int):
        """Add new price data and update indicators"""
        # Store previous price for RSI
        prev_price = self._prices[-1] if self._prices else price
        
        # Add to history
        self._prices.append(price)
        self._highs.append(high)
        self._lows.append(low)
        self._timestamps.append(timestamp_ms)
        
        # Update EMAs
        if len(self._prices) == 1:
            self._ema_20 = price
            self._ema_50 = price
            self._ema_200 = price
        else:
            self._ema_20 = price * self._ema_20_mult + self._ema_20 * (1 - self._ema_20_mult)
            self._ema_50 = price * self._ema_50_mult + self._ema_50 * (1 - self._ema_50_mult)
            self._ema_200 = price * self._ema_200_mult + self._ema_200 * (1 - self._ema_200_mult)
        
        # Update RSI components
        change = price - prev_price
        if change > 0:
            self._gains.append(change)
            self._losses.append(0)
        else:
            self._gains.append(0)
            self._losses.append(abs(change))
        
        # Update swing points (simple detection)
        self._update_swings()
    
    def _update_swings(self):
        """Update swing high/low tracking"""
        if len(self._highs) < 5:
            return
        
        # Look for swing high (higher than neighbors)
        highs = list(self._highs)[-10:]
        lows = list(self._lows)[-10:]
        
        # Find local max in last 10 bars
        max_idx = np.argmax(highs)
        if 2 <= max_idx <= 7:  # Not at edges
            new_swing_high = highs[max_idx]
            if new_swing_high != self._recent_swing_high:
                self._prev_swing_high = self._recent_swing_high
                self._recent_swing_high = new_swing_high
                self._swing_high_count += 1
        
        # Find local min in last 10 bars
        min_idx = np.argmin(lows)
        if 2 <= min_idx <= 7:
            new_swing_low = lows[min_idx]
            if new_swing_low != self._recent_swing_low:
                self._prev_swing_low = self._recent_swing_low
                self._recent_swing_low = new_swing_low
                self._swing_low_count += 1
    
    def get_state(self, current_price: float) -> TrendState:
        """Get current trend state"""
        if len(self._prices) < 20:
            return TrendState()  # Not enough data
        
        # Calculate RSI
        avg_gain = sum(self._gains) / len(self._gains) if self._gains else 0
        avg_loss = sum(self._losses) / len(self._losses) if self._losses else 0.0001
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        # Price vs EMA (percentage)
        price_vs_ema20 = (current_price - self._ema_20) / self._ema_20 * 100 if self._ema_20 > 0 else 0
        price_vs_ema50 = (current_price - self._ema_50) / self._ema_50 * 100 if self._ema_50 > 0 else 0
        
        # Structure detection - FIXED: Also infer from price vs EMA when swings not yet established
        if self._swing_high_count >= 2 and self._swing_low_count >= 2:
            # We have enough swings for proper structure detection
            higher_high = self._recent_swing_high > self._prev_swing_high
            higher_low = self._recent_swing_low > self._prev_swing_low
            lower_high = self._recent_swing_high < self._prev_swing_high
            lower_low = self._recent_swing_low < self._prev_swing_low
        else:
            # Infer structure from EMA relationship and price position
            # This prevents being stuck with all False during warmup
            if self._ema_20 > self._ema_50 and current_price > self._ema_20:
                higher_high = True
                higher_low = True
                lower_high = False
                lower_low = False
            elif self._ema_20 < self._ema_50 and current_price < self._ema_20:
                higher_high = False
                higher_low = False
                lower_high = True
                lower_low = True
            else:
                higher_high = False
                higher_low = False
                lower_high = False
                lower_low = False
        
        # Determine trend direction
        direction, strength = self._determine_trend(
            current_price, price_vs_ema20, price_vs_ema50,
            higher_high, higher_low, lower_high, lower_low
        )
        
        return TrendState(
            direction=direction,
            strength=strength,
            ema_20=self._ema_20,
            ema_50=self._ema_50,
            ema_200=self._ema_200,
            higher_high=higher_high,
            higher_low=higher_low,
            lower_high=lower_high,
            lower_low=lower_low,
            price_vs_ema20=price_vs_ema20,
            price_vs_ema50=price_vs_ema50,
            rsi_14=rsi,
        )
    
    def _determine_trend(
        self,
        price: float,
        price_vs_ema20: float,
        price_vs_ema50: float,
        higher_high: bool,
        higher_low: bool,
        lower_high: bool,
        lower_low: bool,
    ) -> Tuple[Direction, float]:
        """Determine trend direction and strength"""
        score = 0.0
        
        # EMA alignment
        if self._ema_20 > self._ema_50 > self._ema_200:
            score += 0.3  # Bullish alignment
        elif self._ema_20 < self._ema_50 < self._ema_200:
            score -= 0.3  # Bearish alignment
        
        # Price vs EMAs
        if price_vs_ema20 > 0.5:
            score += 0.2
        elif price_vs_ema20 < -0.5:
            score -= 0.2
        
        if price_vs_ema50 > 1.0:
            score += 0.2
        elif price_vs_ema50 < -1.0:
            score -= 0.2
        
        # Structure
        if higher_high and higher_low:
            score += 0.3
        elif lower_high and lower_low:
            score -= 0.3
        
        # Determine direction
        if score > 0.3:
            return Direction.LONG, min(1.0, score)
        elif score < -0.3:
            return Direction.SHORT, min(1.0, abs(score))
        else:
            return Direction.NEUTRAL, 0.0
    
    def is_pullback_entry(self, direction: Direction, threshold_pct: float = 0.3) -> bool:
        """Check if price has pulled back to EMA for entry"""
        if len(self._prices) < 20:
            return False
        
        current = self._prices[-1]
        price_vs_ema = (current - self._ema_20) / self._ema_20 * 100
        
        if direction == Direction.LONG:
            # For longs: price should be near or just above EMA20
            return -threshold_pct <= price_vs_ema <= threshold_pct * 2
        elif direction == Direction.SHORT:
            # For shorts: price should be near or just below EMA20
            return -threshold_pct * 2 <= price_vs_ema <= threshold_pct
        
        return False
    
    def get_stop_level(self, direction: Direction, atr: float) -> float:
        """Get suggested stop level based on structure"""
        if len(self._prices) < 5:
            return 0
        
        current = self._prices[-1]
        
        if direction == Direction.LONG:
            # Stop below recent swing low or 1.5 ATR
            structure_stop = self._recent_swing_low if self._recent_swing_low < float('inf') else current - atr * 1.5
            atr_stop = current - atr * 1.5
            return max(structure_stop, atr_stop)  # Use closer stop
        else:
            # Stop above recent swing high or 1.5 ATR
            structure_stop = self._recent_swing_high if self._recent_swing_high > 0 else current + atr * 1.5
            atr_stop = current + atr * 1.5
            return min(structure_stop, atr_stop)
    
    def get_recent_swing_high(self) -> float:
        return self._recent_swing_high
    
    def get_recent_swing_low(self) -> float:
        return self._recent_swing_low
