"""
Order Flow Processor - MOI, DeltaVelocity, AggressionPersistence
Optimized for speed with numpy and rolling buffers

Key calculations per spec:
- MOI_250ms: sum of last 3 bars (750ms)
- MOI_1s: sum of last 4 bars (1 second)
- DeltaVelocity: MOI_1s(t) - MOI_1s(t-1)
- AggressionPersistence: mean(|MOI|) / std(|MOI|)
- moi_std: std of raw MOI values over 60 sec rolling
- delta_flip_rate: count sign flips in DeltaVelocity / window_minutes (1 min rolling)
- price_change_5m: from 5 min price buffer
"""
import numpy as np
from collections import deque
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

from src.core.models import Bar, Trade
from src.stage2.models import OrderFlowFeatures


class OrderFlowProcessor:
    """
    Computes order flow features from trades and bars
    Updates every 250ms
    """
    
    def __init__(self, symbol: str, history_bars: int = 240):
        """
        Args:
            symbol: Trading pair
            history_bars: Number of 250ms bars to keep (240 = 1 minute)
        """
        self.symbol = symbol
        
        # Rolling buffers for 250ms bars
        self._bars_250ms: deque[Bar] = deque(maxlen=history_bars)
        self._signed_volumes: deque[float] = deque(maxlen=history_bars)
        
        # MOI history for std calculation (60 sec = 240 bars at 250ms)
        self._moi_history: deque[float] = deque(maxlen=240)  # 60 sec of raw MOI
        
        # MOI_1s history for velocity/persistence
        self._moi_1s_history: deque[float] = deque(maxlen=60)  # 1 min of 1s MOI
        
        # DeltaVelocity history for flip rate (1 min rolling)
        self._delta_velocity_history: deque[float] = deque(maxlen=240)  # 1 min at 250ms
        
        # Price buffer for price_change_5m (5 min = 1200 bars at 250ms)
        self._price_buffer: deque[float] = deque(maxlen=1200)
        
        # Cached features
        self._last_features = OrderFlowFeatures()
        self._last_moi_1s = 0.0
    
    def add_bar(self, bar: Bar) -> OrderFlowFeatures:
        """
        Add new 250ms bar and compute features
        Returns updated OrderFlowFeatures
        """
        if bar.interval != "250ms":
            return self._last_features
        
        self._bars_250ms.append(bar)
        self._signed_volumes.append(bar.delta)  # buy_volume - sell_volume
        
        # Track price for price_change_5m
        self._price_buffer.append(bar.close)
        
        return self._compute_features()
    
    def add_trade(self, trade: Trade) -> None:
        """For real-time trade-by-trade processing if needed"""
        pass
    
    def _compute_features(self) -> OrderFlowFeatures:
        """Compute all order flow features"""
        n = len(self._signed_volumes)
        if n < 3:
            return self._last_features
        
        # Convert to numpy for fast computation
        vols = np.array(self._signed_volumes, dtype=np.float64)
        
        # MOI_250ms: sum of last 3 bars (750ms)
        moi_250ms = float(np.sum(vols[-3:]))
        
        # MOI_1s: sum of last 4 bars (1 second)
        moi_1s = float(np.sum(vols[-4:])) if n >= 4 else moi_250ms
        
        # Store raw MOI for moi_std calculation
        self._moi_history.append(moi_1s)
        
        # Delta velocity: MOI_1s change
        delta_velocity = moi_1s - self._last_moi_1s
        self._moi_1s_history.append(moi_1s)
        self._last_moi_1s = moi_1s
        
        # Store delta velocity for flip rate
        self._delta_velocity_history.append(delta_velocity)
        
        # moi_std: std of raw MOI values (60 sec rolling)
        moi_std = 0.0
        if len(self._moi_history) >= 10:
            moi_arr = np.array(self._moi_history, dtype=np.float64)
            moi_std = float(np.std(moi_arr))
        
        # Aggression persistence: mean(|MOI|) / std(|MOI|)
        aggression_persistence = 0.0
        if len(self._moi_1s_history) >= 10:
            moi_arr = np.array(self._moi_1s_history, dtype=np.float64)
            abs_moi = np.abs(moi_arr)
            std_abs = float(np.std(abs_moi))
            if std_abs > 1e-9:
                aggression_persistence = float(np.mean(abs_moi) / std_abs)
        
        # delta_flip_rate: sign flips in DeltaVelocity per minute
        delta_flip_rate = 0.0
        if len(self._delta_velocity_history) >= 10:
            dv_arr = np.array(self._delta_velocity_history, dtype=np.float64)
            signs = np.sign(dv_arr)
            flips = int(np.sum(np.abs(np.diff(signs)) > 0))
            # Window is 1 min (240 bars at 250ms), convert to flips/min
            window_minutes = len(self._delta_velocity_history) / 240.0
            delta_flip_rate = float(flips / window_minutes) if window_minutes > 0 else 0.0
        
        self._last_features = OrderFlowFeatures(
            moi_250ms=moi_250ms,
            moi_1s=moi_1s,
            delta_velocity=delta_velocity,
            aggression_persistence=aggression_persistence,
            moi_std=moi_std,
            moi_flip_rate=delta_flip_rate,  # Using delta_flip_rate here
        )
        
        return self._last_features
    
    def get_features(self) -> OrderFlowFeatures:
        """Get current features without recomputing"""
        return self._last_features
    
    def get_moi_series(self, count: int = 60) -> List[float]:
        """Get recent MOI values for regime detection"""
        return list(self._moi_1s_history)[-count:]
    
    def get_delta_velocity_series(self, count: int = 30) -> List[float]:
        """Get delta velocity series for regime detection"""
        if len(self._moi_1s_history) < 2:
            return []
        moi_arr = np.array(self._moi_1s_history, dtype=np.float64)
        velocities = np.diff(moi_arr)
        return list(velocities[-count:])
    
    def get_price_change_5m(self) -> float:
        """
        Get price change over 5 minutes
        price_change_5m = (price_now - price_5m_ago) / price_5m_ago
        """
        if len(self._price_buffer) < 2:
            return 0.0
        
        price_now = self._price_buffer[-1]
        # 5 min = 1200 bars at 250ms
        idx = min(len(self._price_buffer) - 1, 1199)
        price_5m_ago = self._price_buffer[-(idx + 1)]
        
        if price_5m_ago == 0:
            return 0.0
        
        return (price_now - price_5m_ago) / price_5m_ago


class MultiSymbolOrderFlowProcessor:
    """Manages order flow processors for multiple symbols"""
    
    def __init__(self, symbols: List[str]):
        self.processors: Dict[str, OrderFlowProcessor] = {
            s: OrderFlowProcessor(s) for s in symbols
        }
    
    def add_bar(self, bar: Bar) -> OrderFlowFeatures:
        """Add bar to appropriate processor"""
        if bar.symbol in self.processors:
            return self.processors[bar.symbol].add_bar(bar)
        return OrderFlowFeatures()
    
    def get_features(self, symbol: str) -> OrderFlowFeatures:
        """Get features for symbol"""
        if symbol in self.processors:
            return self.processors[symbol].get_features()
        return OrderFlowFeatures()
