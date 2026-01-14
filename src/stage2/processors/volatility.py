"""
Volatility Processor - ATR, realized volatility, vol regime
Optimized with numpy for fast rolling calculations

Key calculations:
- Vol_5m: rolling std of 250ms returns over 5 minutes (1200 bars)
- ATR_5m: mean of last 14 True Range values (5m bars) - bootstrapped from klines
- ATR_1h: mean of last 14 True Range values (1h bars) - bootstrapped from klines
- vol_expansion_ratio: ATR_5m / ATR_1h
- vol_regime: based on ATR ratio (LOW < 0.7, MID < 1.3, HIGH >= 1.3)
"""
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import time

from src.core.models import Bar
from src.stage2.models import VolatilityFeatures


class VolatilityProcessor:
    """
    Computes volatility features from bars
    
    ATR bootstrapped from historical klines at startup, then updated live.
    Vol_5m computed from live 250ms returns (5 min rolling).
    Vol regime based on ATR_5m / ATR_1h ratio.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # ATR True Range deques (14-period for ATR calculation)
        self._tr_5m: deque[float] = deque(maxlen=14)
        self._tr_1h: deque[float] = deque(maxlen=14)
        
        # 250ms returns for vol_5m calculation (5 min = 1200 bars at 250ms)
        self._returns_250ms: deque[float] = deque(maxlen=1200)
        
        # Vol_5m history for percentile-based regime (rolling 1500 values = 25 hours)
        self._vol_5m_history: deque[float] = deque(maxlen=1500)
        
        # Last closes for TR/return calculation
        self._last_close_5m = 0.0
        self._last_close_1h = 0.0
        self._last_close_250ms = 0.0
        
        # Cached ATR values (bootstrapped from klines)
        self._atr_5m = 0.0
        self._atr_1h = 0.0
        
        # Cached features
        self._last_features = VolatilityFeatures()
        
        # Bootstrap flag
        self._bootstrapped = False
    
    def bootstrap(
        self,
        tr_5m_values: List[float],
        tr_1h_values: List[float],
        atr_5m: float,
        atr_1h: float,
        last_close_5m: float = 0.0,
        last_close_1h: float = 0.0,
        vol_5m_history: Optional[List[float]] = None,
    ) -> None:
        """
        Bootstrap ATR and volatility from historical klines data
        
        Args:
            tr_5m_values: Last 14 True Range values for 5m ATR
            tr_1h_values: Last 14 True Range values for 1h ATR
            atr_5m: Current ATR_5m
            atr_1h: Current ATR_1h
            last_close_5m: Last 5m candle close for TR continuation
            last_close_1h: Last 1h candle close for TR continuation
            vol_5m_history: Historical vol_5m values for percentile regime
        """
        # Load TR values for live continuation
        for tr in tr_5m_values:
            self._tr_5m.append(tr)
        for tr in tr_1h_values:
            self._tr_1h.append(tr)
        
        # Set ATR values
        self._atr_5m = atr_5m
        self._atr_1h = atr_1h
        
        # Set last closes for live TR calculation continuity
        self._last_close_5m = last_close_5m
        self._last_close_1h = last_close_1h
        
        # Load vol_5m history for percentile-based regime detection
        if vol_5m_history:
            for vol in vol_5m_history[-1500:]:  # Keep only last 1500
                self._vol_5m_history.append(vol)
        
        self._bootstrapped = True
        
        # Compute initial features
        self._compute_features()
    
    def add_bar(self, bar: Bar) -> VolatilityFeatures:
        """Add bar and update volatility features"""
        if bar.interval == "250ms":
            self._add_250ms_bar(bar)
        elif bar.interval == "5m":
            self._add_5m_bar(bar)
        elif bar.interval == "1h":
            self._add_1h_bar(bar)
        
        return self._compute_features()
    
    def _add_250ms_bar(self, bar: Bar) -> None:
        """Process 250ms bar for returns tracking (vol_5m calculation)"""
        if self._last_close_250ms > 0:
            ret = (bar.close - self._last_close_250ms) / self._last_close_250ms
            self._returns_250ms.append(ret)
        self._last_close_250ms = bar.close
    
    def _add_5m_bar(self, bar: Bar) -> None:
        """Process 5m bar for ATR_5m only"""
        if self._last_close_5m > 0:
            # Compute True Range
            tr = self._compute_true_range(
                bar.high, bar.low, bar.close, self._last_close_5m
            )
            self._tr_5m.append(tr)
            
            # Update ATR_5m (mean of last 14 TR)
            if len(self._tr_5m) >= 1:
                self._atr_5m = float(np.mean(self._tr_5m))
        
        self._last_close_5m = bar.close
    
    def _add_1h_bar(self, bar: Bar) -> None:
        """Process 1h bar for ATR_1h"""
        if self._last_close_1h > 0:
            tr = self._compute_true_range(
                bar.high, bar.low, bar.close, self._last_close_1h
            )
            self._tr_1h.append(tr)
            
            # Update ATR_1h (mean of last 14 TR)
            if len(self._tr_1h) >= 1:
                self._atr_1h = float(np.mean(self._tr_1h))
        
        self._last_close_1h = bar.close
    
    def _compute_true_range(
        self, high: float, low: float, close: float, prev_close: float
    ) -> float:
        """
        Compute True Range for ATR
        TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
        """
        return max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
    
    def _compute_features(self) -> VolatilityFeatures:
        """Compute all volatility features"""
        # Vol_5m: std of 250ms returns over 5 minutes
        vol_5m = 0.0
        if len(self._returns_250ms) >= 10:
            vol_5m = float(np.std(self._returns_250ms))
        
        # Add current vol_5m to rolling history
        if vol_5m > 0:
            self._vol_5m_history.append(vol_5m)
        
        # Vol expansion ratio = ATR_5m / ATR_1h
        vol_expansion_ratio = 0.0
        if self._atr_1h > 1e-9:
            vol_expansion_ratio = self._atr_5m / self._atr_1h
        
        # Vol regime based on percentile of vol_5m vs history
        # LOW: vol_5m < 30th percentile
        # MID: 30th-70th percentile
        # HIGH: > 70th percentile
        vol_rank = 0.0
        vol_regime = "MID"
        
        if len(self._vol_5m_history) >= 100:  # Need enough history
            history_array = np.array(self._vol_5m_history)
            vol_rank = float(np.sum(history_array < vol_5m) / len(history_array) * 100)
            
            if vol_rank < 30:
                vol_regime = "LOW"
            elif vol_rank > 70:
                vol_regime = "HIGH"
            else:
                vol_regime = "MID"
        
        self._last_features = VolatilityFeatures(
            vol_5m=vol_5m,
            vol_1h=vol_5m,  # Same as vol_5m for now
            vol_rank=vol_rank,
            vol_regime=vol_regime,
            atr_5m=self._atr_5m,
            atr_1h=self._atr_1h,
            vol_expansion_ratio=vol_expansion_ratio,
        )
        
        return self._last_features
    
    def get_features(self) -> VolatilityFeatures:
        """Get current features"""
        return self._last_features
    
    def get_atr_5m(self) -> float:
        """Get current ATR 5m for normalization"""
        return self._atr_5m
    
    def get_atr_1h(self) -> float:
        """Get current ATR 1h"""
        return self._atr_1h


class MultiSymbolVolatilityProcessor:
    """Manages volatility processors for multiple symbols"""
    
    def __init__(self, symbols: List[str]):
        self.processors: Dict[str, VolatilityProcessor] = {
            s: VolatilityProcessor(s) for s in symbols
        }
    
    def bootstrap(
        self,
        symbol: str,
        tr_5m_values: List[float],
        tr_1h_values: List[float],
        atr_5m: float,
        atr_1h: float,
        last_close_5m: float = 0.0,
        last_close_1h: float = 0.0,
        vol_5m_history: Optional[List[float]] = None,
    ) -> None:
        """Bootstrap a symbol's processor with ATR and volatility data"""
        if symbol in self.processors:
            self.processors[symbol].bootstrap(
                tr_5m_values, tr_1h_values, atr_5m, atr_1h,
                last_close_5m, last_close_1h, vol_5m_history
            )
    
    def add_bar(self, bar: Bar) -> VolatilityFeatures:
        if bar.symbol in self.processors:
            return self.processors[bar.symbol].add_bar(bar)
        return VolatilityFeatures()
    
    def get_features(self, symbol: str) -> VolatilityFeatures:
        if symbol in self.processors:
            return self.processors[symbol].get_features()
        return VolatilityFeatures()
    
    def get_atr_5m(self, symbol: str) -> float:
        if symbol in self.processors:
            return self.processors[symbol].get_atr_5m()
        return 0.0
    
    def get_atr_1h(self, symbol: str) -> float:
        if symbol in self.processors:
            return self.processors[symbol].get_atr_1h()
        return 0.0
