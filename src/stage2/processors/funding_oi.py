"""
Funding & OI Processor - Z-scores, crowd detection, participation type
"""
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import time

from src.core.models import FundingRate, OpenInterest
from src.stage2.models import FundingFeatures, OIFeatures, CrowdSide, ParticipationType


class FundingOIProcessor:
    """
    Computes funding and OI features
    Detects crowd positioning and participation type
    """
    
    def __init__(
        self, 
        symbol: str,
        funding_history: int = 100,    # ~33 days of funding (3/day)
        oi_history: int = 300,         # 5 hours of 1-min OI
        funding_z_threshold: float = 1.5,  # Z-score threshold for crowd detection
    ):
        self.symbol = symbol
        self.funding_z_threshold = funding_z_threshold
        
        # Funding history
        self._funding_rates: deque[Tuple[int, float]] = deque(maxlen=funding_history)
        
        # OI history with timestamps
        self._oi_history: deque[Tuple[int, float]] = deque(maxlen=oi_history)
        
        # Price history for OI-price correlation
        self._price_history: deque[Tuple[int, float]] = deque(maxlen=oi_history)
        
        # Stats
        self._funding_mean = 0.0
        self._funding_std = 0.0001  # Small default to avoid div by zero
        
        # Cached features
        self._last_funding = FundingFeatures()
        self._last_oi = OIFeatures()
        
        # Staleness tracking
        self._last_funding_update_ms: int = 0
        self._last_oi_update_ms: int = 0
        self._oi_stale_threshold_ms: int = 120_000  # 2 minutes = stale
        self._funding_stale_threshold_ms: int = 28_800_000  # 8 hours = stale
    
    def add_funding(self, funding: FundingRate) -> FundingFeatures:
        """Add funding rate and compute features"""
        self._funding_rates.append((funding.timestamp_ms, funding.funding_rate))
        self._last_funding_update_ms = int(time.time() * 1000)
        
        # Update stats
        if len(self._funding_rates) >= 10:
            rates = np.array([r for _, r in self._funding_rates], dtype=np.float64)
            self._funding_mean = float(np.mean(rates))
            self._funding_std = float(np.std(rates))
            if self._funding_std < 1e-9:
                self._funding_std = 0.0001
        
        # Compute z-score
        funding_z = (funding.funding_rate - self._funding_mean) / self._funding_std
        
        # Determine crowd side
        crowd_side = CrowdSide.NEUTRAL
        if funding_z > self.funding_z_threshold:
            crowd_side = CrowdSide.LONG  # Longs paying = crowd is long
        elif funding_z < -self.funding_z_threshold:
            crowd_side = CrowdSide.SHORT  # Shorts paying = crowd is short
        
        self._last_funding = FundingFeatures(
            rate=funding.funding_rate,
            funding_z=funding_z,
            crowd_side=crowd_side,
            annualized_pct=funding.annualized_rate,
        )
        
        return self._last_funding
    
    def add_oi(self, oi: OpenInterest, current_price: float) -> OIFeatures:
        """Add OI reading and compute features"""
        self._oi_history.append((oi.timestamp_ms, oi.open_interest))
        self._price_history.append((oi.timestamp_ms, current_price))
        self._last_oi_update_ms = int(time.time() * 1000)
        
        # Current OI
        current_oi = oi.open_interest
        now_ms = oi.timestamp_ms
        
        # OI delta (vs previous reading)
        oi_delta = 0.0
        if len(self._oi_history) >= 2:
            prev_oi = self._oi_history[-2][1]
            if prev_oi > 0:
                oi_delta = (current_oi - prev_oi) / prev_oi
        
        # OI delta 1m (vs 1 min ago)
        oi_delta_1m = 0.0
        if len(self._oi_history) >= 2:
            target_ms = now_ms - 60_000
            for ts, old_oi in reversed(self._oi_history):
                if ts <= target_ms:
                    if old_oi > 0:
                        oi_delta_1m = (current_oi - old_oi) / old_oi
                    break
        
        # OI delta 5m (vs 5 min ago)
        oi_delta_5m = 0.0
        if len(self._oi_history) >= 5:
            target_ms = now_ms - 300_000
            for ts, old_oi in reversed(self._oi_history):
                if ts <= target_ms:
                    if old_oi > 0:
                        oi_delta_5m = (current_oi - old_oi) / old_oi
                    break
        
        # Determine participation type from OI-price relationship
        participation = self._determine_participation(oi_delta_5m)
        
        self._last_oi = OIFeatures(
            oi=current_oi,
            oi_delta=oi_delta,
            oi_delta_1m=oi_delta_1m,
            oi_delta_5m=oi_delta_5m,
            participation_type=participation,
        )
        
        return self._last_oi
    
    def _determine_participation(self, oi_delta: float) -> ParticipationType:
        """
        Determine participation type from OI-price relationship
        Price up + OI up   → new longs
        Price up + OI down → short covering
        Price down + OI up   → new shorts
        Price down + OI down → long covering
        """
        if len(self._price_history) < 5:
            return ParticipationType.NEUTRAL
        
        # Get price change over same period
        prices = [p for _, p in self._price_history]
        if len(prices) < 5:
            return ParticipationType.NEUTRAL
        
        # Price change (current vs 5 readings ago)
        price_change = prices[-1] - prices[-5] if len(prices) >= 5 else 0.0
        price_pct = price_change / prices[-5] if prices[-5] > 0 else 0.0
        
        # Thresholds
        oi_thresh = 0.001  # 0.1% OI change
        price_thresh = 0.0005  # 0.05% price change
        
        if abs(oi_delta) < oi_thresh and abs(price_pct) < price_thresh:
            return ParticipationType.NEUTRAL
        
        if price_pct > price_thresh:
            # Price up
            if oi_delta > oi_thresh:
                return ParticipationType.NEW_LONGS
            elif oi_delta < -oi_thresh:
                return ParticipationType.SHORT_COVERING
        elif price_pct < -price_thresh:
            # Price down
            if oi_delta > oi_thresh:
                return ParticipationType.NEW_SHORTS
            elif oi_delta < -oi_thresh:
                return ParticipationType.LONG_COVERING
        
        return ParticipationType.NEUTRAL
    
    def get_funding_features(self) -> FundingFeatures:
        """Get current funding features"""
        return self._last_funding
    
    def get_oi_features(self) -> OIFeatures:
        """Get current OI features"""
        return self._last_oi
    
    def is_oi_stale(self) -> bool:
        """Check if OI data is stale (no update in 2+ minutes)"""
        if self._last_oi_update_ms == 0:
            return True  # Never updated
        age_ms = int(time.time() * 1000) - self._last_oi_update_ms
        return age_ms > self._oi_stale_threshold_ms
    
    def is_funding_stale(self) -> bool:
        """Check if funding data is stale (no update in 8+ hours)"""
        if self._last_funding_update_ms == 0:
            return True  # Never updated
        age_ms = int(time.time() * 1000) - self._last_funding_update_ms
        return age_ms > self._funding_stale_threshold_ms
    
    def get_oi_age_ms(self) -> int:
        """Get age of last OI update in milliseconds"""
        if self._last_oi_update_ms == 0:
            return 999999999
        return int(time.time() * 1000) - self._last_oi_update_ms


class MultiSymbolFundingOIProcessor:
    """Manages funding/OI processors for multiple symbols"""
    
    def __init__(self, symbols: List[str]):
        self.processors: Dict[str, FundingOIProcessor] = {
            s: FundingOIProcessor(s) for s in symbols
        }
    
    def add_funding(self, funding: FundingRate) -> FundingFeatures:
        if funding.symbol in self.processors:
            return self.processors[funding.symbol].add_funding(funding)
        return FundingFeatures()
    
    def add_oi(self, oi: OpenInterest, current_price: float) -> OIFeatures:
        if oi.symbol in self.processors:
            return self.processors[oi.symbol].add_oi(oi, current_price)
        return OIFeatures()
    
    def get_funding_features(self, symbol: str) -> FundingFeatures:
        if symbol in self.processors:
            return self.processors[symbol].get_funding_features()
        return FundingFeatures()
    
    def get_oi_features(self, symbol: str) -> OIFeatures:
        if symbol in self.processors:
            return self.processors[symbol].get_oi_features()
        return OIFeatures()
