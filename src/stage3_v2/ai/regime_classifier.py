"""
Positioning Regime Classifier
=============================

Classifies market state into positioning regimes (not price direction).

Classes:
- NEUTRAL: No clear positioning imbalance
- CROWDED_LONG: Longs crowded, vulnerable to squeeze
- CROWDED_SHORT: Shorts crowded, vulnerable to squeeze
- UNWINDING: Active position unwind in progress
- POST_UNWIND: Fresh state after major unwind

This is learnable because regimes PERSIST (hours to days).
"""
import numpy as np
from typing import Optional, List
from collections import deque

from src.stage3_v2.models import LongHorizonState, PositioningRegime


class PositioningRegimeClassifier:
    """
    Classify current market state into positioning regimes.
    
    Rule-based implementation (can be upgraded to ML later).
    ML would use: RandomForest or XGBoost on features below.
    """
    
    def __init__(self):
        # Thresholds for rule-based classification
        self.funding_z_crowded = 1.5
        self.funding_z_extreme = 2.5
        self.oi_drop_unwind = -0.05  # 5% OI drop = unwind
        self.liq_imbalance_significant = 0.5
        
        # State tracking for POST_UNWIND
        self._recent_unwind = False
        self._unwind_end_time: Optional[float] = None
        self._post_unwind_window_h = 4.0  # 4 hours after unwind
    
    def classify(self, state: LongHorizonState) -> PositioningRegime:
        """
        Classify positioning regime from state.
        
        Feature vector for ML:
        - funding_z, funding_z_8h_avg, funding_z_change_8h
        - oi_delta_1h, oi_delta_4h, oi_delta_24h
        - liq_imbalance_1h, liq_imbalance_4h
        - liq_total_usd_24h
        - vol_expansion_ratio
        """
        import time
        
        # Check for UNWINDING (active OI drop + liquidations)
        if self._is_unwinding(state):
            self._recent_unwind = True
            self._unwind_end_time = None
            return PositioningRegime.UNWINDING
        
        # Check for POST_UNWIND (just finished unwinding)
        if self._recent_unwind and not self._is_unwinding(state):
            if self._unwind_end_time is None:
                self._unwind_end_time = time.time()
            
            hours_since_unwind = (time.time() - self._unwind_end_time) / 3600
            if hours_since_unwind < self._post_unwind_window_h:
                return PositioningRegime.POST_UNWIND
            else:
                self._recent_unwind = False
        
        # Check for CROWDED_LONG
        if self._is_crowded_long(state):
            return PositioningRegime.CROWDED_LONG
        
        # Check for CROWDED_SHORT
        if self._is_crowded_short(state):
            return PositioningRegime.CROWDED_SHORT
        
        return PositioningRegime.NEUTRAL
    
    def _is_unwinding(self, state: LongHorizonState) -> bool:
        """Check if market is actively unwinding positions"""
        # OI dropping significantly
        oi_dropping = state.oi_delta_4h < self.oi_drop_unwind
        
        # Liquidations happening
        liqs_active = state.liq_total_usd_4h > 50_000  # $50k+ in liquidations
        
        # Strong liquidation imbalance
        liq_imbalanced = abs(state.liq_imbalance_4h) > self.liq_imbalance_significant
        
        return oi_dropping and (liqs_active or liq_imbalanced)
    
    def _is_crowded_long(self, state: LongHorizonState) -> bool:
        """Check if longs are crowded"""
        # High positive funding
        funding_high = state.funding_z > self.funding_z_crowded
        
        # OI elevated or expanding
        oi_elevated = state.oi_delta_24h > 0.02
        
        # Longs getting liquidated (early warning)
        longs_liquidating = state.liq_imbalance_4h > 0.3
        
        return funding_high and (oi_elevated or longs_liquidating)
    
    def _is_crowded_short(self, state: LongHorizonState) -> bool:
        """Check if shorts are crowded"""
        # High negative funding
        funding_low = state.funding_z < -self.funding_z_crowded
        
        # OI elevated or expanding
        oi_elevated = state.oi_delta_24h > 0.02
        
        # Shorts getting liquidated (early warning)
        shorts_liquidating = state.liq_imbalance_4h < -0.3
        
        return funding_low and (oi_elevated or shorts_liquidating)
    
    def get_feature_vector(self, state: LongHorizonState) -> np.ndarray:
        """
        Extract feature vector for ML training.
        
        Use this to collect training data for future ML classifier.
        """
        return np.array([
            state.funding_z,
            state.funding_z_8h_avg,
            state.funding_z_change_8h,
            state.oi_delta_1h,
            state.oi_delta_4h,
            state.oi_delta_24h,
            state.oi_price_correlation_24h,
            state.liq_imbalance_1h,
            state.liq_imbalance_4h,
            state.liq_total_usd_24h,
            state.liq_asymmetry_persistence_h,
            state.vol_expansion_ratio,
        ])
