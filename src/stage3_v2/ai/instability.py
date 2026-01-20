"""
Instability / Forced-Resolution Detector
========================================

Detects when positioning is UNSTABLE and likely to resolve violently.

Output:
- instability_score: 0-1 (higher = more unstable)
- direction_bias: -1 to +1 (which way resolution likely goes)

This is learnable because instability precedes major moves (hours lag).
"""
import numpy as np
from typing import Tuple
from collections import deque

from src.stage3_v2.models import LongHorizonState


class InstabilityDetector:
    """
    Detect when positioning is unstable and likely to resolve.
    
    Rule-based implementation (can be upgraded to ML later).
    ML would use: Gradient Boosting regression on features below.
    
    Ground truth for training: Did price move >3% in next 24h?
    """
    
    def __init__(self):
        # Scoring weights
        self.weights = {
            "funding_stress": 0.25,
            "oi_concentration": 0.20,
            "liq_proximity": 0.25,
            "vol_compression": 0.15,
            "recent_buildup": 0.15,
        }
        
        # History for baseline
        self._score_history: deque[float] = deque(maxlen=1000)
    
    def detect(self, state: LongHorizonState) -> Tuple[float, float]:
        """
        Detect instability and direction bias.
        
        Returns:
            (instability_score, direction_bias)
        """
        # Component scores
        funding_stress = self._funding_stress_score(state)
        oi_concentration = self._oi_concentration_score(state)
        liq_proximity = self._liq_proximity_score(state)
        vol_compression = self._vol_compression_score(state)
        recent_buildup = self._recent_buildup_score(state)
        
        # Weighted combination
        score = (
            funding_stress * self.weights["funding_stress"] +
            oi_concentration * self.weights["oi_concentration"] +
            liq_proximity * self.weights["liq_proximity"] +
            vol_compression * self.weights["vol_compression"] +
            recent_buildup * self.weights["recent_buildup"]
        )
        
        # Track for baseline
        self._score_history.append(score)
        
        # Direction bias based on positioning
        direction_bias = self._compute_direction_bias(state)
        
        return min(1.0, max(0.0, score)), direction_bias
    
    def _funding_stress_score(self, state: LongHorizonState) -> float:
        """Score based on funding stress (extreme = unstable)"""
        # Extreme funding = positions paying heavily = stressed
        funding_abs = abs(state.funding_z)
        
        if funding_abs > 3.0:
            return 1.0
        elif funding_abs > 2.0:
            return 0.8
        elif funding_abs > 1.5:
            return 0.5
        elif funding_abs > 1.0:
            return 0.3
        return 0.1
    
    def _oi_concentration_score(self, state: LongHorizonState) -> float:
        """Score based on OI built at risky levels"""
        # OI displacement = positions underwater = stressed
        displacement = abs(state.oi_entry_displacement_pct)
        
        if displacement > 0.05:  # 5%+ underwater
            return 1.0
        elif displacement > 0.03:
            return 0.7
        elif displacement > 0.02:
            return 0.4
        return 0.1
    
    def _liq_proximity_score(self, state: LongHorizonState) -> float:
        """Score based on proximity to liquidation levels"""
        # Use liq imbalance as proxy (one-sided = near cascade)
        imbalance = abs(state.liq_imbalance_4h)
        
        # Also consider recent liquidation activity
        liq_activity = min(1.0, state.liq_total_usd_4h / 500_000)  # Normalize to $500k
        
        return max(imbalance * 0.7, liq_activity * 0.5)
    
    def _vol_compression_score(self, state: LongHorizonState) -> float:
        """Score based on volatility compression (coiled spring)"""
        # Long compression = energy building
        compression_hours = state.vol_compression_duration_h
        vol_ratio = state.vol_expansion_ratio
        
        if compression_hours > 24 and vol_ratio < 0.6:
            return 1.0
        elif compression_hours > 12 and vol_ratio < 0.7:
            return 0.7
        elif compression_hours > 6 and vol_ratio < 0.8:
            return 0.4
        return 0.1
    
    def _recent_buildup_score(self, state: LongHorizonState) -> float:
        """Score based on recent OI buildup (new vulnerable positions)"""
        # Large recent OI additions = new positions that may be vulnerable
        oi_growth = state.oi_delta_24h
        
        if oi_growth > 0.10:  # 10%+ OI growth
            return 1.0
        elif oi_growth > 0.05:
            return 0.6
        elif oi_growth > 0.02:
            return 0.3
        return 0.1
    
    def _compute_direction_bias(self, state: LongHorizonState) -> float:
        """
        Compute which direction instability likely resolves.
        
        Returns: -1 (bearish) to +1 (bullish)
        """
        bias = 0.0
        
        # Funding direction
        if state.funding_z > 1.5:
            bias -= 0.3  # Longs crowded, bearish bias
        elif state.funding_z < -1.5:
            bias += 0.3  # Shorts crowded, bullish bias
        
        # Liquidation direction
        if state.liq_imbalance_4h > 0.3:
            bias -= 0.3  # Longs being liquidated
        elif state.liq_imbalance_4h < -0.3:
            bias += 0.3  # Shorts being liquidated
        
        # OI entry direction
        if state.oi_concentration_above_pct > 0.6:
            bias -= 0.2  # OI built high, bearish bias
        elif state.oi_concentration_above_pct < 0.4:
            bias += 0.2  # OI built low, bullish bias
        
        return max(-1.0, min(1.0, bias))
    
    def get_feature_vector(self, state: LongHorizonState) -> np.ndarray:
        """Extract feature vector for ML training"""
        return np.array([
            abs(state.funding_z),
            state.cumulative_funding_24h,
            abs(state.oi_entry_displacement_pct),
            state.oi_concentration_above_pct,
            abs(state.liq_imbalance_4h),
            state.liq_total_usd_24h,
            state.vol_compression_duration_h,
            state.vol_expansion_ratio,
            state.oi_delta_24h,
        ])
