"""
Bias Calculator
===============

Determines directional bias from positioning data.
This is the STRUCTURAL layer that tells us WHICH DIRECTION to look.

Components:
- Funding: Which side is paying? (crowded side is vulnerable)
- Liquidations: Which side is being forced out?
- OI: Is positioning building with or against price?
- Trend: What's the 4h/24h direction?

Output: LONG / SHORT / NEUTRAL with strength 0-1

FIXED (Audit): Added shared threshold constants for consistency between
bias calculator and individual signals.
"""
from typing import Optional
from src.stage3_v3.models import Bias, Direction


# =============================================================================
# SHARED THRESHOLD CONSTANTS (FIXED: Audit alignment)
# =============================================================================
# These should be used by both BiasCalculator and individual signals
# to ensure consistent behavior across the system.

FUNDING_Z_SIGNIFICANT = 1.0   # Z-score for meaningful funding pressure
FUNDING_Z_EXTREME = 1.5       # Z-score for strong veto (signals use this)
FUNDING_Z_DANGEROUS = 2.0     # Z-score for dangerous crowding

OI_CHANGE_SIGNIFICANT = 0.02  # 2% OI change is meaningful
OI_CHANGE_STRONG = 0.03       # 3% OI change is strong

LIQ_IMBALANCE_THRESHOLD = 0.3  # 30% imbalance is notable
LIQ_IMBALANCE_STRONG = 0.5     # 50% imbalance is strong


class BiasCalculator:
    """
    Calculate directional bias from positioning data.
    
    This runs on every update but bias changes slowly (hours).
    """
    
    def __init__(self):
        # Thresholds
        self.funding_z_threshold = 1.0  # Z-score for significant funding
        self.funding_z_extreme = 2.0
        self.liq_imbalance_threshold = 0.4  # 40% one-sided
        self.liq_imbalance_strong = 0.6
        self.oi_divergence_threshold = 0.02  # 2% OI change
        self.trend_threshold = 0.005  # 0.5% price change
        
        # Weights for combining signals
        self.weights = {
            "funding": 0.35,
            "liquidation": 0.25,
            "oi": 0.20,
            "trend": 0.20,
        }
    
    def calculate(
        self,
        funding_z: float,
        liq_imbalance_4h: float,
        oi_delta_24h: float,
        price_change_4h: float,
        price_change_24h: float,
    ) -> Bias:
        """
        Calculate directional bias.
        
        Args:
            funding_z: Funding rate z-score (positive = longs paying)
            liq_imbalance_4h: Liquidation imbalance (positive = more long liqs)
            oi_delta_24h: OI change over 24h (percentage)
            price_change_4h: Price change over 4h (percentage)
            price_change_24h: Price change over 24h (percentage)
        
        Returns:
            Bias object with direction, strength, and component scores
        """
        # Calculate component scores (-1 to +1)
        funding_score = self._funding_score(funding_z)
        liq_score = self._liquidation_score(liq_imbalance_4h)
        oi_score = self._oi_score(oi_delta_24h, price_change_24h)
        trend_score = self._trend_score(price_change_4h, price_change_24h)
        
        # Weighted combination
        total_score = (
            funding_score * self.weights["funding"] +
            liq_score * self.weights["liquidation"] +
            oi_score * self.weights["oi"] +
            trend_score * self.weights["trend"]
        )
        
        # Determine direction and strength
        # FIXED: Lower threshold (0.08 instead of 0.15) and better normalization
        if total_score > 0.08:
            direction = Direction.LONG
            strength = min(1.0, total_score / 0.35)  # Normalize to 0-1 (was /0.6)
        elif total_score < -0.08:
            direction = Direction.SHORT
            strength = min(1.0, abs(total_score) / 0.35)
        else:
            direction = Direction.NEUTRAL
            strength = 0.0
        
        # Build reason string
        reasons = []
        if abs(funding_score) > 0.3:
            if funding_score > 0:
                reasons.append(f"funding bearish (shorts crowded, z={funding_z:.1f})")
            else:
                reasons.append(f"funding bullish (longs crowded, z={funding_z:.1f})")
        if abs(liq_score) > 0.3:
            if liq_score > 0:
                reasons.append("shorts liquidating")
            else:
                reasons.append("longs liquidating")
        if abs(trend_score) > 0.3:
            if trend_score > 0:
                reasons.append("uptrend")
            else:
                reasons.append("downtrend")
        
        return Bias(
            direction=direction,
            strength=strength,
            funding_score=funding_score,
            liquidation_score=liq_score,
            oi_score=oi_score,
            trend_score=trend_score,
            reason=" + ".join(reasons) if reasons else "no clear bias",
        )
    
    def _funding_score(self, funding_z: float) -> float:
        """
        Funding score: NEGATIVE funding_z = shorts paying = bullish bias
        Because crowded shorts will eventually cover.
        
        Returns: -1 (bearish) to +1 (bullish)
        
        FIXED: Use continuous scaling instead of discrete buckets
        """
        # Continuous scaling with soft thresholds
        if abs(funding_z) < 0.5:
            return 0.0  # Dead zone for noise
        
        # Scale linearly: z=1 -> 0.4, z=2 -> 0.8, z=3 -> 1.0
        score = min(1.0, abs(funding_z) * 0.4)
        
        # Flip sign: positive funding_z (longs paying) = bearish
        return -score if funding_z > 0 else score
    
    def _liquidation_score(self, liq_imbalance: float) -> float:
        """
        Liquidation score: If longs are being liquidated, more may follow = bearish
        If shorts are being liquidated, more may follow = bullish
        
        liq_imbalance > 0 means more long liquidations
        
        Returns: -1 (bearish) to +1 (bullish)
        """
        if liq_imbalance > self.liq_imbalance_strong:
            return -0.6  # Longs getting crushed = bearish
        elif liq_imbalance > self.liq_imbalance_threshold:
            return -0.3
        elif liq_imbalance < -self.liq_imbalance_strong:
            return 0.6  # Shorts getting crushed = bullish
        elif liq_imbalance < -self.liq_imbalance_threshold:
            return 0.3
        else:
            return 0.0
    
    def _oi_score(self, oi_delta: float, price_change: float) -> float:
        """
        OI score: OI expanding against price direction = trapped positions
        
        OI up + price down = trapped longs = bearish
        OI up + price up = healthy long building = neutral to bullish
        OI down = unwind happening = follow price direction
        
        Returns: -1 (bearish) to +1 (bullish)
        """
        if abs(oi_delta) < self.oi_divergence_threshold:
            return 0.0  # No significant OI change
        
        if oi_delta > 0:  # OI expanding
            if price_change < -0.01:  # Price down while OI up = trapped longs
                return -0.4
            elif price_change > 0.01:  # Price up while OI up = healthy
                return 0.2
        else:  # OI contracting (unwind)
            if price_change > 0.01:  # Unwind + price up = short covering = bullish
                return 0.4
            elif price_change < -0.01:  # Unwind + price down = long capitulation = bearish
                return -0.4
        
        return 0.0
    
    def _trend_score(self, price_change_4h: float, price_change_24h: float) -> float:
        """
        Trend score: Simple momentum alignment.
        
        Returns: -1 (bearish) to +1 (bullish)
        
        FIXED: Lower thresholds, continuous scaling
        """
        score = 0.0
        
        # 4h trend (primary) - continuous scaling
        # 0.3% move = 0.15, 0.6% = 0.3, 1% = 0.5
        if abs(price_change_4h) > 0.002:  # > 0.2% to escape noise
            trend_4h = min(0.5, abs(price_change_4h) * 50)  # Scale: 1% = 0.5
            score += trend_4h if price_change_4h > 0 else -trend_4h
        
        # 24h trend (context) - adds up to 0.5
        if abs(price_change_24h) > 0.005:  # > 0.5% to escape noise
            trend_24h = min(0.5, abs(price_change_24h) * 25)  # Scale: 2% = 0.5
            score += trend_24h if price_change_24h > 0 else -trend_24h
        
        return max(-1.0, min(1.0, score))


# Convenience function
def calculate_bias(
    funding_z: float,
    liq_imbalance_4h: float,
    oi_delta_24h: float,
    price_change_4h: float,
    price_change_24h: float,
) -> Bias:
    """Quick bias calculation without instantiating class"""
    calc = BiasCalculator()
    return calc.calculate(
        funding_z, liq_imbalance_4h, oi_delta_24h,
        price_change_4h, price_change_24h
    )
