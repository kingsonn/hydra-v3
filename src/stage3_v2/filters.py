"""
Stage 3 V2 Filters: Signal Parameter Adjustment
===============================================

Filters modify signal parameters based on market context.
They do NOT generate or block signals - they adjust:
- Position size
- Stop width
- Target
- Holding time expectation
"""
from src.stage3_v2.models import (
    LongHorizonState, PositioningSignal, PositioningRegime, DEFAULT_THRESHOLDS
)


class VolatilityFilter:
    """
    Adjusts trade parameters based on volatility regime.
    
    HIGH VOL: Widen stops, reduce size, shorten holding expectation
    LOW VOL: Tighten stops, increase size, longer holding expectation
    """
    
    def __init__(self):
        self.thresholds = DEFAULT_THRESHOLDS
    
    def adjust(self, signal: PositioningSignal, state: LongHorizonState) -> PositioningSignal:
        """Adjust signal parameters based on volatility"""
        vol_ratio = state.vol_expansion_ratio
        
        if vol_ratio > 2.0:
            # High volatility: defensive parameters
            signal.stop_pct *= 1.5
            signal.size_multiplier *= 0.5
            signal.expected_holding_hours *= 0.6
            signal.reason += " [HIGH_VOL: widened stops 1.5x, reduced size 0.5x]"
        
        elif vol_ratio > 1.5:
            # Elevated volatility: slightly defensive
            signal.stop_pct *= 1.2
            signal.size_multiplier *= 0.75
            signal.reason += " [ELEVATED_VOL: widened stops 1.2x, reduced size 0.75x]"
        
        elif vol_ratio < 0.7:
            # Low volatility: aggressive parameters
            signal.stop_pct *= 0.7
            signal.size_multiplier *= 1.3
            signal.expected_holding_hours *= 1.4
            signal.reason += " [LOW_VOL: tightened stops 0.7x, increased size 1.3x]"
        
        return signal


class InstabilityFilter:
    """
    Adjusts parameters based on AI instability score.
    
    HIGH INSTABILITY: Increase size (opportunity), widen stops (volatility coming)
    LOW INSTABILITY: Standard parameters
    """
    
    def adjust(self, signal: PositioningSignal, state: LongHorizonState) -> PositioningSignal:
        """Adjust signal parameters based on instability"""
        score = state.ai_instability_score
        
        if score > 0.7:
            # High instability: big opportunity, but more volatile
            signal.size_multiplier *= 1.3
            signal.stop_pct *= 1.2
            signal.confidence = min(0.85, signal.confidence + 0.05)
            signal.reason += f" [HIGH_INSTABILITY: {score:.2f}, boosted size 1.3x]"
        
        elif score > 0.5:
            # Moderate instability
            signal.size_multiplier *= 1.1
            signal.reason += f" [MOD_INSTABILITY: {score:.2f}]"
        
        elif score < 0.3:
            # Low instability: reduce confidence
            signal.confidence = max(0.50, signal.confidence - 0.05)
            signal.reason += f" [LOW_INSTABILITY: {score:.2f}, reduced confidence]"
        
        return signal


class HistoricalContextFilter:
    """
    Adjusts parameters based on historical context similarity.
    
    If similar historical setups had high win rate, boost confidence.
    If similar setups had low win rate, reduce confidence or size.
    """
    
    def adjust(self, signal: PositioningSignal, state: LongHorizonState) -> PositioningSignal:
        """Adjust signal parameters based on historical win rate"""
        win_rate = state.ai_historical_win_rate
        
        if win_rate > 0.65:
            # Strong historical support
            signal.confidence = min(0.85, signal.confidence + 0.08)
            signal.size_multiplier *= 1.2
            signal.reason += f" [HISTORICAL: {win_rate:.0%} win rate, boosted]"
        
        elif win_rate > 0.55:
            # Moderate historical support
            signal.confidence = min(0.80, signal.confidence + 0.03)
            signal.reason += f" [HISTORICAL: {win_rate:.0%} win rate]"
        
        elif win_rate < 0.40:
            # Weak historical support - caution
            signal.confidence = max(0.45, signal.confidence - 0.10)
            signal.size_multiplier *= 0.7
            signal.reason += f" [HISTORICAL WARNING: {win_rate:.0%} win rate, reduced]"
        
        return signal


class AnomalyFilter:
    """
    Adjusts parameters based on anomaly detection.
    
    If current state is anomalous (unusual), it's either a big opportunity
    or a trap. Increase stops but maintain size.
    """
    
    def adjust(self, signal: PositioningSignal, state: LongHorizonState) -> PositioningSignal:
        """Adjust signal parameters based on anomaly score"""
        anomaly = state.ai_anomaly_score
        
        if anomaly > 0.7:
            # Highly anomalous state - widen stops, flag for review
            signal.stop_pct *= 1.3
            signal.reason += f" [ANOMALY: {anomaly:.2f}, widened stops - unusual state]"
        
        elif anomaly > 0.5:
            # Moderately anomalous
            signal.stop_pct *= 1.1
            signal.reason += f" [ANOMALY: {anomaly:.2f}]"
        
        return signal


class CombinedFilter:
    """
    Applies all filters in sequence.
    Order matters: volatility → instability → historical → anomaly
    """
    
    def __init__(self):
        self.volatility = VolatilityFilter()
        self.instability = InstabilityFilter()
        self.historical = HistoricalContextFilter()
        self.anomaly = AnomalyFilter()
    
    def adjust(self, signal: PositioningSignal, state: LongHorizonState) -> PositioningSignal:
        """Apply all filters to signal"""
        signal = self.volatility.adjust(signal, state)
        signal = self.instability.adjust(signal, state)
        signal = self.historical.adjust(signal, state)
        signal = self.anomaly.adjust(signal, state)
        
        # Clamp final values to reasonable ranges
        signal.stop_pct = max(0.01, min(0.10, signal.stop_pct))  # 1-10%
        signal.target_pct = max(0.02, min(0.15, signal.target_pct))  # 2-15%
        signal.size_multiplier = max(0.25, min(2.0, signal.size_multiplier))  # 0.25x-2x
        signal.confidence = max(0.45, min(0.90, signal.confidence))  # 45-90%
        signal.expected_holding_hours = max(6, min(96, signal.expected_holding_hours))  # 6h-4d
        
        return signal
