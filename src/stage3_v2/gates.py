"""
Stage 3 V2 Gates: Entry Permission / Veto Signals
=================================================

Gates do not generate entry signals - they ALLOW or BLOCK entries
from the primary signals.

Gates:
- OIExpansionGate: Veto crowded entries
- RegimeGate: Veto based on AI positioning regime
"""
from typing import Tuple
from src.stage3_v2.models import (
    LongHorizonState, Direction, PositioningRegime, DEFAULT_THRESHOLDS
)


class OIExpansionGate:
    """
    VETO signal: blocks entries when we would be joining a crowded trade.
    
    If OI is expanding rapidly into our intended direction with aligned
    funding, we are joining the crowd too late.
    """
    
    def __init__(self):
        self.thresholds = DEFAULT_THRESHOLDS
    
    def should_allow_entry(
        self, 
        direction: Direction, 
        state: LongHorizonState
    ) -> Tuple[bool, str]:
        """
        Check if entry should be allowed.
        
        Returns:
            (allowed: bool, reason: str)
        """
        # If going LONG and OI expanding with positive funding → crowded long
        if direction == Direction.LONG:
            if (state.oi_delta_4h > self.thresholds.oi_expansion_significant and 
                state.funding_z > 1.0):
                return False, "VETO: Crowded long - OI expanding with positive funding"
        
        # If going SHORT and OI expanding with negative funding → crowded short
        if direction == Direction.SHORT:
            if (state.oi_delta_4h > self.thresholds.oi_expansion_significant and 
                state.funding_z < -1.0):
                return False, "VETO: Crowded short - OI expanding with negative funding"
        
        return True, "OK"


class RegimeGate:
    """
    Veto based on AI-classified positioning regime.
    
    Prevents entries that conflict with detected regime:
    - CROWDED_LONG → block LONG entries
    - CROWDED_SHORT → block SHORT entries
    - UNWINDING → reduce all entries (size adjustment, not veto)
    """
    
    def should_allow_entry(
        self, 
        direction: Direction, 
        state: LongHorizonState
    ) -> Tuple[bool, str]:
        """
        Check if entry should be allowed based on positioning regime.
        
        Returns:
            (allowed: bool, reason: str)
        """
        regime = state.ai_positioning_regime
        
        # Block entries that join the crowd
        if regime == PositioningRegime.CROWDED_LONG and direction == Direction.LONG:
            return False, "VETO: AI regime is CROWDED_LONG, cannot enter LONG"
        
        if regime == PositioningRegime.CROWDED_SHORT and direction == Direction.SHORT:
            return False, "VETO: AI regime is CROWDED_SHORT, cannot enter SHORT"
        
        return True, "OK"


class VolatilityGate:
    """
    Gate based on volatility conditions.
    
    Blocks entries during extreme volatility spikes (chaos)
    or when volatility has been compressed too long (about to break).
    """
    
    def __init__(self):
        self.thresholds = DEFAULT_THRESHOLDS
    
    def should_allow_entry(
        self, 
        direction: Direction, 
        state: LongHorizonState
    ) -> Tuple[bool, str]:
        """
        Check if volatility conditions allow entry.
        
        Returns:
            (allowed: bool, reason: str)
        """
        # Extreme volatility spike - chaos, wait
        if state.vol_expansion_ratio > 3.0:
            return False, "VETO: Extreme volatility spike (>3x), wait for clarity"
        
        # Long compression about to break - risky
        if state.vol_compression_duration_h > 24 and state.vol_expansion_ratio < 0.5:
            return False, "VETO: Extended compression (>24h), breakout imminent - wait"
        
        return True, "OK"


class CombinedGate:
    """
    Combines all gates into a single evaluation.
    Entry is blocked if ANY gate vetoes.
    """
    
    def __init__(self):
        self.oi_gate = OIExpansionGate()
        self.regime_gate = RegimeGate()
        self.vol_gate = VolatilityGate()
    
    def should_allow_entry(
        self, 
        direction: Direction, 
        state: LongHorizonState
    ) -> Tuple[bool, str]:
        """
        Check all gates and return combined result.
        
        Returns:
            (allowed: bool, reason: str) - reason is first veto if blocked
        """
        gates = [
            ("OI", self.oi_gate),
            ("Regime", self.regime_gate),
            ("Volatility", self.vol_gate),
        ]
        
        for name, gate in gates:
            allowed, reason = gate.should_allow_entry(direction, state)
            if not allowed:
                return False, f"[{name}] {reason}"
        
        return True, "All gates passed"
