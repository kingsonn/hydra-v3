"""
Stage 3 Thesis Signals
5 independent signals that detect market pressures

Each signal returns:
- Signal(direction, confidence, name) if condition met
- None if no signal
"""
from typing import Optional
from src.stage3.models import Signal, Direction, ThesisState


# ============================================================
# SIGNAL 1: FUNDING SQUEEZE
# ============================================================
# Idea: Crowded side paying → vulnerable
# Used in: late compression, early expansion

def funding_squeeze(state: ThesisState) -> Optional[Signal]:
    """
    Detect funding squeeze conditions
    
    Crowded side paying high funding AND price not moving their way
    = vulnerable to squeeze
    """
    if state.regime == "CHOP":
        return None
    
    # Longs crowded & price stalling/falling
    if (
        state.funding_z > 1.2 and
        state.oi_delta_5m > 0.01 and
        state.price_change_5m <= 0
    ):
        return Signal(Direction.SHORT, 0.7, "Funding squeeze (crowded longs)")
    
    # Shorts crowded & price stalling/rising
    if (
        state.funding_z < -1.2 and
        state.oi_delta_5m > 0.01 and
        state.price_change_5m >= 0
    ):
        return Signal(Direction.LONG, 0.7, "Funding squeeze (crowded shorts)")
    
    return None


# ============================================================
# SIGNAL 2: LIQUIDATION EXHAUSTION
# ============================================================
# Idea: Forced sellers exhausted → bounce
# Only valid in expansion, never in chop

def liquidation_exhaustion(state: ThesisState) -> Optional[Signal]:
    """
    Detect liquidation exhaustion
    
    Heavy liquidations + absorption = forced sellers exhausted
    Price likely to reverse
    """
    if state.regime != "EXPANSION":
        return None
    
    # Need significant absorption to confirm exhaustion
    if state.absorption_z < 1.0:
        return None
    
    # Heavy long liquidations = bounce opportunity
    if state.liq_imbalance > 0.6:
        return Signal(Direction.LONG, 0.65, "Long liquidation exhaustion")
    
    # Heavy short liquidations = fade opportunity
    if state.liq_imbalance < -0.6:
        return Signal(Direction.SHORT, 0.65, "Short liquidation exhaustion")
    
    return None


# ============================================================
# SIGNAL 3: OI DIVERGENCE
# ============================================================
# Idea: Price moving without participation → weak move
# This is a FADE signal, not a chase signal

# Minimum price move threshold (relative)
MIN_PRICE_MOVE = 0.002  # 0.2%

def oi_divergence(state: ThesisState) -> Optional[Signal]:
    """
    Detect OI divergence
    
    Price moving but OI declining = positions closing, not new conviction
    Move is weak and likely to fade
    """
    # Need meaningful price move to evaluate
    if abs(state.price_change_5m) < MIN_PRICE_MOVE:
        return None
    
    # Price up but OI down = weak rally
    if state.price_change_5m > 0 and state.oi_delta_5m < -0.01:
        return Signal(Direction.SHORT, 0.55, "OI divergence (weak rally)")
    
    # Price down but OI down = weak selloff
    if state.price_change_5m < 0 and state.oi_delta_5m < -0.01:
        return Signal(Direction.LONG, 0.55, "OI divergence (weak selloff)")
    
    return None


# ============================================================
# SIGNAL 4: CROWDING FADE
# ============================================================
# Idea: Everyone on one side → fade
# Stronger in: compression, late expansion

def crowding_fade(state: ThesisState) -> Optional[Signal]:
    """
    Detect crowding conditions for fade
    
    Extreme one-sided positioning = mean reversion opportunity
    """
    if state.regime == "CHOP":
        return None
    
    # Don't fade too early in expansion
    if state.regime == "EXPANSION" and state.time_in_regime < 120:
        return None
    
    # Extremely crowded longs
    if state.funding_z > 1.5:
        return Signal(Direction.SHORT, 0.6, "Crowded longs")
    
    # Extremely crowded shorts
    if state.funding_z < -1.5:
        return Signal(Direction.LONG, 0.6, "Crowded shorts")
    
    return None


# ============================================================
# SIGNAL 5: FUNDING CARRY (SPECIAL)
# ============================================================
# Idea: In quiet ranges, get paid to wait
# Only active in compression

def funding_carry(state: ThesisState) -> Optional[Signal]:
    """
    Detect funding carry opportunity
    
    In quiet compression, take the side that gets paid funding
    Low risk while waiting for breakout
    """
    if state.regime != "COMPRESSION":
        return None
    
    # Need meaningful funding to earn
    if abs(state.funding_z) < 0.8:
        return None
    
    # Positive funding = go short to earn
    if state.funding_z > 0:
        return Signal(Direction.SHORT, 0.5, "Funding carry (range)")
    
    # Negative funding = go long to earn
    if state.funding_z < 0:
        return Signal(Direction.LONG, 0.5, "Funding carry (range)")
    
    return None
