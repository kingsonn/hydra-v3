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


# ============================================================
# SIGNAL 6: INVENTORY LOCK (ILI)
# ============================================================
# Idea: Detect trapped inventory via aggression absorbed at key levels
# Core concept: OI rising + price flat + aggression absorbed = inventory lock

def _ili_regime_ok(state: ThesisState) -> bool:
    """
    Regime gate for inventory lock signal.
    Valid in: COMPRESSION, late EXPANSION (failed continuation)
    """
    if state.regime == "COMPRESSION":
        return True
    
    # Late expansion = failed continuation
    if state.regime == "EXPANSION" and state.time_in_regime > 120:
        return True
    
    return False


def _ili_location_ok(state: ThesisState) -> bool:
    """
    Location gate: Must be at LVN or value edge.
    Dynamic threshold based on volatility.
    """
    lvn_threshold = 0.3 * min(1.0, state.vol_expansion_ratio)
    
    at_lvn = state.dist_lvn <= lvn_threshold
    at_edge = (
        state.price >= state.vah or
        state.price <= state.val or
        state.dist_poc >= 0.8
    )
    
    return at_lvn and at_edge


def _ili_inventory_build(state: ThesisState) -> bool:
    """
    Inventory build detection: OI rising but price not moving.
    This is the structural core - new positions entering but no follow through.
    """
    return (
        state.oi_delta_5m > 0.01 and  # New positions coming in
        abs(state.price_change_5m) < 0.0015  # ~0.15% - price stalling
    )


def _ili_aggression(state: ThesisState) -> bool:
    """
    Aggression detection (Fabio's "bubbles").
    Significant order flow pressure being applied.
    """
    return (
        state.moi_z > 2.0 and
        state.delta_vel_z > 1.5 and
        state.aggression_persistence > 1.2
    )


def _ili_absorption(state: ThesisState) -> bool:
    """
    Absorption detection - MOST IMPORTANT.
    Aggression hits liquidity, liquidity refills, price does not advance.
    """
    return (
        state.absorption_z > 1.0 and
        state.refill_rate > 0 and
        not state.liquidity_sweep
    )


def _ili_aggression_valid(state: ThesisState) -> bool:
    """
    Anti-trap filter: Reject noisy/fake aggression.
    Prevents: algo spoofing, retail chop, microstructure noise.
    """
    # High flip noise with aggression = fake/noisy
    if state.moi_z > 2.0 and state.flip_noise > 3.0:
        return False
    return True


def inventory_lock(state: ThesisState) -> Optional[Signal]:
    """
    SIGNAL 6: INVENTORY LOCK (ILI)
    
    Detects trapped inventory via aggression absorbed at key levels.
    
    Logic:
    - OI rising + price flat = positions building
    - Aggression happening = someone pushing
    - Absorption confirmed = liquidity absorbing the push
    - No liquidity sweep = absorption successful
    
    Result: The aggressive side is trapped.
    """
    # Gate checks
    if not _ili_regime_ok(state):
        return None
    if not _ili_location_ok(state):
        return None
    if not _ili_inventory_build(state):
        return None
    if not _ili_aggression(state):
        return None
    if not _ili_absorption(state):
        return None
    if not _ili_aggression_valid(state):
        return None
    
    # Determine direction based on who is aggressive
    # Positive MOI = aggressive buyers → they get trapped → SHORT
    if state.moi_1s > 0:
        return Signal(
            Direction.SHORT, 
            0.72, 
            "ILI: Aggressive longs absorbed at LVN"
        )
    
    # Negative MOI = aggressive sellers → they get trapped → LONG
    if state.moi_1s < 0:
        return Signal(
            Direction.LONG, 
            0.72, 
            "ILI: Aggressive shorts absorbed at LVN"
        )
    
    return None


# ============================================================
# SIGNAL 7: FAILED ACCEPTANCE REVERSAL (FAR)
# ============================================================
# Idea: Price tries to accept outside value area, fails, order flow confirms rejection
# Classic Auction Market Theory - high win rate, slow, very clean
# Works best: COMPRESSION regime, low-mid volatility, no strong trend

def failed_acceptance_reversal(state: ThesisState) -> Optional[Signal]:
    """
    SIGNAL 7: FAILED ACCEPTANCE REVERSAL (FAR)
    
    Detects failed attempts to accept outside value area.
    
    Logic:
    - Price tried to leave value (VAH/VAL) but failed to accept
    - Must be near value edge
    - Absorption confirms rejection
    - MOI shows enough pressure in expected direction
    
    Result: Fade the failed breakout.
    """
    # Only valid in COMPRESSION (balanced market)
    if state.regime != "COMPRESSION":
        return None
    
    # Price tried to accept outside but failed
    # If acceptance_outside_value is True, price DID accept → no signal
    if state.acceptance_outside_value:
        return None
    
    # Must be at value edge to have attempted breakout
    at_vah = state.price >= state.vah and state.vah > 0
    at_val = state.price <= state.val and state.val > 0
    
    if not at_vah and not at_val:
        return None
    
    # Absorption must confirm rejection
    if state.absorption_z < 0.8:
        return None
    
    # Direction based on which edge we're at
    if at_vah:
        # At VAH, failed to accept above → SHORT
        # Need some buy pressure that's being absorbed
        if state.moi_z < 0.8:  # Not enough buy pressure to absorb
            return None
        return Signal(
            Direction.SHORT,
            0.70,
            "FAR: Failed acceptance above VAH"
        )
    
    if at_val:
        # At VAL, failed to accept below → LONG
        # Need some sell pressure that's being absorbed
        if state.moi_z < 0.8:  # Not enough sell pressure to absorb
            return None
        return Signal(
            Direction.LONG,
            0.70,
            "FAR: Failed acceptance below VAL"
        )
    
    return None
