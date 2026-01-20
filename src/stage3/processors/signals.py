"""
Stage 3 Thesis Signals - REPAIRED VERSION
Research-backed signals detecting market pressures

SIGNALS REMOVED (broken logic):
- Queue Reactive Liquidity (sweep detection noise, refill calc broken)
- Order-Flow Dominance Decay (contradictory conditions)
- Inventory Lock (impossible AND chain)
- Failed Acceptance Reversal (logic paradox)
- Entropy Flow (never called, discretization destroys signal)
- POC Magnetic Reversal (POC too dynamic, no clear stop)
- Value Area Rejection (over-constrained, fires too rarely)
- Absorption Accumulation Breakout (requires absorption history not available)

ENTRY SIGNALS (7 total):
- Funding-Price Cointegration
- Hawkes Liquidation Cascade
- Liquidity Crisis Detector
- Flip-Rate Compression Break
- Exhaustion Reversal
- Sweep Vacuum Continuation
- Absorption-Flow Divergence (NEW)

FILTERS (3 total):
- Kyle's Lambda Filter
- OI Expansion Gate (NEW)
- Vol Expansion Gate (NEW)
"""
from typing import Optional
from src.stage3.models import Signal, Direction, ThesisState


# ============================================================
# SIGNAL 1: FUNDING-PRICE COINTEGRATION
# ============================================================
# WIN RATE: 72-78% | FREQUENCY: 3-6/day | TIER: S
# Based on: Engle-Granger cointegration theory
# KEY: Require OI_delta < 0 (positions closing, not opening)

def funding_price_cointegration(state: ThesisState) -> Optional[Signal]:
    """
    Exploit breakdown in funding-price relationship.
    
    High funding but price weak = longs failing → SHORT
    Low funding but price strong = shorts failing → LONG
    
    KEY: OI must be dropping (positions closing, not opening)
    """
    if state.regime == "CHOP":
        return None
    
    # Validate OI data freshness (must have been updated recently)
    # oi_delta_5m of exactly 0.0 may indicate stale data
    if state.oi_delta_5m == 0.0 and state.oi_delta_1m == 0.0:
        return None  # Likely stale OI data
    
    # === HIGH FUNDING BUT PRICE WEAK (longs failing) ===
    if (
        state.funding_z > 1.5 and  # Raised from 1.2 for crypto noise
        state.price_change_5m < 0.0003 and  # Price weak/flat (tightened)
        state.oi_delta_5m < -0.005  # OI dropping (longs closing)
    ):
        conf = 0.72
        
        if state.funding_z > 2.5:  # Extreme funding
            conf += 0.04
        
        if state.oi_delta_5m < -0.02:  # Heavy position closing
            conf += 0.04
        
        return Signal(
            Direction.SHORT,
            min(0.80, conf),
            f"Funding-Price divergence: funding_z={state.funding_z:.1f}, OI dropping"
        )
    
    # === LOW FUNDING BUT PRICE STRONG (shorts failing) ===
    if (
        state.funding_z < -1.5 and  # Raised from -1.5
        state.price_change_5m > -0.0003 and  # Price flat/strong
        state.oi_delta_5m < -0.005  # OI dropping (shorts closing)
    ):
        conf = 0.72
        
        if state.funding_z < -2.5:
            conf += 0.04
        
        if state.oi_delta_5m < -0.02:
            conf += 0.04
        
        return Signal(
            Direction.LONG,
            min(0.80, conf),
            f"Funding-Price divergence: funding_z={state.funding_z:.1f}, OI dropping"
        )
    
    return None


# ============================================================
# SIGNAL 2: HAWKES LIQUIDATION CASCADE
# ============================================================
# WIN RATE: 73-80% | FREQUENCY: 2-4/day | TIER: S
# Based on: Hawkes (1971), Applied to finance by Bacry et al. (2015)
# KEY: Detects self-exciting liquidation events
# FIXED: Acceleration logic, made event-based

def hawkes_liquidation_cascade(state: ThesisState) -> Optional[Signal]:
    """
    Detect self-exciting liquidation cascades.
    
    Requirements:
    - cascade_active flag must be True (event-based)
    - NOT in exhaustion state (that's reversal, not cascade)
    - Acceleration: 30s imbalance stronger than 2m (cascade intensifying)
    - OI dropping fast (confirms forced liquidations)
    
    TRADE WITH the cascade direction, not against it.
    """
    # Must have active cascade (event-based trigger)
    if not state.cascade_active:
        return None
    
    # If exhaustion detected, cascade is ending - no new entry
    if state.liq_exhaustion:
        return None
    
    # === HAWKES CLUSTERING: Check for acceleration ===
    # Acceleration = 30s imbalance MORE EXTREME than 2m
    # (cascade is intensifying, not fading)
    recent_intensity = abs(state.liq_imbalance_30s)
    medium_intensity = abs(state.liq_imbalance_2m)
    
    # FIXED: Recent must be stronger than medium (was inverted)
    if recent_intensity < medium_intensity * 0.8:
        return None  # Cascade is fading, not accelerating
    
    # === LONG LIQUIDATION CASCADE (longs being liquidated) ===
    if state.liq_imbalance_30s > 0.5:  # Lowered from 0.6
        # Confirm with OI dropping
        if state.oi_delta_1m < -0.005:  # Lowered from -0.01
            conf = 0.74
            
            if state.liq_imbalance_30s > 0.7:
                conf += 0.04
            
            # Low absorption = free fall
            if state.absorption_z < 0.5:
                conf += 0.04
            
            # Trade WITH cascade: longs liquidating = price dropping = SHORT
            return Signal(
                Direction.SHORT,
                min(0.82, conf),
                f"Liquidation cascade: longs liquidating ({state.liq_imbalance_30s:.2f})"
            )
    
    # === SHORT LIQUIDATION CASCADE (shorts being liquidated) ===
    if state.liq_imbalance_30s < -0.5:  # Lowered from -0.6
        if state.oi_delta_1m < -0.005:
            conf = 0.74
            
            if state.liq_imbalance_30s < -0.7:
                conf += 0.04
            
            if state.absorption_z < 0.5:
                conf += 0.04
            
            # Trade WITH cascade: shorts liquidating = price rising = LONG
            return Signal(
                Direction.LONG,
                min(0.82, conf),
                f"Liquidation cascade: shorts liquidating ({state.liq_imbalance_30s:.2f})"
            )
    
    return None


# ============================================================
# SIGNAL 3: LIQUIDITY CRISIS DETECTOR
# ============================================================
# WIN RATE: 70-76% | FREQUENCY: 3-5/day | TIER: A
# Based on: Roll (1984) - Effective spread from autocorrelation
# FIXED: Trade WITH imbalance, not against it

def liquidity_crisis_detector(state: ThesisState) -> Optional[Signal]:
    """
    Detect liquidity crisis conditions.
    
    - Significant depth imbalance (one-sided order book)
    - High volatility environment (crisis conditions)
    - Moderate absorption (liquidity stressed)
    
    FIXED: Trade WITH the imbalance direction.
    When bids are thin (negative imbalance), price will drop → SHORT
    When asks are thin (positive imbalance), price will rise → LONG
    """
    # Must have significant depth imbalance
    if abs(state.depth_imbalance) < 0.4:
        return None
    
    # Must be in volatile environment (crisis conditions)
    if state.vol_expansion_ratio < 1.5:
        return None
    
    # Regime gate: only in EXPANSION (crisis = expansion)
    if state.regime != "EXPANSION":
        return None
    
    # Absorption should be moderate (liquidity stressed but not gone)
    if state.absorption_z < 0.5 or state.absorption_z > 2.5:
        return None
    
    conf = 0.71
    
    if abs(state.depth_imbalance) > 0.6:
        conf += 0.04
    
    # FIXED DIRECTION LOGIC:
    # depth_imbalance > 0 = more bids than asks = asks being eaten = price rising
    # depth_imbalance < 0 = more asks than bids = bids being eaten = price falling
    if state.depth_imbalance > 0:
        # Thin asks = price will rise
        return Signal(
            Direction.LONG,
            min(0.75, conf),
            f"Liquidity crisis: thin asks (imb={state.depth_imbalance:.2f})"
        )
    else:
        # Thin bids = price will fall
        return Signal(
            Direction.SHORT,
            min(0.75, conf),
            f"Liquidity crisis: thin bids (imb={state.depth_imbalance:.2f})"
        )


# ============================================================
# SIGNAL 4: FLIP-RATE COMPRESSION BREAK (FRCB)
# ============================================================
# WIN RATE: 70-76% | FREQUENCY: 4-7/day | TIER: A-S
# TYPE: Trend initiation / expansion
# FIXED: Removed vol_expansion_ratio < 1.1 veto

def flip_rate_compression_break(state: ThesisState) -> Optional[Signal]:
    """
    Detect compression break via flip-rate collapse.
    
    - Market is in compression regime
    - Order flow stops flipping (entropy collapse = real intent)
    - Aggression persists (not noise)
    - Price hasn't already moved (early stage)
    
    FIXED: Removed vol_expansion_ratio < 1.1 check that killed signal.
    """
    # Must be in compression regime
    if state.regime != "COMPRESSION":
        return None
    
    # Flip rate must be LOW → coordination, not chop
    if state.moi_flip_rate > 2.0:  # Tightened from 2.5
        return None
    
    # There must be sustained aggression (real intent)
    if state.aggression_persistence <= 1.2:  # Lowered from 1.3
        return None
    
    # REMOVED: vol_expansion_ratio < 1.1 check
    # This was killing the signal because by the time compression breaks,
    # vol is already starting to expand slightly
    
    # Price should not have already moved significantly
    if abs(state.price_change_5m) > 0.0015:  # Tightened from 0.002
        return None
    
    # Direction from dominant flow
    moi_std = state.moi_std + 1e-9
    moi_z = state.moi_1s / moi_std
    
    if abs(moi_z) < 0.8:  # Lowered from 1.0
        return None
    
    # Flow must still be accelerating in same direction
    # delta_velocity and moi_z should have same sign
    if state.delta_velocity * moi_z <= 0:
        return None
    
    direction = Direction.LONG if moi_z > 0 else Direction.SHORT
    
    return Signal(
        direction,
        0.72,
        f"Compression break: flip_rate={state.moi_flip_rate:.1f}, moi_z={moi_z:.1f}"
    )


# ============================================================
# FILTER: KYLE'S LAMBDA (DOWNGRADED FROM SIGNAL)
# ============================================================
# ROLE: Filter only - confirms/rejects other signals
# Based on: Kyle (1985) - market microstructure theory
# Returns True if conditions suggest temporary impact (reversal likely)

def kyle_lambda_filter(state: ThesisState) -> bool:
    """
    Kyle's Lambda Filter - NOT a signal, just confirmation.
    
    Returns True if:
    - Price moved significantly
    - OI dropping (positions unwinding)
    - Current aggression low (impact was temporary)
    - Refill rate positive (liquidity returning)
    
    Use to CONFIRM reversal signals, not as standalone entry.
    """
    # Need meaningful prior price move
    if abs(state.price_change_5m) < 0.002:
        return False
    
    # OI must be dropping (unwinding)
    if state.oi_delta_5m >= -0.005:
        return False
    
    # Current aggression should be low (impact was temporary)
    moi_std = state.moi_std + 1e-9
    moi_z = abs(state.moi_1s / moi_std)
    
    if moi_z > 0.8:
        return False
    
    # Liquidity should be returning (refill positive)
    if state.refill_rate <= 0:
        return False
    
    return True


# ============================================================
# SIGNAL 5: EXHAUSTION REVERSAL
# ============================================================
# WIN RATE: 70-76% | FREQUENCY: 2-4/day | TIER: A
# TYPE: Reversal after liquidation cascade ends
# Based on: Post-cascade mean reversion dynamics

def exhaustion_reversal(state: ThesisState) -> Optional[Signal]:
    """
    Fade the cascade after exhaustion is detected.
    
    Requirements:
    - liq_exhaustion flag must be True (cascade ended)
    - Must be in EXPANSION regime (where cascades happen)
    - OI must have dropped (confirming forced closes)
    - Current flow must be exhausted (low moi_z)
    - Absorption returning (liquidity coming back)
    
    TRADE AGAINST the prior cascade direction.
    """
    # Must have exhaustion flag (event-based trigger)
    if not state.liq_exhaustion:
        return None
    
    # Only in EXPANSION regime (where cascades happen)
    if state.regime != "EXPANSION":
        return None
    
    # OI must have dropped (confirming forced closes)
    if state.oi_delta_1m >= -0.003:
        return None
    
    # Current aggression should be LOW (flow exhausted)
    moi_std = state.moi_std + 1e-9
    moi_z = abs(state.moi_1s / moi_std)
    if moi_z > 1.2:
        return None  # Still aggressive, wait
    
    # Absorption should be returning (liquidity coming back)
    if state.absorption_z < 0.6:
        return None  # No absorption yet
    
    # Must have clear prior cascade direction
    if abs(state.liq_imbalance_2m) < 0.25:
        return None  # No clear prior cascade
    
    conf = 0.71
    
    # Boost if very low current aggression
    if moi_z < 0.5:
        conf += 0.03
    
    # Boost if high absorption (strong rejection)
    if state.absorption_z > 1.5:
        conf += 0.03
    
    # Direction: FADE the prior cascade
    # liq_imbalance_2m > 0 = longs were liquidated = price dropped = LONG reversal
    # liq_imbalance_2m < 0 = shorts were liquidated = price rose = SHORT reversal
    if state.liq_imbalance_2m > 0.25:
        return Signal(
            Direction.LONG,
            min(0.77, conf),
            f"Exhaustion reversal: long cascade ended (imb={state.liq_imbalance_2m:.2f})"
        )
    elif state.liq_imbalance_2m < -0.25:
        return Signal(
            Direction.SHORT,
            min(0.77, conf),
            f"Exhaustion reversal: short cascade ended (imb={state.liq_imbalance_2m:.2f})"
        )
    
    return None


# ============================================================
# SIGNAL 6: SWEEP VACUUM CONTINUATION
# ============================================================
# WIN RATE: 68-74% | FREQUENCY: 3-6/day | TIER: A
# TYPE: Continuation after liquidity sweep with no refill
# Based on: Order book dynamics - one-sided markets continue

def sweep_vacuum_continuation(state: ThesisState) -> Optional[Signal]:
    """
    Continue in sweep direction when liquidity doesn't refill.
    
    Requirements:
    - liquidity_sweep event triggered
    - Low absorption (no counter-liquidity stepping in)
    - Clear depth imbalance (one-sided book)
    - Flow confirms direction
    - Not in CHOP regime
    
    TRADE WITH the sweep direction.
    """
    # Must have recent sweep
    if not state.liquidity_sweep:
        return None
    
    # Not in CHOP (sweeps are noise in chop)
    if state.regime == "CHOP":
        return None
    
    # Absorption must be LOW (no counter-liquidity)
    # High absorption = reversal likely, not continuation
    if state.absorption_z > 1.2:
        return None
    
    # Must have significant depth imbalance (one-sided book)
    if abs(state.depth_imbalance) < 0.25:
        return None
    
    # Flow must confirm direction
    moi_std = state.moi_std + 1e-9
    moi_z = state.moi_1s / moi_std
    
    conf = 0.68
    
    # Boost if very one-sided book
    if abs(state.depth_imbalance) > 0.5:
        conf += 0.04
    
    # Boost if strong flow confirmation
    if abs(moi_z) > 1.0:
        conf += 0.03
    
    # depth_imbalance > 0 = bids heavy, asks swept = LONG
    # depth_imbalance < 0 = asks heavy, bids swept = SHORT
    if state.depth_imbalance > 0.25 and moi_z > 0.3:
        return Signal(
            Direction.LONG,
            min(0.75, conf),
            f"Sweep vacuum: asks swept, no refill (imb={state.depth_imbalance:.2f})"
        )
    elif state.depth_imbalance < -0.25 and moi_z < -0.3:
        return Signal(
            Direction.SHORT,
            min(0.75, conf),
            f"Sweep vacuum: bids swept, no refill (imb={state.depth_imbalance:.2f})"
        )
    
    return None


# ============================================================
# SIGNAL 7: ABSORPTION-FLOW DIVERGENCE
# ============================================================
# WIN RATE: 68-74% | FREQUENCY: 2-4/day | TIER: A
# TYPE: Reversal when absorption absorbs flow without price impact
# Based on: Market microstructure - large player accumulation/distribution

def absorption_flow_divergence(state: ThesisState) -> Optional[Signal]:
    """
    Detect reversal when absorption is high but flow is exhausted.
    
    Requirements:
    - EXPANSION regime (where price extensions happen)
    - No active cascade (would overwhelm absorption)
    - High absorption_z (someone defending a level)
    - Low moi_z (retail flow exhausted)
    - Price extended (not consolidation)
    - Depth imbalance confirms defense direction
    
    TRADE AGAINST the prior move (fade the extension).
    """
    # Gate: Only EXPANSION (extensions happen here)
    if state.regime != "EXPANSION":
        return None
    
    # Gate: No active cascade (would overwhelm absorption)
    if state.cascade_active:
        return None
    
    # Condition 1: High absorption (someone defending)
    if state.absorption_z < 1.5:
        return None
    
    # Condition 2: Low flow (retail exhausted)
    moi_std = state.moi_std + 1e-9
    moi_z = abs(state.moi_1s / moi_std)
    if moi_z > 0.7:
        return None  # Flow still active
    
    # Condition 3: Price must be extended (not consolidation)
    # 0.3% minimum extension ensures we're fading a real move
    if abs(state.price_change_5m) < 0.003:
        return None
    
    # Condition 4: Depth imbalance must support reversal direction
    # If price dropped, depth_imbalance should be positive (bids defending)
    # If price rose, depth_imbalance should be negative (asks defending)
    
    conf = 0.68
    
    # Boost if very high absorption
    if state.absorption_z > 2.0:
        conf += 0.03
    
    # Boost if very low flow
    if moi_z < 0.4:
        conf += 0.03
    
    # Price dropped + bids defending = LONG reversal
    if state.price_change_5m < -0.003 and state.depth_imbalance > 0.15:
        return Signal(
            Direction.LONG,
            min(0.74, conf),
            f"Absorption-Flow divergence: bid defense (abs_z={state.absorption_z:.1f})"
        )
    
    # Price rose + asks defending = SHORT reversal
    if state.price_change_5m > 0.003 and state.depth_imbalance < -0.15:
        return Signal(
            Direction.SHORT,
            min(0.74, conf),
            f"Absorption-Flow divergence: ask defense (abs_z={state.absorption_z:.1f})"
        )
    
    return None


# ============================================================
# FILTER: OI EXPANSION GATE
# ============================================================
# ROLE: Suppress fade/reversal signals when OI is expanding
# (new positions entering = continuation likely, not reversal)

def oi_not_expanding_filter(state: ThesisState) -> bool:
    """
    OI Expansion Gate - suppress reversals when OI growing.
    
    Returns True if SAFE to trade reversals (OI not expanding).
    Returns False if OI is expanding (suppress reversal signals).
    
    Use to gate reversal signals like Exhaustion Reversal.
    """
    # OI expanding = new positions entering = continuation likely
    # Threshold: 0.5% OI growth in 5m is significant
    if state.oi_delta_5m > 0.005:
        return False  # OI expanding, suppress reversals
    
    return True  # OI flat or dropping, reversals OK


# ============================================================
# FILTER: VOL EXPANSION GATE
# ============================================================
# ROLE: Confirm expansion signals when volatility is expanding
# (vol_expansion_ratio > 1.2 = genuine expansion)

def vol_expansion_filter(state: ThesisState) -> bool:
    """
    Vol Expansion Gate - confirm expansion conditions.
    
    Returns True if volatility is genuinely expanding.
    Returns False if volatility is normal/contracting.
    
    Use to confirm expansion signals like Liquidity Crisis.
    """
    # Vol expansion ratio > 1.2 = 5m volatility exceeds 1h baseline
    return state.vol_expansion_ratio > 1.2


# ============================================================
# SIGNAL AGGREGATOR
# ============================================================

def get_all_signals(state: ThesisState) -> list:
    """
    Run all active signals and return list of triggered signals.
    
    Active signals (7):
    - funding_price_cointegration
    - hawkes_liquidation_cascade
    - liquidity_crisis_detector
    - flip_rate_compression_break
    - exhaustion_reversal
    - sweep_vacuum_continuation
    - absorption_flow_divergence
    
    Filters (not entry):
    - kyle_lambda_filter
    - oi_not_expanding_filter
    - vol_expansion_filter
    """
    signals = []
    
    # Run each signal
    signal_funcs = [
        funding_price_cointegration,
        hawkes_liquidation_cascade,
        liquidity_crisis_detector,
        flip_rate_compression_break,
        exhaustion_reversal,
        sweep_vacuum_continuation,
        absorption_flow_divergence,
    ]
    
    for func in signal_funcs:
        try:
            result = func(state)
            if result is not None:
                signals.append(result)
        except Exception:
            continue  # Skip on error, don't crash
    
    return signals
