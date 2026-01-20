"""
Stage 3 V2 Signals: Long-Horizon Positioning Signals
====================================================

Entry signals based on positioning resolution alpha.
Each signal has MEMORY and tracks regime persistence.

Signals:
- PositioningDivergence: Funding vs price divergence
- TrappedRegime: Underwater OI detection
- LiquidationRegime: Post-cascade setups
"""
import time
from typing import Optional, Dict, List, Tuple
from collections import deque
import numpy as np

from src.stage3_v2.models import (
    LongHorizonState, PositioningSignal, SignalMemory, Direction,
    SignalType, PositioningRegime, OIPriceSnapshot,
    DEFAULT_THRESHOLDS, SIGNAL_COOLDOWNS_HOURS
)


class PositioningDivergenceSignal:
    """
    Detects OI/Funding/Price divergence regimes.
    Fires when crowd positioning conflicts with price action.
    
    LONG: Extreme negative funding, price not dropping, shorts closing
    SHORT: Extreme positive funding, price not rising, longs closing
    
    Requires 2+ hours of regime persistence before signaling.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.memory = SignalMemory(signal_type="POSITIONING_DIVERGENCE")
        self.thresholds = DEFAULT_THRESHOLDS
    
    def evaluate(self, state: LongHorizonState) -> Optional[PositioningSignal]:
        """Evaluate positioning divergence conditions"""
        
        # Check cooldown
        if self.memory.hours_since_last_signal() < SIGNAL_COOLDOWNS_HOURS.get("POSITIONING_DIVERGENCE", 8):
            return None
        
        # Condition 1: Extreme funding (crowd positioned)
        if abs(state.funding_z) < self.thresholds.funding_z_extreme:
            self.memory.reset()
            return None
        
        crowd_is_long = state.funding_z > self.thresholds.funding_z_extreme
        
        # Condition 2: Price not confirming crowd direction
        if crowd_is_long:
            price_confirms = state.price_change_8h > 0.01  # Price rising with longs
        else:
            price_confirms = state.price_change_8h < -0.01  # Price falling with shorts
        
        if price_confirms:
            self.memory.reset()
            return None
        
        # Condition 3: OI not expanding into direction (crowd not adding)
        if state.oi_delta_8h > 0.02:
            self.memory.reset()
            return None
        
        # Condition 4: Liquidations confirming crowd is wrong
        if crowd_is_long and state.liq_imbalance_8h < 0.2:
            return None  # Need longs getting liquidated
        if not crowd_is_long and state.liq_imbalance_8h > -0.2:
            return None  # Need shorts getting liquidated
        
        # REGIME DETECTED - track persistence
        direction = Direction.SHORT if crowd_is_long else Direction.LONG
        
        if self.memory.regime_direction != direction:
            self.memory.regime_active = True
            self.memory.regime_start_time = time.time()
            self.memory.regime_direction = direction
            self.memory.confirmation_count = 0
        
        self.memory.confirmation_count += 1
        
        # Require minimum regime persistence before signaling
        regime_duration_h = self.memory.regime_duration_hours()
        
        if regime_duration_h < self.thresholds.regime_min_persistence_h:
            return None
        
        # Fire signal
        confidence = 0.65 + min(0.15, regime_duration_h * 0.02)
        
        # Update memory
        self.memory.last_signal_time = time.time()
        
        return PositioningSignal(
            direction=direction,
            confidence=confidence,
            signal_type=SignalType.POSITIONING_DIVERGENCE,
            reason=f"Funding {state.funding_z:.1f}, price not confirming, regime for {regime_duration_h:.1f}h",
            expected_holding_hours=24,
            stop_pct=0.03,
            target_pct=0.05,
            regime_at_signal=state.regime,
            positioning_regime=state.ai_positioning_regime,
        )
    
    def get_memory(self) -> SignalMemory:
        return self.memory


class TrappedRegimeSignal:
    """
    Detects when significant OI is underwater.
    Uses OI-weighted average entry estimation.
    
    TRAPPED LONGS: OI built higher than current price, longs paying funding
    TRAPPED SHORTS: OI built lower than current price, shorts paying funding
    
    Requires 4+ hours of trapped regime persistence.
    """
    
    def __init__(self, symbol: str, history_size: int = 1440):
        self.symbol = symbol
        self.memory = SignalMemory(signal_type="TRAPPED_REGIME")
        self.thresholds = DEFAULT_THRESHOLDS
        
        # OI-price history for entry estimation (24h of 1-min data)
        self.oi_price_history: deque[OIPriceSnapshot] = deque(maxlen=history_size)
        self._last_oi: float = 0.0
    
    def add_oi_snapshot(self, price: float, oi: float, timestamp_ms: int):
        """Track OI changes with price for entry estimation"""
        oi_delta = oi - self._last_oi if self._last_oi > 0 else 0.0
        self._last_oi = oi
        
        self.oi_price_history.append(OIPriceSnapshot(
            timestamp_ms=timestamp_ms,
            price=price,
            oi=oi,
            oi_delta=oi_delta,
        ))
    
    def estimate_avg_entry(self, current_price: float) -> Tuple[float, float]:
        """
        Estimate average entry price of recent OI additions.
        Returns (avg_entry_price, concentration_above_pct)
        """
        if len(self.oi_price_history) < 100:
            return 0.0, 0.5
        
        # Calculate OI additions at each price level
        oi_additions = []
        for snapshot in self.oi_price_history:
            if snapshot.oi_delta > 0:  # OI was added
                oi_additions.append((snapshot.price, snapshot.oi_delta))
        
        if not oi_additions:
            return 0.0, 0.5
        
        total_oi_added = sum(d for _, d in oi_additions)
        if total_oi_added <= 0:
            return 0.0, 0.5
        
        # OI-weighted average entry
        avg_entry = sum(p * d for p, d in oi_additions) / total_oi_added
        
        # Concentration: what % of OI was added above current price
        oi_above = sum(d for p, d in oi_additions if p > current_price)
        concentration_above = oi_above / total_oi_added
        
        return avg_entry, concentration_above
    
    def evaluate(self, state: LongHorizonState) -> Optional[PositioningSignal]:
        """Evaluate trapped positioning conditions"""
        
        # Check cooldown
        if self.memory.hours_since_last_signal() < SIGNAL_COOLDOWNS_HOURS.get("TRAPPED_REGIME", 12):
            return None
        
        # Add current OI snapshot
        self.add_oi_snapshot(state.current_price, state.oi, state.timestamp_ms)
        
        # Estimate average entry
        avg_entry, concentration_above = self.estimate_avg_entry(state.current_price)
        
        if avg_entry == 0:
            return None
        
        displacement_pct = (avg_entry - state.current_price) / state.current_price
        
        # TRAPPED LONGS: avg entry significantly above current price
        if (displacement_pct > self.thresholds.trapped_displacement_min and 
            concentration_above > self.thresholds.trapped_concentration_min):
            
            # Longs underwater, still paying funding
            if state.funding_z > 0.3:
                return self._check_regime_and_signal(
                    state, Direction.SHORT, displacement_pct, concentration_above,
                    "TRAPPED_LONGS"
                )
        
        # TRAPPED SHORTS: avg entry significantly below current price
        elif (displacement_pct < -self.thresholds.trapped_displacement_min and 
              concentration_above < (1 - self.thresholds.trapped_concentration_min)):
            
            if state.funding_z < -0.3:
                return self._check_regime_and_signal(
                    state, Direction.LONG, abs(displacement_pct), 1 - concentration_above,
                    "TRAPPED_SHORTS"
                )
        
        else:
            self.memory.reset()
        
        return None
    
    def _check_regime_and_signal(
        self, state: LongHorizonState, direction: Direction, 
        displacement: float, concentration: float, trap_type: str
    ) -> Optional[PositioningSignal]:
        """Check regime persistence and generate signal if conditions met"""
        
        if self.memory.regime_direction != direction:
            self.memory.regime_active = True
            self.memory.regime_start_time = time.time()
            self.memory.regime_direction = direction
        
        regime_hours = self.memory.regime_duration_hours()
        
        if regime_hours < self.thresholds.trapped_min_persistence_h:
            return None
        
        # Fire signal
        confidence = 0.60 + min(0.15, displacement * 2)
        
        self.memory.last_signal_time = time.time()
        
        return PositioningSignal(
            direction=direction,
            confidence=confidence,
            signal_type=SignalType.TRAPPED_REGIME,
            reason=f"{trap_type}: OI avg entry {displacement*100:.1f}% displaced, {concentration*100:.0f}% trapped",
            expected_holding_hours=36,
            stop_pct=0.04,
            target_pct=0.06,
            regime_at_signal=state.regime,
            positioning_regime=state.ai_positioning_regime,
        )
    
    def get_memory(self) -> SignalMemory:
        return self.memory


class LiquidationRegimeSignal:
    """
    Detects post-liquidation continuation or reversal setups.
    State-based: tracks cascade phase across multiple updates.
    
    POST_LIQ_CONTINUATION: After liquidation cascade, more unwind expected
    POST_LIQ_REVERSAL: After liquidation cascade, mean reversion expected
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.memory = SignalMemory(signal_type="LIQUIDATION_REGIME")
        self.thresholds = DEFAULT_THRESHOLDS
        
        # Cascade tracking
        self.in_cascade = False
        self.cascade_start: Optional[float] = None
        self.cascade_direction: Optional[str] = None  # "long_liq" or "short_liq"
        self.cascade_peak_imbalance: float = 0.0
        
        # Post-cascade tracking
        self.post_cascade_mode = False
        self.post_cascade_start: Optional[float] = None
    
    def evaluate(self, state: LongHorizonState) -> Optional[PositioningSignal]:
        """Evaluate liquidation regime conditions"""
        
        # Check cooldown
        if self.memory.hours_since_last_signal() < SIGNAL_COOLDOWNS_HOURS.get("LIQUIDATION_REGIME", 4):
            return None
        
        # Phase 1: Detect active cascade
        if state.cascade_active and abs(state.liq_imbalance_30s) > self.thresholds.liq_imbalance_significant:
            if not self.in_cascade:
                self.in_cascade = True
                self.cascade_start = time.time()
                self.cascade_direction = "long_liq" if state.liq_imbalance_30s > 0 else "short_liq"
                self.cascade_peak_imbalance = abs(state.liq_imbalance_30s)
            else:
                # Track peak imbalance during cascade
                self.cascade_peak_imbalance = max(self.cascade_peak_imbalance, abs(state.liq_imbalance_30s))
            return None  # Don't trade during active cascade
        
        # Phase 2: Cascade ended, enter post-cascade mode
        if self.in_cascade and state.liq_exhaustion:
            self.in_cascade = False
            self.post_cascade_mode = True
            self.post_cascade_start = time.time()
        
        if not self.post_cascade_mode:
            return None
        
        # Phase 3: Wait appropriate time after cascade
        post_cascade_minutes = (time.time() - self.post_cascade_start) / 60
        
        if post_cascade_minutes < self.thresholds.post_liq_wait_min_m:
            return None  # Too early
        
        if post_cascade_minutes > self.thresholds.post_liq_wait_max_m:
            self._reset_cascade_state()
            return None  # Too late, regime expired
        
        # Evaluate: continuation or reversal?
        signal = self._evaluate_post_cascade(state, post_cascade_minutes)
        
        if signal:
            self.memory.last_signal_time = time.time()
            self._reset_cascade_state()
        
        return signal
    
    def _evaluate_post_cascade(
        self, state: LongHorizonState, minutes_since: float
    ) -> Optional[PositioningSignal]:
        """Evaluate continuation vs reversal after cascade"""
        
        if self.cascade_direction == "long_liq":
            # Longs were liquidated → price dropped
            
            # CONTINUATION: keep shorting if funding still elevated
            if state.funding_z > 0.8 and state.oi_delta_1h < 0:
                return PositioningSignal(
                    direction=Direction.SHORT,
                    confidence=0.65,
                    signal_type=SignalType.POST_LIQ_CONTINUATION,
                    reason=f"Longs liquidated {minutes_since:.0f}m ago, funding still positive ({state.funding_z:.1f}), more to unwind",
                    expected_holding_hours=18,
                    stop_pct=0.025,
                    target_pct=0.04,
                    regime_at_signal=state.regime,
                    positioning_regime=state.ai_positioning_regime,
                )
            
            # REVERSAL: funding normalized, absorption high
            elif state.funding_z < 0.3 and state.absorption_z > 1.5:
                return PositioningSignal(
                    direction=Direction.LONG,
                    confidence=0.55,
                    signal_type=SignalType.POST_LIQ_REVERSAL,
                    reason=f"Longs liquidated {minutes_since:.0f}m ago, funding normalized, absorption present",
                    expected_holding_hours=12,
                    stop_pct=0.02,
                    target_pct=0.03,
                    regime_at_signal=state.regime,
                    positioning_regime=state.ai_positioning_regime,
                )
        
        else:  # short_liq
            # Shorts were liquidated → price rose
            
            # CONTINUATION: keep longing if funding still negative
            if state.funding_z < -0.8 and state.oi_delta_1h < 0:
                return PositioningSignal(
                    direction=Direction.LONG,
                    confidence=0.65,
                    signal_type=SignalType.POST_LIQ_CONTINUATION,
                    reason=f"Shorts liquidated {minutes_since:.0f}m ago, funding still negative ({state.funding_z:.1f}), more to cover",
                    expected_holding_hours=18,
                    stop_pct=0.025,
                    target_pct=0.04,
                    regime_at_signal=state.regime,
                    positioning_regime=state.ai_positioning_regime,
                )
            
            # REVERSAL: funding normalized, absorption high
            elif state.funding_z > -0.3 and state.absorption_z > 1.5:
                return PositioningSignal(
                    direction=Direction.SHORT,
                    confidence=0.55,
                    signal_type=SignalType.POST_LIQ_REVERSAL,
                    reason=f"Shorts liquidated {minutes_since:.0f}m ago, funding normalized, absorption present",
                    expected_holding_hours=12,
                    stop_pct=0.02,
                    target_pct=0.03,
                    regime_at_signal=state.regime,
                    positioning_regime=state.ai_positioning_regime,
                )
        
        return None
    
    def _reset_cascade_state(self):
        """Reset cascade tracking state"""
        self.in_cascade = False
        self.cascade_start = None
        self.cascade_direction = None
        self.cascade_peak_imbalance = 0.0
        self.post_cascade_mode = False
        self.post_cascade_start = None
    
    def get_memory(self) -> SignalMemory:
        return self.memory


# ============================================================
# SIGNAL AGGREGATOR
# ============================================================

class PositioningSignalAggregator:
    """
    Aggregates all positioning signals for a symbol.
    Manages signal memory and evaluation.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.signals = {
            "positioning_divergence": PositioningDivergenceSignal(symbol),
            "trapped_regime": TrappedRegimeSignal(symbol),
            "liquidation_regime": LiquidationRegimeSignal(symbol),
        }
    
    def evaluate_all(self, state: LongHorizonState) -> List[PositioningSignal]:
        """Evaluate all signals and return any that fire"""
        results = []
        
        for name, signal in self.signals.items():
            try:
                result = signal.evaluate(state)
                if result is not None:
                    results.append(result)
            except Exception as e:
                # Log but don't crash
                pass
        
        return results
    
    def get_all_memory(self) -> Dict[str, SignalMemory]:
        """Get memory state for all signals"""
        return {name: sig.get_memory() for name, sig in self.signals.items()}
    
    def get_regime_summary(self) -> Dict[str, str]:
        """Get summary of current regime states"""
        summary = {}
        for name, sig in self.signals.items():
            mem = sig.get_memory()
            if mem.regime_active:
                duration = mem.regime_duration_hours()
                summary[name] = f"{mem.regime_direction.value if mem.regime_direction else 'N/A'} for {duration:.1f}h"
            else:
                summary[name] = "inactive"
        return summary
