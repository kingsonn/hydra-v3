"""
Regime Classifier - CHOP, COMPRESSION, EXPANSION detection
Score-based classification with per-pair thresholds
"""
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from src.stage2.models import (
    Regime, RegimeClassification, OrderFlowFeatures, AbsorptionFeatures,
    VolatilityFeatures, StructureFeatures
)


# ============================================================
# PER-PAIR THRESHOLDS - Tuned for each asset's microstructure
# ============================================================

@dataclass
class PairThresholds:
    """Per-pair thresholds for regime classification"""
    # Volatility
    vol_expansion_high: float = 1.30   # Above = expansion signal
    vol_compression_low: float = 0.75  # Below = compression signal
    
    # Aggression
    aggression_high: float = 1.15      # Above = expansion signal
    aggression_low: float = 0.85       # Below = compression signal
    
    # MOI std percentiles (relative thresholds)
    moi_std_high_pct: float = 60.0     # P60 = high std
    moi_std_low_pct: float = 40.0      # P40 = low std
    
    # Delta flip rate (flips/min)
    delta_flip_low: float = 2.0        # Below = directional
    delta_flip_high: float = 5.0       # Above = choppy
    
    # Absorption
    absorption_z_noise: float = 1.0    # Below = noise
    absorption_z_spike: float = 2.5    # Above = significant
    
    # Value area width
    value_width_compression: float = 0.65  # Below = tight/compressing
    value_width_wide: float = 1.05         # Above = wide/choppy
    
    # Time inside value (0-1)
    time_inside_compression: float = 0.60  # Above = compressing


# Per-pair threshold configurations
PAIR_THRESHOLDS: Dict[str, PairThresholds] = {
    # BTC - Baseline, clean microstructure
    "BTCUSDT": PairThresholds(
        vol_expansion_high=1.30,
        vol_compression_low=0.75,
        aggression_high=1.15,
        aggression_low=0.85,
        moi_std_high_pct=60.0,
        moi_std_low_pct=40.0,
        delta_flip_low=2.0,
        delta_flip_high=5.0,
        absorption_z_noise=1.0,
        absorption_z_spike=2.5,
        value_width_compression=0.65,
        value_width_wide=1.05,
        time_inside_compression=0.60,
    ),
    
    # ETH - Close to BTC, slightly noisier (hedging, options flows)
    "ETHUSDT": PairThresholds(
        vol_expansion_high=1.25,
        vol_compression_low=0.80,
        aggression_high=1.10,
        aggression_low=0.80,
        moi_std_high_pct=60.0,
        moi_std_low_pct=40.0,
        delta_flip_low=2.5,
        delta_flip_high=6.0,
        absorption_z_noise=1.0,
        absorption_z_spike=2.7,
        value_width_compression=0.65,
        value_width_wide=1.05,
        time_inside_compression=0.60,
    ),
    
    # SOL - Fast, aggressive, momentum-heavy
    "SOLUSDT": PairThresholds(
        vol_expansion_high=1.20,  # Expands earlier
        vol_compression_low=0.70,
        aggression_high=1.00,
        aggression_low=0.75,
        moi_std_high_pct=55.0,
        moi_std_low_pct=35.0,
        delta_flip_low=2.2,
        delta_flip_high=6.5,
        absorption_z_noise=1.0,
        absorption_z_spike=2.2,
        value_width_compression=0.65,
        value_width_wide=1.05,
        time_inside_compression=0.60,
    ),
    
    # BNB - Thick book, slower reaction, fewer fake breakouts
    "BNBUSDT": PairThresholds(
        vol_expansion_high=1.35,  # Requires stronger confirmation
        vol_compression_low=0.78,
        aggression_high=1.25,
        aggression_low=0.90,
        moi_std_high_pct=60.0,
        moi_std_low_pct=40.0,
        delta_flip_low=1.8,
        delta_flip_high=4.5,
        absorption_z_noise=1.0,
        absorption_z_spike=3.0,
        value_width_compression=0.65,
        value_width_wide=1.05,
        time_inside_compression=0.60,
    ),
    
    # DOGE - Noisy, spoof-heavy, retail driven - FILTER AGGRESSIVELY
    "DOGEUSDT": PairThresholds(
        vol_expansion_high=1.40,
        vol_compression_low=0.85,
        aggression_high=1.35,
        aggression_low=1.00,
        moi_std_high_pct=65.0,
        moi_std_low_pct=45.0,
        delta_flip_low=3.5,
        delta_flip_high=8.0,
        absorption_z_noise=1.0,
        absorption_z_spike=3.2,
        value_width_compression=0.65,
        value_width_wide=1.05,
        time_inside_compression=0.60,
    ),
    
    # ADA - Low price, high unit noise, constant churn
    "ADAUSDT": PairThresholds(
        vol_expansion_high=1.35,
        vol_compression_low=0.82,
        aggression_high=1.30,
        aggression_low=0.95,
        moi_std_high_pct=65.0,
        moi_std_low_pct=45.0,
        delta_flip_low=3.0,
        delta_flip_high=7.0,
        absorption_z_noise=1.0,
        absorption_z_spike=3.0,
        value_width_compression=0.65,
        value_width_wide=1.05,
        time_inside_compression=0.60,
    ),
    
    # XRP - Mean-reverting, news-driven, sudden spikes
    "XRPUSDT": PairThresholds(
        vol_expansion_high=1.45,  # Treat expansions cautiously
        vol_compression_low=0.88,
        aggression_high=1.40,
        aggression_low=1.00,
        moi_std_high_pct=60.0,
        moi_std_low_pct=40.0,
        delta_flip_low=3.5,
        delta_flip_high=7.5,
        absorption_z_noise=1.0,
        absorption_z_spike=3.3,
        value_width_compression=0.65,
        value_width_wide=1.05,
        time_inside_compression=0.60,
    ),
    
    # LTC - Thin but clean, slower but honest moves
    "LTCUSDT": PairThresholds(
        vol_expansion_high=1.25,
        vol_compression_low=0.75,
        aggression_high=1.10,
        aggression_low=0.80,
        moi_std_high_pct=60.0,
        moi_std_low_pct=40.0,
        delta_flip_low=2.0,
        delta_flip_high=5.5,
        absorption_z_noise=1.0,
        absorption_z_spike=2.6,
        value_width_compression=0.65,
        value_width_wide=1.05,
        time_inside_compression=0.60,
    ),
}

# Default thresholds (use BTC as baseline)
DEFAULT_THRESHOLDS = PAIR_THRESHOLDS["BTCUSDT"]


def get_thresholds(symbol: str) -> PairThresholds:
    """Get thresholds for a symbol, fallback to BTC baseline"""
    return PAIR_THRESHOLDS.get(symbol, DEFAULT_THRESHOLDS)


@dataclass
class RegimeInputs:
    """All inputs needed for regime classification"""
    # Volatility
    atr_5m: float = 0.0
    atr_1h: float = 0.0
    vol_expansion_ratio: float = 0.0  # ATR_5m / ATR_1h
    
    # Structure
    value_area_width: float = 0.0
    value_width_ratio: float = 0.0    # width / median_width
    time_inside_value_pct: float = 0.0  # 0-100
    acceptance_outside_value: bool = False
    
    # Order flow
    moi_std: float = 0.0
    moi_flip_rate: float = 0.0        # flips/min
    aggression_persistence: float = 0.0
    delta_velocity: float = 0.0
    
    # Absorption
    absorption_z: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "atr_5m": self.atr_5m,
            "atr_1h": self.atr_1h,
            "vol_expansion_ratio": self.vol_expansion_ratio,
            "value_area_width": self.value_area_width,
            "value_width_ratio": self.value_width_ratio,
            "time_inside_value_pct": self.time_inside_value_pct,
            "acceptance_outside_value": float(self.acceptance_outside_value),
            "moi_std": self.moi_std,
            "moi_flip_rate": self.moi_flip_rate,
            "aggression_persistence": self.aggression_persistence,
            "delta_velocity": self.delta_velocity,
            "absorption_z": self.absorption_z,
        }


class RegimeClassifier:
    """
    Score-based regime classification with per-pair thresholds
    
    EXPANSION >= 3 points → EXPANSION
    COMPRESSION >= 3 points → COMPRESSION
    Explicit CHOP conditions → CHOP
    Default fallback → COMPRESSION (safe state)
    """
    
    def __init__(
        self,
        symbol: str,
        update_interval_ms: int = 60_000,  # 60 seconds
    ):
        self.symbol = symbol
        self.update_interval_ms = update_interval_ms
        
        # Get per-pair thresholds
        self.thresholds = get_thresholds(symbol)
        
        # State
        self._current_regime = Regime.COMPRESSION  # Default SAFE state
        self._last_classification_ms = 0
        self._regime_history: deque[Tuple[int, Regime]] = deque(maxlen=100)
        
        # Rolling MOI std history for percentile calculation
        self._moi_std_history: deque[float] = deque(maxlen=200)
        
        # Last scores for transparency
        self._last_expansion_score = 0
        self._last_compression_score = 0
        self._last_is_chop = False
    
    def classify(
        self,
        volatility: VolatilityFeatures,
        structure: StructureFeatures,
        order_flow: OrderFlowFeatures,
        absorption: AbsorptionFeatures,
        force_update: bool = False,
    ) -> RegimeClassification:
        """
        Classify market regime using score-based approach
        """
        now_ms = int(time.time() * 1000)
        
        # Check if we should update
        if not force_update:
            if now_ms - self._last_classification_ms < self.update_interval_ms:
                return RegimeClassification(
                    regime=self._current_regime,
                    confidence=0.8,
                    compression_score=float(self._last_compression_score),
                    expansion_score=float(self._last_expansion_score),
                    chop_score=1.0 if self._last_is_chop else 0.0,
                    reason="No update (within interval)",
                )
        
        # Build inputs
        inputs = RegimeInputs(
            atr_5m=volatility.atr_5m,
            atr_1h=volatility.atr_1h,
            vol_expansion_ratio=volatility.vol_expansion_ratio,
            value_area_width=structure.value_area_width,
            value_width_ratio=structure.value_width_ratio,
            time_inside_value_pct=structure.time_inside_value_pct / 100.0,  # Convert to 0-1
            acceptance_outside_value=structure.acceptance_outside_value,
            moi_std=order_flow.moi_std,
            moi_flip_rate=order_flow.moi_flip_rate,
            aggression_persistence=order_flow.aggression_persistence,
            delta_velocity=order_flow.delta_velocity,
            absorption_z=absorption.absorption_z,
        )
        
        # Track MOI std for percentile calculation
        if inputs.moi_std > 0:
            self._moi_std_history.append(inputs.moi_std)
        
        # Calculate MOI std percentile thresholds
        moi_std_threshold = self._get_moi_std_threshold()
        
        # Compute scores
        expansion_score = self._compute_expansion_score(inputs, moi_std_threshold)
        compression_score = self._compute_compression_score(inputs, moi_std_threshold)
        is_chop = self._is_chop(inputs)
        
        # Store for transparency
        self._last_expansion_score = expansion_score
        self._last_compression_score = compression_score
        self._last_is_chop = is_chop
        
        # REGIME RESOLUTION (priority order)
        regime = Regime.COMPRESSION  # Default SAFE state
        confidence = 0.5
        reason = "Default safe state"
        
        if expansion_score >= 3:
            regime = Regime.EXPANSION
            confidence = min(1.0, expansion_score / 5.0)
            reason = f"Expansion score {expansion_score}/5"
        elif compression_score >= 3:
            regime = Regime.COMPRESSION
            confidence = min(1.0, compression_score / 5.0)
            reason = f"Compression score {compression_score}/5"
        elif is_chop:
            regime = Regime.CHOP
            confidence = 0.7
            reason = "Explicit CHOP conditions met"
        else:
            regime = Regime.COMPRESSION  # Transitional/neutral → safe state
            confidence = 0.5
            reason = "Transitional state, defaulting to COMPRESSION"
        
        # Update state
        self._current_regime = regime
        self._last_classification_ms = now_ms
        self._regime_history.append((now_ms, regime))
        
        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            compression_score=float(compression_score),
            expansion_score=float(expansion_score),
            chop_score=1.0 if is_chop else 0.0,
            reason=reason,
        )
    
    def _get_moi_std_threshold(self) -> float:
        """Get MOI std threshold based on rolling percentile"""
        if len(self._moi_std_history) < 20:
            return 0.5  # Default until we have history
        
        arr = np.array(self._moi_std_history)
        # Use midpoint between low and high percentiles
        low_pct = self.thresholds.moi_std_low_pct
        high_pct = self.thresholds.moi_std_high_pct
        mid_pct = (low_pct + high_pct) / 2.0
        return float(np.percentile(arr, mid_pct))
    
    def _compute_expansion_score(self, inputs: RegimeInputs, moi_std_threshold: float) -> int:
        """
        Compute EXPANSION score (0-5 points)
        
        +1 if vol_expansion_ratio > threshold
        +1 if acceptance_outside_value
        +1 if AggressionPersistence > threshold
        +1 if moi_std > moi_std_threshold (directional flow)
        +1 if delta_flip_rate < mid_flip_threshold (not choppy)
        """
        t = self.thresholds
        score = 0
        
        # Vol expansion
        if inputs.vol_expansion_ratio > t.vol_expansion_high:
            score += 1
        
        # Acceptance outside value
        if inputs.acceptance_outside_value:
            score += 1
        
        # High aggression persistence
        if inputs.aggression_persistence > t.aggression_high:
            score += 1
        
        # High MOI std (directional, not canceling out)
        if inputs.moi_std > moi_std_threshold:
            score += 1
        
        # Low flip rate (directional, not choppy)
        mid_flip = (t.delta_flip_low + t.delta_flip_high) / 2.0
        if inputs.moi_flip_rate < mid_flip:
            score += 1
        
        return score
    
    def _compute_compression_score(self, inputs: RegimeInputs, moi_std_threshold: float) -> int:
        """
        Compute COMPRESSION score (0-5 points)
        
        +1 if vol_expansion_ratio < threshold
        +1 if value_width_ratio < threshold
        +1 if time_inside_value_pct > threshold
        +1 if moi_std < moi_std_threshold (flow canceling out)
        +1 if abs(Absorption_Z) < noise_threshold
        """
        t = self.thresholds
        score = 0
        
        # Vol compression
        if inputs.vol_expansion_ratio < t.vol_compression_low:
            score += 1
        
        # Tight value area
        if inputs.value_width_ratio < t.value_width_compression:
            score += 1
        
        # Time inside value (inputs already normalized to 0-1)
        if inputs.time_inside_value_pct > t.time_inside_compression:
            score += 1
        
        # Low MOI std (flow canceling out)
        if inputs.moi_std < moi_std_threshold:
            score += 1
        
        # Low absorption (noise, not significant)
        if abs(inputs.absorption_z) < t.absorption_z_noise:
            score += 1
        
        return score
    
    def _is_chop(self, inputs: RegimeInputs) -> bool:
        """
        Explicit CHOP detection - only if actively choppy
        
        CHOP = high flip rate AND absorption spikes AND wide value
        """
        t = self.thresholds
        
        return (
            inputs.moi_flip_rate > t.delta_flip_high and
            abs(inputs.absorption_z) > t.absorption_z_spike and
            inputs.value_width_ratio > t.value_width_wide
        )
    
    def get_current_regime(self) -> Regime:
        """Get current regime without reclassifying"""
        return self._current_regime
    
    def get_regime_history(self, count: int = 10) -> List[Tuple[int, Regime]]:
        """Get recent regime history"""
        return list(self._regime_history)[-count:]
    
    def get_regime_duration_ms(self) -> int:
        """Get how long current regime has been active"""
        if len(self._regime_history) < 2:
            return 0
        
        current = self._current_regime
        duration = 0
        
        for ts, regime in reversed(self._regime_history):
            if regime == current:
                if duration == 0:
                    duration = int(time.time() * 1000) - ts
            else:
                break
        
        return duration


class MultiSymbolRegimeClassifier:
    """Manages regime classifiers for multiple symbols"""
    
    def __init__(self, symbols: List[str], update_interval_ms: int = 60_000):
        self.classifiers: Dict[str, RegimeClassifier] = {
            s: RegimeClassifier(s, update_interval_ms) for s in symbols
        }
    
    def classify(
        self,
        symbol: str,
        volatility: VolatilityFeatures,
        structure: StructureFeatures,
        order_flow: OrderFlowFeatures,
        absorption: AbsorptionFeatures,
        force_update: bool = False,
    ) -> RegimeClassification:
        if symbol in self.classifiers:
            return self.classifiers[symbol].classify(
                volatility, structure, order_flow, absorption, force_update
            )
        return RegimeClassification(regime=Regime.CHOP)
    
    def get_current_regime(self, symbol: str) -> Regime:
        if symbol in self.classifiers:
            return self.classifiers[symbol].get_current_regime()
        return Regime.CHOP
    
    def get_all_regimes(self) -> Dict[str, Regime]:
        return {s: c.get_current_regime() for s, c in self.classifiers.items()}
