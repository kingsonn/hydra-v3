"""Regime Classifier with Persistence and Smooth Transitions
============================================================

Simplified to 4 actionable regimes:
- TRENDING_UP: Clear uptrend, trade longs with trend
- TRENDING_DOWN: Clear downtrend, trade shorts with trend  
- RANGING: Bounded price action, wait or fade extremes
- CHOPPY: No clear structure, NO TRADE

Key features:
- Regime persistence: Requires confirmation bars to change regime
- Smooth transitions: Avoids whipsawing between regimes
- Proper recomputation timing: Every minute for regime, hourly for bias
- Clear scoring system for regime determination
"""
import time
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import deque
import structlog

from src.stage3_v3.models import MarketRegime, Direction

logger = structlog.get_logger(__name__)


@dataclass
class RegimeState:
    """Persisted regime state with transition tracking"""
    current_regime: MarketRegime = MarketRegime.CHOPPY
    confidence: float = 0.5
    regime_start_time: float = 0.0
    bars_in_regime: int = 0
    pending_regime: Optional[MarketRegime] = None
    pending_confidence: float = 0.0
    pending_bars: int = 0
    last_update_time: float = 0.0
    
    # Regime scoring history for smoothing
    trend_scores: deque = field(default_factory=lambda: deque(maxlen=10))
    chop_scores: deque = field(default_factory=lambda: deque(maxlen=10))


class RegimeClassifier:
    """
    Robust regime classifier with persistence and smooth transitions.
    
    Computation frequency: Every 1 minute (on bar close)
    Regime change: Requires 3 confirmation bars to switch
    
    Scoring system:
    - Trend score: Based on structure (HH/HL or LH/LL) + price change + EMA alignment
    - Chop score: Based on MOI flip rate + directional inconsistency + vol spikes
    - Range score: Low volatility + bounded price action
    """
    
    # Thresholds for regime determination - AUDIT FIX: More aggressive
    TREND_SCORE_THRESHOLD = 0.25     # Score > 0.25 = trending (was 0.35)
    CHOP_SCORE_THRESHOLD = 0.7       # Score > 0.7 = choppy (raised - harder to trigger)
    RANGE_SCORE_THRESHOLD = 0.6      # Score > 0.6 = ranging (raised)
    
    # Transition settings - AUDIT FIX: Faster transitions
    BARS_TO_CONFIRM = 2              # Bars needed to confirm regime change (was 3)
    MIN_REGIME_DURATION_SEC = 120    # Minimum 2 min in a regime (was 5 min)
    
    # Volatility thresholds
    VOL_HIGH = 1.8                   # Vol ratio > 1.8 = high volatility
    VOL_LOW = 0.6                    # Vol ratio < 0.6 = low volatility
    
    # MOI chop threshold
    MOI_FLIP_CHOP = 0.4              # Flips per bar > 0.4 = choppy
    
    def __init__(self):
        # Per-symbol regime state
        self._states: Dict[str, RegimeState] = {}
        
    def get_state(self, symbol: str) -> RegimeState:
        """Get or create regime state for symbol"""
        if symbol not in self._states:
            self._states[symbol] = RegimeState()
        return self._states[symbol]
    
    def classify(
        self,
        symbol: str,
        # Price structure
        higher_high: bool,
        higher_low: bool,
        lower_high: bool,
        lower_low: bool,
        # Price changes
        price_change_1h: float,
        price_change_4h: float,
        # Volatility
        vol_expansion_ratio: float,
        range_vs_atr: float,
        # EMA alignment
        ema_20: float,
        ema_50: float,
        current_price: float,
        # Chop indicators
        moi_flip_rate: float = 0.0,
        directional_consistency: float = 1.0,  # 1.0 = all same direction, 0.0 = random
    ) -> Tuple[MarketRegime, float]:
        """
        Classify market regime with persistence.
        
        Returns:
            (regime, confidence) tuple
        """
        state = self.get_state(symbol)
        now = time.time()
        
        # Calculate scores for each regime
        trend_score, trend_direction = self._calc_trend_score(
            higher_high, higher_low, lower_high, lower_low,
            price_change_1h, price_change_4h,
            vol_expansion_ratio, ema_20, ema_50, current_price
        )
        
        chop_score = self._calc_chop_score(
            moi_flip_rate, directional_consistency,
            vol_expansion_ratio, higher_high, higher_low,
            lower_high, lower_low
        )
        
        range_score = self._calc_range_score(
            vol_expansion_ratio, range_vs_atr,
            price_change_1h, price_change_4h
        )
        
        # Store scores for smoothing
        state.trend_scores.append(trend_score)
        state.chop_scores.append(chop_score)
        
        # Use smoothed scores - AUDIT FIX: Weight recent scores more heavily
        if len(state.trend_scores) >= 3:
            # Weighted average: recent scores count more (50% last, 30% prev, 20% rest)
            recent = list(state.trend_scores)[-3:]
            smooth_trend = recent[-1] * 0.5 + recent[-2] * 0.3 + recent[-3] * 0.2
        else:
            smooth_trend = trend_score
        
        if len(state.chop_scores) >= 3:
            recent = list(state.chop_scores)[-3:]
            smooth_chop = recent[-1] * 0.5 + recent[-2] * 0.3 + recent[-3] * 0.2
        else:
            smooth_chop = chop_score
        
        # Determine proposed regime
        proposed_regime, proposed_confidence = self._determine_regime(
            smooth_trend, trend_direction, smooth_chop, range_score
        )
        
        # Apply transition logic
        final_regime, final_confidence = self._apply_transition(
            state, proposed_regime, proposed_confidence, now
        )
        
        # Update state
        state.last_update_time = now
        
        return final_regime, final_confidence
    
    def _calc_trend_score(
        self,
        higher_high: bool,
        higher_low: bool, 
        lower_high: bool,
        lower_low: bool,
        price_change_1h: float,
        price_change_4h: float,
        vol_ratio: float,
        ema_20: float,
        ema_50: float,
        price: float,
    ) -> Tuple[float, Direction]:
        """
        Calculate trend score (0-1) and direction.
        
        Components:
        - Structure: HH+HL (bullish) or LH+LL (bearish) = 0.4
        - Price momentum: 4h change direction = 0.3
        - EMA alignment: Price > EMA20 > EMA50 = 0.2
        - Volatility support: Vol ratio 1.0-2.0 = 0.1
        """
        score = 0.0
        direction = Direction.NEUTRAL
        
        # Structure component (0.4)
        if higher_high and higher_low:
            score += 0.4
            direction = Direction.LONG
        elif lower_high and lower_low:
            score += 0.4
            direction = Direction.SHORT
        elif higher_high or higher_low:  # Partial structure
            score += 0.2
            direction = Direction.LONG
        elif lower_high or lower_low:
            score += 0.2
            direction = Direction.SHORT
        
        # Price momentum component (0.3)
        if abs(price_change_4h) > 0.015:  # 1.5%+ move
            score += 0.3
            if direction == Direction.NEUTRAL:
                direction = Direction.LONG if price_change_4h > 0 else Direction.SHORT
        elif abs(price_change_4h) > 0.008:  # 0.8%+ move
            score += 0.2
            if direction == Direction.NEUTRAL:
                direction = Direction.LONG if price_change_4h > 0 else Direction.SHORT
        elif abs(price_change_4h) > 0.003:  # 0.3%+ move
            score += 0.15
            if direction == Direction.NEUTRAL:
                direction = Direction.LONG if price_change_4h > 0 else Direction.SHORT
        
        # EMA alignment component (0.2)
        if ema_20 > 0 and ema_50 > 0:
            if price > ema_20 > ema_50:
                score += 0.2
                if direction == Direction.NEUTRAL:
                    direction = Direction.LONG
            elif price < ema_20 < ema_50:
                score += 0.2
                if direction == Direction.NEUTRAL:
                    direction = Direction.SHORT
            elif price > ema_20 or price > ema_50:
                score += 0.1
            elif price < ema_20 or price < ema_50:
                score += 0.1
        
        # Volatility support (0.15) - trending needs some vol but not chaos
        if 0.7 <= vol_ratio <= 2.5:
            score += 0.15
        elif 0.5 <= vol_ratio <= 3.0:
            score += 0.1
        elif vol_ratio > 0:
            score += 0.05  # Some baseline for any vol
        
        return min(1.0, score), direction
    
    def _calc_chop_score(
        self,
        moi_flip_rate: float,
        directional_consistency: float,
        vol_ratio: float,
        higher_high: bool,
        higher_low: bool,
        lower_high: bool,
        lower_low: bool,
    ) -> float:
        """
        Calculate chop score (0-1).
        
        Components:
        - MOI flip rate: High flip rate = 0.4
        - Directional inconsistency: Random direction = 0.3  
        - Conflicting structure: HH+LL or LH+HL = 0.2
        - Extreme volatility: Vol > 2.5 = 0.1
        """
        score = 0.0
        
        # MOI flip rate (0.35) - needs higher flip rate to be choppy
        if moi_flip_rate > 0.6:  # High threshold
            score += min(0.35, (moi_flip_rate - 0.4) * 0.5)
        
        # Directional inconsistency (0.25)
        # directional_consistency: 1.0 = all same, 0.0 = random
        inconsistency = 1.0 - directional_consistency
        if inconsistency > 0.5:  # Only count high inconsistency
            score += (inconsistency - 0.5) * 0.5
        
        # Conflicting structure (0.25)
        # If we have BOTH strong bullish AND strong bearish structure = confusion
        bullish_structure = higher_high and higher_low  # Need both for strong
        bearish_structure = lower_high and lower_low    # Need both for strong
        if bullish_structure and bearish_structure:
            score += 0.25
        elif (higher_high or higher_low) and (lower_high or lower_low):
            score += 0.15  # Partial conflict
        
        # Extreme volatility without direction (0.15)
        if vol_ratio > 3.0:
            score += 0.15
        elif vol_ratio > 2.5:
            score += 0.1
        
        return min(1.0, score)
    
    def _calc_range_score(
        self,
        vol_ratio: float,
        range_vs_atr: float,
        price_change_1h: float,
        price_change_4h: float,
    ) -> float:
        """
        Calculate range score (0-1).
        
        Components:
        - Low volatility: Vol < 0.8 = 0.4
        - Bounded price: Range < 1.5 ATR = 0.3
        - Minimal net movement: Price change < 0.5% = 0.3
        """
        score = 0.0
        
        # Low volatility (0.4)
        if vol_ratio < self.VOL_LOW:
            score += 0.4
        elif vol_ratio < 0.8:
            score += 0.25
        elif vol_ratio < 1.0:
            score += 0.1
        
        # Bounded price (0.3)
        if range_vs_atr < 1.2:
            score += 0.3
        elif range_vs_atr < 1.5:
            score += 0.2
        elif range_vs_atr < 2.0:
            score += 0.1
        
        # Minimal net movement (0.3)
        if abs(price_change_4h) < 0.003:  # < 0.3%
            score += 0.3
        elif abs(price_change_4h) < 0.007:  # < 0.7%
            score += 0.2
        elif abs(price_change_4h) < 0.01:  # < 1%
            score += 0.1
        
        return min(1.0, score)
    
    def _determine_regime(
        self,
        trend_score: float,
        trend_direction: Direction,
        chop_score: float,
        range_score: float,
    ) -> Tuple[MarketRegime, float]:
        """
        Determine regime from scores.
        
        Priority:
        1. CHOPPY if chop_score > threshold (safety first)
        2. TRENDING if trend_score > threshold
        3. RANGING if range_score > threshold
        4. CHOPPY as default (conservative)
        """
        # Chop check first (safety)
        if chop_score > self.CHOP_SCORE_THRESHOLD:
            return MarketRegime.CHOPPY, 0.5 + chop_score * 0.4
        
        # Trend check
        if trend_score > self.TREND_SCORE_THRESHOLD:
            confidence = 0.5 + trend_score * 0.4
            if trend_direction == Direction.LONG:
                return MarketRegime.TRENDING_UP, confidence
            elif trend_direction == Direction.SHORT:
                return MarketRegime.TRENDING_DOWN, confidence
        
        # Range check
        if range_score > self.RANGE_SCORE_THRESHOLD:
            return MarketRegime.RANGING, 0.5 + range_score * 0.4
        
        # Weak trend still beats nothing
        if trend_score > 0.25:
            confidence = 0.4 + trend_score * 0.4
            if trend_direction == Direction.LONG:
                return MarketRegime.TRENDING_UP, confidence
            elif trend_direction == Direction.SHORT:
                return MarketRegime.TRENDING_DOWN, confidence
        
        # Check if we have ANY directional bias
        if trend_direction != Direction.NEUTRAL and trend_score > 0.15:
            confidence = 0.3 + trend_score * 0.4
            if trend_direction == Direction.LONG:
                return MarketRegime.TRENDING_UP, confidence
            else:
                return MarketRegime.TRENDING_DOWN, confidence
        
        # Default: ranging (less restrictive than choppy)
        if range_score > 0.3:
            return MarketRegime.RANGING, 0.4 + range_score * 0.3
        
        # True default: choppy
        return MarketRegime.CHOPPY, 0.5
    
    def _apply_transition(
        self,
        state: RegimeState,
        proposed: MarketRegime,
        proposed_confidence: float,
        now: float,
    ) -> Tuple[MarketRegime, float]:
        """
        Apply transition logic to avoid whipsawing.
        
        Rules:
        1. Same regime: Increment bars, update confidence
        2. Different regime: Start pending counter
        3. Pending reaches threshold: Switch regime
        4. Minimum duration: Don't switch too fast
        """
        # Same as current regime
        if proposed == state.current_regime:
            state.bars_in_regime += 1
            state.confidence = proposed_confidence
            state.pending_regime = None
            state.pending_bars = 0
            return state.current_regime, state.confidence
        
        # Different regime - check minimum duration
        time_in_regime = now - state.regime_start_time
        if time_in_regime < self.MIN_REGIME_DURATION_SEC:
            # Too soon to switch, stay in current
            state.bars_in_regime += 1
            return state.current_regime, state.confidence
        
        # Start or continue pending transition
        if state.pending_regime == proposed:
            state.pending_bars += 1
            state.pending_confidence = proposed_confidence
        else:
            # New pending regime
            state.pending_regime = proposed
            state.pending_bars = 1
            state.pending_confidence = proposed_confidence
        
        # Check if confirmed
        if state.pending_bars >= self.BARS_TO_CONFIRM:
            # Transition confirmed!
            old_regime = state.current_regime
            state.current_regime = proposed
            state.confidence = proposed_confidence
            state.regime_start_time = now
            state.bars_in_regime = 1
            state.pending_regime = None
            state.pending_bars = 0
            
            return state.current_regime, state.confidence
        
        # Still pending, return current
        state.bars_in_regime += 1
        return state.current_regime, state.confidence
    
    def force_regime(
        self,
        symbol: str,
        regime: MarketRegime,
        confidence: float = 0.7,
    ):
        """Force regime for testing or manual override"""
        state = self.get_state(symbol)
        state.current_regime = regime
        state.confidence = confidence
        state.regime_start_time = time.time()
        state.bars_in_regime = 1
        state.pending_regime = None
        state.pending_bars = 0
    
    def get_regime_info(self, symbol: str) -> Dict:
        """Get detailed regime info for dashboard"""
        state = self.get_state(symbol)
        return {
            "regime": state.current_regime.value,
            "confidence": round(state.confidence, 2),
            "bars_in_regime": state.bars_in_regime,
            "regime_duration_sec": int(time.time() - state.regime_start_time) if state.regime_start_time > 0 else 0,
            "pending_regime": state.pending_regime.value if state.pending_regime else None,
            "pending_bars": state.pending_bars,
            "bars_to_confirm": self.BARS_TO_CONFIRM,
        }
    
    def is_tradeable(self, regime: MarketRegime) -> bool:
        """CHOPPY = NO TRADE, everything else is tradeable"""
        return regime != MarketRegime.CHOPPY


# Singleton instance for the classifier
_classifier_instance: Optional[RegimeClassifier] = None


def get_regime_classifier() -> RegimeClassifier:
    """Get singleton regime classifier"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = RegimeClassifier()
    return _classifier_instance


def classify_regime(
    symbol: str,
    higher_high: bool,
    higher_low: bool,
    lower_high: bool,
    lower_low: bool,
    price_change_1h: float,
    price_change_4h: float,
    vol_expansion_ratio: float,
    range_vs_atr: float,
    ema_20: float = 0.0,
    ema_50: float = 0.0,
    current_price: float = 0.0,
    moi_flip_rate: float = 0.0,
    directional_consistency: float = 1.0,
) -> Tuple[MarketRegime, float]:
    """Quick regime classification using singleton"""
    classifier = get_regime_classifier()
    return classifier.classify(
        symbol=symbol,
        higher_high=higher_high,
        higher_low=higher_low,
        lower_high=lower_high,
        lower_low=lower_low,
        price_change_1h=price_change_1h,
        price_change_4h=price_change_4h,
        vol_expansion_ratio=vol_expansion_ratio,
        range_vs_atr=range_vs_atr,
        ema_20=ema_20,
        ema_50=ema_50,
        current_price=current_price,
        moi_flip_rate=moi_flip_rate,
        directional_consistency=directional_consistency,
    )
