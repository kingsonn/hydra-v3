"""
Hybrid Signals V2 — Audit-Driven Redesign
==========================================

Based on intensive alpha brainstorm (Jan 2026).
Each signal has clear economic thesis, not just indicator conditions.

FIXED (Audit): Reduced to 5 core signals after removing:
- ADXExpansionMomentum (fake ADX - used trend.strength * 50)
- EMATrendContinuation (duplicate of TrendPullback)
- CompressionBreakout (merged with RangeBreakout)

Active Signals:
1. FundingPressureContinuation: Structural funding tax + trend alignment
2. TrendPullbackSignal: Pullback to EMA in confirmed trend
3. LiquidationCascade: Enter DURING cascade with trend confirmation  
4. CompressedRangeBreakout: Tight range + OI/liq confirmation on break
5. TrendExhaustionReversal: Extended move + structure break + unwind
6. SMACrossover: 1H bar SMA10/100 crossover (best implementation)

Key Principles:
- 4-36 hour holding period target
- Minimum 0.3-0.6% edge per trade after fees
- No microstructure noise (that's HFT territory)
- Forced flows > discretionary flows
"""
import time
from typing import Optional, Dict, List
from src.stage3_v3.models import (
    MarketState, HybridSignal, Direction, MarketRegime, SignalType
)
from src.stage3_v3.bias import (
    FUNDING_Z_SIGNIFICANT, FUNDING_Z_EXTREME, FUNDING_Z_DANGEROUS,
    LIQ_IMBALANCE_THRESHOLD, LIQ_IMBALANCE_STRONG,
)


# Minimum liquidation thresholds by symbol tier (lowered for realistic firing)
LIQ_THRESHOLDS: Dict[str, float] = {
    "BTCUSDT": 2_500_000,   # $2.5M for BTC (realistic cascade threshold)
    "ETHUSDT": 1_000_000,   # $1M for ETH
    "default": 250_000,     # $250k for alts
}


class FundingPressureContinuation:
    """
    FUNDING PRESSURE TREND CONTINUATION
    ====================================
    
    Economic Thesis:
    When funding is extreme, the crowded side pays 0.03-0.1% every 8 hours.
    At 0.1%, that's 0.3%/day bleeding. After 2-3 days, cumulative cost forces
    them to close. This is a STRUCTURAL TAX that cannot be avoided.
    
    The unwind happens in waves over 24-72h, not instantly.
    We trade WITH the trend, against the crowded side.
    
    Entry: Pullback to EMA20 in confirmed trend, funding extreme
    Stop: Beyond recent swing (1.2-2.0%)
    Target: 2.5R or funding normalization
    Hold: Up to 36 hours
    
    Frequency: 2-5/week across all symbols
    """
    
    def __init__(self):
        self.name = "FUNDING_PRESSURE"
        # FIXED (Audit): Use shared threshold constants for consistency
        self.funding_z_threshold = FUNDING_Z_SIGNIFICANT  # z > 1.0 = crowded
        self.funding_z_veto = FUNDING_Z_EXTREME           # z > 1.5 = veto opposite trades
        self.cumulative_threshold = 0.001   # 0.1% cumulative 24h funding
        self.min_trend_strength = 0.35
        self.max_vol_expansion = 2.5        # Avoid volatility spikes
    
    def evaluate(self, state: MarketState) -> Optional[HybridSignal]:
        """
        Entry requires ALL of:
        1. Funding z-score > 1.5 OR cumulative 24h funding > 0.15%
        2. Trend aligned with bias direction
        3. EMA20 > EMA50 (for longs) or EMA20 < EMA50 (for shorts)
        4. Price structure confirms (HH/HL or LH/LL)
        5. Pullback to within 0.3% of EMA20 (optimal entry)
        6. Not in volatility spike
        """
        
        # GATE 1: Need trending regime
        if state.regime not in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            return None
        
        # GATE 2: Avoid volatility spikes
        if state.vol_expansion_ratio > self.max_vol_expansion:
            return None
        
        # GATE 3: Check for funding pressure (z-score OR cumulative)
        funding_z_extreme = abs(state.funding_z) > self.funding_z_threshold
        cumulative_extreme = abs(state.cumulative_funding_24h) > self.cumulative_threshold
        
        if not (funding_z_extreme or cumulative_extreme):
            return None
        
        # Determine direction from funding
        # Positive funding = longs paying = longs crowded = SHORT bias
        # Negative funding = shorts paying = shorts crowded = LONG bias
        if state.funding_z > self.funding_z_threshold or state.cumulative_funding_24h > self.cumulative_threshold:
            funding_direction = Direction.SHORT  # Fade the crowded longs
        elif state.funding_z < -self.funding_z_threshold or state.cumulative_funding_24h < -self.cumulative_threshold:
            funding_direction = Direction.LONG   # Fade the crowded shorts
        else:
            return None
        
        # GATE 4: Trend must CONFIRM funding direction
        trend = state.trend
        if funding_direction == Direction.LONG:
            if not trend.is_uptrend():
                return None
          
            direction = Direction.LONG
        else:
            if not trend.is_downtrend():
                return None
            
            direction = Direction.SHORT
        
        # GATE 5: Trend strength minimum
        if trend.strength < self.min_trend_strength:
            return None
        
        # GATE 6: Pullback entry (price near EMA20) - RELAXED
        # RSI should be neutral (not overbought/oversold)
        has_pullback = trend.is_pullback_to_ema(threshold_pct=0.8)  # Widened from 0.3%
        rsi_ok = 35 < trend.rsi_14 < 65  # Widened from 40-60
        
        if not has_pullback and not rsi_ok:
            return None  # Need at least one timing condition
        
        # Calculate stop: beyond recent swing OR 1.2 × ATR
        if direction == Direction.LONG:
            swing_stop = (state.current_price - state.low_4h) / state.current_price
            stop_pct = max(0.012, min(0.020, swing_stop * 1.2))
        else:
            swing_stop = (state.high_4h - state.current_price) / state.current_price
            stop_pct = max(0.012, min(0.020, swing_stop * 1.2))
        
        # Target: 2.5R
        target_pct = stop_pct * 2.5
        
        # Confidence calculation
        confidence = 0.55
        confidence += min(0.15, abs(state.funding_z) / 8)    # Funding extremity
        confidence += min(0.10, trend.strength * 0.15)        # Trend strength
        if has_pullback:
            confidence += 0.10  # Pullback = better entry
        if rsi_ok:
            confidence += 0.05
        if funding_z_extreme and cumulative_extreme:
            confidence += 0.05  # Both conditions = stronger
        
        return HybridSignal(
            direction=direction,
            signal_type=SignalType.FUNDING_TREND,
            confidence=min(0.85, confidence),
            entry_price=state.current_price,
            stop_pct=stop_pct,
            target_pct=target_pct,
            reason=f"Funding z={state.funding_z:.1f}, cumul={state.cumulative_funding_24h*100:.2f}%, pullback entry",
            bias_strength=state.bias.strength,
            regime=state.regime,
        )


class TrendPullbackSignal:
    """
    TREND PULLBACK Signal (Daily Driver)
    
    Economic Thesis:
    In established trends, pullbacks to moving averages offer low-risk entries.
    The trend provides edge; the pullback provides timing.
    This is pure trend-following without requiring positioning data.
    
    Entry: Pullback to EMA20 in established trend with structure
    Stop: Beyond recent swing
    Target: 2R (previous impulse size)
    
    Frequency: 1-2/day - THIS IS THE MAIN SIGNAL
    """
    
    def __init__(self):
        self.name = "TREND_PULLBACK"
        self.min_trend_strength = 0.5
        self.rsi_range = (35, 65)  # Neutral RSI for optimal entry
    
    def evaluate(self, state: MarketState) -> Optional[HybridSignal]:
        """Evaluate trend pullback conditions"""
        
        # Need trending regime
        if state.regime not in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            return None
        
        # Need established trend
        trend = state.trend
        if trend.direction == Direction.NEUTRAL:
            return None
        if trend.strength < self.min_trend_strength:
            return None
        
        # RSI should be neutral (not chasing extreme)
        if not (self.rsi_range[0] < trend.rsi_14 < self.rsi_range[1]):
            return None
        
        # Need pullback to EMA (0.6% threshold for crypto volatility)
        if not trend.is_pullback_to_ema(threshold_pct=0.6):
            return None
        
        # Structure confirmation
        if trend.direction == Direction.LONG:
            if not (trend.higher_high or trend.higher_low):
                return None
            direction = Direction.LONG
        else:
            if not (trend.lower_high or trend.lower_low):
                return None
            direction = Direction.SHORT
        
        # Bias alignment bonus (not required but helps)
        bias_aligned = (
            (direction == Direction.LONG and state.bias.direction == Direction.LONG) or
            (direction == Direction.SHORT and state.bias.direction == Direction.SHORT)
        )
        
        # Calculate stop: beyond recent swing
        if direction == Direction.LONG:
            swing_stop = (state.current_price - state.low_4h) / state.current_price
            stop_pct = max(0.010, min(0.018, swing_stop * 1.1))
        else:
            swing_stop = (state.high_4h - state.current_price) / state.current_price
            stop_pct = max(0.010, min(0.018, swing_stop * 1.1))
        
        target_pct = stop_pct * 2.0  # 2R target
        
        # Confidence
        confidence = 0.55
        confidence += min(0.15, trend.strength * 0.2)
        if bias_aligned:
            confidence += 0.10
        if state.regime_confidence > 0.7:
            confidence += 0.05
        
        return HybridSignal(
            direction=direction,
            signal_type=SignalType.TREND_PULLBACK,
            confidence=min(0.80, confidence),
            entry_price=state.current_price,
            stop_pct=stop_pct,
            target_pct=target_pct,
            reason=f"Trend {direction.value}, pullback to EMA20, RSI={trend.rsi_14:.0f}",
            bias_strength=state.bias.strength if bias_aligned else 0,
            regime=state.regime,
        )


class LiquidationCascadeSignal:
    """
    LIQUIDATION CASCADE CONTINUATION
    =================================
    
    Economic Thesis:
    When price approaches liquidation clusters, positions are FORCIBLY closed
    by exchange. This is not discretionary — it's automatic. Liquidations 
    create market orders that push price further, triggering more liquidations.
    
    Large cascades ($5M+ for BTC) take 1-4 hours to fully work through.
    Liquidated positions are GONE — the opposing supply/demand is permanently
    removed until new positions are built.
    
    KEY CHANGE: Enter DURING cascade with trend confirmation, not after.
    
    Entry: Cascade detected + trend aligned + ATR expansion
    Stop: 1.0 × ATR from entry
    Target: 2R or flow reversal
    Hold: Up to 12 hours (momentum fades)
    
    Frequency: 3-8/week across all symbols
    """
    
    def __init__(self):
        self.name = "LIQUIDATION_CASCADE"
        self.liq_imbalance_threshold = 0.5   # 50%+ one-sided (was 60%)
        self.min_atr_expansion = 1.0         # Need volatility (was 1.2)
        self.max_cascade_age_minutes = 45    # Freshness filter (was 30)
        
        # Per-symbol state tracking
        self._cascade_state: Dict[str, dict] = {}
    
    def _get_liq_threshold(self, symbol: str) -> float:
        """Get liquidation USD threshold for symbol"""
        return LIQ_THRESHOLDS.get(symbol, LIQ_THRESHOLDS["default"])
    
    def evaluate(self, state: MarketState) -> Optional[HybridSignal]:
        """
        Entry requires ALL of:
        1. Liquidation USD > threshold ($5M BTC, $2M ETH, $500k alts)
        2. Liquidation imbalance > 60% one-sided
        3. Trend CONFIRMS cascade direction
        4. ATR expansion > 1.2 (volatility present)
        5. Cascade is fresh (< 30 min old)
        """
        
        symbol = state.symbol
        now = time.time()
        
        # Initialize state for symbol
        if symbol not in self._cascade_state:
            self._cascade_state[symbol] = {
                "detected": False,
                "direction": None,
                "detect_time": 0,
                "detect_price": 0,
            }
        
        cs = self._cascade_state[symbol]
        
        # GATE 1: Check for active cascade
        liq_threshold = self._get_liq_threshold(symbol)
        cascade_active = (
            state.liq_total_1h > liq_threshold and
            abs(state.liq_imbalance_1h) > self.liq_imbalance_threshold
        )
        
        if cascade_active:
            # Determine cascade direction
            # liq_imbalance > 0 = more long liquidations = longs getting wiped = price DOWN = SHORT
            # liq_imbalance < 0 = more short liquidations = shorts getting wiped = price UP = LONG
            cascade_dir = Direction.SHORT if state.liq_imbalance_1h > 0 else Direction.LONG
            
            if not cs["detected"]:
                # New cascade detected
                cs["detected"] = True
                cs["direction"] = cascade_dir
                cs["detect_time"] = now
                cs["detect_price"] = state.current_price
            
            # Check if still fresh
            age_minutes = (now - cs["detect_time"]) / 60
            if age_minutes > self.max_cascade_age_minutes:
                self._reset_state(symbol)
                return None
            
            # GATE 2: Trend must CONFIRM cascade direction
            trend = state.trend
            if cascade_dir == Direction.LONG:
                if not trend.is_uptrend():
                    return None  # Don't trade cascade against trend
            else:
                if not trend.is_downtrend():
                    return None
            
            # GATE 3: ATR expansion (need volatility)
            if state.vol_expansion_ratio < self.min_atr_expansion:
                return None
            
            # GATE 4: Don't enter if too extended already
            price_move = abs(state.current_price - cs["detect_price"]) / cs["detect_price"]
            if price_move > 0.02:  # Already moved 2%+
                return None
            
            direction = cascade_dir
            
            # Calculate stop: 1.0 × ATR from entry
            atr_pct = state.atr_14 / state.current_price if state.current_price > 0 else 0.015
            stop_pct = max(0.010, min(0.015, atr_pct))
            
            # Target: 2R
            target_pct = stop_pct * 2.0
            
            # Confidence based on cascade strength
            confidence = 0.60
            # Stronger imbalance = higher confidence
            confidence += min(0.15, (abs(state.liq_imbalance_1h) - self.liq_imbalance_threshold) * 0.5)
            # Larger cascade = higher confidence
            confidence += min(0.10, (state.liq_total_1h / liq_threshold - 1) * 0.1)
            
            liq_type = "shorts" if cascade_dir == Direction.LONG else "longs"
            
            # Reset state after signal (one signal per cascade)
            self._reset_state(symbol)
            
            return HybridSignal(
                direction=direction,
                signal_type=SignalType.LIQUIDATION_FOLLOW,
                confidence=min(0.80, confidence),
                entry_price=state.current_price,
                stop_pct=stop_pct,
                target_pct=target_pct,
                reason=f"Cascade: ${state.liq_total_1h/1e6:.1f}M {liq_type} liquidated, imbal={state.liq_imbalance_1h:.0%}",
                bias_strength=state.bias.strength,
                regime=state.regime,
            )
        
        else:
            # No active cascade, reset if we had one
            if cs["detected"]:
                self._reset_state(symbol)
        
        return None
    
    def _reset_state(self, symbol: str):
        """Reset cascade state for symbol"""
        self._cascade_state[symbol] = {
            "detected": False,
            "direction": None,
            "detect_time": 0,
            "detect_price": 0,
        }


class CompressedRangeBreakout:
    """
    COMPRESSED RANGE BREAKOUT
    ==========================
    
    Economic Thesis:
    After 18-48h of tight range (<3%), orders cluster outside the range:
    - Breakout buyers with stop-entries above
    - Breakout sellers with stop-entries below
    - Stop-losses from range traders
    
    When price breaks, these orders trigger simultaneously, creating a
    LIQUIDITY VACUUM in the breakout direction. This structural change
    persists for hours as trapped traders capitulate.
    
    KEY: Must confirm with OI expansion + liquidations on break.
    Fake breakouts don't have these.
    
    Entry: Break of tight range + OI increase + liquidations
    Stop: Just inside broken range (0.8-1.2%)
    Target: 1.5× range width (measured move)
    Hold: Up to 24 hours
    
    Frequency: 1-2/week per symbol
    """
    
    def __init__(self):
        self.name = "RANGE_BREAKOUT"
        self.max_range_pct = 0.04           # Range must be < 4% (was 3%)
        self.min_range_hours = 12           # Range must persist 12h+ for meaningful compression
        self.min_oi_increase = 0.002        # 0.2% OI increase on breakout (was 0.5%)
        self.min_liq_on_break = 50_000      # $50k liquidations on break (was $100k)
        
        # Per-symbol state tracking
        self._range_state: Dict[str, dict] = {}
    
    def evaluate(self, state: MarketState) -> Optional[HybridSignal]:
        """
        Entry requires ALL of:
        1. 24h range width < 3% (tight compression)
        2. Range persisted for 18+ hours
        3. Price breaks range by > 0.3%
        4. OI increased during breakout (new positions)
        5. Liquidations on breakout (stops triggered)
        6. Not counter to HTF trend
        """
        
        symbol = state.symbol
        now = time.time()
        
        # Initialize state for symbol
        if symbol not in self._range_state:
            self._range_state[symbol] = {
                "in_range": False,
                "range_start": 0,
                "range_high": 0,
                "range_low": 0,
                "last_oi": 0,
            }
        
        rs = self._range_state[symbol]
        
        # Calculate 24h range width
        if state.high_24h <= 0 or state.low_24h <= 0:
            return None
        
        mid_price = (state.high_24h + state.low_24h) / 2
        range_width_pct = (state.high_24h - state.low_24h) / mid_price
        
        # Phase 1: Detect compression
        if range_width_pct < self.max_range_pct:
            if not rs["in_range"]:
                # New range detected
                rs["in_range"] = True
                rs["range_start"] = now
                rs["range_high"] = state.high_24h
                rs["range_low"] = state.low_24h
                rs["last_oi"] = state.oi_delta_1h  # Track OI at range start
            else:
                # Update range bounds
                rs["range_high"] = state.high_24h
                rs["range_low"] = state.low_24h
            
            # Still in range, no signal yet
            return None
        
        # Phase 2: Check if breaking out from established range
        if not rs["in_range"]:
            return None
        
        range_hours = (now - rs["range_start"]) / 3600
        if range_hours < self.min_range_hours:
            # Range not established long enough, reset
            self._reset_state(symbol)
            return None
        
        # Phase 3: Detect breakout direction
        breakout_threshold = (rs["range_high"] - rs["range_low"]) * 0.1  # 10% of range
        
        if state.current_price > rs["range_high"] + breakout_threshold:
            direction = Direction.LONG
        elif state.current_price < rs["range_low"] - breakout_threshold:
            direction = Direction.SHORT
        else:
            # Price moved but not clear breakout yet
            return None
        
        # GATE 4: OI must increase on breakout (new positions, not just stops)
        oi_change = state.oi_delta_1h
        if oi_change < self.min_oi_increase:
            # No new positioning, likely fake breakout
            self._reset_state(symbol)
            return None
        
        # GATE 5: Liquidations on breakout (confirms stops triggered)
        if state.liq_total_1h < self.min_liq_on_break:
            # No liquidations, weak breakout
            self._reset_state(symbol)
            return None
        
        # GATE 6: Don't trade against HTF trend
        trend = state.trend
        if direction == Direction.LONG and trend.is_downtrend():
            self._reset_state(symbol)
            return None
        if direction == Direction.SHORT and trend.is_uptrend():
            self._reset_state(symbol)
            return None
        
        # Calculate stop: just inside broken range
        range_size = rs["range_high"] - rs["range_low"]
        if direction == Direction.LONG:
            stop_pct = (state.current_price - rs["range_low"]) / state.current_price
        else:
            stop_pct = (rs["range_high"] - state.current_price) / state.current_price
        
        stop_pct = max(0.008, min(0.012, stop_pct * 0.5))  # Tight stop inside range
        
        # Target: 1.5× range width (measured move)
        target_pct = (range_size / state.current_price) * 1.5
        target_pct = max(stop_pct * 2.0, min(0.03, target_pct))
        
        # Confidence based on confirmation strength
        confidence = 0.60
        if state.liq_total_1h > self.min_liq_on_break * 2:
            confidence += 0.10  # Strong liquidation confirmation
        if oi_change > self.min_oi_increase * 2:
            confidence += 0.05  # Strong OI confirmation
        if trend.direction == direction:
            confidence += 0.05  # Trend aligned
        
        breakout_type = "bullish" if direction == Direction.LONG else "bearish"
        
        # Reset state after signal
        self._reset_state(symbol)
        
        return HybridSignal(
            direction=direction,
            signal_type=SignalType.RANGE_BREAKOUT,
            confidence=min(0.80, confidence),
            entry_price=state.current_price,
            stop_pct=stop_pct,
            target_pct=target_pct,
            reason=f"{breakout_type} breakout after {range_hours:.0f}h range, OI+{oi_change*100:.1f}%, liq ${state.liq_total_1h/1000:.0f}k",
            bias_strength=state.bias.strength,
            regime=state.regime,
        )
    
    def _reset_state(self, symbol: str):
        """Reset range state for symbol"""
        self._range_state[symbol] = {
            "in_range": False,
            "range_start": 0,
            "range_high": 0,
            "range_low": 0,
            "last_oi": 0,
        }


class TrendExhaustionReversal:
    """
    TREND EXHAUSTION REVERSAL
    ==========================
    
    Economic Thesis:
    After an extended trend move (5-10%+ over 48h), late entrants are positioned
    at worst prices. When momentum stalls, they become trapped. As price starts
    reversing, their stops create fuel for the reversal.
    
    The unwind plays out over 8-24 hours as:
    1. Profit-taking from early trend followers
    2. Fading from contrarians
    3. Stop-hunting of late entrants
    
    KEY: Requires STRUCTURE BREAK confirmation, not just oscillator extreme.
    
    Entry: After 5%+ 48h move + structure break + OI declining
    Stop: Beyond recent swing high/low (1.5-2.5%)
    Target: 38-50% retracement of prior move
    Hold: Up to 48 hours
    
    Frequency: 2-4/month per symbol (countertrend = selective)
    """
    
    def __init__(self):
        self.name = "EXHAUSTION_REVERSAL"
        self.min_extension_48h = 0.05       # 5%+ move in 48h
        # FIXED (Audit): Use shared threshold constant for consistency
        self.funding_extreme = FUNDING_Z_EXTREME  # z-score threshold
        self.rsi_extreme = (25, 75)         # RSI extremes
    
    def evaluate(self, state: MarketState) -> Optional[HybridSignal]:
        """
        Entry requires ALL of:
        1. Price moved > 5% in 48h in one direction
        2. RSI extreme (> 75 or < 25)
        3. Funding extreme in direction of move
        4. Structure break formed (LH after uptrend, HL after downtrend)
        5. OI declining (positions closing, not adding)
        """
        
        trend = state.trend
        
        # Use 48h price change, fallback to 2x 24h if not available
        price_change_48h = state.price_change_48h
        if price_change_48h == 0:
            price_change_48h = state.price_change_24h * 1.5  # Rough estimate
        
        # GATE 1: Check for extension (5%+ in 48h)
        extended_up = price_change_48h > self.min_extension_48h
        extended_down = price_change_48h < -self.min_extension_48h
        
        if not (extended_up or extended_down):
            return None
        
        # GATE 2: RSI must be extreme
        if extended_up and trend.rsi_14 < self.rsi_extreme[1]:
            return None  # Not overbought enough
        if extended_down and trend.rsi_14 > self.rsi_extreme[0]:
            return None  # Not oversold enough
        
        # GATE 3: Funding must be extreme in direction of move
        if extended_up and state.funding_z < self.funding_extreme:
            return None  # Longs not crowded enough
        if extended_down and state.funding_z > -self.funding_extreme:
            return None  # Shorts not crowded enough
        
        # GATE 4: Structure break must be forming
        # After uptrend: need lower_high (LH) to form
        # After downtrend: need higher_low (HL) to form
        if extended_up:
            if not trend.lower_high:
                return None  # No reversal structure yet
            direction = Direction.SHORT
        else:
            if not trend.higher_low:
                return None  # No reversal structure yet
            direction = Direction.LONG
        
        # GATE 5: OI should be declining (unwind, not new shorts/longs)
        if state.oi_delta_24h > 0:
            return None  # OI still increasing, trend may continue
        
        # GATE 6: Price should be below EMA20 (for shorts) or above (for longs)
        if direction == Direction.SHORT and trend.price_vs_ema20 > 0.5:
            return None  # Still too far above EMA
        if direction == Direction.LONG and trend.price_vs_ema20 < -0.5:
            return None  # Still too far below EMA
        
        # Calculate stop: beyond recent swing
        if direction == Direction.SHORT:
            stop_pct = (state.high_4h - state.current_price) / state.current_price
            stop_pct = max(0.015, min(0.025, stop_pct * 1.2))
        else:
            stop_pct = (state.current_price - state.low_4h) / state.current_price
            stop_pct = max(0.015, min(0.025, stop_pct * 1.2))
        
        # Target: 38-50% retracement of 48h move
        retracement = abs(price_change_48h) * 0.38
        target_pct = max(stop_pct * 2.0, min(0.04, retracement))
        
        # Confidence based on evidence strength
        confidence = 0.55
        # RSI extremity
        if extended_up and trend.rsi_14 > 80:
            confidence += 0.05
        elif extended_down and trend.rsi_14 < 20:
            confidence += 0.05
        # Funding extremity
        if abs(state.funding_z) > 2.0:
            confidence += 0.05
        # OI unwind strength
        if state.oi_delta_24h < -0.02:  # 2%+ OI drop
            confidence += 0.05
        
        move_dir = "up" if extended_up else "down"
        
        return HybridSignal(
            direction=direction,
            signal_type=SignalType.EXHAUSTION_REVERSAL,
            confidence=min(0.75, confidence),
            entry_price=state.current_price,
            stop_pct=stop_pct,
            target_pct=target_pct,
            reason=f"Exhaustion: {abs(price_change_48h)*100:.1f}% {move_dir} 48h, RSI={trend.rsi_14:.0f}, funding z={state.funding_z:.1f}, structure break",
            bias_strength=0,  # Countertrend, no bias alignment
            regime=state.regime,
        )


# =============================================================================
# ENTRY-FIRST SIGNALS (Frequency Alpha)
# =============================================================================
# These fire more frequently and use positional data as GATES, not triggers.
# Philosophy: Trend systems create opportunities, our data decides safety.
# =============================================================================


class EMATrendContinuation:
    """
    DEPRECATED (Audit): Duplicate of TrendPullbackSignal - both detect EMA pullbacks.
    This signal is no longer used in the engine and should be deleted.
    Use TrendPullbackSignal instead.
    
    EMA TREND CONTINUATION (PRIMARY WORKHORSE)
    ==========================================
    
    Entry: EMA20 > EMA50 alignment + pullback to EMA20 + structure confirms
    
    Economic Thesis:
    Trend continuation after healthy pullback. Weak hands exit on pullback,
    strong hands reload. Price resumes trend direction.
    
    Gates: Funding opposing = veto, cascade = veto, chop = veto
    
    Frequency: 1-2/day across symbols
    Hold: 12-24 hours
    """
    
    def __init__(self):
        self.name = "EMA_CONTINUATION"
        self.pullback_threshold = 0.8  # Within 0.8% of EMA20
        self.min_trend_strength = 0.3
        self.max_extension = 0.03  # Don't enter if already moved 3%+ in 1h
        
        # Cooldown tracking per symbol
        self._last_signal: Dict[str, dict] = {}
    
    def evaluate(self, state: MarketState) -> Optional[HybridSignal]:
        """
        Entry requires:
        1. EMA20 > EMA50 (longs) or EMA20 < EMA50 (shorts)
        2. Price within 0.8% of EMA20 (pullback)
        3. Structure confirms (HH/HL or LH/LL)
        4. Not in chop regime
        5. Passes safety gates
        """
        trend = state.trend
        symbol = state.symbol
        
        # === HARD VETOES ===
        
        # Veto 1: Chop regime
        if state.regime == MarketRegime.CHOPPY:
            return None
        
        # Veto 2: Cascade active
        if state.cascade_active:
            return None
        
        # Veto 3: Already extended (missed the move)
        if abs(state.price_change_1h) > self.max_extension:
            return None
        
        # Veto 4: Volatility explosion
        if state.vol_expansion_ratio > 3.0:
            return None
        
        # Veto 5: Cooldown check (no signal within 2 hours)
        last = self._last_signal.get(symbol, {})
        if last.get("time", 0) > 0:
            hours_since = (state.timestamp_ms - last["time"]) / (1000 * 3600)
            if hours_since < 2:
                return None
            # Flip prevention: if direction changed within 4 hours, skip
            if hours_since < 4 and last.get("direction") is not None:
                # Will check direction below
                pass
        
        # === DIRECTION DETERMINATION ===
        
        # Check EMA alignment
        ema_bullish = trend.ema_20 > trend.ema_50
        ema_bearish = trend.ema_20 < trend.ema_50
        
        if not (ema_bullish or ema_bearish):
            return None  # EMAs too close
        
        # Check pullback to EMA20
        has_pullback = trend.is_pullback_to_ema(threshold_pct=self.pullback_threshold)
        if not has_pullback:
            return None
        
        # Determine direction and check structure
        if ema_bullish:
            # Need bullish structure
            if not (trend.higher_high or trend.higher_low):
                return None
            direction = Direction.LONG
            
            # Veto: Funding extreme against us
            if state.funding_z > FUNDING_Z_EXTREME:  # Longs crowded, risky to long
                return None
        else:
            # Need bearish structure
            if not (trend.lower_high or trend.lower_low):
                return None
            direction = Direction.SHORT
            
            # Veto: Funding extreme against us
            if state.funding_z < -FUNDING_Z_EXTREME:  # Shorts crowded, risky to short
                return None
        
        # Flip prevention
        last = self._last_signal.get(symbol, {})
        if last.get("direction") is not None:
            hours_since = (state.timestamp_ms - last.get("time", 0)) / (1000 * 3600)
            if hours_since < 4 and last["direction"] != direction:
                return None  # Don't flip direction within 4 hours
        
        # === SOFT GATES (affect confidence/size) ===
        
        confidence = 0.55
        size_modifier = 1.0
        
        # RSI check
        rsi_ok = 35 < trend.rsi_14 < 65
        if rsi_ok:
            confidence += 0.05
        
        # Funding aligned = boost
        if direction == Direction.LONG and state.funding_z < -0.5:
            confidence += 0.10  # Shorts crowded = fuel for longs
        elif direction == Direction.SHORT and state.funding_z > 0.5:
            confidence += 0.10  # Longs crowded = fuel for shorts
        
        # OI expanding in direction = boost
        if state.oi_delta_4h > 0.003:  # 0.3%+ OI growth
            confidence += 0.05
        elif state.oi_delta_4h < -0.003:  # OI contracting
            size_modifier = 0.7  # Reduce size
        
        # Trend strength
        if trend.strength > 0.5:
            confidence += 0.05
        elif trend.strength < self.min_trend_strength:
            return None  # Too weak
        
        # === CALCULATE STOP & TARGET ===
        
        # Stop: Below EMA50 or recent swing
        if direction == Direction.LONG:
            swing_stop = (state.current_price - state.low_4h) / state.current_price
            ema_stop = (state.current_price - trend.ema_50) / state.current_price
            stop_pct = max(0.010, min(0.018, max(swing_stop, ema_stop) * 1.1))
        else:
            swing_stop = (state.high_4h - state.current_price) / state.current_price
            ema_stop = (trend.ema_50 - state.current_price) / state.current_price
            stop_pct = max(0.010, min(0.018, max(swing_stop, ema_stop) * 1.1))
        
        # Target: 2R base, adjust for confidence
        target_pct = stop_pct * 2.0
        if confidence > 0.65:
            target_pct = stop_pct * 2.5
        
        # Record signal for cooldown
        self._last_signal[symbol] = {
            "time": state.timestamp_ms,
            "direction": direction
        }
        
        return HybridSignal(
            direction=direction,
            signal_type=SignalType.EMA_CONTINUATION,
            confidence=min(0.80, confidence),
            entry_price=state.current_price,
            stop_pct=stop_pct,
            target_pct=target_pct,
            reason=f"EMA continuation: EMA20{'>' if ema_bullish else '<'}EMA50, pullback entry, funding_z={state.funding_z:.2f}",
            bias_strength=state.bias.strength if state.bias else 0,
            regime=state.regime,
        )


class ADXExpansionMomentum:
    """
    DEPRECATED (Audit): This signal uses fake ADX (trend.strength * 50).
    Real ADX requires +DI/-DI calculation which is not implemented.
    This signal is no longer used in the engine and should be deleted.
    
    ADX EXPANSION MOMENTUM
    ======================
    
    Entry: ADX > 25 and rising (from <20), clear DI separation
    
    Economic Thesis:
    ADX expansion = volatility clustering. When ADX breaks above 25 from
    a low base, trend strength is confirmed and momentum tends to continue.
    
    Gates: Funding opposing = veto, late entry (ADX>40) = veto
    
    Frequency: 0.5-1/day across symbols
    Hold: 12-18 hours
    """
    
    def __init__(self):
        self.name = "ADX_EXPANSION"
        self.adx_entry_threshold = 25
        self.adx_late_threshold = 40  # Don't enter if ADX already high
        self.min_di_separation = 8  # +DI and -DI must be 8+ apart
        
        self._last_signal: Dict[str, dict] = {}
        self._adx_state: Dict[str, dict] = {}  # Track ADX values per bar
    
    def evaluate(self, state: MarketState) -> Optional[HybridSignal]:
        """
        Entry requires:
        1. ADX > 25 AND was < 20 recently (expansion)
        2. +DI/-DI clearly separated
        3. Price confirms direction
        4. Not in chop, not in cascade
        """
        trend = state.trend
        symbol = state.symbol
        
        # We need ADX data - derive from trend strength as proxy
        # In real implementation, would have actual ADX
        # Proxy: trend.strength * 50 approximates ADX
        adx_proxy = trend.strength * 50
        
        # Initialize state for symbol
        if symbol not in self._adx_state:
            self._adx_state[symbol] = {"history": [], "last_bar_count": 0}
        
        adx_st = self._adx_state[symbol]
        
        # Only update history when bar count changes (new 1H bar from bootstrap)
        bar_count = len(state.bar_closes_1h)
        if bar_count > adx_st["last_bar_count"] and bar_count > 0:
            adx_st["history"].append(adx_proxy)
            adx_st["last_bar_count"] = bar_count
            if len(adx_st["history"]) > 10:
                adx_st["history"] = adx_st["history"][-10:]
        
        # === HARD VETOES ===
        
        if state.regime == MarketRegime.CHOPPY:
            return None
        
        if state.cascade_active:
            return None
        
        if state.vol_expansion_ratio > 3.0:
            return None
        
        # Cooldown
        last = self._last_signal.get(symbol, {})
        if last.get("time", 0) > 0:
            hours_since = (state.timestamp_ms - last["time"]) / (1000 * 3600)
            if hours_since < 3:
                return None
        
        # === ADX EXPANSION CHECK ===
        
        # Current ADX must be > 25
        if adx_proxy < self.adx_entry_threshold:
            return None
        
        # ADX must be rising (was lower recently)
        if len(adx_st["history"]) >= 5:
            recent_min = min(adx_st["history"][-5:-1])
            if recent_min >= 20:
                return None  # Wasn't low enough recently
        else:
            return None  # Not enough history
        
        # Late entry filter
        if adx_proxy > self.adx_late_threshold:
            return None
        
        # === DIRECTION FROM EMA ===
        
        if trend.ema_20 > trend.ema_50 * 1.002:  # Clear bullish
            direction = Direction.LONG
            if state.funding_z > FUNDING_Z_SIGNIFICANT:  # Longs crowded
                return None
        elif trend.ema_20 < trend.ema_50 * 0.998:  # Clear bearish
            direction = Direction.SHORT
            if state.funding_z < -FUNDING_Z_SIGNIFICANT:  # Shorts crowded
                return None
        else:
            return None  # No clear direction
        
        # Flip prevention
        last = self._last_signal.get(symbol, {})
        if last.get("direction") and last["direction"] != direction:
            hours_since = (state.timestamp_ms - last.get("time", 0)) / (1000 * 3600)
            if hours_since < 4:
                return None
        
        # === CONFIDENCE CALCULATION ===
        
        confidence = 0.55
        
        # Fresh ADX cross (just crossed 25) = boost
        if len(adx_st["history"]) >= 2:
            prev_adx = adx_st["history"][-2]
            if prev_adx < 25 and adx_proxy >= 25:
                confidence += 0.10  # Fresh cross
        
        # Funding aligned
        if direction == Direction.LONG and state.funding_z < -0.5:
            confidence += 0.08
        elif direction == Direction.SHORT and state.funding_z > 0.5:
            confidence += 0.08
        
        # Vol expansion in goldilocks zone
        if 1.2 < state.vol_expansion_ratio < 2.0:
            confidence += 0.05
        
        # === STOP & TARGET ===
        
        # ATR-based stop
        atr_stop = state.atr_14 / state.current_price if state.current_price > 0 else 0.015
        stop_pct = max(0.012, min(0.020, atr_stop * 1.2))
        
        # Target: 2R
        target_pct = stop_pct * 2.0
        
        self._last_signal[symbol] = {
            "time": state.timestamp_ms,
            "direction": direction
        }
        
        return HybridSignal(
            direction=direction,
            signal_type=SignalType.ADX_EXPANSION,
            confidence=min(0.75, confidence),
            entry_price=state.current_price,
            stop_pct=stop_pct,
            target_pct=target_pct,
            reason=f"ADX expansion: strength={trend.strength:.2f}, vol_exp={state.vol_expansion_ratio:.2f}",
            bias_strength=state.bias.strength if state.bias else 0,
            regime=state.regime,
        )


class StructureBreakRetest:
    """
    STRUCTURE BREAK + RETEST
    ========================
    
    Entry: Price breaks 4H high/low, retests, holds
    
    Economic Thesis:
    Support becomes resistance (and vice versa). When price breaks a level
    and retests it, trapped traders from the wrong side capitulate, providing
    fuel for continuation.
    
    Frequency: 0.5-1/day across symbols
    Hold: 8-12 hours
    """
    
    def __init__(self):
        self.name = "STRUCTURE_BREAK"
        self.break_threshold = 0.003  # 0.3% beyond level to confirm break
        self.retest_threshold = 0.003  # Within 0.3% of level for retest
        self.max_retest_time_hours = 6  # Retest must happen within 6h of break
        
        self._break_state: Dict[str, dict] = {}
        self._last_signal: Dict[str, dict] = {}
    
    def evaluate(self, state: MarketState) -> Optional[HybridSignal]:
        """
        Entry requires:
        1. Price broke 4H high/low
        2. Price retested the level
        3. Price holding above/below (2+ bars)
        """
        symbol = state.symbol
        price = state.current_price
        
        # === HARD VETOES ===
        
        if state.cascade_active:
            return None
        
        if state.vol_expansion_ratio > 2.5:
            return None
        
        if state.regime == MarketRegime.CHOPPY:
            return None
        
        # Cooldown
        last = self._last_signal.get(symbol, {})
        if last.get("time", 0) > 0:
            hours_since = (state.timestamp_ms - last["time"]) / (1000 * 3600)
            if hours_since < 2:
                return None
        
        # === TRACK BREAKS ===
        
        if symbol not in self._break_state:
            self._break_state[symbol] = {
                "level": 0,
                "direction": None,
                "break_time": 0,
                "retested": False,
                "hold_count": 0,
                "last_bar_count": 0  # Track bar changes for hold_count
            }
        
        bs = self._break_state[symbol]
        
        # Check for new break
        high_4h = state.high_4h
        low_4h = state.low_4h
        
        # Bullish break (above 4H high)
        if price > high_4h * (1 + self.break_threshold) and bs["direction"] != Direction.LONG:
            bs["level"] = high_4h
            bs["direction"] = Direction.LONG
            bs["break_time"] = state.timestamp_ms
            bs["retested"] = False
            bs["hold_count"] = 0
            return None  # Wait for retest
        
        # Bearish break (below 4H low)
        if price < low_4h * (1 - self.break_threshold) and bs["direction"] != Direction.SHORT:
            bs["level"] = low_4h
            bs["direction"] = Direction.SHORT
            bs["break_time"] = state.timestamp_ms
            bs["retested"] = False
            bs["hold_count"] = 0
            return None  # Wait for retest
        
        # No active break
        if bs["direction"] is None:
            return None
        
        # Check retest timeout
        hours_since_break = (state.timestamp_ms - bs["break_time"]) / (1000 * 3600)
        if hours_since_break > self.max_retest_time_hours:
            bs["direction"] = None  # Reset, too old
            return None
        
        level = bs["level"]
        direction = bs["direction"]
        
        # Check for retest
        if not bs["retested"]:
            if direction == Direction.LONG:
                # Retest = price came back down near the level
                if abs(price - level) / level < self.retest_threshold:
                    bs["retested"] = True
                    bs["hold_count"] = 0
            else:
                # Retest = price came back up near the level
                if abs(price - level) / level < self.retest_threshold:
                    bs["retested"] = True
                    bs["hold_count"] = 0
            return None  # Wait for hold confirmation
        
        # Check hold (price back in direction) - only increment on new bars
        bar_count = len(state.bar_closes_1h)
        is_new_bar = bar_count > bs["last_bar_count"] and bar_count > 0
        
        if is_new_bar:
            bs["last_bar_count"] = bar_count
            if direction == Direction.LONG:
                if price > level * (1 + self.retest_threshold / 2):
                    bs["hold_count"] += 1
                else:
                    bs["hold_count"] = 0
            else:
                if price < level * (1 - self.retest_threshold / 2):
                    bs["hold_count"] += 1
                else:
                    bs["hold_count"] = 0
        
        # Need 2+ bars holding
        if bs["hold_count"] < 2:
            return None
        
        # === FUNDING VETO ===
        
        if direction == Direction.LONG and state.funding_z > FUNDING_Z_EXTREME:
            return None
        if direction == Direction.SHORT and state.funding_z < -FUNDING_Z_EXTREME:
            return None
        
        # Flip prevention
        last = self._last_signal.get(symbol, {})
        if last.get("direction") and last["direction"] != direction:
            hours_since = (state.timestamp_ms - last.get("time", 0)) / (1000 * 3600)
            if hours_since < 4:
                return None
        
        # === CONFIDENCE ===
        
        confidence = 0.58
        
        # Liquidations aligned = boost
        if direction == Direction.LONG and state.liq_imbalance_1h > 0.3:
            confidence += 0.08  # Shorts getting liquidated
        elif direction == Direction.SHORT and state.liq_imbalance_1h < -0.3:
            confidence += 0.08  # Longs getting liquidated
        
        # Funding aligned
        if direction == Direction.LONG and state.funding_z < -0.5:
            confidence += 0.05
        elif direction == Direction.SHORT and state.funding_z > 0.5:
            confidence += 0.05
        
        # === STOP & TARGET ===
        
        # Stop: Just below retested level
        stop_pct = max(0.008, min(0.012, self.retest_threshold * 1.5))
        
        # Target: Next structure or 1.5R
        target_pct = stop_pct * 1.8
        
        # Reset break state
        self._break_state[symbol] = {
            "level": 0, "direction": None, "break_time": 0,
            "retested": False, "hold_count": 0
        }
        
        self._last_signal[symbol] = {
            "time": state.timestamp_ms,
            "direction": direction
        }
        
        return HybridSignal(
            direction=direction,
            signal_type=SignalType.STRUCTURE_BREAK,
            confidence=min(0.75, confidence),
            entry_price=price,
            stop_pct=stop_pct,
            target_pct=target_pct,
            reason=f"Structure break retest: level={level:.2f}, held {bs['hold_count']} bars",
            bias_strength=state.bias.strength if state.bias else 0,
            regime=state.regime,
        )


class CompressionBreakout:
    """
    COMPRESSION BREAKOUT
    ====================
    
    Entry: Tight range (<2.5%) for 6+ hours, breaks with vol expansion
    
    Economic Thesis:
    Energy builds during compression as stops cluster above/below range.
    When range breaks with volume, those stops trigger and fuel continuation.
    
    Frequency: 0.3-0.5/day across symbols
    Hold: 6-12 hours
    """
    
    def __init__(self):
        self.name = "COMPRESSION_BREAK"
        self.max_range_pct = 0.025  # Range must be < 2.5%
        self.min_range_hours = 6    # Range must persist 6h+
        self.break_threshold = 0.004  # 0.4% beyond range to confirm
        self.min_vol_expansion = 1.3  # Need vol expansion on break
        
        self._range_state: Dict[str, dict] = {}
        self._last_signal: Dict[str, dict] = {}
    
    def evaluate(self, state: MarketState) -> Optional[HybridSignal]:
        """
        Entry requires:
        1. 4H range < 2.5% for 6+ hours
        2. Price breaks range by > 0.4%
        3. Vol expansion > 1.3 on break
        4. Not counter to HTF trend
        """
        symbol = state.symbol
        price = state.current_price
        
        # === HARD VETOES ===
        
        if state.cascade_active:
            return None
        
        if state.regime == MarketRegime.CHOPPY:
            return None
        
        # Cooldown
        last = self._last_signal.get(symbol, {})
        if last.get("time", 0) > 0:
            hours_since = (state.timestamp_ms - last["time"]) / (1000 * 3600)
            if hours_since < 3:
                return None
        
        # === RANGE DETECTION ===
        
        if symbol not in self._range_state:
            self._range_state[symbol] = {
                "in_range": False,
                "range_start": 0,
                "range_high": 0,
                "range_low": float('inf'),
            }
        
        rs = self._range_state[symbol]
        high_4h = state.high_4h
        low_4h = state.low_4h
        
        # Calculate current range
        if low_4h > 0:
            current_range = (high_4h - low_4h) / low_4h
        else:
            return None
        
        # Check if in compression
        if current_range < self.max_range_pct:
            if not rs["in_range"]:
                # Start tracking range
                rs["in_range"] = True
                rs["range_start"] = state.timestamp_ms
                rs["range_high"] = high_4h
                rs["range_low"] = low_4h
            else:
                # Update range bounds
                rs["range_high"] = max(rs["range_high"], high_4h)
                rs["range_low"] = min(rs["range_low"], low_4h)
        else:
            # Range too wide, reset
            rs["in_range"] = False
            return None
        
        # Check range duration
        hours_in_range = (state.timestamp_ms - rs["range_start"]) / (1000 * 3600)
        if hours_in_range < self.min_range_hours:
            return None  # Not long enough
        
        # === CHECK FOR BREAKOUT ===
        
        range_high = rs["range_high"]
        range_low = rs["range_low"]
        
        # Bullish breakout
        if price > range_high * (1 + self.break_threshold):
            direction = Direction.LONG
        # Bearish breakout
        elif price < range_low * (1 - self.break_threshold):
            direction = Direction.SHORT
        else:
            return None  # No breakout yet
        
        # === CONFIRMATION GATES ===
        
        # Vol expansion required
        if state.vol_expansion_ratio < self.min_vol_expansion:
            return None
        
        # Don't trade against HTF trend
        if direction == Direction.LONG and state.regime == MarketRegime.TRENDING_DOWN:
            return None
        if direction == Direction.SHORT and state.regime == MarketRegime.TRENDING_UP:
            return None
        
        # Funding veto (FIXED: Audit - use shared threshold)
        if direction == Direction.LONG and state.funding_z > FUNDING_Z_EXTREME:
            return None
        if direction == Direction.SHORT and state.funding_z < -FUNDING_Z_EXTREME:
            return None
        
        # OI should be increasing (new positions)
        if state.oi_delta_1h < 0:
            return None  # No new money
        
        # Flip prevention
        last = self._last_signal.get(symbol, {})
        if last.get("direction") and last["direction"] != direction:
            hours_since = (state.timestamp_ms - last.get("time", 0)) / (1000 * 3600)
            if hours_since < 4:
                return None
        
        # === CONFIDENCE ===
        
        confidence = 0.60
        
        # OI increase boost
        if state.oi_delta_1h > 0.003:
            confidence += 0.08
        
        # Liquidations on break
        if state.liq_total_1h > 50000:
            if direction == Direction.LONG and state.liq_imbalance_1h > 0.3:
                confidence += 0.07
            elif direction == Direction.SHORT and state.liq_imbalance_1h < -0.3:
                confidence += 0.07
        
        # Funding aligned
        if direction == Direction.LONG and state.funding_z < -0.5:
            confidence += 0.05
        elif direction == Direction.SHORT and state.funding_z > 0.5:
            confidence += 0.05
        
        # === STOP & TARGET ===
        
        # Stop: Inside range
        range_width = (range_high - range_low) / range_low
        stop_pct = max(0.006, min(0.010, range_width * 0.5))
        
        # Target: 1.5× range width
        target_pct = range_width * 1.5
        
        # Reset range state
        rs["in_range"] = False
        
        self._last_signal[symbol] = {
            "time": state.timestamp_ms,
            "direction": direction
        }
        
        return HybridSignal(
            direction=direction,
            signal_type=SignalType.COMPRESSION_BREAK,
            confidence=min(0.78, confidence),
            entry_price=price,
            stop_pct=stop_pct,
            target_pct=target_pct,
            reason=f"Compression breakout: {hours_in_range:.1f}h range, vol_exp={state.vol_expansion_ratio:.2f}",
            bias_strength=state.bias.strength if state.bias else 0,
            regime=state.regime,
        )


class SMACrossover:
    """
    SMA 10/100 CROSSOVER
    ====================
    
    Entry: SMA10 crosses SMA100 (golden cross = long, death cross = short)
    
    Economic Thesis:
    Classic momentum signal. When fast MA crosses slow MA, it confirms
    trend change. Works because millions watch these levels.
    
    Uses 1H bar closes from bootstrap (up to 250 bars available).
    
    Timeframe: 1H bars
    Frequency: ~1-2/week per symbol (real crossovers are rare)
    Hold: 12-24 hours
    """
    
    def __init__(self):
        self.name = "SMA_CROSSOVER"
        self.sma_fast_period = 10
        self.sma_slow_period = 100
        
        # Track last signal and previous SMA values per symbol
        self._last_signal: Dict[str, dict] = {}
        self._prev_sma: Dict[str, dict] = {}  # Store previous bar's SMAs for crossover detection
    
    def evaluate(self, state: MarketState) -> Optional[HybridSignal]:
        """
        Entry on SMA10/SMA100 crossover with safety gates.
        Uses bar_closes_1h from MarketState (populated from bootstrap).
        """
        symbol = state.symbol
        price = state.current_price
        
        # Get 1H bar closes from bootstrap (via MarketState)
        closes = state.bar_closes_1h
        
        # Need at least 101 bars (100 for SMA100 + 1 for previous bar comparison)
        if len(closes) < 101:
            return None
        
        # Calculate current SMAs from bootstrap data
        sma10 = sum(closes[-10:]) / 10
        sma100 = sum(closes[-100:]) / 100
        
        # Calculate previous bar's SMAs (shift window back by 1)
        prev_sma10 = sum(closes[-11:-1]) / 10
        prev_sma100 = sum(closes[-101:-1]) / 100
        
        # Detect crossover
        golden_cross = prev_sma10 <= prev_sma100 and sma10 > sma100
        death_cross = prev_sma10 >= prev_sma100 and sma10 < sma100
        
        if not (golden_cross or death_cross):
            return None  # No crossover
        
        direction = Direction.LONG if golden_cross else Direction.SHORT
        
        # Prevent firing multiple times for same bar count (same crossover)
        last = self._last_signal.get(symbol, {})
        if last.get("bar_count") == len(closes):
            return None  # Already fired for this bar
        
        # === HARD VETOES ===
        
        # Chop regime
        if state.regime == MarketRegime.CHOPPY:
            return None
        
        # Cascade active
        if state.cascade_active:
            return None
        
        # Volatility explosion
        if state.vol_expansion_ratio > 3.0:
            return None
        
        # Already extended
        if abs(state.price_change_1h) > 0.025:  # >2.5% move
            return None
        
        # Funding extreme against us (FIXED: Audit - use shared threshold)
        if direction == Direction.LONG and state.funding_z > FUNDING_Z_EXTREME:
            return None
        if direction == Direction.SHORT and state.funding_z < -FUNDING_Z_EXTREME:
            return None
        
        # Cooldown: 24 hours between signals (crossovers should be rare)
        if last.get("time", 0) > 0:
            hours_since = (state.timestamp_ms - last["time"]) / (1000 * 3600)
            if hours_since < 24:
                return None
        
        # Flip prevention: 48 hours if direction changed
        if last.get("direction") and last["direction"] != direction:
            hours_since = (state.timestamp_ms - last.get("time", 0)) / (1000 * 3600)
            if hours_since < 48:
                return None
        
        # === SOFT GATES & CONFIDENCE ===
        
        confidence = 0.55  # Base confidence for crossover
        
        # Trend alignment boost
        if direction == Direction.LONG and state.regime == MarketRegime.TRENDING_UP:
            confidence += 0.10
        elif direction == Direction.SHORT and state.regime == MarketRegime.TRENDING_DOWN:
            confidence += 0.10
        
        # Funding aligned boost
        if direction == Direction.LONG and state.funding_z < -0.5:
            confidence += 0.08
        elif direction == Direction.SHORT and state.funding_z > 0.5:
            confidence += 0.08
        
        # OI expanding
        if state.oi_delta_4h > 0.003:
            confidence += 0.05
        elif state.oi_delta_4h < -0.003:
            confidence -= 0.05
        
        # Structure confirms
        if state.trend:
            if direction == Direction.LONG and (state.trend.higher_high or state.trend.higher_low):
                confidence += 0.05
            elif direction == Direction.SHORT and (state.trend.lower_high or state.trend.lower_low):
                confidence += 0.05
        
        # Vol expansion in good zone
        if 1.2 < state.vol_expansion_ratio < 2.0:
            confidence += 0.05
        
        # Minimum confidence threshold
        if confidence < 0.50:
            return None
        
        # === STOP & TARGET ===
        
        # Stop: 1.5% base for crossover trades (wider stops)
        atr_pct = state.atr_14 / price if price > 0 else 0.015
        stop_pct = max(0.012, min(0.020, atr_pct * 1.2))
        
        # Target: 2R base
        target_pct = stop_pct * 2.0
        
        # If highly confident, extend target
        if confidence > 0.65:
            target_pct = stop_pct * 2.5
        
        # Record signal with bar count to prevent re-firing
        self._last_signal[symbol] = {
            "time": state.timestamp_ms,
            "bar_count": len(closes),
            "direction": direction
        }
        
        cross_type = "Golden" if direction == Direction.LONG else "Death"
        return HybridSignal(
            direction=direction,
            signal_type=SignalType.SMA_CROSSOVER,
            confidence=min(0.75, confidence),
            entry_price=price,
            stop_pct=stop_pct,
            target_pct=target_pct,
            reason=f"SMA {cross_type} Cross (1H): SMA10={sma10:.2f}, SMA100={sma100:.2f}, bars={len(closes)}",
            bias_strength=state.bias.strength if state.bias else 0,
            regime=state.regime,
        )


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Positional signals (background alpha)
FundingTrendSignal = FundingPressureContinuation
LiquidationFollowSignal = LiquidationCascadeSignal
RangeBreakoutSignal = CompressedRangeBreakout
ExhaustionReversalSignal = TrendExhaustionReversal
