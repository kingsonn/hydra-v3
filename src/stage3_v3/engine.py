"""Hybrid Alpha Engine
===================

Main orchestrator implementing the 3-layer gating decision hierarchy:

LAYER 1 - BIAS (Hourly): Directional bias from positioning data
  - Computed hourly (slow-changing structural view)
  - HARD GATE: Blocks trades against strong bias
  
LAYER 2 - REGIME (Per-Minute): Market structure classification  
  - Computed every minute on bar close
  - HARD GATE: CHOPPY = NO TRADE
  - 4 regimes: TRENDING_UP, TRENDING_DOWN, RANGING, CHOPPY

LAYER 3 - ENTRY TIMING: Tactical entry optimization
  - Pullback to EMA as confidence modifier (not hard gate)
  - Applied per-signal based on strategy type

Target: 1-2 signals per day, 0.15%+ edge per trade
"""
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import structlog

from src.stage3_v3.models import (
    MarketState, HybridSignal, Direction, MarketRegime, SignalType, Bias, TrendState
)
from src.stage3_v3.bias import BiasCalculator
from src.stage3_v3.regime import RegimeClassifier, get_regime_classifier
from src.stage3_v3.trend import TrendAnalyzer
from src.stage3_v3.signals import (
    FundingTrendSignal,
    TrendPullbackSignal,
    LiquidationFollowSignal,
    RangeBreakoutSignal,
    ExhaustionReversalSignal,
    SMACrossover,
)
# FIXED (Audit): Removed ADXExpansionMomentum (fake ADX), EMATrendContinuation (duplicate),
# and CompressionBreakout (merged with RangeBreakout). Kept 5 core signals.

logger = structlog.get_logger(__name__)


# =============================================================================
# RECOMPUTATION TIMING CONSTANTS
# =============================================================================
BIAS_RECOMPUTE_INTERVAL_SEC = 3600      # Recompute bias every hour
REGIME_RECOMPUTE_INTERVAL_SEC = 60      # Recompute regime every minute
ENTRY_CHECK_INTERVAL_SEC = 60           # Check entry conditions every minute


@dataclass
class GatingState:
    """
    Persisted state for the 3-layer gating system.
    Tracks when each layer was last computed and current values.
    """
    # Layer 1: Bias
    bias: Bias = field(default_factory=Bias)
    bias_last_computed: float = 0.0
    
    # Layer 2: Regime  
    regime: MarketRegime = MarketRegime.CHOPPY
    regime_confidence: float = 0.5
    regime_last_computed: float = 0.0
    
    # Layer 3: Entry timing
    pullback_active: bool = False
    price_vs_ema20: float = 0.0
    entry_state: str = "WAITING"  # WAITING, PULLBACK_READY, MOMENTUM_ENTRY
    
    def needs_bias_update(self) -> bool:
        """Check if bias needs recomputation (hourly)"""
        return (time.time() - self.bias_last_computed) > BIAS_RECOMPUTE_INTERVAL_SEC
    
    def needs_regime_update(self) -> bool:
        """Check if regime needs recomputation (per-minute)"""
        return (time.time() - self.regime_last_computed) > REGIME_RECOMPUTE_INTERVAL_SEC
    
    def to_dict(self) -> Dict:
        """Convert to dict for dashboard display"""
        return {
            "bias": {
                "direction": self.bias.direction.value,
                "strength": round(self.bias.strength, 2),
                "funding_score": round(self.bias.funding_score, 2),
                "liquidation_score": round(self.bias.liquidation_score, 2),
                "oi_score": round(self.bias.oi_score, 2),
                "trend_score": round(self.bias.trend_score, 2),
                "reason": self.bias.reason,
                "last_computed_ago_sec": int(time.time() - self.bias_last_computed),
            },
            "regime": {
                "current": self.regime.value,
                "confidence": round(self.regime_confidence, 2),
                "last_computed_ago_sec": int(time.time() - self.regime_last_computed),
            },
            "entry": {
                "pullback_active": self.pullback_active,
                "price_vs_ema20_pct": round(self.price_vs_ema20, 3),
                "state": self.entry_state,
            },
        }


@dataclass
class PositionInfo:
    """Track open position"""
    symbol: str
    direction: Direction
    entry_price: float
    entry_time: float
    size: float
    stop_price: float
    target_price: float
    signal_type: SignalType


@dataclass
class EngineConfig:
    """Engine configuration"""
    # Risk
    base_risk_pct: float = 0.02  # 2% base risk per trade
    max_risk_pct: float = 0.05  # 5% max risk
    max_daily_trades: int = 4
    max_consecutive_losses: int = 3
    
    # Time filters
    avoid_funding_minutes: int = 5  # Avoid 5 min around funding
    min_volume_ratio: float = 0.5  # Minimum volume vs average
    
    # Volatility
    max_vol_expansion: float = 3.0  # Don't trade in chaos
    
    # Correlation
    max_correlation_positions: int = 2  # Max positions in same direction
    
    # Bias gating (LAYER 1)
    bias_hard_gate: bool = True           # Enable bias as hard gate
    bias_gate_threshold: float = 0.5      # Block trades if bias strength > this AND opposite direction


class HybridAlphaEngine:
    """
    Main engine orchestrating all components.
    
    This is the brain that decides:
    - Whether to trade
    - What direction
    - When to enter
    - How much to risk
    """
    
    def __init__(self, symbols: List[str], config: Optional[EngineConfig] = None):
        self.symbols = symbols
        self.config = config or EngineConfig()
        
        # Core components
        self.bias_calculator = BiasCalculator()
        self.regime_classifier = get_regime_classifier()  # Singleton with persistence
        self.trend_analyzers: Dict[str, TrendAnalyzer] = {
            s: TrendAnalyzer() for s in symbols
        }
        
        # Signals (per symbol)
        # FIXED (Audit): Reduced to 5 core signals. Removed:
        # - ADXExpansionMomentum (fake ADX using trend.strength * 50)
        # - EMATrendContinuation (duplicate of TrendPullback)
        # - CompressionBreakout (merged into RangeBreakout)
        # - StructureBreakRetest (kept but not in core set)
        self.signals: Dict[str, Dict[str, object]] = {
            s: {
                "funding_trend": FundingTrendSignal(),
                "trend_pullback": TrendPullbackSignal(),
                "liquidation_follow": LiquidationFollowSignal(),
                "range_breakout": RangeBreakoutSignal(),
                "exhaustion_reversal": ExhaustionReversalSignal(),
                "sma_crossover": SMACrossover(),  # Best implemented signal per audit
            }
            for s in symbols
        }
        
        # State tracking
        self.positions: Dict[str, PositionInfo] = {}
        self.daily_trades: int = 0
        self.consecutive_losses: int = 0
        self.last_trade_day: int = 0
        
        # 3-Layer Gating State (per symbol)
        self.gating_states: Dict[str, GatingState] = {
            s: GatingState() for s in symbols
        }
        
        # Stats
        self.total_signals_generated = 0
        self.total_signals_vetoed = 0
        self.total_bias_blocked = 0
        self.total_regime_blocked = 0
        
        logger.info("hybrid_alpha_engine_initialized", symbols=symbols)
    
    def update_price(
        self,
        symbol: str,
        price: float,
        high: float,
        low: float,
        timestamp_ms: int,
    ):
        """Update price data for trend analysis"""
        if symbol in self.trend_analyzers:
            self.trend_analyzers[symbol].update(price, high, low, timestamp_ms)
    
    def evaluate(
        self,
        symbol: str,
        # Core data
        current_price: float,
        timestamp_ms: int,
        # Positioning data
        funding_z: float,
        funding_rate: float,
        cumulative_funding_24h: float,
        oi_delta_1h: float,
        oi_delta_4h: float,
        oi_delta_24h: float,
        liq_imbalance_1h: float,
        liq_imbalance_4h: float,
        liq_total_1h: float,
        liq_long_1h: float,
        liq_short_1h: float,
        cascade_active: bool,
        liq_exhaustion: bool,
        # Volatility
        atr_14: float,
        vol_expansion_ratio: float,
        # Price data
        high_4h: float,
        low_4h: float,
        high_24h: float,
        low_24h: float,
        price_change_1h: float,
        price_change_4h: float,
        price_change_24h: float,
        price_change_48h: float = 0.0,
        # Optional
        volume_ratio: float = 1.0,
        moi_flip_rate: float = 0.0,
        # 1H bar data from bootstrap (FIXED: Audit - needed for proper timeframe alignment)
        bar_closes_1h: List[float] = None,
    ) -> Optional[HybridSignal]:
        """
        Main evaluation function implementing decision hierarchy.
        
        Returns HybridSignal if conditions are met, None otherwise.
        """
        
        # ==========================================
        # LEVEL 1: SYSTEM CHECK
        # ==========================================
        if not self._system_healthy():
            return None
        
        # ==========================================
        # LEVEL 2: TIME FILTER
        # ==========================================
        if not self._time_filter_passed(timestamp_ms, volume_ratio):
            return None
        
        # ==========================================
        # LEVEL 3: POSITION CHECK
        # ==========================================
        if symbol in self.positions:
            # Already have position in this symbol
            return None
        
        # ==========================================
        # GET GATING STATE
        # ==========================================
        gating = self.gating_states.get(symbol)
        if gating is None:
            gating = GatingState()
            self.gating_states[symbol] = gating
        
        # ==========================================
        # LAYER 1: BIAS DETERMINATION (Hourly)
        # ==========================================
        if gating.needs_bias_update():
            gating.bias = self.bias_calculator.calculate(
                funding_z=funding_z,
                liq_imbalance_4h=liq_imbalance_4h,
                oi_delta_24h=oi_delta_24h,
                price_change_4h=price_change_4h,
                price_change_24h=price_change_24h,
            )
            gating.bias_last_computed = time.time()
        
        bias = gating.bias
        
        # ==========================================
        # LAYER 2: REGIME CLASSIFICATION (Per-Minute)
        # ==========================================
        # FIXED (Audit): Pass 1H bar data to TrendAnalyzer for proper timeframe alignment
        trend = self.trend_analyzers[symbol].get_state(current_price, bar_closes_1h)
        range_vs_atr = (high_4h - low_4h) / atr_14 if atr_14 > 0 else 1.0
        
        if gating.needs_regime_update():
            regime, regime_confidence = self.regime_classifier.classify(
                symbol=symbol,
                higher_high=trend.higher_high,
                higher_low=trend.higher_low,
                lower_high=trend.lower_high,
                lower_low=trend.lower_low,
                price_change_1h=price_change_1h,
                price_change_4h=price_change_4h,
                vol_expansion_ratio=vol_expansion_ratio,
                range_vs_atr=range_vs_atr,
                ema_20=trend.ema_20,
                ema_50=trend.ema_50,
                current_price=current_price,
                moi_flip_rate=moi_flip_rate,
            )
            gating.regime = regime
            gating.regime_confidence = regime_confidence
            gating.regime_last_computed = time.time()
        else:
            regime = gating.regime
            regime_confidence = gating.regime_confidence
        
        # LAYER 2 HARD GATE: CHOPPY = NO TRADE
        if regime == MarketRegime.CHOPPY:
            self.total_regime_blocked += 1
            return None
        
        # ==========================================
        # LAYER 3: ENTRY TIMING STATE
        # FIXED (Audit): Now used as actual gate, not just dashboard display
        # ==========================================
        gating.price_vs_ema20 = trend.price_vs_ema20
        
        # Use ATR-relative pullback threshold (0.5 ATR)
        atr_pct = (atr_14 / current_price * 100) if current_price > 0 else 0.5
        pullback_threshold = min(0.8, max(0.3, atr_pct * 0.5))  # 0.3-0.8% range
        gating.pullback_active = abs(trend.price_vs_ema20) < pullback_threshold
        
        # Determine entry state
        if gating.pullback_active:
            gating.entry_state = "PULLBACK_READY"
        elif abs(trend.price_vs_ema20) > 1.5:  # Extended from EMA
            gating.entry_state = "EXTENDED"
        else:
            gating.entry_state = "WAITING"
        
        # FIXED (Audit): EXTENDED state now blocks trend-following signals
        # Only momentum/breakout signals allowed when EXTENDED
        entry_allows_trend_signals = gating.entry_state != "EXTENDED"
        
        # ==========================================
        # BUILD MARKET STATE
        # ==========================================
        state = MarketState(
            symbol=symbol,
            timestamp_ms=timestamp_ms,
            current_price=current_price,
            bias=bias,
            regime=regime,
            regime_confidence=regime_confidence,
            trend=trend,
            funding_z=funding_z,
            funding_rate=funding_rate,
            cumulative_funding_24h=cumulative_funding_24h,
            oi_delta_1h=oi_delta_1h,
            oi_delta_4h=oi_delta_4h,
            oi_delta_24h=oi_delta_24h,
            liq_imbalance_1h=liq_imbalance_1h,
            liq_imbalance_4h=liq_imbalance_4h,
            liq_total_1h=liq_total_1h,
            liq_long_1h=liq_long_1h,
            liq_short_1h=liq_short_1h,
            cascade_active=cascade_active,
            liq_exhaustion=liq_exhaustion,
            atr_14=atr_14,
            vol_expansion_ratio=vol_expansion_ratio,
            range_4h=high_4h - low_4h,
            range_vs_atr=range_vs_atr,
            high_4h=high_4h,
            low_4h=low_4h,
            high_24h=high_24h,
            low_24h=low_24h,
            price_change_1h=price_change_1h,
            price_change_4h=price_change_4h,
            price_change_24h=price_change_24h,
            price_change_48h=price_change_48h,
            volume_ratio=volume_ratio,
            bar_closes_1h=bar_closes_1h or [],  # FIXED (Audit): Include 1H bars for signals
        )
        
        # ==========================================
        # SIGNAL EVALUATION
        # FIXED (Audit): Entry timing + regime now gate signals properly
        # ==========================================
        signals = []
        
        # Signal categories for regime-based gating
        TREND_FOLLOWING_SIGNALS = {"funding_trend", "trend_pullback", "sma_crossover"}
        RANGE_SIGNALS = {"range_breakout"}
        REVERSAL_SIGNALS = {"exhaustion_reversal", "liquidation_follow"}
        
        for name, signal_obj in self.signals[symbol].items():
            try:
                # FIXED (Audit): Block trend-following signals when price is EXTENDED
                if name in TREND_FOLLOWING_SIGNALS and not entry_allows_trend_signals:
                    continue  # Skip trend signals when extended from EMA
                
                # FIXED (Audit): RANGING regime only allows range-specific signals
                if regime == MarketRegime.RANGING:
                    if name not in RANGE_SIGNALS:
                        continue  # Only range_breakout in ranging markets
                
                result = signal_obj.evaluate(state)
                if result is not None:
                    signals.append(result)
            except Exception as e:
                logger.error("signal_evaluation_error", signal=name, error=str(e)[:50])
        
        if not signals:
            return None
        
        # Take highest confidence signal
        best_signal = max(signals, key=lambda s: s.confidence)
        
        # ==========================================
        # LAYER 1 HARD GATE: BIAS CHECK
        # Block trades against strong positioning bias
        # ==========================================
        if self.config.bias_hard_gate:
            bias_blocks = self._check_bias_gate(best_signal.direction, bias)
            if bias_blocks:
                self.total_bias_blocked += 1
                logger.info(
                    "signal_blocked_by_bias",
                    symbol=symbol,
                    signal_direction=best_signal.direction.value,
                    bias_direction=bias.direction.value,
                    bias_strength=f"{bias.strength:.2f}",
                )
                return None
        
        # ==========================================
        # VETO CHECK
        # ==========================================
        vetoed, veto_reason = self._veto_check(best_signal, state)
        
        if vetoed:
            self.total_signals_vetoed += 1
            logger.info("signal_vetoed", 
                       symbol=symbol,
                       signal_type=best_signal.signal_type.value,
                       reason=veto_reason)
            return None
        
        # ==========================================
        # LEVEL 8: SIZE CALCULATION
        # ==========================================
        best_signal = self._calculate_size(best_signal, state)
        
        # ==========================================
        # SIGNAL APPROVED
        # ==========================================
        self.total_signals_generated += 1
        self._update_daily_stats()
        
        logger.info("signal_generated",
                   symbol=symbol,
                   direction=best_signal.direction.value,
                   signal_type=best_signal.signal_type.value,
                   confidence=f"{best_signal.confidence:.2f}",
                   stop_pct=f"{best_signal.stop_pct:.3f}",
                   target_pct=f"{best_signal.target_pct:.3f}",
                   reason=best_signal.reason[:80])
        
        return best_signal
    
    def _system_healthy(self) -> bool:
        """Check if system is healthy enough to trade"""
        # Add actual health checks here
        return True
    
    def _time_filter_passed(self, timestamp_ms: int, volume_ratio: float) -> bool:
        """Check if current time is good for trading"""
        # Skip low volume periods
        if volume_ratio < self.config.min_volume_ratio:
            return False
        
        # Avoid 5 minutes around funding time (00:00, 08:00, 16:00 UTC)
        hour = (timestamp_ms // 3600000) % 24
        minute = (timestamp_ms // 60000) % 60
        
        funding_hours = [0, 8, 16]
        for fh in funding_hours:
            if hour == fh and minute < self.config.avoid_funding_minutes:
                return False
            if hour == (fh - 1) % 24 and minute >= (60 - self.config.avoid_funding_minutes):
                return False
        
        return True
    
    def _check_bias_gate(self, signal_direction: Direction, bias: Bias) -> bool:
        """
        LAYER 1 HARD GATE: Check if bias blocks this trade direction.
        
        Rules:
        - If bias is STRONG (>threshold) and signal is OPPOSITE direction → BLOCK
        - If bias is NEUTRAL or ALIGNED → ALLOW
        
        Returns:
            True if trade should be BLOCKED, False if allowed
        """
        # Neutral bias allows all trades
        if bias.direction == Direction.NEUTRAL:
            return False
        
        # Weak bias allows all trades
        if bias.strength < self.config.bias_gate_threshold:
            return False
        
        # Strong bias blocks opposite direction
        if signal_direction == Direction.LONG and bias.direction == Direction.SHORT:
            return True  # Block long when strong short bias
        
        if signal_direction == Direction.SHORT and bias.direction == Direction.LONG:
            return True  # Block short when strong long bias
        
        return False  # Aligned direction, allow
    
    def _veto_check(self, signal: HybridSignal, state: MarketState) -> Tuple[bool, str]:
        """Check for veto conditions"""
        
        # Veto 1: Extreme volatility
        if state.vol_expansion_ratio > self.config.max_vol_expansion:
            return True, f"Extreme volatility ({state.vol_expansion_ratio:.1f}x)"
        
        # Veto 2: Daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            return True, f"Daily trade limit ({self.config.max_daily_trades})"
        
        # Veto 3: Consecutive losses
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            return True, f"Consecutive losses ({self.consecutive_losses})"
        
        # Veto 4: Correlation check (too many positions same direction)
        same_direction_count = sum(
            1 for p in self.positions.values() 
            if p.direction == signal.direction
        )
        if same_direction_count >= self.config.max_correlation_positions:
            return True, f"Too many {signal.direction.value} positions"
        
        # Veto 5: Signal confidence too low
        if signal.confidence < 0.50:
            return True, f"Confidence too low ({signal.confidence:.2f})"
        
        # Veto 6: R:R too low
        if signal.risk_reward_ratio() < 1.5:
            return True, f"R:R too low ({signal.risk_reward_ratio():.1f})"
        
        return False, ""
    
    def _calculate_size(self, signal: HybridSignal, state: MarketState) -> HybridSignal:
        """Calculate position size based on context"""
        
        base_risk = self.config.base_risk_pct
        
        # Adjustments
        multiplier = 1.0
        
        # Bias alignment boost
        bias_aligned = (
            (signal.direction == Direction.LONG and state.bias.is_bullish()) or
            (signal.direction == Direction.SHORT and state.bias.is_bearish())
        )
        if bias_aligned:
            multiplier *= (1.0 + state.bias.strength * 0.3)  # Up to 1.3x
        
        # Regime confidence adjustment
        multiplier *= (0.7 + state.regime_confidence * 0.4)  # 0.7x to 1.1x
        
        # Volatility adjustment
        if state.vol_expansion_ratio > 1.5:
            multiplier *= 0.7  # Reduce in high vol
        elif state.vol_expansion_ratio < 0.7:
            multiplier *= 1.2  # Increase in low vol
        
        # Recent performance adjustment
        if self.consecutive_losses > 0:
            multiplier *= (1.0 - self.consecutive_losses * 0.1)  # Reduce after losses
        
        # Calculate final risk
        final_risk = base_risk * multiplier
        final_risk = max(0.01, min(self.config.max_risk_pct, final_risk))
        
        # Convert to size multiplier
        signal.size_multiplier = final_risk / self.config.base_risk_pct
        
        return signal
    
    def _update_daily_stats(self):
        """Update daily trade counter"""
        today = int(time.time() / 86400)
        if today != self.last_trade_day:
            self.daily_trades = 0
            self.last_trade_day = today
        self.daily_trades += 1
    
    def record_trade_result(self, symbol: str, won: bool):
        """Record trade result for performance tracking"""
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        
        if symbol in self.positions:
            del self.positions[symbol]
    
    def open_position(
        self,
        symbol: str,
        signal: HybridSignal,
        size: float,
    ):
        """Record new position"""
        self.positions[symbol] = PositionInfo(
            symbol=symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            entry_time=time.time(),
            size=size,
            stop_price=signal.entry_price * (1 - signal.stop_pct) if signal.direction == Direction.LONG else signal.entry_price * (1 + signal.stop_pct),
            target_price=signal.entry_price * (1 + signal.target_pct) if signal.direction == Direction.LONG else signal.entry_price * (1 - signal.target_pct),
            signal_type=signal.signal_type,
        )
    
    def get_stats(self) -> Dict:
        """Get engine statistics"""
        return {
            "total_signals_generated": self.total_signals_generated,
            "total_signals_vetoed": self.total_signals_vetoed,
            "total_bias_blocked": self.total_bias_blocked,
            "total_regime_blocked": self.total_regime_blocked,
            "open_positions": len(self.positions),
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
        }
    
    def get_gating_state(self, symbol: str) -> Dict:
        """
        Get current gating state for dashboard display.
        
        Returns dict with bias, regime, and entry timing info.
        """
        gating = self.gating_states.get(symbol)
        if gating is None:
            return {"error": "Symbol not tracked"}
        
        # Also get regime details from classifier
        regime_info = self.regime_classifier.get_regime_info(symbol)
        
        result = gating.to_dict()
        result["regime"].update(regime_info)
        
        return result
    
    def get_all_gating_states(self) -> Dict[str, Dict]:
        """Get gating states for all symbols"""
        return {
            symbol: self.get_gating_state(symbol)
            for symbol in self.symbols
        }
    
    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position info for symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PositionInfo]:
        """Get all open positions"""
        return self.positions.copy()
