"""
Stage 3 Thesis Engine
Signal stacking and directional bias generation
"""
import time
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass
import structlog

from src.stage2.models import MarketState, Regime
from src.stage3.models import Signal, Thesis, Direction, ThesisState
from src.stage3.processors.signals import (
    funding_price_cointegration,
    hawkes_liquidation_cascade,
    kyle_lambda_divergence,
    inventory_lock,
    failed_acceptance_reversal,
    queue_reactive_liquidity,
    liquidity_crisis_detector,
    flip_rate_compression_break,
    order_flow_dominance_decay,
)
from src.stage4.filter import StructuralFilter, FilterResult, orderflow_confirmation
from src.stage5.predictor_v3 import MLPredictorV3, PredictionResultV3
from src.utils.meta_logger import log_signal_event

# Probability threshold for Stage 5 gate (V3 models output probability of TP hit)
# Minimum probability to allow a trade - 50% = break-even threshold
PROB_MIN_THRESHOLD = 0.50  # Both prob_60 and prob_300 must be >= 50%

logger = structlog.get_logger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ThesisConfig:
    """Configuration for thesis generation"""
    # Minimum score to allow trading (lowered from 0.6 to allow single strong signals)
    min_score_threshold: float = 0.55
    
    # Minimum asymmetry between long/short scores (lowered for clearer signals)
    min_asymmetry: float = 0.25
    
    # Extreme funding level for veto in expansion
    extreme_funding_z: float = 2.5
    
    # Price tracking window (seconds)
    price_window_5m: int = 300
    price_window_15m: int = 900


# Default configuration
DEFAULT_CONFIG = ThesisConfig()


# ============================================================
# PRICE TRACKER
# ============================================================

class PriceTracker:
    """Track price changes over time windows"""
    
    def __init__(self, max_age_seconds: int = 900):
        self.max_age_seconds = max_age_seconds
        self._prices: deque = deque(maxlen=1000)
    
    def add_price(self, price: float, timestamp_ms: Optional[int] = None) -> None:
        """Add a price observation"""
        ts = timestamp_ms or int(time.time() * 1000)
        self._prices.append((ts, price))
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Remove old prices"""
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - (self.max_age_seconds * 1000)
        
        while self._prices and self._prices[0][0] < cutoff_ms:
            self._prices.popleft()
    
    def get_price_change(self, window_seconds: int) -> float:
        """Get price change over window as percentage"""
        if len(self._prices) < 2:
            return 0.0
        
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - (window_seconds * 1000)
        
        current_price = self._prices[-1][1]
        
        # Find price at start of window
        start_price = current_price
        for ts, price in self._prices:
            if ts >= cutoff_ms:
                start_price = price
                break
        
        if start_price == 0:
            return 0.0
        
        return (current_price - start_price) / start_price


# ============================================================
# REGIME TRACKER
# ============================================================

class RegimeTracker:
    """Track time in current regime"""
    
    def __init__(self):
        self._current_regime: Optional[Regime] = None
        self._regime_start_ms: int = 0
    
    def update(self, regime: Regime) -> float:
        """Update regime and return time in current regime (seconds)"""
        now_ms = int(time.time() * 1000)
        
        if regime != self._current_regime:
            self._current_regime = regime
            self._regime_start_ms = now_ms
            return 0.0
        
        return (now_ms - self._regime_start_ms) / 1000.0


# ============================================================
# THESIS ENGINE
# ============================================================

class ThesisEngine:
    """
    Main Stage 3 engine
    
    Converts MarketState into Thesis with signal stacking
    """
    
    def __init__(
        self,
        symbols: List[str],
        config: Optional[ThesisConfig] = None,
        on_thesis: Optional[Callable[[str, Thesis], Any]] = None,
    ):
        self.symbols = symbols
        self.config = config or DEFAULT_CONFIG
        self.on_thesis = on_thesis
        
        # Per-symbol trackers
        self._price_trackers: Dict[str, PriceTracker] = {
            s: PriceTracker() for s in symbols
        }
        self._regime_trackers: Dict[str, RegimeTracker] = {
            s: RegimeTracker() for s in symbols
        }
        
        # Stage 4 structural filter
        self._structural_filter = StructuralFilter()
        
        # Stage 5 ML predictor (V3 with probability-based classification)
        self._ml_predictor = MLPredictorV3()
        
        # Current thesis per symbol
        self._theses: Dict[str, Thesis] = {}
        
        # Last filter result per symbol (for transparency)
        self._filter_results: Dict[str, FilterResult] = {}
        
        # Last ML prediction per symbol (for transparency)
        self._ml_predictions: Dict[str, PredictionResultV3] = {}
        
        # Orderflow filter details per symbol (for transparency)
        self._orderflow_details: Dict[str, Dict] = {}
        
        # Stats
        self._signal_counts: Dict[str, int] = {s: 0 for s in symbols}
        self._thesis_count = 0
        
        logger.info("thesis_engine_initialized", symbols=symbols)
    
    def process(self, state: MarketState) -> Thesis:
        """
        Process MarketState and generate Thesis
        
        This is the main entry point called for each MarketState update
        """
        symbol = state.symbol
        
        if symbol not in self._price_trackers:
            # Add new symbol on the fly
            self._price_trackers[symbol] = PriceTracker()
            self._regime_trackers[symbol] = RegimeTracker()
            self.symbols.append(symbol)
        
        # Update price tracker
        self._price_trackers[symbol].add_price(state.price, state.timestamp_ms)
        
        # Update regime tracker
        time_in_regime = self._regime_trackers[symbol].update(state.regime)
        
        # Build ThesisState from MarketState
        thesis_state = self._build_thesis_state(state, time_in_regime)
        
        # Generate thesis (Stage 3)
        thesis = self._generate_thesis(thesis_state)
        
        # Apply Stage 4 structural location filter
        thesis = self._apply_structural_filter(thesis, thesis_state)
        
        # Apply Stage 4.5 orderflow confirmation filter
        if thesis.allowed and thesis.direction != Direction.NONE:
            thesis = self._apply_orderflow_filter(thesis, state)
        
        # If signal passed Stage 4.5, run Stage 5 ML predictions
        if thesis.allowed and thesis.direction != Direction.NONE:
            prediction = self._run_ml_prediction(thesis, thesis_state, state)
            
            # Gate: Require minimum probability on both horizons
            # No percentile gate for now (no historical data yet)
            if prediction:
                prob_60_ok = prediction.prob_60 >= PROB_MIN_THRESHOLD
                prob_300_ok = prediction.prob_300 >= PROB_MIN_THRESHOLD
                
                if not prob_60_ok:
                    thesis = Thesis(
                        allowed=False,
                        direction=thesis.direction,
                        strength=thesis.strength,
                        reasons=thesis.reasons,
                        veto_reason=f"Stage 5: prob_60 ({prediction.prob_60:.1%}) < {PROB_MIN_THRESHOLD:.0%}",
                    )
                elif not prob_300_ok:
                    thesis = Thesis(
                        allowed=False,
                        direction=thesis.direction,
                        strength=thesis.strength,
                        reasons=thesis.reasons,
                        veto_reason=f"Stage 5: prob_300 ({prediction.prob_300:.1%}) < {PROB_MIN_THRESHOLD:.0%}",
                    )
        
        # Store and callback
        self._theses[symbol] = thesis
        self._thesis_count += 1
        
        if self.on_thesis:
            try:
                self.on_thesis(symbol, thesis)
            except Exception as e:
                logger.error("thesis_callback_error", symbol=symbol, error=str(e))
        
        return thesis
    
    def _build_thesis_state(
        self,
        state: MarketState,
        time_in_regime: float,
    ) -> ThesisState:
        """Build ThesisState from MarketState with computed values"""
        symbol = state.symbol
        
        # Use price_change_5m from MarketState (computed by Stage 2)
        # This ensures consistency with dashboard display
        price_change_5m = state.price_change_5m
        
        # For 15m, use internal tracker (Stage 2 doesn't compute this yet)
        price_change_15m = self._price_trackers[symbol].get_price_change(900)
        
        # Compute derived order flow z-scores for inventory lock signal
        moi_std = state.order_flow.moi_std + 1e-9
        aggression_persistence = state.order_flow.aggression_persistence + 1e-9
        
        moi_z = abs(state.order_flow.moi_1s) / moi_std
        delta_vel_z = abs(state.order_flow.delta_velocity) / moi_std
        flip_noise = state.order_flow.moi_flip_rate / aggression_persistence
        
        return ThesisState(
            symbol=symbol,
            price=state.price,
            regime=state.regime.value,
            funding_z=state.funding.funding_z,
            oi_delta_1m=state.oi.oi_delta_1m,
            oi_delta_5m=state.oi.oi_delta_5m,
            oi_delta_15m=state.oi.oi_delta_5m,  # Use 5m for now, TODO: add 15m to Stage 2
            liq_imbalance=state.liquidations.imbalance_2m,
            liq_imbalance_30s=state.liquidations.imbalance_30s,
            liq_imbalance_2m=state.liquidations.imbalance_2m,
            cascade_active=state.liquidations.cascade_active,
            liq_exhaustion=state.liquidations.exhaustion,
            price_change_5m=price_change_5m,
            price_change_15m=price_change_15m,
            vol_regime=state.volatility.vol_regime,
            vol_rank=state.volatility.vol_rank,
            time_in_regime=time_in_regime,
            # Absorption
            absorption_z=state.absorption.absorption_z,
            refill_rate=state.absorption.refill_rate,
            liquidity_sweep=state.absorption.liquidity_sweep,
            # Structure for Stage 4 filtering
            dist_lvn=state.structure.dist_lvn,
            dist_poc=state.structure.dist_poc,
            vah=state.structure.vah,
            val=state.structure.val,
            # Order flow
            moi_1s=state.order_flow.moi_1s,
            moi_z=moi_z,
            moi_std=state.order_flow.moi_std,
            delta_velocity=state.order_flow.delta_velocity,
            delta_vel_z=delta_vel_z,
            moi_flip_rate=state.order_flow.moi_flip_rate,
            flip_noise=flip_noise,
            aggression_persistence=state.order_flow.aggression_persistence,
            # Absorption extras
            depth_imbalance=state.absorption.depth_imbalance,
            vol_expansion_ratio=state.volatility.vol_expansion_ratio,
            # Structure for FAR signal
            acceptance_outside_value=state.structure.acceptance_outside_value,
        )
    
    def _generate_thesis(self, state: ThesisState) -> Thesis:
        """
        Generate thesis from ThesisState using signal stacking
        
        This implements the exact logic from the spec:
        1. Collect signals (with double-counting prevention)
        2. Score directions
        3. Apply veto rules
        4. Decide thesis
        """
        # ========== HARD VETO RULES (NON-NEGOTIABLE) ==========
        
        # CHOP = no trading
        if state.regime == "CHOP":
            return Thesis(
                allowed=False,
                direction=Direction.NONE,
                strength=0.0,
                reasons=[],
                veto_reason="CHOP regime - no trading",
            )
        
        # Extreme funding in expansion = do not fade parabolic moves
        if (
            abs(state.funding_z) > self.config.extreme_funding_z and
            state.regime == "EXPANSION"
        ):
            return Thesis(
                allowed=False,
                direction=Direction.NONE,
                strength=0.0,
                reasons=[],
                veto_reason=f"Extreme funding ({state.funding_z:.2f}) in expansion - no fade",
            )
        
        # ========== COLLECT SIGNALS ==========
        
        signals: List[Signal] = []
        
        # Signal 1: Funding-Price Cointegration (S-tier)
        fpc = funding_price_cointegration(state)
        if fpc:
            signals.append(fpc)
        
        # Signal 2: Hawkes Liquidation Cascade (S-tier)
        hlc = hawkes_liquidation_cascade(state)
        if hlc:
            signals.append(hlc)
        
        # Signal 3: Kyle's Lambda Divergence (B-tier)
        kld = kyle_lambda_divergence(state)
        if kld:
            signals.append(kld)
        
        # Signal 4: Inventory Lock (ILI)
        ili = inventory_lock(state)
        if ili:
            signals.append(ili)
        
        # Signal 5: Failed Acceptance Reversal (FAR)
        far = failed_acceptance_reversal(state)
        if far:
            signals.append(far)
        
        # Signal 6: Queue Reactive Liquidity (S-tier)
        qrl = queue_reactive_liquidity(state)
        if qrl:
            signals.append(qrl)
        
        # Signal 7: Liquidity Crisis Detector (A-tier)
        lcd = liquidity_crisis_detector(state)
        if lcd:
            signals.append(lcd)
        
        # Signal 8: Flip-Rate Compression Break (A-S tier)
        frcb = flip_rate_compression_break(state)
        if frcb:
            signals.append(frcb)
        
        # Signal 9: Order-Flow Dominance Decay (S-tier)
        ofdd = order_flow_dominance_decay(state)
        if ofdd:
            signals.append(ofdd)
        
        # Log all detected signals
        # if signals:
        #     logger.info(
        #         "thesis_signals_detected",
        #         symbol=state.symbol,
        #         count=len(signals),
        #         signals=[f"{s.direction.value}:{s.name}:{s.confidence}" for s in signals],
        #     )
        
        # ========== SCORE DIRECTIONS ==========
        
        long_score = sum(s.confidence for s in signals if s.direction == Direction.LONG)
        short_score = sum(s.confidence for s in signals if s.direction == Direction.SHORT)
        
        # ========== DECIDE THESIS ==========
        
        max_score = max(long_score, short_score)
        score_diff = abs(long_score - short_score)
        
        # Not enough conviction
        if max_score < self.config.min_score_threshold:
            return Thesis(
                allowed=False,
                direction=Direction.NONE,
                strength=0.0,
                reasons=signals,
                veto_reason=f"Insufficient score ({max_score:.2f} < {self.config.min_score_threshold})",
            )
        
        # Not enough asymmetry
        if score_diff < self.config.min_asymmetry:
            return Thesis(
                allowed=False,
                direction=Direction.NONE,
                strength=0.0,
                reasons=signals,
                veto_reason=f"Conflicting signals (diff={score_diff:.2f} < {self.config.min_asymmetry})",
            )
        
        # We have a thesis!
        if long_score > short_score:
            return Thesis(
                allowed=True,
                direction=Direction.LONG,
                strength=long_score,
                reasons=[s for s in signals if s.direction == Direction.LONG],
            )
        else:
            return Thesis(
                allowed=True,
                direction=Direction.SHORT,
                strength=short_score,
                reasons=[s for s in signals if s.direction == Direction.SHORT],
            )
    
    def _apply_structural_filter(self, thesis: Thesis, state: ThesisState) -> Thesis:
        """
        Apply Stage 4 structural location filter to thesis
        
        Only allows trades:
        - Near 5m LVN (dist_lvn < threshold)
        - OR at 30m value extremes (VAL for LONG, VAH for SHORT)
        """
        # If thesis already blocked, skip filter
        if not thesis.allowed or thesis.direction == Direction.NONE:
            return thesis
        
        # Create a dummy signal for the filter
        filter_signal = Signal(thesis.direction, thesis.strength, "thesis")
        
        # Apply filter
        result = self._structural_filter.filter_signal(
            signal=filter_signal,
            regime=state.regime,
            price=state.price,
            dist_lvn=state.dist_lvn,
            vah=state.vah,
            val=state.val,
        )
        
        # Store result for transparency
        self._filter_results[state.symbol] = result
        
        # If filter rejects, update thesis with detailed reason
        if not result.allowed:
            # Build detailed rejection reason
            lvn_threshold = self._structural_filter.lvn_threshold
            details = []
            
            if result.reason.value == "regime_chop":
                details.append("regime=CHOP")
            elif result.reason.value == "bad_structural_location":
                details.append(f"dist_lvn={state.dist_lvn:.3f}>={lvn_threshold}")
                if thesis.direction == Direction.LONG:
                    details.append(f"price={state.price:.2f}>VAL={state.val:.2f}")
                else:  # SHORT
                    details.append(f"price={state.price:.2f}<VAH={state.vah:.2f}")
            
            reason_str = ", ".join(details) if details else result.reason.value
            
            return Thesis(
                allowed=False,
                direction=thesis.direction,  # Keep direction for transparency
                strength=thesis.strength,
                reasons=thesis.reasons,
                veto_reason=f"Stage 4: {reason_str}",
            )
        
        # Add pass reason to result for transparency
        return thesis
    
    def _apply_orderflow_filter(
        self,
        thesis: Thesis,
        market_state: MarketState,
    ) -> Thesis:
        """
        Apply Stage 4.5 orderflow confirmation filter.
        
        Confirms that orderflow aligns with signal direction.
        """
        from src.stage2.processors.regime import PAIR_THRESHOLDS, PairThresholds
        from src.stage4.filter import DEPTH_IMBALANCE_THRESHOLD
        
        symbol = market_state.symbol
        delta_velocity = market_state.order_flow.delta_velocity
        depth_imbalance = market_state.absorption.depth_imbalance
        absorption_z = market_state.absorption.absorption_z
        
        # Get pair-specific thresholds
        thresholds = PAIR_THRESHOLDS.get(symbol, PairThresholds())
        absorb_threshold = thresholds.absorption_z_spike
        
        confirmed = orderflow_confirmation(
            direction=thesis.direction,
            symbol=symbol,
            delta_velocity=delta_velocity,
            depth_imbalance=depth_imbalance,
            absorption_z=absorption_z,
        )
        
        # Store orderflow filter details for transparency
        self._orderflow_details[symbol] = {
            "delta_velocity": delta_velocity,
            "depth_imbalance": depth_imbalance,
            "absorption_z": absorption_z,
            "depth_threshold": DEPTH_IMBALANCE_THRESHOLD,
            "absorb_threshold": absorb_threshold,
            "confirmed": confirmed,
        }
        
        if not confirmed:
            # Build detailed rejection reason
            reasons = []
            if thesis.direction == Direction.LONG:
                if delta_velocity <= 0:
                    reasons.append(f"delta_vel={delta_velocity:.3f}<=0")
                if depth_imbalance <= DEPTH_IMBALANCE_THRESHOLD:
                    reasons.append(f"depth_imb={depth_imbalance:.3f}<={DEPTH_IMBALANCE_THRESHOLD}")
                if absorption_z >= absorb_threshold:
                    reasons.append(f"absorb_z={absorption_z:.2f}>={absorb_threshold}")
            else:  # SHORT
                if delta_velocity >= 0:
                    reasons.append(f"delta_vel={delta_velocity:.3f}>=0")
                if depth_imbalance >= -DEPTH_IMBALANCE_THRESHOLD:
                    reasons.append(f"depth_imb={depth_imbalance:.3f}>={-DEPTH_IMBALANCE_THRESHOLD}")
                if absorption_z <= -absorb_threshold:
                    reasons.append(f"absorb_z={absorption_z:.2f}<={-absorb_threshold}")
            
            reason_str = ", ".join(reasons) if reasons else "conditions_not_met"
            
            # Clear ML prediction when Stage 4.5 fails
            if symbol in self._ml_predictions:
                del self._ml_predictions[symbol]
            
            return Thesis(
                allowed=False,
                direction=thesis.direction,
                strength=thesis.strength,
                reasons=thesis.reasons,
                veto_reason=f"Stage 4.5: {reason_str}",
            )
        
        # Store success reason
        self._orderflow_details[symbol]["reason"] = "confirmed"
        
        return thesis
    
    def _run_ml_prediction(
        self,
        thesis: Thesis,
        thesis_state: ThesisState,
        market_state: MarketState,
    ) -> Optional[PredictionResultV3]:
        """
        Run Stage 5 ML predictions for a signal that passed Stage 4.5.
        
        V3 uses probability-based classification models.
        Returns prediction result for probability + percentile gating.
        """
        symbol = thesis_state.symbol
        
        try:
            # Build features dict for V3 predictor
            # V3 expects features from update_bar(), but we can build them from MarketState
            import datetime
            now = datetime.datetime.now(datetime.timezone.utc)
            
            atr_5m = market_state.volatility.atr_5m if market_state.volatility.atr_5m > 0 else 1.0
            moi_std = market_state.order_flow.moi_std + 1e-9
            
            features = {
                # Order flow
                "MOI_250ms": market_state.order_flow.moi_250ms,
                "MOI_1s": market_state.order_flow.moi_1s,
                "MOI_5s": market_state.order_flow.moi_1s * 5,  # Approximate
                "MOI_20s": market_state.order_flow.moi_1s * 20,  # Approximate
                "MOI_z": abs(market_state.order_flow.moi_1s) / moi_std,
                "MOI_std": moi_std,
                "delta_velocity": market_state.order_flow.delta_velocity,
                "delta_velocity_5s": market_state.order_flow.delta_velocity * 5,
                "AggressionPersistence": market_state.order_flow.aggression_persistence,
                
                # Momentum
                "MOI_roc_1s": 0.0,  # Would need history
                "MOI_roc_5s": 0.0,
                "MOI_acceleration": 0.0,
                "MOI_flip_rate": market_state.order_flow.moi_flip_rate,
                
                # Volatility
                "vol_1m": market_state.volatility.vol_5m,  # Approximate
                "vol_5m": market_state.volatility.vol_5m,
                "vol_ratio": market_state.volatility.vol_expansion_ratio,
                "vol_rank": market_state.volatility.vol_rank,
                "ATR_5m": atr_5m,
                "ATR_5m_pct": atr_5m / market_state.price if market_state.price > 0 else 0,
                
                # Absorption
                "absorption_z": market_state.absorption.absorption_z,
                "price_impact_z": 0.0,  # Would need computation
                
                # Structure
                "dist_poc": market_state.structure.dist_poc,
                "dist_lvn": market_state.structure.dist_lvn,
                "dist_poc_atr": market_state.structure.dist_poc / atr_5m if atr_5m > 0 else 0,
                "dist_lvn_atr": market_state.structure.dist_lvn / atr_5m if atr_5m > 0 else 0,
                
                # Trade intensity
                "trade_intensity": 0.0,
                "trade_intensity_z": 0.0,
                
                # Cumulative
                "cum_delta_1m": market_state.order_flow.moi_1s * 240,  # Approximate
                "cum_delta_5m": market_state.order_flow.moi_1s * 1200,
                
                # Time
                "hour_sin": np.sin(2 * np.pi * now.hour / 24),
                "hour_cos": np.cos(2 * np.pi * now.hour / 24),
                "is_weekend": 1 if now.weekday() >= 5 else 0,
            }
            
            prediction = self._ml_predictor.predict(
                direction=thesis.direction,
                vol_regime=thesis_state.vol_regime,
                symbol=symbol,
                features=features,
            )
            
            # Store prediction for transparency
            self._ml_predictions[symbol] = prediction
            
            logger.info(
                "stage5_ml_prediction_v3",
                symbol=symbol,
                direction=thesis.direction.value,
                vol_regime=thesis_state.vol_regime,
                prob_60=f"{prediction.prob_60:.1%}",
                prob_300=f"{prediction.prob_300:.1%}",
                pct_60=prediction.percentile_60,
                pct_300=prediction.percentile_300,
                model=prediction.model_300,
            )
            
            # Log to meta.json for future model building
            passed_gate = prediction.prob_60 >= PROB_MIN_THRESHOLD and prediction.prob_300 >= PROB_MIN_THRESHOLD
            risk_tier = None
            if passed_gate:
                # Determine risk tier based on probabilities
                if prediction.prob_60 >= 0.65 and prediction.prob_300 >= 0.65:
                    risk_tier = "max"
                elif (prediction.prob_60 >= 0.65 and prediction.prob_300 >= 0.55) or \
                     (prediction.prob_60 >= 0.55 and prediction.prob_300 >= 0.65):
                    risk_tier = "med"
                else:
                    risk_tier = "min"
            
            try:
                log_signal_event(
                    symbol=symbol,
                    direction=thesis.direction.value,
                    signal_names=[s.name for s in thesis.reasons] if thesis.reasons else [],
                    vol_regime=thesis_state.vol_regime,
                    prob_60=prediction.prob_60,
                    prob_300=prediction.prob_300,
                    model_60=prediction.model_60,
                    model_300=prediction.model_300,
                    features=features,
                    price=market_state.price,
                    passed_gate=passed_gate,
                    risk_tier=risk_tier,
                )
            except Exception as log_err:
                logger.warning("meta_log_failed", error=str(log_err))
            
            return prediction
        except Exception as e:
            logger.error(
                "ml_prediction_error",
                symbol=symbol,
                error=str(e),
            )
            return PredictionResultV3()
    
    # ========== PUBLIC API ==========
    
    def get_thesis(self, symbol: str) -> Optional[Thesis]:
        """Get current thesis for symbol"""
        return self._theses.get(symbol)
    
    def get_all_theses(self) -> Dict[str, Thesis]:
        """Get all current theses"""
        return self._theses.copy()
    
    def get_filter_result(self, symbol: str) -> Optional[FilterResult]:
        """Get last Stage 4 filter result for symbol"""
        return self._filter_results.get(symbol)
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get Stage 4 filter statistics"""
        return self._structural_filter.get_stats()
    
    def get_ml_prediction(self, symbol: str) -> Optional[PredictionResultV3]:
        """Get last Stage 5 ML prediction for symbol"""
        return self._ml_predictions.get(symbol)
    
    def get_orderflow_details(self, symbol: str) -> Optional[Dict]:
        """Get last Stage 4.5 orderflow filter details for symbol"""
        return self._orderflow_details.get(symbol)
    
    def get_all_ml_predictions(self) -> Dict[str, PredictionResultV3]:
        """Get all ML predictions"""
        return self._ml_predictions.copy()
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get engine health metrics"""
        allowed_count = sum(1 for t in self._theses.values() if t.allowed)
        long_count = sum(1 for t in self._theses.values() if t.direction == Direction.LONG)
        short_count = sum(1 for t in self._theses.values() if t.direction == Direction.SHORT)
        
        return {
            "thesis_count": self._thesis_count,
            "symbols_active": len(self.symbols),
            "allowed_count": allowed_count,
            "long_count": long_count,
            "short_count": short_count,
            "theses": {s: t.to_dict() for s, t in self._theses.items()},
            "stage4_filter": self._structural_filter.get_stats(),
            "stage5_ml_predictions": {s: p.to_dict() for s, p in self._ml_predictions.items()},
        }
