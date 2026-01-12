"""
Stage 3 Thesis Engine
Signal stacking and directional bias generation
"""
import time
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass
import structlog

from src.stage2.models import MarketState, Regime
from src.stage3.models import Signal, Thesis, Direction, ThesisState
from src.stage3.processors.signals import (
    funding_squeeze,
    liquidation_exhaustion,
    oi_divergence,
    crowding_fade,
    funding_carry,
)
from src.stage4.filter import StructuralFilter, FilterResult, orderflow_confirmation
from src.stage5.predictor import MLPredictor, PredictionResult

# Percentile threshold for Stage 5 gate (top 20% = >= 80)
PERCENTILE_300_THRESHOLD = 80.0

logger = structlog.get_logger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ThesisConfig:
    """Configuration for thesis generation"""
    # Minimum score to allow trading
    min_score_threshold: float = 0.6
    
    # Minimum asymmetry between long/short scores
    min_asymmetry: float = 0.35
    
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
        
        # Stage 5 ML predictor
        self._ml_predictor = MLPredictor()
        
        # Current thesis per symbol
        self._theses: Dict[str, Thesis] = {}
        
        # Last filter result per symbol (for transparency)
        self._filter_results: Dict[str, FilterResult] = {}
        
        # Last ML prediction per symbol (for transparency)
        self._ml_predictions: Dict[str, PredictionResult] = {}
        
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
            
            # Gate: Only continue if percentile_300 is in top 20%
            if prediction and prediction.percentile_300 < PERCENTILE_300_THRESHOLD:
                thesis = Thesis(
                    allowed=False,
                    direction=thesis.direction,
                    strength=thesis.strength,
                    reasons=thesis.reasons,
                    veto_reason=f"Stage 5: percentile_300 ({prediction.percentile_300:.1f}%) < {PERCENTILE_300_THRESHOLD}%",
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
        
        # Get price changes
        price_change_5m = self._price_trackers[symbol].get_price_change(300)
        price_change_15m = self._price_trackers[symbol].get_price_change(900)
        
        return ThesisState(
            symbol=symbol,
            price=state.price,
            regime=state.regime.value,
            funding_z=state.funding.funding_z,
            oi_delta_5m=state.oi.oi_delta_5m,
            oi_delta_15m=state.oi.oi_delta_5m,  # Use 5m for now, TODO: add 15m to Stage 2
            liq_imbalance=state.liquidations.imbalance_2m,
            price_change_5m=price_change_5m,
            price_change_15m=price_change_15m,
            vol_regime=state.volatility.vol_regime,
            vol_rank=state.volatility.vol_rank,
            time_in_regime=time_in_regime,
            absorption_z=state.absorption.absorption_z,
            # Structure for Stage 4 filtering
            dist_lvn=state.structure.dist_lvn,
            vah=state.structure.vah,
            val=state.structure.val,
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
        
        # ========== COLLECT SIGNALS (with double-counting prevention) ==========
        
        signals: List[Signal] = []
        
        # Signal 1: Funding squeeze
        fs = funding_squeeze(state)
        if fs:
            signals.append(fs)
        
        # Signal 2: Liquidation exhaustion
        le = liquidation_exhaustion(state)
        if le:
            signals.append(le)
        
        # Signal 3: OI divergence
        oi = oi_divergence(state)
        if oi:
            signals.append(oi)
        
        # Signal 4: Crowding fade (only if funding squeeze didn't fire)
        # This prevents double-counting funding signals
        if not fs:
            cf = crowding_fade(state)
            if cf:
                signals.append(cf)
        
        # Signal 5: Funding carry
        fc = funding_carry(state)
        if fc:
            signals.append(fc)
        
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
        
        # If filter rejects, update thesis
        if not result.allowed:
            return Thesis(
                allowed=False,
                direction=thesis.direction,  # Keep direction for transparency
                strength=thesis.strength,
                reasons=thesis.reasons,
                veto_reason=f"Stage 4: {result.reason.value}",
            )
        
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
        confirmed = orderflow_confirmation(
            direction=thesis.direction,
            symbol=market_state.symbol,
            delta_velocity=market_state.order_flow.delta_velocity,
            depth_imbalance=market_state.absorption.depth_imbalance,
            absorption_z=market_state.absorption.absorption_z,
        )
        
        if not confirmed:
            return Thesis(
                allowed=False,
                direction=thesis.direction,
                strength=thesis.strength,
                reasons=thesis.reasons,
                veto_reason="Stage 4.5: orderflow_not_confirmed",
            )
        
        return thesis
    
    def _run_ml_prediction(
        self,
        thesis: Thesis,
        thesis_state: ThesisState,
        market_state: MarketState,
    ) -> Optional[PredictionResult]:
        """
        Run Stage 5 ML predictions for a signal that passed Stage 4.5.
        
        Returns prediction result for percentile gating.
        """
        symbol = thesis_state.symbol
        
        prediction = self._ml_predictor.predict(
            direction=thesis.direction,
            vol_regime=thesis_state.vol_regime,
            symbol=symbol,
            moi_250ms=market_state.order_flow.moi_250ms,
            moi_1s=market_state.order_flow.moi_1s,
            delta_velocity=market_state.order_flow.delta_velocity,
            aggression_persistence=market_state.order_flow.aggression_persistence,
            absorption_z=market_state.absorption.absorption_z,
            dist_lvn=market_state.structure.dist_lvn,
            vol_5m=market_state.volatility.vol_5m,
        )
        
        # Store prediction for transparency
        self._ml_predictions[symbol] = prediction
        
        logger.info(
            "stage5_ml_prediction",
            symbol=symbol,
            direction=thesis.direction.value,
            vol_regime=thesis_state.vol_regime,
            pred_60=prediction.pred_60,
            pred_300=prediction.pred_300,
            pct_60=prediction.percentile_60,
            pct_300=prediction.percentile_300,
        )
        
        return prediction
    
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
    
    def get_ml_prediction(self, symbol: str) -> Optional[PredictionResult]:
        """Get last Stage 5 ML prediction for symbol"""
        return self._ml_predictions.get(symbol)
    
    def get_all_ml_predictions(self) -> Dict[str, PredictionResult]:
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
