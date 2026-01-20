"""
Stage 3 V2 Engine: Main Orchestrator
====================================

Coordinates all signals, gates, and filters for long-horizon positioning alpha.

Flow:
1. Build LongHorizonState from Stage 2 features
2. Run AI modules to enrich state
3. Evaluate all entry signals
4. Apply gates (veto if needed)
5. Apply filters (adjust parameters)
6. Return final signals for execution
"""
import time
from typing import Dict, List, Optional, Tuple
import structlog

from src.stage3_v2.models import (
    LongHorizonState, PositioningSignal, SignalMemory, Direction,
    SignalType, PositioningRegime, SignalEvaluation
)
from src.stage3_v2.signals import PositioningSignalAggregator
from src.stage3_v2.gates import CombinedGate
from src.stage3_v2.filters import CombinedFilter

logger = structlog.get_logger(__name__)


class Stage3V2Engine:
    """
    Main orchestrator for Stage 3 V2 positioning signals.
    
    Key differences from Stage 3 V1:
    - Signals have memory (track regime persistence)
    - Gates can veto entries
    - Filters adjust parameters based on context
    - AI modules enrich state (pluggable)
    - Much rarer signal firing (0-2 per day per symbol)
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        
        # Signal aggregators per symbol
        self.signal_aggregators: Dict[str, PositioningSignalAggregator] = {
            s: PositioningSignalAggregator(s) for s in symbols
        }
        
        # Gates and filters (shared across symbols)
        self.gate = CombinedGate()
        self.filter = CombinedFilter()
        
        # AI modules (pluggable, None by default)
        self.ai_regime_classifier = None
        self.ai_instability_detector = None
        self.ai_anomaly_detector = None
        self.ai_context_similarity = None
        
        # Stats
        self._signals_generated = 0
        self._signals_vetoed = 0
        self._evaluations_total = 0
        
        logger.info("stage3_v2_engine_initialized", symbols=symbols)
    
    def register_ai_modules(
        self,
        regime_classifier=None,
        instability_detector=None,
        anomaly_detector=None,
        context_similarity=None,
    ):
        """Register AI modules for state enrichment"""
        self.ai_regime_classifier = regime_classifier
        self.ai_instability_detector = instability_detector
        self.ai_anomaly_detector = anomaly_detector
        self.ai_context_similarity = context_similarity
        
        logger.info("ai_modules_registered",
                   regime=regime_classifier is not None,
                   instability=instability_detector is not None,
                   anomaly=anomaly_detector is not None,
                   similarity=context_similarity is not None)
    
    def evaluate(self, symbol: str, state: LongHorizonState) -> List[SignalEvaluation]:
        """
        Evaluate all signals for a symbol.
        
        Returns list of SignalEvaluation (may include vetoed signals for logging).
        """
        self._evaluations_total += 1
        
        if symbol not in self.signal_aggregators:
            logger.warning("unknown_symbol", symbol=symbol)
            return []
        
        # Step 1: Enrich state with AI modules
        state = self._enrich_state_with_ai(state)
        
        # Step 2: Get raw signals from aggregator
        raw_signals = self.signal_aggregators[symbol].evaluate_all(state)
        
        if not raw_signals:
            return []
        
        # Step 3: Apply gates and filters to each signal
        evaluations = []
        for signal in raw_signals:
            eval_result = self._process_signal(signal, state)
            evaluations.append(eval_result)
            
            if eval_result.triggered:
                self._signals_generated += 1
                logger.info("signal_triggered",
                           symbol=symbol,
                           direction=signal.direction.value,
                           type=signal.signal_type.value,
                           confidence=f"{signal.confidence:.2f}",
                           reason=signal.reason[:100])
            elif eval_result.vetoed:
                self._signals_vetoed += 1
                logger.info("signal_vetoed",
                           symbol=symbol,
                           direction=signal.direction.value,
                           type=signal.signal_type.value,
                           veto_reason=eval_result.veto_reason)
        
        return evaluations
    
    def _enrich_state_with_ai(self, state: LongHorizonState) -> LongHorizonState:
        """Apply AI modules to enrich state"""
        
        # Positioning regime classifier
        if self.ai_regime_classifier:
            try:
                state.ai_positioning_regime = self.ai_regime_classifier.classify(state)
            except Exception as e:
                logger.error("ai_regime_error", error=str(e)[:50])
        
        # Instability detector
        if self.ai_instability_detector:
            try:
                score, bias = self.ai_instability_detector.detect(state)
                state.ai_instability_score = score
                state.ai_instability_direction_bias = bias
            except Exception as e:
                logger.error("ai_instability_error", error=str(e)[:50])
        
        # Anomaly detector
        if self.ai_anomaly_detector:
            try:
                state.ai_anomaly_score = self.ai_anomaly_detector.score(state)
            except Exception as e:
                logger.error("ai_anomaly_error", error=str(e)[:50])
        
        # Context similarity
        if self.ai_context_similarity:
            try:
                state.ai_historical_win_rate = self.ai_context_similarity.get_historical_win_rate(state)
            except Exception as e:
                logger.error("ai_similarity_error", error=str(e)[:50])
        
        return state
    
    def _process_signal(
        self, 
        signal: PositioningSignal, 
        state: LongHorizonState
    ) -> SignalEvaluation:
        """Process a single signal through gates and filters"""
        
        # Apply gates (may veto)
        allowed, veto_reason = self.gate.should_allow_entry(signal.direction, state)
        
        if not allowed:
            return SignalEvaluation(
                signal=signal,
                vetoed=True,
                veto_reason=veto_reason,
                regime_status=self._get_regime_status(state),
            )
        
        # Apply filters (adjust parameters)
        adjusted_signal = self.filter.adjust(signal, state)
        
        return SignalEvaluation(
            signal=adjusted_signal,
            vetoed=False,
            veto_reason="",
            regime_status=self._get_regime_status(state),
        )
    
    def _get_regime_status(self, state: LongHorizonState) -> str:
        """Get human-readable regime status"""
        return f"Regime: {state.regime}, Positioning: {state.ai_positioning_regime.value}"
    
    def get_triggered_signals(self, evaluations: List[SignalEvaluation]) -> List[PositioningSignal]:
        """Extract only triggered (non-vetoed) signals from evaluations"""
        return [e.signal for e in evaluations if e.triggered]
    
    def get_all_memory(self, symbol: str) -> Dict[str, SignalMemory]:
        """Get memory state for all signals for a symbol"""
        if symbol in self.signal_aggregators:
            return self.signal_aggregators[symbol].get_all_memory()
        return {}
    
    def get_regime_summary(self, symbol: str) -> Dict[str, str]:
        """Get summary of current regime states for a symbol"""
        if symbol in self.signal_aggregators:
            return self.signal_aggregators[symbol].get_regime_summary()
        return {}
    
    def get_stats(self) -> Dict[str, int]:
        """Get engine statistics"""
        return {
            "evaluations_total": self._evaluations_total,
            "signals_generated": self._signals_generated,
            "signals_vetoed": self._signals_vetoed,
        }


# ============================================================
# STATE BUILDER (from Stage 2 features)
# ============================================================

def build_long_horizon_state(
    symbol: str,
    timestamp_ms: int,
    current_price: float,
    # Funding features
    funding_features: dict,
    # OI features
    oi_features: dict,
    # Liquidation features
    liq_features: dict,
    # Volatility features
    vol_features: dict,
    # Order book features
    book_features: dict,
    # Absorption features
    absorption_features: dict,
    # Order flow features
    flow_features: dict,
    # Regime
    regime: str,
    # Extended features (from new processors)
    extended_features: Optional[dict] = None,
) -> LongHorizonState:
    """
    Build LongHorizonState from Stage 2 processor outputs.
    
    This is the bridge between Stage 2 and Stage 3 V2.
    """
    ext = extended_features or {}
    
    return LongHorizonState(
        symbol=symbol,
        timestamp_ms=timestamp_ms,
        current_price=current_price,
        
        # Funding
        funding_rate=funding_features.get("rate", 0.0),
        funding_z=funding_features.get("funding_z", 0.0),
        funding_z_8h_avg=ext.get("funding_z_8h_avg", funding_features.get("funding_z", 0.0)),
        funding_z_change_8h=ext.get("funding_z_change_8h", 0.0),
        cumulative_funding_24h=ext.get("cumulative_funding_24h", 0.0),
        
        # OI
        oi=oi_features.get("oi", 0.0),
        oi_delta_1m=oi_features.get("oi_delta_1m", 0.0),
        oi_delta_5m=oi_features.get("oi_delta_5m", 0.0),
        oi_delta_1h=ext.get("oi_delta_1h", oi_features.get("oi_delta_5m", 0.0)),
        oi_delta_4h=ext.get("oi_delta_4h", 0.0),
        oi_delta_8h=ext.get("oi_delta_8h", 0.0),
        oi_delta_24h=ext.get("oi_delta_24h", 0.0),
        
        # OI entry estimation
        oi_avg_entry_price=ext.get("oi_avg_entry_price", 0.0),
        oi_entry_displacement_pct=ext.get("oi_entry_displacement_pct", 0.0),
        oi_concentration_above_pct=ext.get("oi_concentration_above_pct", 0.5),
        
        # Price
        price_change_5m=flow_features.get("price_change_5m", 0.0),
        price_change_1h=ext.get("price_change_1h", 0.0),
        price_change_4h=ext.get("price_change_4h", 0.0),
        price_change_8h=ext.get("price_change_8h", 0.0),
        price_change_24h=ext.get("price_change_24h", 0.0),
        
        # Liquidations
        liq_imbalance_30s=liq_features.get("imbalance_30s", 0.0),
        liq_imbalance_2m=liq_features.get("imbalance_2m", 0.0),
        liq_imbalance_5m=liq_features.get("imbalance_5m", 0.0),
        liq_imbalance_1h=ext.get("liq_imbalance_1h", liq_features.get("imbalance_5m", 0.0)),
        liq_imbalance_4h=ext.get("liq_imbalance_4h", 0.0),
        liq_imbalance_8h=ext.get("liq_imbalance_8h", 0.0),
        liq_total_usd_1h=ext.get("liq_total_usd_1h", 0.0),
        liq_total_usd_4h=ext.get("liq_total_usd_4h", 0.0),
        liq_total_usd_24h=ext.get("liq_total_usd_24h", 0.0),
        cascade_active=liq_features.get("cascade_active", False),
        liq_exhaustion=liq_features.get("exhaustion", False),
        
        # Volatility
        vol_expansion_ratio=vol_features.get("vol_expansion_ratio", 1.0),
        atr_5m=vol_features.get("atr_5m", 0.0),
        atr_1h=vol_features.get("atr_1h", 0.0),
        vol_compression_duration_h=ext.get("vol_compression_duration_h", 0.0),
        
        # Order book
        depth_imbalance=book_features.get("depth_imbalance", 0.0),
        bid_depth_usd=book_features.get("bid_depth_usd", 0.0),
        ask_depth_usd=book_features.get("ask_depth_usd", 0.0),
        
        # Absorption
        absorption_z=absorption_features.get("absorption_z", 0.0),
        refill_rate=absorption_features.get("refill_rate", 0.0),
        liquidity_sweep=absorption_features.get("liquidity_sweep", False),
        
        # Order flow
        moi_1s=flow_features.get("moi_1s", 0.0),
        moi_std=flow_features.get("moi_std", 1.0),
        moi_flip_rate=flow_features.get("moi_flip_rate", 0.0),
        aggression_persistence=flow_features.get("aggression_persistence", 0.0),
        delta_velocity=flow_features.get("delta_velocity", 0.0),
        
        # Regime
        regime=regime,
    )
