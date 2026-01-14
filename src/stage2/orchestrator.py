"""
Stage 2 Orchestrator - Coordinates all feature processors and regime classification
Manages rolling memory and emits MarketState updates
"""
import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from collections import deque
from dataclasses import dataclass
import structlog

from config import settings
from src.core.models import Trade, Bar, OrderBookSnapshot, VolumeProfile
from src.core.models import FundingRate, OpenInterest, Liquidation

from src.stage2.models import (
    MarketState, Regime, OrderFlowFeatures, AbsorptionFeatures,
    VolatilityFeatures, StructureFeatures, FundingFeatures,
    OIFeatures, LiquidationFeatures
)
from src.stage2.processors.order_flow import OrderFlowProcessor
from src.stage2.processors.absorption import AbsorptionProcessor
from src.stage2.processors.volatility import VolatilityProcessor
from src.stage2.processors.structure import StructureProcessor
from src.stage2.processors.funding_oi import FundingOIProcessor
from src.stage2.processors.liquidations import LiquidationProcessor
from src.stage2.processors.regime import RegimeClassifier

logger = structlog.get_logger(__name__)


@dataclass
class SymbolState:
    """Current state for a single symbol"""
    symbol: str
    price: float = 0.0
    order_flow: OrderFlowFeatures = None
    absorption: AbsorptionFeatures = None
    volatility: VolatilityFeatures = None
    structure: StructureFeatures = None
    funding: FundingFeatures = None
    oi: OIFeatures = None
    liquidations: LiquidationFeatures = None
    regime: Regime = Regime.CHOP
    last_update_ms: int = 0
    
    def __post_init__(self):
        if self.order_flow is None:
            self.order_flow = OrderFlowFeatures()
        if self.absorption is None:
            self.absorption = AbsorptionFeatures()
        if self.volatility is None:
            self.volatility = VolatilityFeatures()
        if self.structure is None:
            self.structure = StructureFeatures()
        if self.funding is None:
            self.funding = FundingFeatures()
        if self.oi is None:
            self.oi = OIFeatures()
        if self.liquidations is None:
            self.liquidations = LiquidationFeatures()


class Stage2Orchestrator:
    """
    Main Stage 2 orchestrator
    
    Responsibilities:
    - Receive data from Stage 1 collectors
    - Route to appropriate processors
    - Manage rolling memory
    - Emit MarketState updates
    - Classify regimes
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        on_market_state: Optional[Callable[[MarketState], Any]] = None,
        update_interval_ms: int = 1000,  # Emit MarketState every 1s
        regime_interval_ms: int = 60_000,  # Update regime every 60s
    ):
        self.symbols = symbols or settings.SYMBOLS
        self.on_market_state = on_market_state
        self.update_interval_ms = update_interval_ms
        self.regime_interval_ms = regime_interval_ms
        
        # Processors per symbol
        self._order_flow: Dict[str, OrderFlowProcessor] = {}
        self._absorption: Dict[str, AbsorptionProcessor] = {}
        self._volatility: Dict[str, VolatilityProcessor] = {}
        self._structure: Dict[str, StructureProcessor] = {}
        self._funding_oi: Dict[str, FundingOIProcessor] = {}
        self._liquidations: Dict[str, LiquidationProcessor] = {}
        self._regime: Dict[str, RegimeClassifier] = {}
        
        # Current state per symbol
        self._states: Dict[str, SymbolState] = {}
        
        # Initialize all processors
        for symbol in self.symbols:
            self._order_flow[symbol] = OrderFlowProcessor(symbol)
            self._absorption[symbol] = AbsorptionProcessor(symbol)
            self._volatility[symbol] = VolatilityProcessor(symbol)
            self._structure[symbol] = StructureProcessor(symbol)
            self._funding_oi[symbol] = FundingOIProcessor(symbol)
            self._liquidations[symbol] = LiquidationProcessor(symbol)
            self._regime[symbol] = RegimeClassifier(symbol, regime_interval_ms)
            self._states[symbol] = SymbolState(symbol=symbol)
        
        # Tracking
        self._running = False
        self._last_emit_ms: Dict[str, int] = {s: 0 for s in self.symbols}
        self._trade_volume_since_book: Dict[str, float] = {s: 0.0 for s in self.symbols}
        self._price_at_last_book: Dict[str, float] = {s: 0.0 for s in self.symbols}
        
        # Regime tracking for time_in_regime
        self._current_regime: Dict[str, Regime] = {s: Regime.COMPRESSION for s in self.symbols}
        self._regime_start_ms: Dict[str, int] = {s: int(time.time() * 1000) for s in self.symbols}
        
        # Stats
        self._market_state_count = 0
        self._processing_times: deque[float] = deque(maxlen=100)
        
        logger.info("stage2_orchestrator_initialized", symbols=self.symbols)
    
    # ========== DATA INGESTION FROM STAGE 1 ==========
    
    def on_trade(self, trade: Trade) -> None:
        """Process incoming trade"""
        symbol = trade.symbol
        if symbol not in self._states:
            return
        
        # Update price
        self._states[symbol].price = trade.price
        
        # Track volume for absorption calculation
        self._trade_volume_since_book[symbol] += trade.quantity
        
        # Add trade to structure processor (builds rolling 5m/30m buffers)
        self._structure[symbol].add_trade(trade)
    
    def on_bar(self, bar: Bar) -> None:
        """Process incoming bar"""
        symbol = bar.symbol
        if symbol not in self._states:
            return
        
        start = time.perf_counter()
        
        # Update price
        self._states[symbol].price = bar.close
        
        # Process through order flow
        self._states[symbol].order_flow = self._order_flow[symbol].add_bar(bar)
        
        # Process through volatility
        self._states[symbol].volatility = self._volatility[symbol].add_bar(bar)
        
        # Update ATR in structure processor for normalization
        self._structure[symbol].set_atr(self._states[symbol].volatility.atr_5m)
        
        # Track processing time
        elapsed = time.perf_counter() - start
        self._processing_times.append(elapsed)
        
        # Maybe emit MarketState
        self._maybe_emit_state(symbol)
    
    def on_book_snapshot(self, snapshot: OrderBookSnapshot) -> None:
        """Process order book snapshot"""
        symbol = snapshot.symbol
        if symbol not in self._states:
            return
        
        # Calculate price change since last book
        price_change = 0.0
        if self._price_at_last_book[symbol] > 0:
            current_price = snapshot.mid_price or self._states[symbol].price
            price_change = current_price - self._price_at_last_book[symbol]
        
        # Get traded volume since last book
        traded_volume = self._trade_volume_since_book[symbol]
        
        # Process absorption
        self._states[symbol].absorption = self._absorption[symbol].add_book_snapshot(
            snapshot, traded_volume, price_change
        )
        
        # Reset tracking
        self._trade_volume_since_book[symbol] = 0.0
        self._price_at_last_book[symbol] = snapshot.mid_price or self._states[symbol].price
    
    def on_volume_profile(self, profile: VolumeProfile) -> None:
        """Process volume profile update (legacy - structure now computed from trades)"""
        pass
    
    def on_funding(self, funding: FundingRate) -> None:
        """Process funding rate update"""
        symbol = funding.symbol
        if symbol not in self._states:
            return
        
        self._states[symbol].funding = self._funding_oi[symbol].add_funding(funding)
    
    def on_oi(self, oi: OpenInterest) -> None:
        """Process open interest update"""
        symbol = oi.symbol
        if symbol not in self._states:
            return
        
        current_price = self._states[symbol].price
        self._states[symbol].oi = self._funding_oi[symbol].add_oi(oi, current_price)
    
    def on_liquidation(self, liq: Liquidation) -> None:
        """Process liquidation event"""
        symbol = liq.symbol
        if symbol not in self._states:
            return
        
        self._states[symbol].liquidations = self._liquidations[symbol].add_liquidation(liq)
    
    # ========== STATE EMISSION ==========
    
    def _maybe_emit_state(self, symbol: str) -> None:
        """Emit MarketState if enough time has passed"""
        now_ms = int(time.time() * 1000)
        
        if now_ms - self._last_emit_ms[symbol] < self.update_interval_ms:
            return
        
        self._emit_state(symbol)
        self._last_emit_ms[symbol] = now_ms
    
    def _emit_state(self, symbol: str) -> None:
        """Build and emit MarketState for symbol"""
        state = self._states[symbol]
        now_ms = int(time.time() * 1000)
        
        # Compute structure features from rolling trade buffers (every 1 sec)
        # Pass absorption_z for acceptance veto logic
        self._structure[symbol].set_atr(self._volatility[symbol].get_atr_5m())
        state.structure = self._structure[symbol].compute(state.absorption.absorption_z)
        
        # Update liquidations (periodic update even without new events)
        state.liquidations = self._liquidations[symbol].update()
        
        # Classify regime
        classification = self._regime[symbol].classify(
            state.volatility,
            state.structure,
            state.order_flow,
            state.absorption,
        )
        state.regime = classification.regime
        
        # Track time_in_regime
        if state.regime != self._current_regime[symbol]:
            self._current_regime[symbol] = state.regime
            self._regime_start_ms[symbol] = now_ms
        
        time_in_regime = (now_ms - self._regime_start_ms[symbol]) / 1000.0
        
        # Get price_change_5m from order flow processor
        price_change_5m = self._order_flow[symbol].get_price_change_5m()
        
        # Build MarketState
        market_state = MarketState(
            timestamp_ms=now_ms,
            symbol=symbol,
            price=state.price,
            regime=state.regime,
            order_flow=state.order_flow,
            absorption=state.absorption,
            structure=state.structure,
            volatility=state.volatility,
            funding=state.funding,
            oi=state.oi,
            liquidations=state.liquidations,
            price_change_5m=price_change_5m,
            time_in_regime=time_in_regime,
            regime_inputs={
                "compression_score": classification.compression_score,
                "expansion_score": classification.expansion_score,
                "chop_score": classification.chop_score,
                "confidence": classification.confidence,
            },
        )
        
        state.last_update_ms = now_ms
        self._market_state_count += 1
        
        # Callback
        if self.on_market_state:
            try:
                result = self.on_market_state(market_state)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error("market_state_callback_error", error=str(e))
    
    # ========== PERIODIC UPDATE LOOP ==========
    
    async def start_update_loop(self) -> None:
        """Start periodic update loop for all symbols"""
        self._running = True
        
        while self._running:
            await asyncio.sleep(1.0)  # Update every second
            
            for symbol in self.symbols:
                try:
                    # Force emit state for all symbols
                    self._emit_state(symbol)
                except Exception as e:
                    logger.error("update_loop_error", symbol=symbol, error=str(e))
    
    async def stop(self) -> None:
        """Stop the orchestrator"""
        self._running = False
        logger.info("stage2_orchestrator_stopped")
    
    # ========== PUBLIC API ==========
    
    def get_market_state(self, symbol: str) -> Optional[MarketState]:
        """Get current MarketState for symbol"""
        if symbol not in self._states:
            return None
        
        state = self._states[symbol]
        now_ms = int(time.time() * 1000)
        
        return MarketState(
            timestamp_ms=now_ms,
            symbol=symbol,
            price=state.price,
            regime=state.regime,
            order_flow=state.order_flow,
            absorption=state.absorption,
            structure=state.structure,
            volatility=state.volatility,
            funding=state.funding,
            oi=state.oi,
            liquidations=state.liquidations,
        )
    
    def get_all_market_states(self) -> Dict[str, MarketState]:
        """Get MarketState for all symbols"""
        return {s: self.get_market_state(s) for s in self.symbols}
    
    def get_all_regimes(self) -> Dict[str, Regime]:
        """Get current regime for all symbols"""
        return {s: self._states[s].regime for s in self.symbols}
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get orchestrator health metrics"""
        avg_processing_time = 0.0
        if self._processing_times:
            avg_processing_time = sum(self._processing_times) / len(self._processing_times)
        
        return {
            "market_state_count": self._market_state_count,
            "avg_processing_time_ms": avg_processing_time * 1000,
            "symbols_active": len(self.symbols),
            "regimes": {s: self._states[s].regime.value for s in self.symbols},
        }
    
    def bootstrap_volatility(
        self,
        symbol: str,
        tr_5m_values: List[float],
        tr_1h_values: List[float],
        atr_5m: float,
        atr_1h: float,
        last_close_5m: float = 0.0,
        last_close_1h: float = 0.0,
        vol_5m_history: Optional[List[float]] = None,
    ) -> None:
        """
        Bootstrap ATR and volatility from historical klines
        
        Called at startup to eliminate cold start problem for ATR.
        Vol_5m history used for percentile-based regime detection.
        """
        if symbol in self._volatility:
            self._volatility[symbol].bootstrap(
                tr_5m_values=tr_5m_values,
                tr_1h_values=tr_1h_values,
                atr_5m=atr_5m,
                atr_1h=atr_1h,
                last_close_5m=last_close_5m,
                last_close_1h=last_close_1h,
                vol_5m_history=vol_5m_history,
            )
            # Update state with bootstrapped values
            self._states[symbol].volatility = self._volatility[symbol].get_features()


# ========== INTEGRATION WITH STAGE 1 ==========

def create_stage2_with_stage1_hooks(
    symbols: List[str],
    on_market_state: Optional[Callable[[MarketState], Any]] = None,
) -> Stage2Orchestrator:
    """
    Create Stage 2 orchestrator configured to receive Stage 1 data
    Returns orchestrator with callback methods ready to be wired to Stage 1
    """
    orchestrator = Stage2Orchestrator(
        symbols=symbols,
        on_market_state=on_market_state,
    )
    
    return orchestrator
