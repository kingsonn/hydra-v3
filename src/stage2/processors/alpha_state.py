"""
Alpha State Processor
=====================

Computes all variables required for the Stage 3 V3 hybrid alpha system:
- funding_z (21-sample window)
- funding_time_at_extreme
- oi_change_1h / 4h / 24h
- price_change_4h / 24h
- trend_direction (1h & 4h EMAs)
- ATR_short (5-period on 1h bars)
- ATR_long (20-period on 1h bars)
- vol_expansion_ratio
- liquidation_usd_1h / 4h / 8h / 24h
- liquidation_imbalance_1h / 4h

This processor maintains the MarketState object for stage3_v3 signals.
"""
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import structlog

from src.stage3_v3.models import MarketState, Bias, TrendState, Direction, MarketRegime

logger = structlog.get_logger(__name__)


@dataclass
class AlphaState:
    """
    Complete alpha state for a symbol.
    This maps to stage3_v3.models.MarketState.
    """
    symbol: str = ""
    timestamp_ms: int = 0
    current_price: float = 0.0
    
    # Funding
    funding_rate: float = 0.0
    funding_z: float = 0.0
    cumulative_funding_24h: float = 0.0  # Sum of last 3 funding rates
    funding_time_at_extreme_hours: float = 0.0
    funding_extreme_start_ms: int = 0
    
    # OI Changes
    oi_change_1h: float = 0.0
    oi_change_4h: float = 0.0
    oi_change_24h: float = 0.0
    
    # Price Changes
    price_change_1h: float = 0.0
    price_change_4h: float = 0.0
    price_change_24h: float = 0.0
    price_change_48h: float = 0.0
    
    # Price Ranges
    high_4h: float = 0.0
    low_4h: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    
    # Trend (from EMAs)
    trend_direction_1h: Direction = Direction.NEUTRAL
    trend_strength_1h: float = 0.0
    ema_20_1h: float = 0.0
    ema_50_1h: float = 0.0
    
    # ATR and Volatility
    atr_short: float = 0.0   # 5-period on 1h bars
    atr_long: float = 0.0    # 20-period on 1h bars
    vol_expansion_ratio: float = 1.0
    
    # Liquidations
    liq_long_usd_1h: float = 0.0
    liq_short_usd_1h: float = 0.0
    liq_long_usd_4h: float = 0.0
    liq_short_usd_4h: float = 0.0
    liq_long_usd_8h: float = 0.0
    liq_short_usd_8h: float = 0.0
    liq_long_usd_24h: float = 0.0
    liq_short_usd_24h: float = 0.0
    liq_imbalance_1h: float = 0.0
    liq_imbalance_4h: float = 0.0
    
    # Cascade detection
    cascade_active: bool = False
    liq_exhaustion: bool = False
    
    # RSI
    rsi_14: float = 50.0
    
    # Volume
    volume_ratio: float = 1.0
    
    # Order Flow (from order_flow.py)
    moi_flip_rate: float = 0.0  # MOI sign flips per minute - used for chop detection
    
    # Price history for structure analysis
    bar_closes_1h: List[float] = field(default_factory=list)
    
    def to_market_state(self) -> MarketState:
        """Convert to stage3_v3.models.MarketState"""
        # Calculate structure from 1H bars if available
        higher_high = False
        higher_low = False
        lower_high = False
        lower_low = False
        
        if len(self.bar_closes_1h) >= 20:
            # Simple structure detection from closes
            closes = self.bar_closes_1h[-30:] if len(self.bar_closes_1h) >= 30 else self.bar_closes_1h
            
            # Find swing points using 3-bar windows
            if len(closes) >= 10:
                # Split into thirds for structure analysis
                third = len(closes) // 3
                first_third = closes[:third]
                middle_third = closes[third:2*third]
                last_third = closes[2*third:]
                
                # Compare peaks and troughs
                first_peak = max(first_third)
                middle_peak = max(middle_third)
                last_peak = max(last_third)
                
                first_trough = min(first_third)
                middle_trough = min(middle_third)
                last_trough = min(last_third)
                
                # Determine structure
                higher_high = last_peak > middle_peak > first_peak
                lower_high = last_peak < middle_peak < first_peak
                higher_low = last_trough > middle_trough > first_trough
                lower_low = last_trough < middle_trough < first_trough
        
        # Build TrendState with structure flags
        trend = TrendState(
            direction=self.trend_direction_1h,
            strength=self.trend_strength_1h,
            ema_20=self.ema_20_1h,
            ema_50=self.ema_50_1h,
            ema_200=0.0,  # Not used in our setup
            price_vs_ema20=((self.current_price - self.ema_20_1h) / self.ema_20_1h * 100) if self.ema_20_1h > 0 else 0,
            price_vs_ema50=((self.current_price - self.ema_50_1h) / self.ema_50_1h * 100) if self.ema_50_1h > 0 else 0,
            rsi_14=self.rsi_14,
            higher_high=higher_high,
            higher_low=higher_low,
            lower_high=lower_high,
            lower_low=lower_low,
        )
        
        # Regime will be calculated by RegimeClassifier in regime.py
        # AlphaState only provides raw data - it does NOT determine regime
        # Default to CHOPPY as a safe fallback (will be overwritten by RegimeClassifier)
        regime = MarketRegime.CHOPPY
        
        # Calculate range_vs_atr
        range_4h = self.high_4h - self.low_4h
        range_vs_atr = range_4h / self.atr_long if self.atr_long > 0 else 1.0
        
        return MarketState(
            symbol=self.symbol,
            timestamp_ms=self.timestamp_ms,
            current_price=self.current_price,
            bias=Bias(),  # Will be calculated by BiasCalculator
            regime=regime,
            regime_confidence=0.7,
            trend=trend,
            funding_z=self.funding_z,
            funding_rate=self.funding_rate,
            cumulative_funding_24h=self.cumulative_funding_24h,
            oi_delta_1h=self.oi_change_1h,
            oi_delta_4h=self.oi_change_4h,
            oi_delta_24h=self.oi_change_24h,
            liq_imbalance_1h=self.liq_imbalance_1h,
            liq_imbalance_4h=self.liq_imbalance_4h,
            liq_total_1h=self.liq_long_usd_1h + self.liq_short_usd_1h,
            liq_long_1h=self.liq_long_usd_1h,
            liq_short_1h=self.liq_short_usd_1h,
            cascade_active=self.cascade_active,
            liq_exhaustion=self.liq_exhaustion,
            atr_14=self.atr_long,
            vol_expansion_ratio=self.vol_expansion_ratio,
            range_4h=range_4h,
            range_vs_atr=range_vs_atr,
            high_4h=self.high_4h,
            low_4h=self.low_4h,
            high_24h=self.high_24h,
            low_24h=self.low_24h,
            price_change_1h=self.price_change_1h,
            price_change_4h=self.price_change_4h,
            price_change_24h=self.price_change_24h,
            price_change_48h=self.price_change_48h,
            volume_ratio=self.volume_ratio,
        )


class AlphaStateProcessor:
    """
    Processes data from bootstrap and live feeds to maintain alpha state.
    
    Integrates with:
    - AlphaDataBootstrap for historical data
    - Live feeds for real-time updates
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        
        # State per symbol
        self._states: Dict[str, AlphaState] = {
            s: AlphaState(symbol=s) for s in symbols
        }
        
        # EMA state per symbol (for incremental updates)
        self._ema_multipliers = {
            20: 2 / (20 + 1),
            50: 2 / (50 + 1),
        }
        
        # RSI state per symbol
        self._rsi_gains: Dict[str, deque] = {s: deque(maxlen=14) for s in symbols}
        self._rsi_losses: Dict[str, deque] = {s: deque(maxlen=14) for s in symbols}
        self._last_close: Dict[str, float] = {s: 0.0 for s in symbols}
        
        # Cascade detection state
        self._liq_rate_baseline: Dict[str, float] = {s: 0.0 for s in symbols}
        self._liq_rate_peak: Dict[str, float] = {s: 0.0 for s in symbols}
        self._cascade_start_ms: Dict[str, int] = {s: 0 for s in symbols}
        
        # Funding extreme tracking
        self._funding_extreme_start: Dict[str, int] = {s: 0 for s in symbols}
        
        logger.info("alpha_state_processor_initialized", symbols=len(symbols))
    
    def initialize_from_bootstrap(self, bootstrap: 'AlphaDataBootstrap'):
        """Initialize state from bootstrap data"""
        for symbol in self.symbols:
            state = self._states[symbol]
            
            # Funding
            state.funding_z = bootstrap.get_funding_z(symbol)
            state.funding_rate = bootstrap.get_funding_rate(symbol)
            state.cumulative_funding_24h = bootstrap.get_cumulative_funding_24h(symbol)
            
            # OI Changes
            state.oi_change_1h = bootstrap.get_oi_change(symbol, 60)
            state.oi_change_4h = bootstrap.get_oi_change(symbol, 240)
            state.oi_change_24h = bootstrap.get_oi_change(symbol, 1440)
            
            # Price Changes
            state.price_change_1h = bootstrap.get_price_change(symbol, 1)
            state.price_change_4h = bootstrap.get_price_change(symbol, 4)
            state.price_change_24h = bootstrap.get_price_change(symbol, 24)
            state.price_change_48h = bootstrap.get_price_change(symbol, 48)
            
            # Price Ranges
            state.high_4h, state.low_4h = bootstrap.get_high_low_4h(symbol)
            state.high_24h, state.low_24h = bootstrap.get_high_low_24h(symbol)
            
            # ATR
            state.atr_short = bootstrap.get_atr_short(symbol)
            state.atr_long = bootstrap.get_atr_long(symbol)
            state.vol_expansion_ratio = bootstrap.get_vol_expansion_ratio(symbol)
            
            # Liquidations
            l1h, s1h = bootstrap.get_liq_totals(symbol, 60)
            l4h, s4h = bootstrap.get_liq_totals(symbol, 240)
            l8h, s8h = bootstrap.get_liq_totals(symbol, 480)
            l24h, s24h = bootstrap.get_liq_totals(symbol, 1440)
            
            state.liq_long_usd_1h = l1h
            state.liq_short_usd_1h = s1h
            state.liq_long_usd_4h = l4h
            state.liq_short_usd_4h = s4h
            state.liq_long_usd_8h = l8h
            state.liq_short_usd_8h = s8h
            state.liq_long_usd_24h = l24h
            state.liq_short_usd_24h = s24h
            state.liq_imbalance_1h = bootstrap.get_liq_imbalance(symbol, 60)
            state.liq_imbalance_4h = bootstrap.get_liq_imbalance(symbol, 240)
            
            # Initialize EMAs from price history
            closes = bootstrap.get_price_closes(symbol)
            if closes:
                state.bar_closes_1h = closes.copy()  # Store for structure analysis
                state.current_price = closes[-1]
                state.ema_20_1h = self._compute_ema(closes, 20)
                state.ema_50_1h = self._compute_ema(closes, 50)
                state.trend_direction_1h, state.trend_strength_1h = self._compute_trend(
                    state.current_price, state.ema_20_1h, state.ema_50_1h
                )
                
                # Initialize RSI
                self._initialize_rsi(symbol, closes)
                state.rsi_14 = self._compute_rsi(symbol)
                self._last_close[symbol] = closes[-1]
            
            # Initialize funding extreme tracking
            if abs(state.funding_z) >= 1.5:
                self._funding_extreme_start[symbol] = int(time.time() * 1000)
            
            state.timestamp_ms = int(time.time() * 1000)
            
            logger.debug(
                "alpha_state_initialized",
                symbol=symbol,
                funding_z=f"{state.funding_z:.2f}",
                oi_change_4h=f"{state.oi_change_4h*100:.2f}%",
                atr_short=f"{state.atr_short:.2f}",
                trend=state.trend_direction_1h.value,
            )
    
    def _compute_ema(self, prices: List[float], period: int) -> float:
        """Compute EMA from price list"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        # Start with SMA
        ema = sum(prices[:period]) / period
        mult = 2 / (period + 1)
        
        # Apply EMA formula
        for price in prices[period:]:
            ema = price * mult + ema * (1 - mult)
        
        return ema
    
    def _compute_trend(self, price: float, ema_20: float, ema_50: float) -> Tuple[Direction, float]:
        """Compute trend direction and strength from EMAs"""
        if ema_20 == 0 or ema_50 == 0:
            return Direction.NEUTRAL, 0.0
        
        score = 0.0
        
        # EMA alignment
        if ema_20 > ema_50:
            score += 0.4  # Bullish
        elif ema_20 < ema_50:
            score -= 0.4  # Bearish
        
        # Price vs EMA20
        price_vs_ema20_pct = (price - ema_20) / ema_20
        if price_vs_ema20_pct > 0.005:
            score += 0.3
        elif price_vs_ema20_pct < -0.005:
            score -= 0.3
        
        # Price vs EMA50
        price_vs_ema50_pct = (price - ema_50) / ema_50
        if price_vs_ema50_pct > 0.01:
            score += 0.3
        elif price_vs_ema50_pct < -0.01:
            score -= 0.3
        
        if score > 0.3:
            return Direction.LONG, min(1.0, score)
        elif score < -0.3:
            return Direction.SHORT, min(1.0, abs(score))
        else:
            return Direction.NEUTRAL, 0.0
    
    def _initialize_rsi(self, symbol: str, prices: List[float]):
        """Initialize RSI from price history"""
        if len(prices) < 15:
            return
        
        for i in range(1, min(15, len(prices))):
            change = prices[i] - prices[i-1]
            if change > 0:
                self._rsi_gains[symbol].append(change)
                self._rsi_losses[symbol].append(0)
            else:
                self._rsi_gains[symbol].append(0)
                self._rsi_losses[symbol].append(abs(change))
    
    def _compute_rsi(self, symbol: str) -> float:
        """Compute RSI from gains/losses"""
        gains = self._rsi_gains[symbol]
        losses = self._rsi_losses[symbol]
        
        if len(gains) < 14:
            return 50.0
        
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    # ========== LIVE UPDATE METHODS ==========
    
    def update_price(self, symbol: str, price: float, timestamp_ms: int):
        """Update with live price"""
        if symbol not in self._states:
            return
        
        state = self._states[symbol]
        state.current_price = price
        state.timestamp_ms = timestamp_ms
        
        # Update RSI incrementally
        if self._last_close[symbol] > 0:
            change = price - self._last_close[symbol]
            if change > 0:
                self._rsi_gains[symbol].append(change)
                self._rsi_losses[symbol].append(0)
            else:
                self._rsi_gains[symbol].append(0)
                self._rsi_losses[symbol].append(abs(change))
            state.rsi_14 = self._compute_rsi(symbol)
        
        self._last_close[symbol] = price
    
    def update_funding(self, symbol: str, funding_rate: float, funding_z: float, timestamp_ms: int):
        """Update with live funding data"""
        if symbol not in self._states:
            return
        
        state = self._states[symbol]
        state.funding_rate = funding_rate
        state.funding_z = funding_z
        
        # Track time at extreme
        is_extreme = abs(funding_z) >= 1.5
        
        if is_extreme:
            if self._funding_extreme_start[symbol] == 0:
                self._funding_extreme_start[symbol] = timestamp_ms
            state.funding_time_at_extreme_hours = (timestamp_ms - self._funding_extreme_start[symbol]) / (3600 * 1000)
        else:
            self._funding_extreme_start[symbol] = 0
            state.funding_time_at_extreme_hours = 0
    
    def update_oi(self, symbol: str, oi_change_1h: float, oi_change_4h: float, oi_change_24h: float):
        """Update with live OI data"""
        if symbol not in self._states:
            return
        
        state = self._states[symbol]
        state.oi_change_1h = oi_change_1h
        state.oi_change_4h = oi_change_4h
        state.oi_change_24h = oi_change_24h
    
    def update_price_bar(self, symbol: str, high: float, low: float, close: float, timestamp_ms: int):
        """Update with new 1h bar - recalculate EMAs and trend"""
        if symbol not in self._states:
            return
        
        state = self._states[symbol]
        
        # Update bar history for structure analysis (maintain rolling window of 250 bars)
        state.bar_closes_1h.append(close)
        # Always maintain exactly 250 bars (rolling window)
        if len(state.bar_closes_1h) > 250:
            state.bar_closes_1h = state.bar_closes_1h[-250:]  # Keep last 250
        
        # Update EMAs incrementally
        if state.ema_20_1h > 0:
            state.ema_20_1h = close * self._ema_multipliers[20] + state.ema_20_1h * (1 - self._ema_multipliers[20])
        else:
            state.ema_20_1h = close
        
        if state.ema_50_1h > 0:
            state.ema_50_1h = close * self._ema_multipliers[50] + state.ema_50_1h * (1 - self._ema_multipliers[50])
        else:
            state.ema_50_1h = close
        
        # Update trend
        state.trend_direction_1h, state.trend_strength_1h = self._compute_trend(
            close, state.ema_20_1h, state.ema_50_1h
        )
        
        state.current_price = close
        state.timestamp_ms = timestamp_ms
    
    def update_atr(self, symbol: str, atr_short: float, atr_long: float):
        """Update ATR values"""
        if symbol not in self._states:
            return
        
        state = self._states[symbol]
        state.atr_short = atr_short
        state.atr_long = atr_long
        state.vol_expansion_ratio = atr_short / atr_long if atr_long > 0 else 1.0
    
    def update_liquidations(
        self, 
        symbol: str,
        long_1h: float, short_1h: float,
        long_4h: float, short_4h: float,
        long_8h: float, short_8h: float,
        long_24h: float, short_24h: float,
    ):
        """Update liquidation totals"""
        if symbol not in self._states:
            return
        
        state = self._states[symbol]
        state.liq_long_usd_1h = long_1h
        state.liq_short_usd_1h = short_1h
        state.liq_long_usd_4h = long_4h
        state.liq_short_usd_4h = short_4h
        state.liq_long_usd_8h = long_8h
        state.liq_short_usd_8h = short_8h
        state.liq_long_usd_24h = long_24h
        state.liq_short_usd_24h = short_24h
        
        # Calculate imbalances
        total_1h = long_1h + short_1h
        total_4h = long_4h + short_4h
        state.liq_imbalance_1h = (long_1h - short_1h) / total_1h if total_1h > 0 else 0.0
        state.liq_imbalance_4h = (long_4h - short_4h) / total_4h if total_4h > 0 else 0.0
        
        # Cascade detection
        self._detect_cascade(symbol, long_1h + short_1h)
    
    def _detect_cascade(self, symbol: str, total_liq_1h: float):
        """Detect liquidation cascade and exhaustion"""
        state = self._states[symbol]
        
        # Update baseline (exponential moving average of liq rate)
        if self._liq_rate_baseline[symbol] == 0:
            self._liq_rate_baseline[symbol] = total_liq_1h
        else:
            self._liq_rate_baseline[symbol] = 0.9 * self._liq_rate_baseline[symbol] + 0.1 * total_liq_1h
        
        baseline = self._liq_rate_baseline[symbol]
        
        # Cascade = 5x baseline
        if total_liq_1h > 5 * baseline and baseline > 0:
            if not state.cascade_active:
                state.cascade_active = True
                self._cascade_start_ms[symbol] = int(time.time() * 1000)
            self._liq_rate_peak[symbol] = max(self._liq_rate_peak[symbol], total_liq_1h)
        else:
            state.cascade_active = False
        
        # Exhaustion = rate dropped to <30% of peak after cascade
        if self._liq_rate_peak[symbol] > 0:
            if total_liq_1h < 0.3 * self._liq_rate_peak[symbol]:
                state.liq_exhaustion = True
                self._liq_rate_peak[symbol] = 0  # Reset peak
            else:
                state.liq_exhaustion = False
    
    def update_price_ranges(self, symbol: str, high_4h: float, low_4h: float, high_24h: float, low_24h: float):
        """Update price ranges"""
        if symbol not in self._states:
            return
        
        state = self._states[symbol]
        state.high_4h = high_4h
        state.low_4h = low_4h
        state.high_24h = high_24h
        state.low_24h = low_24h
    
    def update_price_changes(self, symbol: str, change_1h: float, change_4h: float, change_24h: float):
        """Update price changes"""
        if symbol not in self._states:
            return
        
        state = self._states[symbol]
        state.price_change_1h = change_1h
        state.price_change_4h = change_4h
        state.price_change_24h = change_24h
    
    def update_volume_ratio(self, symbol: str, ratio: float):
        """Update volume ratio"""
        if symbol not in self._states:
            return
        self._states[symbol].volume_ratio = ratio
    
    # ========== GETTERS ==========
    
    def get_state(self, symbol: str) -> Optional[AlphaState]:
        """Get current alpha state for symbol"""
        return self._states.get(symbol)
    
    def get_market_state(self, symbol: str) -> Optional[MarketState]:
        """Get MarketState for stage3_v3 signals"""
        state = self._states.get(symbol)
        if state is None:
            return None
        return state.to_market_state()
    
    def get_all_states(self) -> Dict[str, AlphaState]:
        """Get all states"""
        return self._states.copy()
    
    def get_health_metrics(self) -> Dict[str, any]:
        """Get health metrics"""
        return {
            "symbols": len(self.symbols),
            "states_with_price": sum(1 for s in self._states.values() if s.current_price > 0),
            "states_with_funding": sum(1 for s in self._states.values() if s.funding_z != 0),
            "cascades_active": sum(1 for s in self._states.values() if s.cascade_active),
        }
