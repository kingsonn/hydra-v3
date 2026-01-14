"""
Exit Strategy Module

Implements hard exit conditions based on market state:
1. CHOP regime exit (if profitable)
2. Liquidation exhaustion exit (liq spike + no price response after 5s)
3. Exit near POC distance (within 0.3 ATR)
4. Exit near LVN distance (within 0.2 ATR)
5. Failed acceptance exit (outside VA but rejected)
6. MOI/Delta reversal exit (sustained 30s reversal with absorption)
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
import structlog

logger = structlog.get_logger(__name__)

# Exit thresholds
EXIT_NEAR_POC_ATR = 0.15  # Exit within 0.15 ATR of POC (tightened from 0.3)
EXIT_NEAR_LVN_ATR = 0.1   # Exit within 0.1 ATR of LVN (tightened from 0.2)
LIQ_SPIKE_THRESHOLD = 2.5  # liq_norm > 2.5 = spike
LIQ_RESPONSE_THRESHOLD = 0.25  # price_response < 0.25 = exhaustion
LIQ_RESPONSE_DELAY_S = 5.0  # Wait 5 seconds after spike
REVERSAL_THRESHOLD = -1.0  # MOI/delta reversal threshold
REVERSAL_SUSTAINED_S = 15.0  # Must sustain for 15 seconds (faster from 30s)
REVERSAL_FAST_THRESHOLD = -1.5  # Faster exit if reversal is strong
REVERSAL_FAST_S = 5.0  # Only need 5s if reversal is strong
ABSORPTION_REVERSAL_THRESHOLD = 1.2  # abs(absorption_z) > 1.2
FAILED_ACCEPTANCE_BUFFER_ATR = 0.15  # 0.15 ATR buffer for failed acceptance
CHOP_MIN_R_FOR_EXIT = 0.3  # Only exit CHOP if at least 0.3R profit (not just breakeven)


@dataclass
class LiqSpike:
    """Tracks a liquidation spike for exhaustion detection"""
    timestamp: float  # When spike detected
    price_at_spike: float  # Price when spike occurred
    liq_norm: float  # Normalized liquidation value
    checked: bool = False  # Whether we've checked price response


@dataclass
class SymbolExitState:
    """Per-symbol exit state tracking"""
    symbol: str
    
    # Liquidation exhaustion tracking
    liq_spike: Optional[LiqSpike] = None
    liquidation_exhaustion: bool = False
    
    # MOI/Delta reversal tracking (30s rolling)
    moi_z_history: deque = field(default_factory=lambda: deque(maxlen=60))  # 0.5s intervals = 30s
    delta_vel_z_history: deque = field(default_factory=lambda: deque(maxlen=60))
    last_history_update: float = 0.0
    
    # Current market state cache
    current_price: float = 0.0
    regime: str = "MID"
    absorption_z: float = 0.0
    dist_poc: float = 0.0
    dist_lvn: float = 0.0
    atr_5m: float = 0.0
    atr_1h: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    acceptance_outside_value: bool = False
    moi_z: float = 0.0
    delta_vel_z: float = 0.0
    liq_long_usd: float = 0.0
    liq_short_usd: float = 0.0


class ExitStrategyManager:
    """
    Manages exit strategies for all symbols with open positions
    
    Call update_market_state() on every price update
    Call check_exit() to evaluate exit conditions for a position
    """
    
    def __init__(self):
        self._states: Dict[str, SymbolExitState] = {}
        logger.info("exit_strategy_manager_initialized")
    
    def _get_state(self, symbol: str) -> SymbolExitState:
        """Get or create state for symbol"""
        if symbol not in self._states:
            self._states[symbol] = SymbolExitState(symbol=symbol)
        return self._states[symbol]
    
    def update_market_state(
        self,
        symbol: str,
        current_price: float,
        regime: str,
        absorption_z: float,
        dist_poc: float,
        dist_lvn: float,
        atr_5m: float,
        atr_1h: float,
        vah: float,
        val: float,
        acceptance_outside_value: bool,
        moi_z: float,
        delta_vel_z: float,
        liq_long_usd: float,
        liq_short_usd: float,
    ) -> None:
        """
        Update market state for a symbol
        Call this on every price update for symbols with open positions
        """
        state = self._get_state(symbol)
        now = time.time()
        
        # Update cached values
        state.current_price = current_price
        state.regime = regime
        state.absorption_z = absorption_z
        state.dist_poc = dist_poc
        state.dist_lvn = dist_lvn
        state.atr_5m = atr_5m
        state.atr_1h = atr_1h
        state.vah = vah
        state.val = val
        state.acceptance_outside_value = acceptance_outside_value
        state.moi_z = moi_z
        state.delta_vel_z = delta_vel_z
        state.liq_long_usd = liq_long_usd
        state.liq_short_usd = liq_short_usd
        
        # Update MOI/Delta history (every 0.5s)
        if now - state.last_history_update >= 0.5:
            state.moi_z_history.append((now, moi_z))
            state.delta_vel_z_history.append((now, delta_vel_z))
            state.last_history_update = now
        
        # Check for liquidation spike
        self._check_liq_spike(state, now)
        
        # Check liquidation exhaustion (5s after spike)
        self._check_liq_exhaustion(state, now)
    
    def _check_liq_spike(self, state: SymbolExitState, now: float) -> None:
        """Detect liquidation spike"""
        if state.atr_1h <= 0:
            return
        
        # Calculate ATR in dollar terms
        atr_1h_dollar = state.atr_1h * state.current_price if state.current_price > 0 else 1.0
        
        # Total liquidation USD
        total_liq_usd = state.liq_long_usd + state.liq_short_usd
        
        # Normalize by ATR
        liq_norm = total_liq_usd / atr_1h_dollar if atr_1h_dollar > 0 else 0
        
        # Detect spike
        if liq_norm > LIQ_SPIKE_THRESHOLD:
            # Only set new spike if none active or previous was checked
            if state.liq_spike is None or state.liq_spike.checked:
                state.liq_spike = LiqSpike(
                    timestamp=now,
                    price_at_spike=state.current_price,
                    liq_norm=liq_norm,
                    checked=False,
                )
                state.liquidation_exhaustion = False  # Reset
                logger.debug(
                    "liq_spike_detected",
                    symbol=state.symbol,
                    liq_norm=f"{liq_norm:.2f}",
                    price=state.current_price,
                )
    
    def _check_liq_exhaustion(self, state: SymbolExitState, now: float) -> None:
        """Check price response after liquidation spike"""
        if state.liq_spike is None or state.liq_spike.checked:
            return
        
        # Wait for delay
        elapsed = now - state.liq_spike.timestamp
        if elapsed < LIQ_RESPONSE_DELAY_S:
            return
        
        # Mark as checked
        state.liq_spike.checked = True
        
        # Calculate price response
        if state.atr_5m <= 0:
            return
        
        atr_5m_dollar = state.atr_5m * state.current_price if state.current_price > 0 else 1.0
        price_response = abs(state.current_price - state.liq_spike.price_at_spike) / atr_5m_dollar
        
        # Check exhaustion condition
        state.liquidation_exhaustion = price_response < LIQ_RESPONSE_THRESHOLD
        
        if state.liquidation_exhaustion:
            logger.info(
                "liq_exhaustion_detected",
                symbol=state.symbol,
                price_response=f"{price_response:.3f}",
                threshold=LIQ_RESPONSE_THRESHOLD,
            )
    
    def _check_moi_reversal(self, state: SymbolExitState, trade_direction: int) -> tuple:
        """
        Check if MOI reversal sustained
        Returns (is_reversal, is_fast_reversal)
        """
        if len(state.moi_z_history) < 5:
            return False, False
        
        now = time.time()
        
        # Check for fast strong reversal (5s at -1.5 threshold)
        fast_cutoff = now - REVERSAL_FAST_S
        fast_reversal = True
        for ts, moi_z in state.moi_z_history:
            if ts < fast_cutoff:
                continue
            if moi_z * trade_direction >= REVERSAL_FAST_THRESHOLD:
                fast_reversal = False
                break
        
        # Check for normal reversal (15s at -1.0 threshold)
        normal_cutoff = now - REVERSAL_SUSTAINED_S
        normal_reversal = True
        for ts, moi_z in state.moi_z_history:
            if ts < normal_cutoff:
                continue
            if moi_z * trade_direction >= REVERSAL_THRESHOLD:
                normal_reversal = False
                break
        
        return normal_reversal, fast_reversal
    
    def _check_delta_reversal(self, state: SymbolExitState, trade_direction: int) -> tuple:
        """
        Check if delta velocity reversal sustained
        Returns (is_reversal, is_fast_reversal)
        """
        if len(state.delta_vel_z_history) < 5:
            return False, False
        
        now = time.time()
        
        # Check for fast strong reversal
        fast_cutoff = now - REVERSAL_FAST_S
        fast_reversal = True
        for ts, delta_z in state.delta_vel_z_history:
            if ts < fast_cutoff:
                continue
            if delta_z * trade_direction >= REVERSAL_FAST_THRESHOLD:
                fast_reversal = False
                break
        
        # Check for normal reversal
        normal_cutoff = now - REVERSAL_SUSTAINED_S
        normal_reversal = True
        for ts, delta_z in state.delta_vel_z_history:
            if ts < normal_cutoff:
                continue
            if delta_z * trade_direction >= REVERSAL_THRESHOLD:
                normal_reversal = False
                break
        
        return normal_reversal, fast_reversal
    
    def check_exit(
        self,
        symbol: str,
        side: str,  # "LONG" or "SHORT"
        entry_price: float,
        breakeven_price: float,  # Already includes fees
        current_r: float,
        tp_2r_price: float,  # Price at 2R
    ) -> Optional[Dict[str, Any]]:
        """
        Check all exit conditions for a position
        
        Returns:
            Dict with exit reason and details if exit triggered, None otherwise
        """
        state = self._get_state(symbol)
        
        if state.current_price <= 0:
            return None
        
        trade_direction = 1 if side == "LONG" else -1
        is_long = side == "LONG"
        
        # 1. CHOP regime exit (only if at least 0.3R profit, not just breakeven)
        if state.regime == "CHOP" and current_r >= CHOP_MIN_R_FOR_EXIT:
            if is_long and state.current_price > breakeven_price:
                return {"reason": "CHOP_EXIT", "details": f"CHOP regime at {current_r:.2f}R, taking profit"}
            elif not is_long and state.current_price < breakeven_price:
                return {"reason": "CHOP_EXIT", "details": f"CHOP regime at {current_r:.2f}R, taking profit"}
        
        # 2. Liquidation exhaustion (only at >= 2R)
        if state.liquidation_exhaustion and current_r >= 2.0:
            if is_long and state.current_price >= tp_2r_price:
                return {"reason": "LIQ_EXHAUSTION", "details": f"Liq exhaustion at {current_r:.1f}R"}
            elif not is_long and state.current_price <= tp_2r_price:
                return {"reason": "LIQ_EXHAUSTION", "details": f"Liq exhaustion at {current_r:.1f}R"}
        
        # 3. Exit near POC (if profitable, tightened threshold)
        if state.dist_poc <= EXIT_NEAR_POC_ATR and state.dist_poc > 0 and current_r >= 0.3:
            if is_long and state.current_price > breakeven_price:
                return {"reason": "NEAR_POC", "details": f"Within {state.dist_poc:.2f} ATR of POC at {current_r:.2f}R"}
            elif not is_long and state.current_price < breakeven_price:
                return {"reason": "NEAR_POC", "details": f"Within {state.dist_poc:.2f} ATR of POC at {current_r:.2f}R"}
        
        # 4. Exit near LVN (if profitable, tightened threshold)
        if state.dist_lvn <= EXIT_NEAR_LVN_ATR and state.dist_lvn > 0 and current_r >= 0.3:
            if is_long and state.current_price > breakeven_price:
                return {"reason": "NEAR_LVN", "details": f"Within {state.dist_lvn:.2f} ATR of LVN at {current_r:.2f}R"}
            elif not is_long and state.current_price < breakeven_price:
                return {"reason": "NEAR_LVN", "details": f"Within {state.dist_lvn:.2f} ATR of LVN at {current_r:.2f}R"}
        
        # 5. Failed acceptance
        if state.atr_5m > 0:
            buffer = FAILED_ACCEPTANCE_BUFFER_ATR * state.atr_5m
            outside_va = False
            
            if is_long and state.current_price >= state.vah + buffer:
                outside_va = True
            elif not is_long and state.current_price <= state.val - buffer:
                outside_va = True
            
            if outside_va and not state.acceptance_outside_value and state.absorption_z > 1.0:
                return {
                    "reason": "FAILED_ACCEPTANCE",
                    "details": f"Outside VA but rejected, absorption_z={state.absorption_z:.2f}"
                }
        
        # 6. MOI/Delta reversal with absorption (supports fast and normal modes)
        moi_normal, moi_fast = self._check_moi_reversal(state, trade_direction)
        delta_normal, delta_fast = self._check_delta_reversal(state, trade_direction)
        
        # Fast exit: strong reversal (5s at -1.5) with high absorption
        if moi_fast and delta_fast and abs(state.absorption_z) > 1.5:
            return {
                "reason": "FLOW_REVERSAL_FAST",
                "details": f"Strong MOI+Delta reversal (5s), absorption_z={state.absorption_z:.2f}"
            }
        
        # Normal exit: sustained reversal (15s at -1.0) with absorption
        if moi_normal and delta_normal and abs(state.absorption_z) > ABSORPTION_REVERSAL_THRESHOLD:
            return {
                "reason": "FLOW_REVERSAL",
                "details": f"MOI+Delta reversal sustained 15s, absorption_z={state.absorption_z:.2f}"
            }
        
        return None
    
    def clear_symbol(self, symbol: str) -> None:
        """Clear state for a symbol (call when position closes)"""
        if symbol in self._states:
            del self._states[symbol]
    
    def get_diagnostics(self, symbol: str) -> Dict[str, Any]:
        """Get diagnostic info for a symbol"""
        if symbol not in self._states:
            return {}
        
        state = self._states[symbol]
        return {
            "regime": state.regime,
            "liquidation_exhaustion": state.liquidation_exhaustion,
            "liq_spike_active": state.liq_spike is not None and not state.liq_spike.checked,
            "moi_history_len": len(state.moi_z_history),
            "delta_history_len": len(state.delta_vel_z_history),
            "dist_poc": state.dist_poc,
            "dist_lvn": state.dist_lvn,
            "absorption_z": state.absorption_z,
        }
