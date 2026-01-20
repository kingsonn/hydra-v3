"""
Stage 4: Structural Location Filter
Stage 4.5: Orderflow Confirmation Filter

Filters signals based on price location relative to volume profile structures.
Only allows trades near LVN or at value area extremes.
Then confirms with orderflow alignment using normalized metrics.
"""
from dataclasses import dataclass
from typing import Optional, Dict
from enum import Enum

from src.stage3.models import Signal, Direction
from dataclasses import dataclass as dc_dataclass


# Stub PairThresholds - actual file deleted, using AlphaStateProcessor for V3
@dc_dataclass
class PairThresholds:
    """Stub thresholds - V3 pipeline uses AlphaStateProcessor"""
    vol_expansion_high: float = 1.30
    vol_compression_low: float = 0.75
    aggression_high: float = 1.15
    aggression_low: float = 0.85
    moi_std_high_pct: float = 60.0
    moi_std_low_pct: float = 40.0
    delta_flip_low: float = 2.0
    delta_flip_high: float = 5.0
    absorption_z_noise: float = 1.0
    absorption_z_spike: float = 2.5
    value_width_compression: float = 0.65
    value_width_wide: float = 1.05
    time_inside_compression: float = 0.60


# Stub - V3 pipeline doesn't use per-pair thresholds
PAIR_THRESHOLDS = {}


# LVN distance threshold (in ATR units)
# If price is within this distance of LVN, signal is allowed
LVN_THRESHOLD = 0.3

# Default thresholds for unknown pairs
DEFAULT_THRESHOLDS = PairThresholds()

# Depth imbalance threshold (book should favor direction)
DEPTH_IMBALANCE_THRESHOLD = 0.05


class FilterReason(Enum):
    """Reasons for filter decision"""
    ALLOWED_NEAR_LVN = "near_lvn"
    ALLOWED_AT_VAL = "at_val_extreme"
    ALLOWED_AT_VAH = "at_vah_extreme"
    REJECTED_CHOP = "regime_chop"
    REJECTED_BAD_LOCATION = "bad_structural_location"
    REJECTED_ORDERFLOW = "orderflow_not_confirmed"


@dataclass
class FilterResult:
    """Result of structural location filter"""
    allowed: bool
    reason: FilterReason
    signal: Optional[Signal]
    
    # Location details for transparency
    dist_lvn: float = 0.0
    price: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason.value,
            "signal": self.signal.to_dict() if self.signal else None,
            "dist_lvn": self.dist_lvn,
            "price": self.price,
            "vah": self.vah,
            "val": self.val,
        }


def structural_location_ok(
    direction: Direction,
    regime: str,
    price: float,
    dist_lvn: float,
    vah: float,
    val: float,
    lvn_threshold: float = LVN_THRESHOLD,
) -> tuple[bool, FilterReason]:
    """
    Check if structural location is valid for the given direction.
    
    We only trade when BOTH conditions are met:
    - Near 5m LVN (dist_lvn < threshold)
    - AND at 30m value extremes (VAL for LONG, VAH for SHORT)
    - Everywhere else → WAIT
    
    Args:
        direction: LONG or SHORT
        regime: Current regime (CHOP, COMPRESSION, EXPANSION)
        price: Current price
        dist_lvn: Distance to LVN in ATR units
        vah: Value Area High (30m)
        val: Value Area Low (30m)
        lvn_threshold: Max distance to LVN to allow trade
    
    Returns:
        (allowed, reason) tuple
    """
    # Block all signals in CHOP
    if regime == "CHOP":
        return False, FilterReason.REJECTED_CHOP
    
    # Must be near LVN first
    near_lvn = dist_lvn < lvn_threshold
    if not near_lvn:
        return False, FilterReason.REJECTED_BAD_LOCATION
    
    # Also must be at value area extreme based on direction
    if direction == Direction.LONG:
        # LONG requires near LVN AND at VAL (value area low)
        if val > 0 and price <= val:
            return True, FilterReason.ALLOWED_AT_VAL
    
    elif direction == Direction.SHORT:
        # SHORT requires near LVN AND at VAH (value area high)
        if vah > 0 and price >= vah:
            return True, FilterReason.ALLOWED_AT_VAH
    
    # Not near LVN and not at value extreme → REJECT
    return False, FilterReason.REJECTED_BAD_LOCATION


def orderflow_confirmation(
    direction: Direction,
    symbol: str,
    delta_velocity: float,
    depth_imbalance: float,
    absorption_z: float,
) -> bool:
    """
    Stage 4.5: Orderflow Confirmation
    
    Answers: "Is the move actually starting RIGHT NOW?"
    
    Uses normalized metrics that work across all pairs:
    - delta_velocity: Sign alignment (momentum building in direction)
    - depth_imbalance: Book favors direction (more bids for LONG, more asks for SHORT)
    - absorption_z: Move not being absorbed (pair-specific threshold)
    
    Args:
        direction: Signal direction (LONG or SHORT)
        symbol: Trading pair for pair-specific thresholds
        delta_velocity: Rate of change of MOI (sign matters)
        depth_imbalance: (bid_depth - ask_depth) / total, range [-1, 1]
        absorption_z: Z-scored absorption ratio
    
    Returns:
        True if orderflow confirms direction, False otherwise
    """
    # Get pair-specific thresholds
    thresholds = PAIR_THRESHOLDS.get(symbol, DEFAULT_THRESHOLDS)
    
    if direction == Direction.LONG:
        # For LONG: momentum up, book favors bids, not being absorbed
        return (
            delta_velocity > 0  # Momentum building upward
            and depth_imbalance > DEPTH_IMBALANCE_THRESHOLD  # More bids than asks
            and absorption_z < thresholds.absorption_z_spike  # Not hitting wall of sells
        )
    
    if direction == Direction.SHORT:
        # For SHORT: momentum down, book favors asks, not being absorbed
        return (
            delta_velocity < 0  # Momentum building downward
            and depth_imbalance < -DEPTH_IMBALANCE_THRESHOLD  # More asks than bids
            and absorption_z > -thresholds.absorption_z_spike  # Not hitting wall of buys
        )
    
    return False


class StructuralFilter:
    """
    Stage 4 Structural Location Filter
    
    Filters Stage 3 signals based on price location.
    Only allows trades near LVN or at value area extremes.
    """
    
    def __init__(self, lvn_threshold: float = LVN_THRESHOLD):
        self.lvn_threshold = lvn_threshold
        
        # Stats tracking
        self._total_signals = 0
        self._allowed_signals = 0
        self._rejected_chop = 0
        self._rejected_location = 0
        self._allowed_lvn = 0
        self._allowed_val = 0
        self._allowed_vah = 0
    
    def filter_signal(
        self,
        signal: Signal,
        regime: str,
        price: float,
        dist_lvn: float,
        vah: float,
        val: float,
    ) -> FilterResult:
        """
        Filter a Stage 3 signal based on structural location.
        
        Args:
            signal: The Stage 3 signal to filter
            regime: Current market regime
            price: Current price
            dist_lvn: Distance to 5m LVN (in ATR units)
            vah: 30m Value Area High
            val: 30m Value Area Low
        
        Returns:
            FilterResult with allowed/rejected decision
        """
        self._total_signals += 1
        
        allowed, reason = structural_location_ok(
            direction=signal.direction,
            regime=regime,
            price=price,
            dist_lvn=dist_lvn,
            vah=vah,
            val=val,
            lvn_threshold=self.lvn_threshold,
        )
        
        # Update stats
        if allowed:
            self._allowed_signals += 1
            if reason == FilterReason.ALLOWED_NEAR_LVN:
                self._allowed_lvn += 1
            elif reason == FilterReason.ALLOWED_AT_VAL:
                self._allowed_val += 1
            elif reason == FilterReason.ALLOWED_AT_VAH:
                self._allowed_vah += 1
        else:
            if reason == FilterReason.REJECTED_CHOP:
                self._rejected_chop += 1
            else:
                self._rejected_location += 1
        
        return FilterResult(
            allowed=allowed,
            reason=reason,
            signal=signal if allowed else None,
            dist_lvn=dist_lvn,
            price=price,
            vah=vah,
            val=val,
        )
    
    def get_stats(self) -> dict:
        """Get filter statistics"""
        return {
            "total_signals": self._total_signals,
            "allowed_signals": self._allowed_signals,
            "rejected_signals": self._total_signals - self._allowed_signals,
            "pass_rate": self._allowed_signals / max(1, self._total_signals),
            "allowed_by_lvn": self._allowed_lvn,
            "allowed_by_val": self._allowed_val,
            "allowed_by_vah": self._allowed_vah,
            "rejected_by_chop": self._rejected_chop,
            "rejected_by_location": self._rejected_location,
        }
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self._total_signals = 0
        self._allowed_signals = 0
        self._rejected_chop = 0
        self._rejected_location = 0
        self._allowed_lvn = 0
        self._allowed_val = 0
        self._allowed_vah = 0
