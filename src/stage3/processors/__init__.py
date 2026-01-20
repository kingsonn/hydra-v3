"""
Stage 3 Signal Processors - REPAIRED VERSION

Active Entry Signals (7):
- funding_price_cointegration
- hawkes_liquidation_cascade
- liquidity_crisis_detector
- flip_rate_compression_break
- exhaustion_reversal
- sweep_vacuum_continuation
- absorption_flow_divergence (NEW)

Filters (3):
- kyle_lambda_filter
- oi_not_expanding_filter
- vol_expansion_filter

REMOVED (broken logic):
- queue_reactive_liquidity
- order_flow_dominance_decay
- inventory_lock
- failed_acceptance_reversal
- entropy_flow_signal
- poc_magnetic_reversal
- value_area_rejection
- absorption_accumulation_breakout
"""
from src.stage3.processors.signals import (
    funding_price_cointegration,
    hawkes_liquidation_cascade,
    liquidity_crisis_detector,
    flip_rate_compression_break,
    exhaustion_reversal,
    sweep_vacuum_continuation,
    absorption_flow_divergence,
    kyle_lambda_filter,
    oi_not_expanding_filter,
    vol_expansion_filter,
    get_all_signals,
)

__all__ = [
    "funding_price_cointegration",
    "hawkes_liquidation_cascade",
    "liquidity_crisis_detector",
    "flip_rate_compression_break",
    "exhaustion_reversal",
    "sweep_vacuum_continuation",
    "absorption_flow_divergence",
    "kyle_lambda_filter",
    "oi_not_expanding_filter",
    "vol_expansion_filter",
    "get_all_signals",
]
