"""
Stage 6: Position Sizing

Takes successful Stage 5 signals and calculates position sizes with two tranches:
- Tranche A: Tight stop (0.6x ATR_5m), single TP at 1R
- Tranche B: Wide stop (1.2x ATR_5m), partial TP at 2R (40%), runner TP at 3R

Rejects trades if ATR_1h_pct < min_stop_pct (0.32%)

V3: Simplified position sizer for hybrid alpha signals (signal provides stop/target)
"""
from src.stage6.models import Position, PositionResult, RejectionResult
from src.stage6.position_sizer import PositionSizer
from src.stage6.position_sizer_v3 import PositionSizerV3, PositionV3, PositionResultV3

__all__ = [
    "Position", "PositionResult", "RejectionResult", "PositionSizer",
    "PositionSizerV3", "PositionV3", "PositionResultV3",
]
