"""
Stage 4: Structural Location Filter
Stage 4.5: Orderflow Confirmation Filter

Filters Stage 3 signals based on price location relative to:
- 5-minute LVN (Low Volume Node)
- 30-minute Value Area (VAH/VAL)

Then confirms orderflow alignment with signal direction.

Only allows trades:
- Near 5m LVN (ML learned this is important)
- OR at 30m value extremes (VAL for LONG, VAH for SHORT)
- AND orderflow confirms direction
- Everywhere else → WAIT (signal rejected)
"""

from src.stage4.filter import (
    StructuralFilter,
    FilterResult,
    structural_location_ok,
    orderflow_confirmation,
)

__all__ = [
    "StructuralFilter",
    "FilterResult",
    "structural_location_ok",
    "orderflow_confirmation",
]
