"""
Stage 4: Structural Location Filter - DEPRECATED/REMOVED
Stage 4.5: Orderflow Confirmation Filter - DEPRECATED/REMOVED

AUDIT VERDICT: DELETE ENTIRELY

Reasons for removal:
1. structural_location_ok: AND logic paradox (LVN ≠ VA boundary)
   - LVNs are low-volume nodes, VAH/VAL are high-volume boundaries
   - Requiring BOTH conditions simultaneously is geometrically impossible
   - Near-zero pass rate destroys all valid signals

2. orderflow_confirmation: Redundant with Stage 3 signals
   - Stage 3 signals already check moi_z, absorption_z, depth_imbalance
   - delta_velocity sign check is too noisy (flips every second)
   - depth_imbalance threshold (0.05) is meaningless noise
   - absorption_z directional interpretation was incorrect

Pipeline now flows: Stage 3 → Stage 5 (ML gate)

This module is kept for reference only. Do not import or use.
"""

# REMOVED - do not import
# from src.stage4.filter import (
#     StructuralFilter,
#     FilterResult,
#     structural_location_ok,
#     orderflow_confirmation,
# )

__all__ = []  # Nothing exported - module deprecated
