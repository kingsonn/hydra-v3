"""
Stage 3 V2: Long-Horizon Positioning Alpha
==========================================

Replaces microstructure-based signals with positioning-based structural alpha.

Key differences from Stage 3 V1:
- Signals fire rarely (0-2 per day per symbol)
- Signals have MEMORY (track regime persistence)
- Holding times: 6h - 72h
- Wide stops: 2-5%
- Based on forced flows and positioning resolution

Components:
- models.py: Data models and state
- signals.py: Entry signals with memory
- gates.py: Veto/permission signals
- filters.py: Parameter adjustment
- engine.py: Main orchestrator

AI modules (in ai/):
- regime_classifier.py: Positioning regime detection
- instability.py: Forced resolution detection
- anomaly.py: Unusual state detection
- similarity.py: Historical context retrieval
"""

from src.stage3_v2.models import (
    LongHorizonState,
    PositioningSignal,
    SignalMemory,
    Direction,
    SignalType,
    PositioningRegime,
)

__all__ = [
    "LongHorizonState",
    "PositioningSignal", 
    "SignalMemory",
    "Direction",
    "SignalType",
    "PositioningRegime",
]
