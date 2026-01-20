"""
Stage 3 V2 AI Modules
=====================

AI modules for state enrichment WITHOUT direction prediction.

Modules:
- RegimeClassifier: Classify positioning regime (5 classes)
- InstabilityDetector: Detect forced-resolution probability
- AnomalyDetector: Detect unusual positioning states
- ContextSimilarity: Find similar historical episodes
"""

from src.stage3_v2.ai.regime_classifier import PositioningRegimeClassifier
from src.stage3_v2.ai.instability import InstabilityDetector
from src.stage3_v2.ai.anomaly import PositioningAnomalyDetector
from src.stage3_v2.ai.similarity import SignalContextSimilarity

__all__ = [
    "PositioningRegimeClassifier",
    "InstabilityDetector", 
    "PositioningAnomalyDetector",
    "SignalContextSimilarity",
]
