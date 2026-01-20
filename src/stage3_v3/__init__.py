"""
Stage 3 V3: Hybrid Alpha System
===============================

Daily-firing signals combining:
- Positional bias (funding, liquidations, OI)
- Trend/momentum timing
- Wide stops, longer holds

Target: 1-2 signals per day, 0.15%+ edge per trade after fees
"""

from src.stage3_v3.bias import BiasCalculator, Bias
from src.stage3_v3.models import MarketRegime
from src.stage3_v3.trend import TrendAnalyzer, TrendState
from src.stage3_v3.signals import (
    FundingTrendSignal,
    TrendPullbackSignal,
    LiquidationFollowSignal,
    RangeBreakoutSignal,
    ExhaustionReversalSignal,
)
from src.stage3_v3.engine import HybridAlphaEngine

__all__ = [
    "BiasCalculator",
    "Bias",
    "MarketRegime",
    "TrendAnalyzer",
    "TrendState",
    "FundingTrendSignal",
    "TrendPullbackSignal",
    "LiquidationFollowSignal",
    "RangeBreakoutSignal",
    "ExhaustionReversalSignal",
    "HybridAlphaEngine",
]
