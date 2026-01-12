"""
Stage 2: Feature Construction & Market Regime Classification

Usage:
    # Run with dashboard UI
    python run_tests.py dashboard
    
    # Then open http://localhost:8080 to see all pairs updating in real-time
    
    # Run ML model testing dashboard
    python -m src.stage2.ml_runner
    
    # Then open http://localhost:8081 to see ML predictions
"""
from .models import (
    MarketState, Regime, RegimeClassification,
    OrderFlowFeatures, AbsorptionFeatures, VolatilityFeatures,
    StructureFeatures, FundingFeatures, OIFeatures, LiquidationFeatures,
    CrowdSide, ParticipationType,
)
from .orchestrator import Stage2Orchestrator

__all__ = [
    # Main
    "Stage2Orchestrator",
    "MarketState",
    "Regime",
    "RegimeClassification",
    # Features
    "OrderFlowFeatures",
    "AbsorptionFeatures",
    "VolatilityFeatures",
    "StructureFeatures",
    "FundingFeatures",
    "OIFeatures",
    "LiquidationFeatures",
    # Enums
    "CrowdSide",
    "ParticipationType",
]
