"""
Run Global Pipeline Dashboard
All 8 pairs through 5 stages with real-time visualization

Usage:
    python run_global_dashboard.py
    
Dashboard will be available at http://localhost:8888
"""
import asyncio
import sys

# Add project root to path
sys.path.insert(0, ".")

from src.dashboard.global_runner import run_global_pipeline


if __name__ == "__main__":
    print("Starting Hydra Global Pipeline Dashboard...")
    asyncio.run(run_global_pipeline())
