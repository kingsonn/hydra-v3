#!/usr/bin/env python
"""
HYDRA V3 - TEST RUNNER

Stage 1 (Data Ingestion):
    python run_tests.py trades      # Test trades collector (5s)
    python run_tests.py orderbook   # Test order book collector (5s)
    python run_tests.py bars        # Test bar aggregation (10s)
    python run_tests.py derivatives # Test funding/OI/liquidations (10s)
    python run_tests.py liquidations # Test liquidations WebSocket (indefinite)
    python run_tests.py profile     # Test volume profile
    python run_tests.py health      # Test health monitor
    python run_tests.py full        # Full Stage 1 test (30s)

Stage 2 (Feature Engine):
    python run_tests.py stage2      # Test Stage 2 console (60s)
    python run_tests.py dashboard   # Run Stage 2 with dashboard UI (http://localhost:8080)

General:
    python run_tests.py all         # Run all unit tests
    python run_tests.py interactive # Interactive test menu
"""
import sys
import asyncio

# Add src to path
sys.path.insert(0, '.')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    test = sys.argv[1].lower()
    
    if test == "trades":
        from src.collectors.trades import test_trades_collector
        asyncio.run(test_trades_collector(5))
    
    elif test == "orderbook":
        from src.collectors.orderbook import test_orderbook_collector
        asyncio.run(test_orderbook_collector(5))
    
    elif test == "bars":
        from src.processors.bars import test_bar_aggregator_live
        asyncio.run(test_bar_aggregator_live(10))
    
    elif test == "bars-sync":
        from src.processors.bars import test_bar_builder, test_bar_aggregator_sync
        test_bar_builder()
        print()
        test_bar_aggregator_sync()
    
    elif test == "derivatives":
        from src.collectors.derivatives import test_derivatives_collector
        asyncio.run(test_derivatives_collector(10))
    
    elif test == "funding":
        from src.collectors.derivatives import test_funding_rate
        asyncio.run(test_funding_rate("BTCUSDT"))
    
    elif test == "oi":
        from src.collectors.derivatives import test_open_interest
        asyncio.run(test_open_interest("BTCUSDT"))
    
    elif test == "liquidations":
        from src.collectors.derivatives import test_liquidations_ws
        asyncio.run(test_liquidations_ws())  # Run indefinitely
    
    elif test == "profile":
        from src.processors.volume_profile import test_volume_profile_sync
        test_volume_profile_sync()
    
    elif test == "profile-live":
        from src.processors.volume_profile import test_rolling_profiler_live
        asyncio.run(test_rolling_profiler_live(30))
    
    elif test == "health":
        from src.health.monitor import test_health_monitor
        asyncio.run(test_health_monitor(10))
    
    elif test == "full":
        from src.stage1 import test_stage1_full
        asyncio.run(test_stage1_full(30))
    
    # ========== STAGE 2 TESTS ==========
    
    elif test == "stage2":
        from src.stage2.runner import test_stage2_standalone
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        asyncio.run(test_stage2_standalone(duration))
    
    elif test == "dashboard":
        from src.stage2.runner import run_stage2_live
        asyncio.run(run_stage2_live())
    
    elif test == "all":
        import pytest
        pytest.main(["tests/test_stage1.py", "-v"])
    
    elif test == "interactive":
        from tests.test_stage1 import run_interactive_tests
        run_interactive_tests()
    
    else:
        print(f"Unknown test: {test}")
        print(__doc__)


if __name__ == "__main__":
    main()
