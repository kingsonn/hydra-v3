#!/usr/bin/env python3
"""
Hydra Global Pipeline Entry Point
Proper signal handling for systemd service

Now runs V3 hybrid alpha system by default.

Usage:
  python run_hydra.py                    # Paper trading (simulation only)
  python run_hydra.py --live             # Live trading with WEEX (dry-run mode)
  python run_hydra.py --live --real      # REAL TRADING - actual orders placed!
"""
import argparse
import asyncio
import multiprocessing
import os
import signal
import sys
from pathlib import Path

# Add project root to path (works with absolute paths)
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)  # Ensure working directory is correct

import structlog
from src.dashboard.global_runner_v3 import run_global_pipeline_v3

logger = structlog.get_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Hydra V3 Trading Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_hydra.py                    # Paper trading (simulation only)
  python run_hydra.py --live             # Live trading with WEEX (dry-run mode)
  python run_hydra.py --live --real      # REAL TRADING - actual orders placed!
        """
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading mode (connects to WEEX API)"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="Execute REAL orders (requires --live). WITHOUT THIS FLAG, orders are simulated."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8889,
        help="Dashboard port (default: 8889)"
    )
    return parser.parse_args()


def main():
    """Main entry point with proper signal handling"""
    args = parse_args()
    
    # Safety check for real trading
    if args.real and not args.live:
        print("ERROR: --real requires --live flag")
        sys.exit(1)
    
    # Determine trading mode
    live_trading = args.live
    dry_run = not args.real  # dry_run=True unless --real is specified
    
    if live_trading:
        if dry_run:
            print("=" * 60)
            print("LIVE TRADING MODE (DRY-RUN)")
            print("Orders will be SIMULATED, not placed on exchange")
            print("=" * 60)
        else:
            print("=" * 60)
            print("⚠️  REAL TRADING MODE - ACTUAL ORDERS WILL BE PLACED!")
            print("⚠️  This will use REAL money on WEEX exchange!")
            print("=" * 60)
            sys.stdout.flush()
            try:
                response = input("Type 'YES' to confirm: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(0)
            if response != "YES":
                print("Aborted.")
                sys.exit(0)
            print("Confirmed. Starting real trading...")
    else:
        print("=" * 60)
        print("PAPER TRADING MODE (simulation only)")
        print("=" * 60)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Global exception handler
    def global_exception_handler(loop, context):
        exception = context.get("exception")
        message = context.get("message", "Unknown error")
        logger.error(
            "uncaught_async_exception",
            message=message,
            exception=str(exception) if exception else "None",
        )
    
    loop.set_exception_handler(global_exception_handler)
    
    # Signal handler for graceful shutdown
    shutdown_event = asyncio.Event()
    
    def signal_handler(signum, frame):
        logger.info("shutdown_signal_received", signal=signum)
        loop.call_soon_threadsafe(shutdown_event.set)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Run the V3 pipeline
        loop.run_until_complete(run_global_pipeline_v3(
            dashboard_port=args.port,
            live_trading=live_trading,
            dry_run=dry_run,
        ))
    except KeyboardInterrupt:
        logger.info("keyboard_interrupt")
    finally:
        logger.info("cleaning_up_tasks")
        
        # Cancel all pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        # Wait for cancellation with timeout
        if pending:
            try:
                loop.run_until_complete(
                    asyncio.wait(pending, timeout=10.0)
                )
            except Exception as e:
                logger.warning("cleanup_error", error=str(e))
        
        # Close the loop
        try:
            loop.close()
        except Exception as e:
            logger.warning("loop_close_error", error=str(e))
        
        logger.info("hydra_shutdown_complete")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
