#!/usr/bin/env python3
"""
Hydra Global Pipeline Entry Point
Proper signal handling for systemd service

Now runs V3 hybrid alpha system by default.
"""
import asyncio
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


def main():
    """Main entry point with proper signal handling"""
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
        loop.run_until_complete(run_global_pipeline_v3())
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
    main()
