#!/usr/bin/env python3
"""
VM Diagnostic Script for Hydra V3
Run this on your EC2 instance to check for common issues.
"""
import asyncio
import sys
import time
import platform
import os

def check_python():
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU count: {os.cpu_count()}")
    return True

def check_memory():
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"Memory: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
        print(f"Memory usage: {mem.percent}%")
        if mem.percent > 80:
            print("  WARNING: High memory usage!")
            return False
        return True
    except ImportError:
        print("Memory: psutil not installed (pip install psutil)")
        return True

def check_cpu():
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=1)
        print(f"CPU usage: {cpu}%")
        if cpu > 80:
            print("  WARNING: High CPU usage - t3 credits may be exhausted!")
            return False
        return True
    except ImportError:
        return True

async def check_binance_latency():
    import aiohttp
    url = "https://fapi.binance.com/fapi/v1/ping"
    try:
        start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                await resp.text()
        latency = (time.time() - start) * 1000
        print(f"Binance API latency: {latency:.0f}ms")
        if latency > 500:
            print("  WARNING: High latency to Binance!")
            return False
        return True
    except Exception as e:
        print(f"Binance API: FAILED - {e}")
        return False

async def check_binance_ws():
    try:
        import websockets
        url = "wss://fstream.binance.com/ws/btcusdt@trade"
        start = time.time()
        async with websockets.connect(url, ping_interval=None, close_timeout=5) as ws:
            msg = await asyncio.wait_for(ws.recv(), timeout=10)
            latency = (time.time() - start) * 1000
            print(f"Binance WebSocket: OK ({latency:.0f}ms to first message)")
            return True
    except Exception as e:
        print(f"Binance WebSocket: FAILED - {e}")
        return False

async def check_weex():
    import aiohttp
    url = "https://api.weex.com/api/v1/public/time"
    try:
        start = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                await resp.text()
        latency = (time.time() - start) * 1000
        print(f"WEEX API latency: {latency:.0f}ms")
        return True
    except Exception as e:
        print(f"WEEX API: FAILED - {e}")
        return False

def check_env():
    from pathlib import Path
    env_file = Path(".env")
    if not env_file.exists():
        print(".env file: MISSING")
        return False
    print(".env file: OK")
    
    from config import settings
    if not settings.WEEX_API_KEY:
        print("  WARNING: WEEX_API_KEY not set")
    if not settings.WEEX_SECRET_KEY:
        print("  WARNING: WEEX_SECRET_KEY not set")
    
    print(f"VM_MODE: {settings.VM_MODE}")
    return True

async def main():
    print("=" * 50)
    print("HYDRA V3 - VM DIAGNOSTIC CHECK")
    print("=" * 50)
    print()
    
    results = []
    
    print("[1/6] Python & System")
    results.append(check_python())
    print()
    
    print("[2/6] Memory")
    results.append(check_memory())
    print()
    
    print("[3/6] CPU")
    results.append(check_cpu())
    print()
    
    print("[4/6] Binance REST API")
    results.append(await check_binance_latency())
    print()
    
    print("[5/6] Binance WebSocket")
    results.append(await check_binance_ws())
    print()
    
    print("[6/6] WEEX API")
    results.append(await check_weex())
    print()
    
    print("[7/7] Environment")
    results.append(check_env())
    print()
    
    print("=" * 50)
    if all(results):
        print("ALL CHECKS PASSED")
        print("\nTo enable VM mode, add to .env:")
        print("  VM_MODE=true")
        print("  UPDATE_INTERVAL_S=2.0")
    else:
        print("SOME CHECKS FAILED - see warnings above")
        print("\nRecommendations for t3.large:")
        print("1. Enable VM_MODE=true in .env")
        print("2. Consider upgrading to t3.xlarge or m5.large")
        print("3. Check CloudWatch for CPU credit balance")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
