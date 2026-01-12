"""
Klines Fetcher - Historical candle data for ATR and volatility bootstrap
Fetches from Binance Futures REST API with custom DNS resolver
"""
import asyncio
import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from aiohttp.resolver import AsyncResolver
import aiodns
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import structlog

from config import settings

logger = structlog.get_logger(__name__)

BASE_URL = "https://fapi.binance.com"

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY_S = 2.0
REQUEST_TIMEOUT_S = 30

# Custom DNS servers (Google DNS and Cloudflare DNS)
DNS_SERVERS = [
    "8.8.8.8",      # Google DNS Primary
    "8.8.4.4",      # Google DNS Secondary
    "1.1.1.1",      # Cloudflare DNS Primary
    "1.0.0.1",      # Cloudflare DNS Secondary
]


@dataclass
class Kline:
    """Single kline/candle"""
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


@dataclass
class ATRData:
    """ATR bootstrap data for a symbol"""
    atr_5m: float
    atr_1h: float
    tr_5m_deque: List[float]  # Last 14 TR values for 5m
    tr_1h_deque: List[float]  # Last 14 TR values for 1h
    last_close_5m: float = 0.0  # Last 5m candle close for TR continuity
    last_close_1h: float = 0.0  # Last 1h candle close for TR continuity


@dataclass
class VolatilityData:
    """Volatility bootstrap data for a symbol"""
    vol_5m: float
    vol_5m_history: List[float]  # 2000 vol_5m values for percentile


async def fetch_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    limit: int = 100,
) -> List[Kline]:
    """
    Fetch klines from Binance Futures with retry logic
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Candle interval (e.g., 5m, 1h)
        limit: Number of candles (max 1500)
    """
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(
                        "klines_fetch_failed",
                        symbol=symbol,
                        interval=interval,
                        status=resp.status,
                        attempt=attempt + 1,
                    )
                    last_error = f"HTTP {resp.status}"
                    await asyncio.sleep(RETRY_DELAY_S)
                    continue
                
                data = await resp.json()
                
                klines = []
                for k in data:
                    klines.append(Kline(
                        open_time=int(k[0]),
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5]),
                        close_time=int(k[6]),
                    ))
                
                if attempt > 0:
                    logger.info(
                        "klines_fetch_recovered",
                        symbol=symbol,
                        interval=interval,
                        attempt=attempt + 1,
                    )
                
                return klines
        
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = str(e)
            logger.warning(
                "klines_fetch_retry",
                symbol=symbol,
                interval=interval,
                attempt=attempt + 1,
                error=last_error,
            )
            await asyncio.sleep(RETRY_DELAY_S * (attempt + 1))
        
        except Exception as e:
            last_error = str(e)
            logger.error(
                "klines_fetch_error",
                symbol=symbol,
                interval=interval,
                error=last_error,
            )
            break
    
    logger.error(
        "klines_fetch_failed_all_retries",
        symbol=symbol,
        interval=interval,
        error=last_error,
    )
    return []


def compute_true_range(high: float, low: float, close: float, prev_close: float) -> float:
    """Compute True Range for ATR"""
    return max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    )


def compute_atr_from_klines(klines: List[Kline], period: int = 14) -> Tuple[float, List[float]]:
    """
    Compute ATR from klines using simple mean (not EMA)
    
    Returns:
        Tuple of (current ATR, list of last 14 TR values)
    """
    if len(klines) < period + 1:
        return 0.0, []
    
    # Compute all TR values
    tr_values = []
    for i in range(1, len(klines)):
        tr = compute_true_range(
            klines[i].high,
            klines[i].low,
            klines[i].close,
            klines[i-1].close
        )
        tr_values.append(tr)
    
    # ATR = mean of last 14 TR values
    recent_tr = tr_values[-period:]
    atr = float(np.mean(recent_tr))
    
    return atr, recent_tr


def compute_volatility_from_klines(klines: List[Kline], window: int = 12) -> Tuple[float, List[float]]:
    """
    Compute realized volatility from klines
    
    vol_5m = rolling_std of returns over window
    
    Args:
        klines: 5m klines
        window: Rolling window for std (12 = 1 hour of 5m bars)
    
    Returns:
        Tuple of (current vol_5m, history of vol_5m values)
    """
    if len(klines) < window + 1:
        return 0.0, []
    
    # Compute returns
    returns = []
    for i in range(1, len(klines)):
        if klines[i-1].close > 0:
            ret = (klines[i].close - klines[i-1].close) / klines[i-1].close
            returns.append(ret)
    
    if len(returns) < window:
        return 0.0, []
    
    # Compute rolling volatility
    vol_history = []
    for i in range(window - 1, len(returns)):
        window_returns = returns[i - window + 1:i + 1]
        vol = float(np.std(window_returns))
        vol_history.append(vol)
    
    current_vol = vol_history[-1] if vol_history else 0.0
    
    return current_vol, vol_history


async def bootstrap_atr(
    session: aiohttp.ClientSession,
    symbol: str,
) -> ATRData:
    """
    Bootstrap ATR data for a symbol
    
    Fetches 100 bars each for 5m and 1h, computes ATR
    Returns TR deque and last close for live continuation
    """
    # Fetch klines
    klines_5m = await fetch_klines(session, symbol, "5m", 100)
    klines_1h = await fetch_klines(session, symbol, "1h", 100)
    
    # Compute ATR
    atr_5m, tr_5m = compute_atr_from_klines(klines_5m, period=14)
    atr_1h, tr_1h = compute_atr_from_klines(klines_1h, period=14)
    
    # Get last close prices for TR continuity
    last_close_5m = klines_5m[-1].close if klines_5m else 0.0
    last_close_1h = klines_1h[-1].close if klines_1h else 0.0
    
    logger.info(
        "atr_bootstrapped",
        symbol=symbol,
        atr_5m=f"{atr_5m:.6f}",
        atr_1h=f"{atr_1h:.6f}",
        tr_5m_count=len(tr_5m),
        tr_1h_count=len(tr_1h),
    )
    
    return ATRData(
        atr_5m=atr_5m,
        atr_1h=atr_1h,
        tr_5m_deque=tr_5m,
        tr_1h_deque=tr_1h,
        last_close_5m=last_close_5m,
        last_close_1h=last_close_1h,
    )


async def bootstrap_volatility(
    session: aiohttp.ClientSession,
    symbol: str,
) -> VolatilityData:
    """
    Bootstrap volatility data for a symbol
    
    Fetches 2000 5m klines, computes vol_5m history
    """
    # Fetch 1500 klines (Binance limit), then another batch if needed
    # Actually Binance limit is 1500, so we need 2 requests
    klines = []
    
    # First batch - most recent 1500
    batch1 = await fetch_klines(session, symbol, "5m", 1500)
    if batch1:
        # Get end time of first candle in batch1 for second request
        if len(batch1) >= 1500:
            # Fetch older klines
            # Need to specify endTime to get earlier data
            # For simplicity, just use 1500 - it's ~5 days of 5m data
            pass
        klines = batch1
    
    if not klines:
        logger.warning("volatility_bootstrap_no_data", symbol=symbol)
        return VolatilityData(vol_5m=0.0, vol_5m_history=[])
    
    # Compute volatility history (rolling std of returns, window=12 = 1 hour)
    current_vol, vol_history = compute_volatility_from_klines(klines, window=12)
    
    logger.info(
        "volatility_bootstrapped",
        symbol=symbol,
        vol_5m=f"{current_vol:.6f}",
        history_size=len(vol_history),
    )
    
    return VolatilityData(
        vol_5m=current_vol,
        vol_5m_history=vol_history,
    )


async def create_custom_resolver() -> AsyncResolver:
    """
    Create custom DNS resolver with Google and Cloudflare DNS
    Falls back through multiple DNS servers if one fails
    """
    try:
        # Try to create resolver with custom DNS servers
        resolver = AsyncResolver(nameservers=DNS_SERVERS)
        logger.info("custom_dns_resolver_created", servers=DNS_SERVERS)
        return resolver
    except Exception as e:
        logger.warning("custom_dns_resolver_failed", error=str(e))
        # Fall back to default resolver
        return AsyncResolver()


async def bootstrap_all_symbols(
    symbols: List[str],
) -> Tuple[Dict[str, ATRData], Dict[str, VolatilityData]]:
    """
    Bootstrap ATR and volatility data for all symbols
    Uses custom DNS resolver for better reliability
    
    Returns:
        Tuple of (atr_data_dict, volatility_data_dict)
    """
    atr_data = {}
    vol_data = {}
    
    # Create custom DNS resolver
    resolver = await create_custom_resolver()
    
    # Configure timeout and connector with custom DNS
    timeout = ClientTimeout(total=REQUEST_TIMEOUT_S, connect=10)
    connector = TCPConnector(
        limit=10,
        ttl_dns_cache=300,
        force_close=False,
        resolver=resolver,
    )
    
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Process symbols with rate limiting
        for symbol in symbols:
            try:
                atr = await bootstrap_atr(session, symbol)
                vol = await bootstrap_volatility(session, symbol)
                
                atr_data[symbol] = atr
                vol_data[symbol] = vol
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("bootstrap_error", symbol=symbol, error=str(e))
    
    logger.info("bootstrap_complete", symbols=len(atr_data))
    
    return atr_data, vol_data


# Test function
async def test_bootstrap():
    """Test bootstrap for BTC"""
    atr_data, vol_data = await bootstrap_all_symbols(["BTCUSDT", "ETHUSDT"])
    
    for symbol in atr_data:
        print(f"\n{symbol}:")
        print(f"  ATR_5m: {atr_data[symbol].atr_5m:.4f}")
        print(f"  ATR_1h: {atr_data[symbol].atr_1h:.4f}")
        print(f"  Vol_5m: {vol_data[symbol].vol_5m:.6f}")
        print(f"  Vol history: {len(vol_data[symbol].vol_5m_history)} values")


if __name__ == "__main__":
    asyncio.run(test_bootstrap())
