"""
Bootstrap Module - Historical Data for Alpha System
====================================================

Fetches historical data on bot startup:
- OI: 24 hours of 5-min data (288 samples) from Binance fapi
- Funding: 7 days (21 samples) from Binance fapi  
- Price/Klines: 1h bars for EMA/trend calculations
- Liquidations: Historical from Coinalyze API

Data is fetched once on startup, then maintained with live updates.
"""
import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from aiohttp.resolver import AsyncResolver
import structlog

from config import settings

logger = structlog.get_logger(__name__)

# API endpoints
BINANCE_FAPI_BASE = "https://fapi.binance.com"
COINALYZE_BASE = "https://api.coinalyze.net/v1"
COINALYZE_API_KEY = "d02ff8e4-16e7-44b1-bcb8-ef663a8de294"

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY_S = 2.0
REQUEST_TIMEOUT_S = 30


@dataclass
class OIHistoryData:
    """Historical OI data (5-min intervals, 24 hours)"""
    symbol: str
    # Rolling deque of (timestamp_ms, oi_value) - 288 samples = 24h at 5min intervals
    history: deque = field(default_factory=lambda: deque(maxlen=288))
    last_update_ms: int = 0
    
    def get_oi_at_offset(self, minutes_ago: int) -> Optional[float]:
        """Get OI value from N minutes ago"""
        if not self.history:
            return None
        samples_back = minutes_ago // 5
        if samples_back >= len(self.history):
            return self.history[0][1] if self.history else None
        idx = len(self.history) - 1 - samples_back
        return self.history[idx][1] if idx >= 0 else None
    
    def get_oi_change(self, minutes: int) -> float:
        """Get OI percentage change over last N minutes"""
        if len(self.history) < 2:
            return 0.0
        current = self.history[-1][1]
        past = self.get_oi_at_offset(minutes)
        if past is None or past == 0:
            return 0.0
        return (current - past) / past


@dataclass
class FundingHistoryData:
    """Historical funding rate data (21 samples = 7 days)"""
    symbol: str
    # Rolling deque of (timestamp_ms, funding_rate) - 21 samples
    history: deque = field(default_factory=lambda: deque(maxlen=21))
    last_update_ms: int = 0
    
    def get_funding_z(self) -> float:
        """Calculate funding z-score from history"""
        if len(self.history) < 10:
            return 0.0
        rates = [r for _, r in self.history]
        import numpy as np
        mean = np.mean(rates)
        std = np.std(rates)
        if std < 1e-9:
            return 0.0
        current = rates[-1]
        return (current - mean) / std
    
    def get_current_rate(self) -> float:
        """Get current funding rate"""
        if not self.history:
            return 0.0
        return self.history[-1][1]
    
    def get_cumulative_24h(self) -> float:
        """Get cumulative funding over last 24h (3 funding periods)"""
        if len(self.history) < 3:
            return 0.0
        # Last 3 funding rates (each is 8h apart)
        rates = [r for _, r in list(self.history)[-3:]]
        return sum(rates)


@dataclass 
class PriceHistoryData:
    """Historical price/kline data for trend analysis"""
    symbol: str
    # 1-hour bars for EMA calculation (need 200+ for EMA200)
    bars_1h: deque = field(default_factory=lambda: deque(maxlen=250))
    # 4-hour high/low tracking
    high_4h: float = 0.0
    low_4h: float = float('inf')
    high_24h: float = 0.0
    low_24h: float = float('inf')
    last_update_ms: int = 0
    
    def add_bar(self, timestamp_ms: int, open_: float, high: float, low: float, close: float):
        """Add a new 1h bar"""
        self.bars_1h.append({
            'timestamp_ms': timestamp_ms,
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
        })
        self._update_ranges()
    
    def _update_ranges(self):
        """Update 4h and 24h high/low from bars"""
        if not self.bars_1h:
            return
        
        now_ms = int(time.time() * 1000)
        cutoff_4h = now_ms - 4 * 3600 * 1000
        cutoff_24h = now_ms - 24 * 3600 * 1000
        
        self.high_4h = 0.0
        self.low_4h = float('inf')
        self.high_24h = 0.0
        self.low_24h = float('inf')
        
        for bar in self.bars_1h:
            if bar['timestamp_ms'] >= cutoff_4h:
                self.high_4h = max(self.high_4h, bar['high'])
                self.low_4h = min(self.low_4h, bar['low'])
            if bar['timestamp_ms'] >= cutoff_24h:
                self.high_24h = max(self.high_24h, bar['high'])
                self.low_24h = min(self.low_24h, bar['low'])
        
        # Handle case where no bars in window
        if self.low_4h == float('inf'):
            self.low_4h = 0.0
        if self.low_24h == float('inf'):
            self.low_24h = 0.0
    
    def get_closes(self) -> List[float]:
        """Get list of close prices"""
        return [bar['close'] for bar in self.bars_1h]
    
    def get_price_change(self, hours: int, current_price: float = 0.0) -> float:
        """Get price change over N hours
        
        Args:
            hours: Number of hours to look back
            current_price: Optional live price. If provided, uses this instead of last bar close
        """
        if len(self.bars_1h) < hours:
            return 0.0
        
        # Use live price if provided, otherwise use last bar close
        current = current_price if current_price > 0 else self.bars_1h[-1]['close']
        
        # Get price from N hours ago
        past = self.bars_1h[-hours]['close'] if hours <= len(self.bars_1h) else self.bars_1h[0]['close']
        if past == 0:
            return 0.0
        return (current - past) / past


@dataclass
class LiquidationHistoryData:
    """Historical liquidation data from Coinalyze"""
    symbol: str
    # Rolling windows of (timestamp_ms, long_usd, short_usd)
    history_1h: deque = field(default_factory=lambda: deque(maxlen=60))  # 1-min intervals
    history_4h: deque = field(default_factory=lambda: deque(maxlen=240))
    history_8h: deque = field(default_factory=lambda: deque(maxlen=480))
    last_update_ms: int = 0
    
    def get_totals(self, window_minutes: int) -> Tuple[float, float]:
        """Get (long_usd_total, short_usd_total) for window"""
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - window_minutes * 60 * 1000
        
        long_total = 0.0
        short_total = 0.0
        
        # Use appropriate history based on window
        if window_minutes <= 60:
            history = self.history_1h
        elif window_minutes <= 240:
            history = self.history_4h
        else:
            history = self.history_8h
        
        for ts, long_usd, short_usd in history:
            if ts >= cutoff:
                long_total += long_usd
                short_total += short_usd
        
        return long_total, short_total
    
    def get_imbalance(self, window_minutes: int) -> float:
        """Get liquidation imbalance for window"""
        long_usd, short_usd = self.get_totals(window_minutes)
        total = long_usd + short_usd
        if total == 0:
            return 0.0
        return (long_usd - short_usd) / total


@dataclass
class ATRData:
    """ATR data with proper 1h bar calculation"""
    symbol: str
    # True range history for ATR calculation
    tr_history_short: deque = field(default_factory=lambda: deque(maxlen=5))  # 5 period (5h)
    tr_history_long: deque = field(default_factory=lambda: deque(maxlen=20))  # 20 period (20h)
    atr_short: float = 0.0  # 5-period ATR on 1h bars
    atr_long: float = 0.0   # 20-period ATR on 1h bars
    last_close: float = 0.0
    last_update_ms: int = 0
    
    def add_bar(self, high: float, low: float, close: float):
        """Add 1h bar and update ATR"""
        if self.last_close > 0:
            tr = max(
                high - low,
                abs(high - self.last_close),
                abs(low - self.last_close)
            )
            self.tr_history_short.append(tr)
            self.tr_history_long.append(tr)
            
            # Update ATRs
            if len(self.tr_history_short) >= 5:
                import numpy as np
                self.atr_short = float(np.mean(list(self.tr_history_short)))
            if len(self.tr_history_long) >= 20:
                import numpy as np
                self.atr_long = float(np.mean(list(self.tr_history_long)))
        
        self.last_close = close
        self.last_update_ms = int(time.time() * 1000)
    
    def get_vol_expansion_ratio(self) -> float:
        """Get volatility expansion ratio (short ATR / long ATR)"""
        if self.atr_long == 0:
            return 1.0
        return self.atr_short / self.atr_long


class AlphaDataBootstrap:
    """
    Bootstrap historical data for the alpha system.
    
    Fetches on startup:
    - OI: 24h of 5-min data (288 samples)
    - Funding: 7 days (21 samples)
    - Price: 250 1h bars for EMA/trend
    - Liquidations: 8h from Coinalyze
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        
        # Data stores per symbol
        self.oi_data: Dict[str, OIHistoryData] = {}
        self.funding_data: Dict[str, FundingHistoryData] = {}
        self.price_data: Dict[str, PriceHistoryData] = {}
        self.liq_data: Dict[str, LiquidationHistoryData] = {}
        self.atr_data: Dict[str, ATRData] = {}
        
        # Initialize stores
        for symbol in symbols:
            self.oi_data[symbol] = OIHistoryData(symbol=symbol)
            self.funding_data[symbol] = FundingHistoryData(symbol=symbol)
            self.price_data[symbol] = PriceHistoryData(symbol=symbol)
            self.liq_data[symbol] = LiquidationHistoryData(symbol=symbol)
            self.atr_data[symbol] = ATRData(symbol=symbol)
        
        self._bootstrapped = False
    
    async def bootstrap_all(self) -> bool:
        """Bootstrap all historical data for all symbols"""
        logger.info("alpha_bootstrap_starting", symbols=len(self.symbols))
        
        timeout = ClientTimeout(total=REQUEST_TIMEOUT_S, connect=10, sock_connect=10)
        
        # Create custom resolver with fallback DNS servers (Google DNS + Cloudflare)
        try:
            resolver = AsyncResolver(nameservers=['8.8.8.8', '8.8.4.4', '1.1.1.1'])
        except Exception:
            resolver = None
        
        # Create connector with DNS caching and custom settings
        connector = TCPConnector(
            ttl_dns_cache=300,  # Cache DNS for 5 minutes
            limit=100,
            limit_per_host=30,
            force_close=False,
            enable_cleanup_closed=True,
            family=0,  # Allow both IPv4 and IPv6
            resolver=resolver,
            use_dns_cache=True
        )
        
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            for symbol in self.symbols:
                try:
                    # Fetch OI history (24h, 5min intervals)
                    await self._fetch_oi_history(session, symbol)
                    await asyncio.sleep(0.2)  # Rate limit
                    
                    # Fetch funding history (7 days, 21 samples)
                    await self._fetch_funding_history(session, symbol)
                    await asyncio.sleep(0.2)
                    
                    # Fetch price history (1h bars, 250 samples)
                    await self._fetch_price_history(session, symbol)
                    await asyncio.sleep(0.2)
                    
                    # Fetch liquidation history from Coinalyze (8h)
                    await self._fetch_liq_history(session, symbol)
                    await asyncio.sleep(0.2)
                    
                    logger.info(
                        "symbol_bootstrapped",
                        symbol=symbol,
                        oi_samples=len(self.oi_data[symbol].history),
                        funding_samples=len(self.funding_data[symbol].history),
                        price_bars=len(self.price_data[symbol].bars_1h),
                        liq_samples=len(self.liq_data[symbol].history_1h),
                    )
                    
                except Exception as e:
                    logger.error("symbol_bootstrap_failed", symbol=symbol, error=str(e)[:100])
        
        self._bootstrapped = True
        logger.info("alpha_bootstrap_complete", symbols=len(self.symbols))
        return True
    
    async def _fetch_oi_history(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch 24h of OI at 5-min intervals from Binance"""
        url = f"{BINANCE_FAPI_BASE}/futures/data/openInterestHist"
        params = {
            "symbol": symbol,
            "period": "5m",
            "limit": 288,  # 24h of 5-min data
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data:
                            ts = int(item["timestamp"])
                            oi = float(item["sumOpenInterest"])
                            self.oi_data[symbol].history.append((ts, oi))
                        self.oi_data[symbol].last_update_ms = int(time.time() * 1000)
                        return
                    else:
                        logger.warning("oi_history_fetch_failed", symbol=symbol, status=resp.status)
            except Exception as e:
                logger.warning("oi_history_fetch_error", symbol=symbol, error=str(e)[:50], attempt=attempt+1)
            await asyncio.sleep(RETRY_DELAY_S)
    
    async def _fetch_funding_history(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch 7 days of funding rates (21 samples) from Binance"""
        url = f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate"
        params = {
            "symbol": symbol,
            "limit": 21,  # 7 days of 8h funding
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for item in data:
                            ts = int(item["fundingTime"])
                            rate = float(item["fundingRate"])
                            self.funding_data[symbol].history.append((ts, rate))
                        self.funding_data[symbol].last_update_ms = int(time.time() * 1000)
                        return
                    else:
                        logger.warning("funding_history_fetch_failed", symbol=symbol, status=resp.status)
            except Exception as e:
                logger.warning("funding_history_fetch_error", symbol=symbol, error=str(e)[:50], attempt=attempt+1)
            await asyncio.sleep(RETRY_DELAY_S)
    
    async def _fetch_price_history(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch 250 1h bars for EMA/trend calculation"""
        url = f"{BINANCE_FAPI_BASE}/fapi/v1/klines"
        params = {
            "symbol": symbol,
            "interval": "1h",
            "limit": 250,
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for k in data:
                            ts = int(k[0])
                            open_ = float(k[1])
                            high = float(k[2])
                            low = float(k[3])
                            close = float(k[4])
                            
                            self.price_data[symbol].add_bar(ts, open_, high, low, close)
                            self.atr_data[symbol].add_bar(high, low, close)
                        
                        self.price_data[symbol].last_update_ms = int(time.time() * 1000)
                        return
                    else:
                        logger.warning("price_history_fetch_failed", symbol=symbol, status=resp.status)
            except Exception as e:
                logger.warning("price_history_fetch_error", symbol=symbol, error=str(e)[:50], attempt=attempt+1)
            await asyncio.sleep(RETRY_DELAY_S)
    
    async def _fetch_liq_history(self, session: aiohttp.ClientSession, symbol: str):
        """Fetch 8h of liquidation history from Coinalyze"""
        # Convert symbol format for Coinalyze (BTCUSDT -> BTCUSDT_PERP.A)
        coinalyze_symbol = f"{symbol}_PERP.A"
        
        now = int(time.time())
        from_ts = now - 8 * 3600  # 8 hours ago
        
        url = f"{COINALYZE_BASE}/liquidation-history"
        params = {
            "symbols": coinalyze_symbol,
            "interval": "1min",
            "from": from_ts,
            "to": now,
            "convert_to_usd": "true",
        }
        headers = {
            "api_key": COINALYZE_API_KEY,
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data and len(data) > 0:
                            history = data[0].get("history", [])
                            for item in history:
                                ts = item["t"] * 1000  # Convert to ms
                                long_usd = float(item.get("l", 0))
                                short_usd = float(item.get("s", 0))
                                
                                self.liq_data[symbol].history_1h.append((ts, long_usd, short_usd))
                                self.liq_data[symbol].history_4h.append((ts, long_usd, short_usd))
                                self.liq_data[symbol].history_8h.append((ts, long_usd, short_usd))
                            
                            self.liq_data[symbol].last_update_ms = int(time.time() * 1000)
                        return
                    elif resp.status == 401:
                        logger.warning("coinalyze_auth_failed", symbol=symbol)
                        return  # Don't retry auth failures
                    else:
                        logger.warning("liq_history_fetch_failed", symbol=symbol, status=resp.status)
            except Exception as e:
                logger.warning("liq_history_fetch_error", symbol=symbol, error=str(e)[:50], attempt=attempt+1)
            await asyncio.sleep(RETRY_DELAY_S)
    
    # ========== LIVE UPDATE METHODS ==========
    
    def update_oi(self, symbol: str, timestamp_ms: int, oi_value: float):
        """Update OI with live data (called every 5 min)"""
        if symbol not in self.oi_data:
            return
        
        # Check if 5 minutes have passed since last update
        last_ts = self.oi_data[symbol].history[-1][0] if self.oi_data[symbol].history else 0
        if timestamp_ms - last_ts >= 5 * 60 * 1000:  # 5 minutes
            self.oi_data[symbol].history.append((timestamp_ms, oi_value))
            self.oi_data[symbol].last_update_ms = timestamp_ms
    
    def update_funding(self, symbol: str, timestamp_ms: int, funding_rate: float):
        """
        Update funding with live data.
        Only update at 00:05, 08:05, 16:05 UTC (5 min after funding payment)
        """
        if symbol not in self.funding_data:
            return
        
        from datetime import datetime, timezone
        now = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
        
        # Check if it's 5 minutes after funding time (00:05, 08:05, 16:05)
        is_funding_update_time = (
            now.minute == 5 and 
            now.hour in [0, 8, 16]
        )
        
        if is_funding_update_time:
            # Check we haven't already added this funding
            last_ts = self.funding_data[symbol].history[-1][0] if self.funding_data[symbol].history else 0
            if timestamp_ms - last_ts >= 7 * 3600 * 1000:  # At least 7 hours since last
                self.funding_data[symbol].history.append((timestamp_ms, funding_rate))
                self.funding_data[symbol].last_update_ms = timestamp_ms
    
    def update_price_bar(self, symbol: str, timestamp_ms: int, open_: float, high: float, low: float, close: float):
        """Update with new 1h bar"""
        if symbol not in self.price_data:
            return
        
        self.price_data[symbol].add_bar(timestamp_ms, open_, high, low, close)
        self.atr_data[symbol].add_bar(high, low, close)
    
    def update_liquidation(self, symbol: str, timestamp_ms: int, long_usd: float, short_usd: float):
        """Update with live liquidation data"""
        if symbol not in self.liq_data:
            return
        
        self.liq_data[symbol].history_1h.append((timestamp_ms, long_usd, short_usd))
        self.liq_data[symbol].history_4h.append((timestamp_ms, long_usd, short_usd))
        self.liq_data[symbol].history_8h.append((timestamp_ms, long_usd, short_usd))
        self.liq_data[symbol].last_update_ms = timestamp_ms
    
    # ========== GETTERS ==========
    
    def get_funding_z(self, symbol: str) -> float:
        """Get current funding z-score"""
        if symbol not in self.funding_data:
            return 0.0
        return self.funding_data[symbol].get_funding_z()
    
    def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate"""
        if symbol not in self.funding_data:
            return 0.0
        return self.funding_data[symbol].get_current_rate()
    
    def get_cumulative_funding_24h(self, symbol: str) -> float:
        """Get cumulative funding over last 24h (3 periods)"""
        if symbol not in self.funding_data:
            return 0.0
        return self.funding_data[symbol].get_cumulative_24h()
    
    def get_oi_change(self, symbol: str, minutes: int) -> float:
        """Get OI change over N minutes"""
        if symbol not in self.oi_data:
            return 0.0
        return self.oi_data[symbol].get_oi_change(minutes)
    
    def get_price_change(self, symbol: str, hours: int, current_price: float = 0.0) -> float:
        """Get price change over N hours
        
        Args:
            symbol: Trading symbol
            hours: Number of hours to look back
            current_price: Optional live price for rolling calculation
        """
        if symbol not in self.price_data:
            return 0.0
        return self.price_data[symbol].get_price_change(hours, current_price)
    
    def get_atr_short(self, symbol: str) -> float:
        """Get 5-period ATR (1h bars)"""
        if symbol not in self.atr_data:
            return 0.0
        return self.atr_data[symbol].atr_short
    
    def get_atr_long(self, symbol: str) -> float:
        """Get 20-period ATR (1h bars)"""
        if symbol not in self.atr_data:
            return 0.0
        return self.atr_data[symbol].atr_long
    
    def get_vol_expansion_ratio(self, symbol: str) -> float:
        """Get volatility expansion ratio"""
        if symbol not in self.atr_data:
            return 1.0
        return self.atr_data[symbol].get_vol_expansion_ratio()
    
    def get_liq_totals(self, symbol: str, window_minutes: int) -> Tuple[float, float]:
        """Get (long_usd, short_usd) for window"""
        if symbol not in self.liq_data:
            return 0.0, 0.0
        return self.liq_data[symbol].get_totals(window_minutes)
    
    def get_bar_closes_1h(self, symbol: str) -> List[float]:
        """Get list of 1H bar close prices (up to 250 bars)"""
        if symbol not in self.price_data:
            return []
        return self.price_data[symbol].get_closes()
    
    def get_liq_imbalance(self, symbol: str, window_minutes: int) -> float:
        """Get liquidation imbalance for window"""
        if symbol not in self.liq_data:
            return 0.0
        return self.liq_data[symbol].get_imbalance(window_minutes)
    
    def get_price_closes(self, symbol: str) -> List[float]:
        """Get list of 1h close prices for EMA calculation"""
        if symbol not in self.price_data:
            return []
        return self.price_data[symbol].get_closes()
    
    def get_high_low_4h(self, symbol: str) -> Tuple[float, float]:
        """Get (high_4h, low_4h)"""
        if symbol not in self.price_data:
            return 0.0, 0.0
        return self.price_data[symbol].high_4h, self.price_data[symbol].low_4h
    
    def get_high_low_24h(self, symbol: str) -> Tuple[float, float]:
        """Get (high_24h, low_24h)"""
        if symbol not in self.price_data:
            return 0.0, 0.0
        return self.price_data[symbol].high_24h, self.price_data[symbol].low_24h


# Test function
async def test_bootstrap():
    """Test bootstrap for BTC and ETH"""
    bootstrap = AlphaDataBootstrap(["BTCUSDT", "ETHUSDT"])
    await bootstrap.bootstrap_all()
    
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        print(f"\n{symbol}:")
        print(f"  Funding Z: {bootstrap.get_funding_z(symbol):.2f}")
        print(f"  Funding Rate: {bootstrap.get_funding_rate(symbol):.6f}")
        print(f"  OI Change 1h: {bootstrap.get_oi_change(symbol, 60)*100:.2f}%")
        print(f"  OI Change 4h: {bootstrap.get_oi_change(symbol, 240)*100:.2f}%")
        print(f"  OI Change 24h: {bootstrap.get_oi_change(symbol, 1440)*100:.2f}%")
        print(f"  Price Change 4h: {bootstrap.get_price_change(symbol, 4)*100:.2f}%")
        print(f"  Price Change 24h: {bootstrap.get_price_change(symbol, 24)*100:.2f}%")
        print(f"  ATR Short (5h): {bootstrap.get_atr_short(symbol):.2f}")
        print(f"  ATR Long (20h): {bootstrap.get_atr_long(symbol):.2f}")
        print(f"  Vol Expansion: {bootstrap.get_vol_expansion_ratio(symbol):.2f}")
        long_1h, short_1h = bootstrap.get_liq_totals(symbol, 60)
        print(f"  Liq 1h: L=${long_1h:,.0f} S=${short_1h:,.0f}")
        print(f"  Liq Imbalance 1h: {bootstrap.get_liq_imbalance(symbol, 60):.2f}")


if __name__ == "__main__":
    asyncio.run(test_bootstrap())
