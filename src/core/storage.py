"""
Storage manager for Stage 1 data
Handles SQLite for metadata + Parquet for time-series data
"""
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, BigInteger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import text
import aiosqlite
import orjson

from config import settings
from .models import Trade, Bar, FundingRate, OpenInterest, Liquidation, VolumeProfile

Base = declarative_base()


class TradeRecord(Base):
    """SQLite table for recent trades (for quick access)"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True)
    timestamp_ms = Column(BigInteger, index=True)
    price = Column(Float)
    quantity = Column(Float)
    is_buyer_maker = Column(Boolean)
    trade_id = Column(BigInteger)


class BarRecord(Base):
    """SQLite table for bars"""
    __tablename__ = "bars"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True)
    interval = Column(String, index=True)
    timestamp_ms = Column(BigInteger, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    notional = Column(Float)
    trade_count = Column(Integer)
    buy_volume = Column(Float)
    sell_volume = Column(Float)
    buy_notional = Column(Float)
    sell_notional = Column(Float)
    buy_count = Column(Integer)
    sell_count = Column(Integer)


class StorageManager:
    """
    Manages all data storage for Stage 1
    - SQLite: metadata, recent data, quick lookups
    - Parquet: historical time-series (trades, bars)
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-directories
        self.trades_dir = self.data_dir / "trades"
        self.bars_dir = self.data_dir / "bars"
        self.derivatives_dir = self.data_dir / "derivatives"
        self.profiles_dir = self.data_dir / "profiles"
        
        for d in [self.trades_dir, self.bars_dir, self.derivatives_dir, self.profiles_dir]:
            d.mkdir(exist_ok=True)
        
        # SQLite for quick access
        self.db_path = self.data_dir / "hydra.db"
        self._engine = None
        self._async_engine = None
    
    async def initialize(self):
        """Initialize database tables"""
        self._async_engine = create_async_engine(
            f"sqlite+aiosqlite:///{self.db_path}",
            echo=False
        )
        
        # Create sync engine for table creation
        sync_engine = create_engine(f"sqlite:///{self.db_path}")
        Base.metadata.create_all(sync_engine)
        sync_engine.dispose()
    
    # ========== TRADES STORAGE ==========
    
    def _trades_parquet_path(self, symbol: str, date: datetime) -> Path:
        """Get parquet file path for trades"""
        return self.trades_dir / symbol / f"{date.strftime('%Y-%m-%d')}.parquet"
    
    async def store_trades(self, trades: List[Trade]) -> int:
        """
        Store trades to Parquet files (partitioned by symbol and date)
        Returns number of trades stored
        """
        if not trades:
            return 0
        
        # Group by symbol and date
        grouped: Dict[str, Dict[str, List[Trade]]] = {}
        for trade in trades:
            dt = datetime.utcfromtimestamp(trade.timestamp_ms / 1000)
            date_key = dt.strftime('%Y-%m-%d')
            
            if trade.symbol not in grouped:
                grouped[trade.symbol] = {}
            if date_key not in grouped[trade.symbol]:
                grouped[trade.symbol][date_key] = []
            grouped[trade.symbol][date_key].append(trade)
        
        # Write to parquet files
        count = 0
        for symbol, dates in grouped.items():
            symbol_dir = self.trades_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            for date_key, trade_list in dates.items():
                file_path = symbol_dir / f"{date_key}.parquet"
                
                # Convert to DataFrame
                df = pd.DataFrame([t.to_dict() for t in trade_list])
                
                # Append to existing file or create new
                if file_path.exists():
                    existing_df = pd.read_parquet(file_path)
                    df = pd.concat([existing_df, df], ignore_index=True)
                    df = df.drop_duplicates(subset=['trade_id'], keep='last')
                    df = df.sort_values('timestamp_ms')
                
                df.to_parquet(file_path, index=False, compression='snappy')
                count += len(trade_list)
        
        return count
    
    async def load_trades(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int
    ) -> List[Trade]:
        """Load trades from Parquet files for a time range"""
        start_dt = datetime.utcfromtimestamp(start_ms / 1000)
        end_dt = datetime.utcfromtimestamp(end_ms / 1000)
        
        trades = []
        symbol_dir = self.trades_dir / symbol
        
        if not symbol_dir.exists():
            return trades
        
        # Iterate through date range
        current = start_dt.date()
        end_date = end_dt.date()
        
        while current <= end_date:
            file_path = symbol_dir / f"{current.strftime('%Y-%m-%d')}.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                df = df[(df['timestamp_ms'] >= start_ms) & (df['timestamp_ms'] <= end_ms)]
                
                for _, row in df.iterrows():
                    trades.append(Trade(
                        symbol=row['symbol'],
                        timestamp_ms=row['timestamp_ms'],
                        price=row['price'],
                        quantity=row['quantity'],
                        is_buyer_maker=row['is_buyer_maker'],
                        trade_id=row['trade_id'],
                        first_trade_id=row.get('first_trade_id', row['trade_id']),
                        last_trade_id=row.get('last_trade_id', row['trade_id']),
                    ))
            
            current += timedelta(days=1)
        
        return sorted(trades, key=lambda t: t.timestamp_ms)
    
    # ========== BARS STORAGE ==========
    
    async def store_bars(self, bars: List[Bar]) -> int:
        """Store bars to Parquet files"""
        if not bars:
            return 0
        
        # Group by symbol and interval
        grouped: Dict[str, Dict[str, List[Bar]]] = {}
        for bar in bars:
            if bar.symbol not in grouped:
                grouped[bar.symbol] = {}
            if bar.interval not in grouped[bar.symbol]:
                grouped[bar.symbol][bar.interval] = []
            grouped[bar.symbol][bar.interval].append(bar)
        
        count = 0
        for symbol, intervals in grouped.items():
            for interval, bar_list in intervals.items():
                file_path = self.bars_dir / f"{symbol}_{interval}.parquet"
                
                df = pd.DataFrame([b.to_dict() for b in bar_list])
                
                if file_path.exists():
                    existing_df = pd.read_parquet(file_path)
                    df = pd.concat([existing_df, df], ignore_index=True)
                    df = df.drop_duplicates(subset=['timestamp_ms'], keep='last')
                    df = df.sort_values('timestamp_ms')
                
                df.to_parquet(file_path, index=False, compression='snappy')
                count += len(bar_list)
        
        return count
    
    async def load_bars(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int
    ) -> List[Bar]:
        """Load bars from Parquet file"""
        file_path = self.bars_dir / f"{symbol}_{interval}.parquet"
        
        if not file_path.exists():
            return []
        
        df = pd.read_parquet(file_path)
        df = df[(df['timestamp_ms'] >= start_ms) & (df['timestamp_ms'] <= end_ms)]
        
        bars = []
        for _, row in df.iterrows():
            bars.append(Bar(
                symbol=row['symbol'],
                interval=row['interval'],
                timestamp_ms=row['timestamp_ms'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                notional=row['notional'],
                trade_count=row['trade_count'],
                buy_volume=row['buy_volume'],
                sell_volume=row['sell_volume'],
                buy_notional=row['buy_notional'],
                sell_notional=row['sell_notional'],
                buy_count=row['buy_count'],
                sell_count=row['sell_count'],
            ))
        
        return bars
    
    # ========== DERIVATIVES STORAGE ==========
    
    async def store_funding_rates(self, rates: List[FundingRate]) -> int:
        """Store funding rates"""
        if not rates:
            return 0
        
        for rate in rates:
            file_path = self.derivatives_dir / f"{rate.symbol}_funding.parquet"
            df = pd.DataFrame([rate.to_dict()])
            
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=['timestamp_ms'], keep='last')
                df = df.sort_values('timestamp_ms')
            
            df.to_parquet(file_path, index=False, compression='snappy')
        
        return len(rates)
    
    async def store_open_interest(self, oi_list: List[OpenInterest]) -> int:
        """Store open interest data"""
        if not oi_list:
            return 0
        
        for oi in oi_list:
            file_path = self.derivatives_dir / f"{oi.symbol}_oi.parquet"
            df = pd.DataFrame([oi.to_dict()])
            
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=['timestamp_ms'], keep='last')
                df = df.sort_values('timestamp_ms')
            
            df.to_parquet(file_path, index=False, compression='snappy')
        
        return len(oi_list)
    
    async def store_liquidations(self, liqs: List[Liquidation]) -> int:
        """Store liquidation events"""
        if not liqs:
            return 0
        
        # Group by symbol
        grouped: Dict[str, List[Liquidation]] = {}
        for liq in liqs:
            if liq.symbol not in grouped:
                grouped[liq.symbol] = []
            grouped[liq.symbol].append(liq)
        
        for symbol, liq_list in grouped.items():
            file_path = self.derivatives_dir / f"{symbol}_liquidations.parquet"
            df = pd.DataFrame([l.to_dict() for l in liq_list])
            
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=['timestamp_ms', 'price', 'quantity'], keep='last')
                df = df.sort_values('timestamp_ms')
            
            df.to_parquet(file_path, index=False, compression='snappy')
        
        return len(liqs)
    
    # ========== VOLUME PROFILES ==========
    
    async def store_volume_profile(self, profile: VolumeProfile) -> None:
        """Store volume profile snapshot"""
        file_path = self.profiles_dir / f"{profile.symbol}_{profile.window}_profiles.parquet"
        df = pd.DataFrame([profile.to_dict()])
        
        if file_path.exists():
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            # Keep last 1000 profiles per symbol/window
            df = df.tail(1000)
        
        df.to_parquet(file_path, index=False, compression='snappy')
    
    # ========== CLEANUP ==========
    
    async def cleanup_old_data(self) -> Dict[str, int]:
        """Remove data older than retention settings"""
        now = datetime.utcnow()
        deleted = {"trades": 0, "bars": 0}
        
        # Cleanup old trade files
        cutoff_trades = now - timedelta(days=settings.RETENTION_TRADES_DAYS)
        for symbol_dir in self.trades_dir.iterdir():
            if symbol_dir.is_dir():
                for file in symbol_dir.glob("*.parquet"):
                    try:
                        file_date = datetime.strptime(file.stem, '%Y-%m-%d')
                        if file_date < cutoff_trades:
                            file.unlink()
                            deleted["trades"] += 1
                    except ValueError:
                        pass
        
        return deleted


# Singleton instance
_storage_manager: Optional[StorageManager] = None


async def get_storage() -> StorageManager:
    """Get or create storage manager singleton"""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
        await _storage_manager.initialize()
    return _storage_manager
