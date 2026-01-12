"""
Hydra V3 Configuration
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    
    # Binance Futures endpoints
    BINANCE_WS_BASE: str = "wss://fstream.binance.com/ws"
    BINANCE_REST_BASE: str = "https://fapi.binance.com"
    
    # Trading pairs (perpetual futures)
    SYMBOLS: List[str] = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "BNBUSDT",
        "XRPUSDT",
        "DOGEUSDT",
        "ADAUSDT",
        "LTCUSDT",
    ]
    
    # Bar resolutions (milliseconds)
    BAR_INTERVALS_MS: dict = {
        "250ms": 250,
        "1s": 1000,
        "5m": 300_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
    }
    
    # Order book settings
    ORDERBOOK_DEPTH_LEVELS: int = 20  # Top 20 levels
    ORDERBOOK_SNAPSHOT_INTERVAL_S: int = 5  # Snapshot every 5 seconds
    
    # Volume profile settings
    VP_BIN_SIZE_TICKS: int = 10  # Price bins for volume profile
    VP_ROLLING_WINDOWS: List[str] = ["5m", "30m"]
    VP_UPDATE_INTERVAL_S: int = 30  # Recompute every 30 seconds
    
    # REST API polling intervals (seconds)
    FUNDING_POLL_INTERVAL_S: int = 60  # Every minute (funding updates every 8h)
    OI_POLL_INTERVAL_S: int = 60  # Every minute
    LIQUIDATION_POLL_INTERVAL_S: int = 5  # Every 5 seconds
    
    # Storage retention (days)
    RETENTION_TRADES_DAYS: int = 60
    RETENTION_BARS_250MS_DAYS: int = 14
    RETENTION_BARS_1S_DAYS: int = 60
    RETENTION_BARS_5M_DAYS: int = 180
    RETENTION_BARS_30M_DAYS: int = 365
    
    # Health monitoring
    MAX_WS_LAG_MS: int = 5000  # Alert if WebSocket lag > 5s
    HEARTBEAT_INTERVAL_S: int = 30
    
    # Database
    DB_PATH: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data" / "hydra.db")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
