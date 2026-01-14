"""
Stage 7: Trade Management System

Manages trades, tracks positions, account balance, PnL, and handles
order execution through exchange APIs.
"""

from src.stage7.exchange_api import DummyExchangeAPI
from src.stage7.database import TradeDatabase
from src.stage7.trade_manager import TradeManager

__all__ = [
    "DummyExchangeAPI",
    "TradeDatabase",
    "TradeManager",
]
