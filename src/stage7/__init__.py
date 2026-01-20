"""
Stage 7: Trade Management System

Manages trades, tracks positions, account balance, PnL, and handles
order execution through exchange APIs.

V3: Simplified trade manager for hybrid alpha (TP/SL/trail only)
"""

from src.stage7.exchange_api import DummyExchangeAPI
from src.stage7.database import TradeDatabase
from src.stage7.trade_manager import TradeManager
from src.stage7.trade_manager_v3 import TradeManagerV3, TradeRecordV3, AccountStateV3

__all__ = [
    "DummyExchangeAPI",
    "TradeDatabase",
    "TradeManager",
    "TradeManagerV3",
    "TradeRecordV3",
    "AccountStateV3",
]
