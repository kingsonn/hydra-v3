"""
Trade Database Module

SQLite database for storing positions, trades, and account state.
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

# Fee percentage (0.08% round trip)
FEE_PCT = 0.0008

# Default database path
DEFAULT_DB_PATH = Path("data/trades.db")


@dataclass
class TrancheRecord:
    """Database record for a single tranche"""
    id: int = 0
    order_id: str = ""
    symbol: str = ""
    side: str = ""  # LONG or SHORT
    tranche: str = ""  # A or B
    mode: str = "test"  # test or live
    
    # Entry details
    entry_price: float = 0.0
    size: float = 0.0
    notional: float = 0.0
    
    # Stop loss and take profit
    stop_loss: float = 0.0
    take_profit: float = 0.0
    breakeven: float = 0.0
    
    # For tranche B specifics
    tp_partial: float = 0.0  # 2R partial TP
    tp_runner: float = 0.0   # 3R runner TP
    partial_size: float = 0.0  # Size to close at partial TP
    
    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    r_multiple: float = 0.0
    risk_amount: float = 0.0
    
    # Status flags
    status: str = "open"  # open, closed, partial
    sl_hit: bool = False
    tp_hit: bool = False
    tp_partial_hit: bool = False
    tp_runner_hit: bool = False
    sl_moved_be: bool = False   # SL moved to breakeven at 0.5R
    sl_moved_1r: bool = False   # SL moved to 1R at 2R partial
    
    # Tracking
    signal_name: str = ""
    created_at: str = ""
    closed_at: str = ""
    close_price: float = 0.0
    close_reason: str = ""


@dataclass
class AccountState:
    """Account state record"""
    id: int = 1
    initial_equity: float = 1000.0
    current_equity: float = 1000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 1000.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_r: float = 0.0
    updated_at: str = ""


class TradeDatabase:
    """
    SQLite database for trade management
    
    Stores:
    - Position tranches (A and B separately)
    - Account state (equity, margin, PnL)
    - Trade history
    """
    
    def __init__(self, db_path: Path = DEFAULT_DB_PATH, reset: bool = False):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        if reset:
            self._drop_tables()
        
        self._create_tables()
        self._init_account()
        
        logger.info("trade_database_initialized", path=str(db_path), reset=reset)
    
    def _drop_tables(self):
        """Drop all tables for fresh start"""
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS tranches")
        cursor.execute("DROP TABLE IF EXISTS account")
        self.conn.commit()
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Tranches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tranches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                tranche TEXT NOT NULL,
                mode TEXT DEFAULT 'test',
                
                entry_price REAL NOT NULL,
                size REAL NOT NULL,
                notional REAL NOT NULL,
                
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                breakeven REAL NOT NULL,
                
                tp_partial REAL DEFAULT 0,
                tp_runner REAL DEFAULT 0,
                partial_size REAL DEFAULT 0,
                
                current_price REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                realized_pnl REAL DEFAULT 0,
                r_multiple REAL DEFAULT 0,
                risk_amount REAL NOT NULL,
                
                status TEXT DEFAULT 'open',
                sl_hit INTEGER DEFAULT 0,
                tp_hit INTEGER DEFAULT 0,
                tp_partial_hit INTEGER DEFAULT 0,
                tp_runner_hit INTEGER DEFAULT 0,
                sl_moved_be INTEGER DEFAULT 0,
                sl_moved_1r INTEGER DEFAULT 0,
                
                signal_name TEXT,
                created_at TEXT NOT NULL,
                closed_at TEXT,
                close_price REAL DEFAULT 0,
                close_reason TEXT
            )
        """)
        
        # Account state table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY,
                initial_equity REAL NOT NULL,
                current_equity REAL NOT NULL,
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                margin_used REAL DEFAULT 0,
                margin_available REAL NOT NULL,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                total_r REAL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
        """)
        
        self.conn.commit()
    
    def _init_account(self):
        """Initialize account with starting equity"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM account")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO account (id, initial_equity, current_equity, margin_available, updated_at)
                VALUES (1, 1000.0, 1000.0, 1000.0, ?)
            """, (datetime.now().isoformat(),))
            self.conn.commit()
    
    # ========== TRANCHE OPERATIONS ==========
    
    def insert_tranche(self, tranche: TrancheRecord) -> int:
        """Insert a new tranche record"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO tranches (
                order_id, symbol, side, tranche, mode,
                entry_price, size, notional,
                stop_loss, take_profit, breakeven,
                tp_partial, tp_runner, partial_size,
                current_price, unrealized_pnl, realized_pnl, r_multiple, risk_amount,
                status, sl_hit, tp_hit, tp_partial_hit, tp_runner_hit,
                signal_name, created_at, closed_at, close_price, close_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tranche.order_id, tranche.symbol, tranche.side, tranche.tranche, tranche.mode,
            tranche.entry_price, tranche.size, tranche.notional,
            tranche.stop_loss, tranche.take_profit, tranche.breakeven,
            tranche.tp_partial, tranche.tp_runner, tranche.partial_size,
            tranche.current_price, tranche.unrealized_pnl, tranche.realized_pnl, tranche.r_multiple, tranche.risk_amount,
            tranche.status, int(tranche.sl_hit), int(tranche.tp_hit), int(tranche.tp_partial_hit), int(tranche.tp_runner_hit),
            tranche.signal_name, tranche.created_at, tranche.closed_at, tranche.close_price, tranche.close_reason,
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def update_tranche(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """Update tranche by order_id"""
        if not updates:
            return False
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [order_id]
        
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE tranches SET {set_clause} WHERE order_id = ?", values)
        self.conn.commit()
        return cursor.rowcount > 0
    
    def get_tranche(self, order_id: str) -> Optional[TrancheRecord]:
        """Get tranche by order_id"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tranches WHERE order_id = ?", (order_id,))
        row = cursor.fetchone()
        if row:
            return self._row_to_tranche(row)
        return None
    
    def get_open_tranches(self) -> List[TrancheRecord]:
        """Get all open tranches"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tranches WHERE status = 'open' OR status = 'partial'")
        return [self._row_to_tranche(row) for row in cursor.fetchall()]
    
    def get_tranches_by_symbol(self, symbol: str) -> List[TrancheRecord]:
        """Get all tranches for a symbol"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tranches WHERE symbol = ? AND (status = 'open' OR status = 'partial')", (symbol,))
        return [self._row_to_tranche(row) for row in cursor.fetchall()]
    
    def get_all_tranches(self) -> List[TrancheRecord]:
        """Get all tranches"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM tranches ORDER BY created_at DESC")
        return [self._row_to_tranche(row) for row in cursor.fetchall()]
    
    def close_tranche(self, order_id: str, close_price: float, close_reason: str, realized_pnl: float, r_multiple: float) -> bool:
        """Close a tranche"""
        return self.update_tranche(order_id, {
            "status": "closed",
            "close_price": close_price,
            "close_reason": close_reason,
            "closed_at": datetime.now().isoformat(),
            "realized_pnl": realized_pnl,
            "r_multiple": r_multiple,
            "unrealized_pnl": 0.0,
        })
    
    def _row_to_tranche(self, row: sqlite3.Row) -> TrancheRecord:
        """Convert database row to TrancheRecord"""
        return TrancheRecord(
            id=row["id"],
            order_id=row["order_id"],
            symbol=row["symbol"],
            side=row["side"],
            tranche=row["tranche"],
            mode=row["mode"],
            entry_price=row["entry_price"],
            size=row["size"],
            notional=row["notional"],
            stop_loss=row["stop_loss"],
            take_profit=row["take_profit"],
            breakeven=row["breakeven"],
            tp_partial=row["tp_partial"],
            tp_runner=row["tp_runner"],
            partial_size=row["partial_size"],
            current_price=row["current_price"],
            unrealized_pnl=row["unrealized_pnl"],
            realized_pnl=row["realized_pnl"],
            r_multiple=row["r_multiple"],
            risk_amount=row["risk_amount"],
            status=row["status"],
            sl_hit=bool(row["sl_hit"]),
            tp_hit=bool(row["tp_hit"]),
            tp_partial_hit=bool(row["tp_partial_hit"]),
            tp_runner_hit=bool(row["tp_runner_hit"]),
            signal_name=row["signal_name"] or "",
            created_at=row["created_at"],
            closed_at=row["closed_at"] or "",
            close_price=row["close_price"],
            close_reason=row["close_reason"] or "",
        )
    
    # ========== ACCOUNT OPERATIONS ==========
    
    def get_account(self) -> AccountState:
        """Get current account state"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM account WHERE id = 1")
        row = cursor.fetchone()
        if row:
            return AccountState(
                id=row["id"],
                initial_equity=row["initial_equity"],
                current_equity=row["current_equity"],
                realized_pnl=row["realized_pnl"],
                unrealized_pnl=row["unrealized_pnl"],
                margin_used=row["margin_used"],
                margin_available=row["margin_available"],
                total_trades=row["total_trades"],
                winning_trades=row["winning_trades"],
                losing_trades=row["losing_trades"],
                total_r=row["total_r"],
                updated_at=row["updated_at"],
            )
        return AccountState()
    
    def update_account(self, updates: Dict[str, Any]) -> bool:
        """Update account state"""
        updates["updated_at"] = datetime.now().isoformat()
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values())
        
        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE account SET {set_clause} WHERE id = 1", values)
        self.conn.commit()
        return cursor.rowcount > 0
    
    def reset_account(self, initial_equity: float = 1000.0):
        """Reset account to initial state"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE account SET
                initial_equity = ?,
                current_equity = ?,
                realized_pnl = 0,
                unrealized_pnl = 0,
                margin_used = 0,
                margin_available = ?,
                total_trades = 0,
                winning_trades = 0,
                losing_trades = 0,
                total_r = 0,
                updated_at = ?
            WHERE id = 1
        """, (initial_equity, initial_equity, initial_equity, datetime.now().isoformat()))
        self.conn.commit()
    
    def clear_all_tranches(self):
        """Clear all tranches (for reset)"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM tranches")
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()
