"""
Data models for Stage 1 ingestion
All models are designed for fast serialization and storage
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
import time


class Side(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(slots=True)
class Trade:
    """
    Single aggregated trade from Binance aggTrade stream
    This is the atomic unit of order flow data
    """
    symbol: str
    timestamp_ms: int          # Event timestamp in milliseconds
    price: float               # Trade price
    quantity: float            # Trade quantity
    is_buyer_maker: bool       # True = seller aggressor, False = buyer aggressor
    trade_id: int              # Aggregate trade ID
    first_trade_id: int        # First trade ID in aggregation
    last_trade_id: int         # Last trade ID in aggregation
    
    @property
    def side(self) -> Side:
        """Aggressor side: buyer aggressor = BUY, seller aggressor = SELL"""
        return Side.SELL if self.is_buyer_maker else Side.BUY
    
    @property
    def signed_quantity(self) -> float:
        """Positive for buyer aggressor, negative for seller aggressor"""
        return self.quantity if self.side == Side.BUY else -self.quantity
    
    @property
    def notional(self) -> float:
        """USD notional value"""
        return self.price * self.quantity
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "price": self.price,
            "quantity": self.quantity,
            "is_buyer_maker": self.is_buyer_maker,
            "trade_id": self.trade_id,
            "first_trade_id": self.first_trade_id,
            "last_trade_id": self.last_trade_id,
        }
    
    @classmethod
    def from_binance(cls, data: dict, symbol: str) -> "Trade":
        """Parse from Binance aggTrade WebSocket message"""
        return cls(
            symbol=symbol,
            timestamp_ms=data["T"],  # Trade time
            price=float(data["p"]),
            quantity=float(data["q"]),
            is_buyer_maker=data["m"],
            trade_id=data["a"],
            first_trade_id=data["f"],
            last_trade_id=data["l"],
        )


@dataclass(slots=True)
class OrderBookLevel:
    """Single price level in order book"""
    price: float
    quantity: float


@dataclass(slots=True)
class OrderBookSnapshot:
    """
    Order book snapshot (top N levels)
    Used for absorption analysis and liquidity detection
    """
    symbol: str
    timestamp_ms: int
    bids: List[OrderBookLevel]  # Best bid first (descending price)
    asks: List[OrderBookLevel]  # Best ask first (ascending price)
    last_update_id: int
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points"""
        if self.mid_price and self.spread:
            return (self.spread / self.mid_price) * 10000
        return None
    
    def bid_depth_usd(self, levels: int = 5) -> float:
        """Total bid liquidity in USD for top N levels"""
        return sum(lvl.price * lvl.quantity for lvl in self.bids[:levels])
    
    def ask_depth_usd(self, levels: int = 5) -> float:
        """Total ask liquidity in USD for top N levels"""
        return sum(lvl.price * lvl.quantity for lvl in self.asks[:levels])
    
    def depth_imbalance(self, levels: int = 5) -> float:
        """(bid_depth - ask_depth) / (bid_depth + ask_depth)"""
        bid = self.bid_depth_usd(levels)
        ask = self.ask_depth_usd(levels)
        total = bid + ask
        return (bid - ask) / total if total > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "bids": [[l.price, l.quantity] for l in self.bids],
            "asks": [[l.price, l.quantity] for l in self.asks],
            "last_update_id": self.last_update_id,
        }


@dataclass(slots=True)
class Bar:
    """
    OHLCV bar with order flow metrics
    Built from raw trades, NOT from exchange candles
    """
    symbol: str
    interval: str              # "250ms", "1s", "5m", "30m"
    timestamp_ms: int          # Bar open time
    open: float
    high: float
    low: float
    close: float
    volume: float              # Total volume
    notional: float            # Total USD volume
    trade_count: int
    buy_volume: float          # Aggressive buy volume
    sell_volume: float         # Aggressive sell volume
    buy_notional: float        # Aggressive buy USD
    sell_notional: float       # Aggressive sell USD
    buy_count: int             # Number of buy trades
    sell_count: int            # Number of sell trades
    
    @property
    def delta(self) -> float:
        """Volume delta (buy - sell)"""
        return self.buy_volume - self.sell_volume
    
    @property
    def delta_notional(self) -> float:
        """USD delta"""
        return self.buy_notional - self.sell_notional
    
    @property
    def delta_pct(self) -> float:
        """Delta as percentage of total volume"""
        return self.delta / self.volume if self.volume > 0 else 0.0
    
    @property
    def vwap(self) -> float:
        """Volume-weighted average price"""
        return self.notional / self.volume if self.volume > 0 else self.close
    
    @property
    def body(self) -> float:
        """Candle body (close - open)"""
        return self.close - self.open
    
    @property
    def range(self) -> float:
        """Candle range (high - low)"""
        return self.high - self.low
    
    @property
    def body_pct(self) -> float:
        """Body as percentage of range"""
        return abs(self.body) / self.range if self.range > 0 else 0.0
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "timestamp_ms": self.timestamp_ms,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "notional": self.notional,
            "trade_count": self.trade_count,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
            "buy_notional": self.buy_notional,
            "sell_notional": self.sell_notional,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
        }


@dataclass
class VolumeProfileBin:
    """Single price bin in volume profile"""
    price_low: float
    price_high: float
    volume: float
    buy_volume: float
    sell_volume: float
    trade_count: int
    
    @property
    def price_mid(self) -> float:
        return (self.price_low + self.price_high) / 2
    
    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume


@dataclass
class VolumeProfile:
    """
    Volume profile with structural levels
    Computed from trades over a rolling window
    """
    symbol: str
    window: str                # "5m", "30m"
    timestamp_ms: int          # Computation time
    start_ms: int              # Window start
    end_ms: int                # Window end
    bins: List[VolumeProfileBin]
    poc_price: float           # Point of Control
    vah_price: float           # Value Area High
    val_price: float           # Value Area Low
    lvn_prices: List[float]    # Low Volume Nodes
    total_volume: float
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "window": self.window,
            "timestamp_ms": self.timestamp_ms,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "poc_price": self.poc_price,
            "vah_price": self.vah_price,
            "val_price": self.val_price,
            "lvn_prices": self.lvn_prices,
            "total_volume": self.total_volume,
        }


@dataclass(slots=True)
class FundingRate:
    """Perpetual funding rate"""
    symbol: str
    timestamp_ms: int          # Funding time
    funding_rate: float        # Funding rate (positive = longs pay shorts)
    mark_price: float
    
    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate (3 fundings per day)"""
        return self.funding_rate * 3 * 365 * 100  # As percentage
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "funding_rate": self.funding_rate,
            "mark_price": self.mark_price,
        }


@dataclass(slots=True)
class OpenInterest:
    """Open interest data"""
    symbol: str
    timestamp_ms: int
    open_interest: float       # In contracts
    open_interest_value: float # In USD
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "open_interest": self.open_interest,
            "open_interest_value": self.open_interest_value,
        }


@dataclass(slots=True)
class Liquidation:
    """Single liquidation event"""
    symbol: str
    timestamp_ms: int
    side: Side                 # Side that was liquidated
    price: float
    quantity: float
    
    @property
    def notional(self) -> float:
        return self.price * self.quantity
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "side": self.side.value,
            "price": self.price,
            "quantity": self.quantity,
        }
    
    @classmethod
    def from_binance(cls, data: dict) -> "Liquidation":
        """Parse from Binance forceOrder"""
        order = data.get("o", data)
        return cls(
            symbol=order["s"],
            timestamp_ms=order["T"],
            side=Side.BUY if order["S"] == "BUY" else Side.SELL,
            price=float(order["p"]),
            quantity=float(order["q"]),
        )


@dataclass
class HealthMetrics:
    """System health metrics"""
    timestamp_ms: int
    websocket_lag_ms: Dict[str, int] = field(default_factory=dict)
    dropped_messages: Dict[str, int] = field(default_factory=dict)
    last_trade_time: Dict[str, int] = field(default_factory=dict)
    last_book_time: Dict[str, int] = field(default_factory=dict)
    connection_status: Dict[str, bool] = field(default_factory=dict)
    
    def is_healthy(self, max_lag_ms: int = 5000) -> bool:
        """Check if all connections are healthy"""
        now = int(time.time() * 1000)
        for symbol, lag in self.websocket_lag_ms.items():
            if lag > max_lag_ms:
                return False
        for symbol, status in self.connection_status.items():
            if not status:
                return False
        return True
