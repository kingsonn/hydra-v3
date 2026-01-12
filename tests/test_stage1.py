"""
STAGE 1 TEST HARNESS
Easy function-by-function testing of all components

Run individual tests:
    python -m pytest tests/test_stage1.py::test_trade_model -v
    python -m pytest tests/test_stage1.py::test_bar_builder -v
    
Run all tests:
    python -m pytest tests/test_stage1.py -v

Or use the interactive test runner:
    python tests/test_stage1.py
"""
import pytest
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# A. TRADES DATA TESTS
# ============================================================

class TestTradeModel:
    """Test Trade model"""
    
    def test_trade_creation(self):
        """Test creating a Trade object"""
        from src.core.models import Trade, Side
        
        trade = Trade(
            symbol="BTCUSDT",
            timestamp_ms=1700000000000,
            price=50000.0,
            quantity=1.5,
            is_buyer_maker=False,  # Buyer aggressor
            trade_id=12345,
            first_trade_id=12345,
            last_trade_id=12345,
        )
        
        assert trade.symbol == "BTCUSDT"
        assert trade.price == 50000.0
        assert trade.quantity == 1.5
        assert trade.side == Side.BUY
        assert trade.signed_quantity == 1.5
        assert trade.notional == 75000.0
    
    def test_trade_seller_aggressor(self):
        """Test seller aggressor trade"""
        from src.core.models import Trade, Side
        
        trade = Trade(
            symbol="ETHUSDT",
            timestamp_ms=1700000000000,
            price=2000.0,
            quantity=10.0,
            is_buyer_maker=True,  # Seller aggressor
            trade_id=1,
            first_trade_id=1,
            last_trade_id=1,
        )
        
        assert trade.side == Side.SELL
        assert trade.signed_quantity == -10.0
    
    def test_trade_from_binance(self):
        """Test parsing from Binance WebSocket format"""
        from src.core.models import Trade
        
        binance_data = {
            "T": 1700000000123,
            "p": "50000.50",
            "q": "0.5",
            "m": False,
            "a": 999,
            "f": 998,
            "l": 999,
        }
        
        trade = Trade.from_binance(binance_data, "BTCUSDT")
        
        assert trade.timestamp_ms == 1700000000123
        assert trade.price == 50000.50
        assert trade.quantity == 0.5
        assert trade.trade_id == 999


# ============================================================
# B. ORDER BOOK TESTS
# ============================================================

class TestOrderBookModel:
    """Test OrderBook model"""
    
    def test_orderbook_creation(self):
        """Test creating OrderBookSnapshot"""
        from src.core.models import OrderBookSnapshot, OrderBookLevel
        
        bids = [
            OrderBookLevel(price=50000.0, quantity=10.0),
            OrderBookLevel(price=49999.0, quantity=5.0),
        ]
        asks = [
            OrderBookLevel(price=50001.0, quantity=8.0),
            OrderBookLevel(price=50002.0, quantity=3.0),
        ]
        
        book = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp_ms=1700000000000,
            bids=bids,
            asks=asks,
            last_update_id=12345,
        )
        
        assert book.best_bid == 50000.0
        assert book.best_ask == 50001.0
        assert book.mid_price == 50000.5
        assert book.spread == 1.0
    
    def test_orderbook_metrics(self):
        """Test order book derived metrics"""
        from src.core.models import OrderBookSnapshot, OrderBookLevel
        
        bids = [
            OrderBookLevel(price=50000.0, quantity=10.0),
            OrderBookLevel(price=49990.0, quantity=20.0),
        ]
        asks = [
            OrderBookLevel(price=50010.0, quantity=5.0),
            OrderBookLevel(price=50020.0, quantity=15.0),
        ]
        
        book = OrderBookSnapshot(
            symbol="BTCUSDT",
            timestamp_ms=1700000000000,
            bids=bids,
            asks=asks,
            last_update_id=1,
        )
        
        # Depth calculations
        bid_depth = book.bid_depth_usd(2)
        ask_depth = book.ask_depth_usd(2)
        
        assert bid_depth == (50000.0 * 10.0) + (49990.0 * 20.0)
        assert ask_depth == (50010.0 * 5.0) + (50020.0 * 15.0)
        
        # Imbalance
        imbalance = book.depth_imbalance(2)
        expected = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        assert abs(imbalance - expected) < 0.0001


# ============================================================
# C. BAR AGGREGATION TESTS
# ============================================================

class TestBarBuilder:
    """Test BarBuilder"""
    
    def test_bar_builder_basic(self):
        """Test basic bar building"""
        from src.processors.bars import BarBuilder
        from src.core.models import Trade
        
        builder = BarBuilder(
            symbol="BTCUSDT",
            interval="1s",
            interval_ms=1000,
            start_ms=1700000000000,
        )
        
        # Add trades
        trades = [
            Trade("BTCUSDT", 1700000000100, 100.0, 1.0, False, 1, 1, 1),
            Trade("BTCUSDT", 1700000000200, 102.0, 2.0, True, 2, 2, 2),
            Trade("BTCUSDT", 1700000000300, 101.0, 1.5, False, 3, 3, 3),
        ]
        
        for t in trades:
            builder.add_trade(t)
        
        bar = builder.to_bar()
        
        assert bar.open == 100.0
        assert bar.high == 102.0
        assert bar.low == 100.0
        assert bar.close == 101.0
        assert bar.volume == 4.5
        assert bar.trade_count == 3
        assert bar.buy_volume == 2.5  # First and third trades
        assert bar.sell_volume == 2.0  # Second trade
    
    def test_bar_delta(self):
        """Test bar delta calculation"""
        from src.processors.bars import BarBuilder
        from src.core.models import Trade
        
        builder = BarBuilder("TEST", "1s", 1000, 0)
        
        # 3 buys, 1 sell
        builder.add_trade(Trade("TEST", 100, 100.0, 1.0, False, 1, 1, 1))  # Buy
        builder.add_trade(Trade("TEST", 200, 100.0, 1.0, False, 2, 2, 2))  # Buy
        builder.add_trade(Trade("TEST", 300, 100.0, 1.0, True, 3, 3, 3))   # Sell
        builder.add_trade(Trade("TEST", 400, 100.0, 1.0, False, 4, 4, 4))  # Buy
        
        bar = builder.to_bar()
        
        assert bar.buy_volume == 3.0
        assert bar.sell_volume == 1.0
        assert bar.delta == 2.0
        assert bar.delta_pct == 0.5


class TestBarAggregator:
    """Test BarAggregator"""
    
    def test_bar_completion(self):
        """Test bars complete when time passes"""
        from src.processors.bars import BarAggregator
        from src.core.models import Trade
        
        completed = []
        
        def on_bar(bar):
            completed.append(bar)
        
        agg = BarAggregator(
            symbols=["TEST"],
            intervals={"1s": 1000},
            on_bar=on_bar,
        )
        
        # Trades in first bar (0-999ms)
        agg.process_trade(Trade("TEST", 100, 100.0, 1.0, False, 1, 1, 1))
        agg.process_trade(Trade("TEST", 500, 101.0, 1.0, False, 2, 2, 2))
        
        assert len(completed) == 0  # Bar not complete yet
        
        # Trade in second bar (1000-1999ms) completes first bar
        agg.process_trade(Trade("TEST", 1100, 102.0, 1.0, False, 3, 3, 3))
        
        assert len(completed) == 1
        assert completed[0].timestamp_ms == 0
        assert completed[0].open == 100.0
        assert completed[0].close == 101.0
    
    def test_multi_interval(self):
        """Test multiple interval aggregation"""
        from src.processors.bars import BarAggregator
        from src.core.models import Trade
        
        completed = []
        
        agg = BarAggregator(
            symbols=["TEST"],
            intervals={"250ms": 250, "1s": 1000},
            on_bar=lambda b: completed.append(b),
        )
        
        # Generate trades over 1.5 seconds
        for i in range(15):
            agg.process_trade(Trade("TEST", i * 100, 100.0, 1.0, False, i, i, i))
        
        # Count completed bars by interval
        bars_250ms = [b for b in completed if b.interval == "250ms"]
        bars_1s = [b for b in completed if b.interval == "1s"]
        
        assert len(bars_250ms) >= 4  # At least 4 x 250ms bars in 1s
        assert len(bars_1s) >= 1


# ============================================================
# D. VOLUME PROFILE TESTS
# ============================================================

class TestVolumeProfile:
    """Test VolumeProfileCalculator"""
    
    def test_poc_detection(self):
        """Test POC (Point of Control) detection"""
        from src.processors.volume_profile import VolumeProfileCalculator
        from src.core.models import Trade
        
        calc = VolumeProfileCalculator(tick_size=1.0)
        
        # Create trades concentrated around 100
        trades = []
        base_time = 1700000000000
        
        # High volume at 100
        for i in range(50):
            trades.append(Trade("TEST", base_time + i, 100.0, 1.0, False, i, i, i))
        
        # Lower volume at 101
        for i in range(10):
            trades.append(Trade("TEST", base_time + 50 + i, 101.0, 1.0, False, 50+i, 50+i, 50+i))
        
        profile = calc.compute_profile(trades, "TEST", "5m")
        
        assert profile is not None
        assert profile.poc_price == 100.5  # Mid of 100-101 bin
        assert profile.total_volume == 60.0
    
    def test_value_area(self):
        """Test VAH/VAL calculation"""
        from src.processors.volume_profile import VolumeProfileCalculator
        from src.core.models import Trade
        
        calc = VolumeProfileCalculator(tick_size=1.0, value_area_pct=0.70)
        
        # Create bell-curve like distribution
        trades = []
        base_time = 1700000000000
        trade_id = 0
        
        # Most volume in middle (100)
        for _ in range(100):
            trades.append(Trade("TEST", base_time, 100.0, 1.0, False, trade_id, trade_id, trade_id))
            trade_id += 1
        
        # Some volume at edges
        for _ in range(20):
            trades.append(Trade("TEST", base_time, 99.0, 1.0, False, trade_id, trade_id, trade_id))
            trade_id += 1
        for _ in range(20):
            trades.append(Trade("TEST", base_time, 101.0, 1.0, False, trade_id, trade_id, trade_id))
            trade_id += 1
        
        profile = calc.compute_profile(trades, "TEST", "5m")
        
        assert profile.vah_price >= profile.poc_price
        assert profile.val_price <= profile.poc_price


# ============================================================
# E. DERIVATIVES TESTS
# ============================================================

class TestDerivativesModels:
    """Test derivatives data models"""
    
    def test_funding_rate(self):
        """Test FundingRate model"""
        from src.core.models import FundingRate
        
        rate = FundingRate(
            symbol="BTCUSDT",
            timestamp_ms=1700000000000,
            funding_rate=0.0001,  # 0.01%
            mark_price=50000.0,
        )
        
        # Annualized: 0.01% * 3 * 365 = 10.95%
        assert abs(rate.annualized_rate - 10.95) < 0.01
    
    def test_liquidation(self):
        """Test Liquidation model"""
        from src.core.models import Liquidation, Side
        
        liq = Liquidation(
            symbol="BTCUSDT",
            timestamp_ms=1700000000000,
            side=Side.BUY,
            price=50000.0,
            quantity=1.0,
        )
        
        assert liq.notional == 50000.0


# ============================================================
# F. HEALTH MONITORING TESTS
# ============================================================

class TestHealthMonitor:
    """Test HealthMonitor"""
    
    def test_trade_lag_tracking(self):
        """Test trade latency tracking"""
        import time
        from src.health.monitor import HealthMonitor
        
        monitor = HealthMonitor(symbols=["TEST"], max_lag_ms=5000)
        
        now_ms = int(time.time() * 1000)
        
        # Record trade with 100ms lag
        monitor.record_trade("TEST", now_ms - 100)
        
        health = monitor.get_symbol_health("TEST")
        assert health is not None
        assert health.trade_lag_ms >= 100
        assert health.trade_lag_ms < 200  # Should be close to 100
    
    def test_unhealthy_detection(self):
        """Test unhealthy state detection"""
        import time
        from src.health.monitor import HealthMonitor, HealthStatus
        
        monitor = HealthMonitor(symbols=["TEST"], max_lag_ms=1000)
        
        now_ms = int(time.time() * 1000)
        
        # Record trade with 2000ms lag (>1000ms threshold)
        monitor.record_trade("TEST", now_ms - 2000)
        
        health = monitor.get_symbol_health("TEST")
        status = health.get_status(max_lag_ms=1000)
        
        assert status == HealthStatus.UNHEALTHY


# ============================================================
# INTEGRATION TESTS
# ============================================================

@pytest.mark.asyncio
async def test_trades_collector_live():
    """Integration test: live trades collection (requires network)"""
    from src.collectors.trades import TradesCollector
    
    trades = []
    
    collector = TradesCollector(
        symbols=["BTCUSDT"],
        on_trade=lambda t: trades.append(t),
    )
    
    task = asyncio.create_task(collector.start())
    await asyncio.sleep(5)  # Collect for 5 seconds
    await collector.stop()
    task.cancel()
    
    assert len(trades) > 0, "Should receive at least some trades"
    assert trades[0].symbol == "BTCUSDT"
    
    health = collector.get_health_metrics()
    assert health["message_count"] > 0


@pytest.mark.asyncio
async def test_orderbook_collector_live():
    """Integration test: live order book collection"""
    from src.collectors.orderbook import OrderBookCollector
    
    snapshots = []
    
    collector = OrderBookCollector(
        symbols=["BTCUSDT"],
        on_snapshot=lambda s: snapshots.append(s),
        snapshot_interval_s=1,
    )
    
    task = asyncio.create_task(collector.start())
    await asyncio.sleep(5)
    await collector.stop()
    task.cancel()
    
    book = collector.get_book("BTCUSDT")
    assert book is not None
    assert book.best_bid is not None
    assert book.best_ask is not None
    assert book.best_bid < book.best_ask


@pytest.mark.asyncio
async def test_full_pipeline():
    """Integration test: full Stage 1 pipeline"""
    from src.collectors.trades import TradesCollector
    from src.processors.bars import BarAggregator
    
    bars_completed = []
    
    aggregator = BarAggregator(
        symbols=["BTCUSDT"],
        intervals={"1s": 1000},
        on_bar=lambda b: bars_completed.append(b),
    )
    
    collector = TradesCollector(
        symbols=["BTCUSDT"],
        on_trade=lambda t: aggregator.process_trade(t),
    )
    
    task = asyncio.create_task(collector.start())
    await asyncio.sleep(10)  # Run for 10 seconds
    await collector.stop()
    task.cancel()
    
    # Flush remaining
    remaining = aggregator.flush_all()
    
    total_bars = len(bars_completed) + len(remaining)
    assert total_bars >= 5, f"Should have at least 5 1s bars in 10s, got {total_bars}"


# ============================================================
# INTERACTIVE TEST RUNNER
# ============================================================

def run_interactive_tests():
    """Interactive test menu"""
    import asyncio
    
    tests = {
        "1": ("Trade Model", lambda: test_trade_model_interactive()),
        "2": ("Order Book Model", lambda: test_orderbook_model_interactive()),
        "3": ("Bar Builder", lambda: test_bar_builder_interactive()),
        "4": ("Volume Profile", lambda: test_volume_profile_interactive()),
        "5": ("Live Trades (5s)", lambda: asyncio.run(test_trades_live_interactive())),
        "6": ("Live OrderBook (5s)", lambda: asyncio.run(test_orderbook_live_interactive())),
        "7": ("Live Bars (10s)", lambda: asyncio.run(test_bars_live_interactive())),
        "8": ("Live Derivatives (10s)", lambda: asyncio.run(test_derivatives_live_interactive())),
        "9": ("Full Stage 1 (30s)", lambda: asyncio.run(test_stage1_interactive())),
        "0": ("Exit", None),
    }
    
    while True:
        print("\n" + "=" * 50)
        print("HYDRA V3 - STAGE 1 TEST HARNESS")
        print("=" * 50)
        for key, (name, _) in tests.items():
            print(f"  [{key}] {name}")
        print("=" * 50)
        
        choice = input("\nSelect test: ").strip()
        
        if choice == "0":
            break
        
        if choice in tests and tests[choice][1]:
            print(f"\n>>> Running: {tests[choice][0]}\n")
            try:
                tests[choice][1]()
            except Exception as e:
                print(f"\n❌ Error: {e}")
            print("\n>>> Test complete")


def test_trade_model_interactive():
    from src.core.models import Trade, Side
    
    # Test buyer aggressor
    trade = Trade("BTCUSDT", 1700000000000, 50000.0, 1.5, False, 1, 1, 1)
    print(f"Trade: {trade.symbol}")
    print(f"  Price: ${trade.price:,.2f}")
    print(f"  Quantity: {trade.quantity}")
    print(f"  Side: {trade.side.value}")
    print(f"  Signed Qty: {trade.signed_quantity}")
    print(f"  Notional: ${trade.notional:,.2f}")
    print("✓ Trade model working")


def test_orderbook_model_interactive():
    from src.core.models import OrderBookSnapshot, OrderBookLevel
    
    bids = [OrderBookLevel(50000.0, 10.0), OrderBookLevel(49999.0, 5.0)]
    asks = [OrderBookLevel(50001.0, 8.0), OrderBookLevel(50002.0, 3.0)]
    
    book = OrderBookSnapshot("BTCUSDT", 1700000000000, bids, asks, 1)
    
    print(f"OrderBook: {book.symbol}")
    print(f"  Best Bid: ${book.best_bid:,.2f}")
    print(f"  Best Ask: ${book.best_ask:,.2f}")
    print(f"  Mid: ${book.mid_price:,.2f}")
    print(f"  Spread: ${book.spread:.2f} ({book.spread_bps:.2f} bps)")
    print(f"  Bid Depth (5): ${book.bid_depth_usd(5):,.0f}")
    print(f"  Ask Depth (5): ${book.ask_depth_usd(5):,.0f}")
    print(f"  Imbalance: {book.depth_imbalance(5):.2%}")
    print("✓ OrderBook model working")


def test_bar_builder_interactive():
    from src.processors.bars import test_bar_builder
    test_bar_builder()
    print("✓ Bar builder working")


def test_volume_profile_interactive():
    from src.processors.volume_profile import test_volume_profile_sync
    test_volume_profile_sync()
    print("✓ Volume profile working")


async def test_trades_live_interactive():
    from src.collectors.trades import test_trades_collector
    await test_trades_collector(5)


async def test_orderbook_live_interactive():
    from src.collectors.orderbook import test_orderbook_collector
    await test_orderbook_collector(5)


async def test_bars_live_interactive():
    from src.processors.bars import test_bar_aggregator_live
    await test_bar_aggregator_live(10)


async def test_derivatives_live_interactive():
    from src.collectors.derivatives import test_derivatives_collector
    await test_derivatives_collector(10)


async def test_stage1_interactive():
    from src.stage1 import test_stage1_full
    await test_stage1_full(30)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive_tests()
    else:
        # Run pytest
        pytest.main([__file__, "-v"])
