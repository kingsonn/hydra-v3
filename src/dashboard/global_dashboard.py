"""
Global Dashboard - All 8 Pairs Pipeline View
Shows all 5 stages: Data Ingestion → Regime → Signal → Filter → ML Prediction → Trade

WebSocket-based real-time updates for all pairs through the complete pipeline.
"""
import asyncio
import json
import time
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import structlog

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from config import settings

logger = structlog.get_logger(__name__)

app = FastAPI(title="Hydra Global Pipeline Dashboard")

# Store connected WebSocket clients
connected_clients: Set[WebSocket] = set()

# Pipeline state for all symbols
pipeline_states: Dict[str, dict] = {}

# Trade history (signals that made it through all stages)
trade_history: List[dict] = []
MAX_TRADE_HISTORY = 100

# Position history (Stage 6 confirmed positions with tranches)
position_history: List[dict] = []
MAX_POSITION_HISTORY = 50

# Rejection history (Stage 6 rejections due to low ATR)
rejection_history: List[dict] = []
MAX_REJECTION_HISTORY = 20


@dataclass
class PipelineState:
    """Complete pipeline state for a single symbol"""
    symbol: str
    timestamp_ms: int = 0
    
    # Stage 1: Data Ingestion
    stage1_ok: bool = False
    trades_per_sec: float = 0.0
    last_trade_ms: int = 0
    last_book_ms: int = 0
    last_funding_ms: int = 0
    price: float = 0.0
    
    # Stage 2: Regime Classification
    regime: str = "UNKNOWN"
    regime_rejected: bool = False
    regime_confidence: float = 0.0
    expansion_score: int = 0
    compression_score: int = 0
    chop_score: int = 0
    time_in_regime: float = 0.0
    
    # Key metrics from Stage 2
    funding_z: float = 0.0
    oi_delta_5m: float = 0.0
    liq_imbalance: float = 0.0
    moi_250ms: float = 0.0
    delta_velocity: float = 0.0
    absorption_z: float = 0.0
    vol_regime: str = "MID"
    
    # Stage 3: Signal Detection
    signal_fired: bool = False
    signal_direction: str = "NONE"
    signal_strength: float = 0.0
    signal_reasons: List[str] = field(default_factory=list)
    stage3_veto: str = ""
    
    # Stage 4: Structural Filter
    stage4_pass: bool = False
    stage4_reason: str = ""
    dist_lvn: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    
    # Stage 4.5: Orderflow Confirmation
    stage45_pass: bool = False
    stage45_reason: str = ""
    
    # Stage 5: ML Prediction
    stage5_pass: bool = False
    pred_60: float = 0.0
    pred_300: float = 0.0
    percentile_60: float = 0.0
    percentile_300: float = 0.0
    model_used: str = ""
    
    # Final Trade
    is_trade: bool = False
    trade_direction: str = ""
    trade_price: float = 0.0
    trade_time: str = ""
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "price": self.price,
            # Stage 1
            "stage1_ok": self.stage1_ok,
            "trades_per_sec": self.trades_per_sec,
            "last_trade_ms": self.last_trade_ms,
            "last_book_ms": self.last_book_ms,
            "last_funding_ms": self.last_funding_ms,
            # Stage 2
            "regime": self.regime,
            "regime_rejected": self.regime_rejected,
            "regime_confidence": self.regime_confidence,
            "expansion_score": self.expansion_score,
            "compression_score": self.compression_score,
            "chop_score": self.chop_score,
            "time_in_regime": self.time_in_regime,
            "funding_z": self.funding_z,
            "oi_delta_5m": self.oi_delta_5m,
            "liq_imbalance": self.liq_imbalance,
            "moi_250ms": self.moi_250ms,
            "delta_velocity": self.delta_velocity,
            "absorption_z": self.absorption_z,
            "vol_regime": self.vol_regime,
            # Stage 3
            "signal_fired": self.signal_fired,
            "signal_direction": self.signal_direction,
            "signal_strength": self.signal_strength,
            "signal_reasons": self.signal_reasons,
            "stage3_veto": self.stage3_veto,
            # Stage 4
            "stage4_pass": self.stage4_pass,
            "stage4_reason": self.stage4_reason,
            "dist_lvn": self.dist_lvn,
            "vah": self.vah,
            "val": self.val,
            # Stage 4.5
            "stage45_pass": self.stage45_pass,
            "stage45_reason": self.stage45_reason,
            # Stage 5
            "stage5_pass": self.stage5_pass,
            "pred_60": self.pred_60,
            "pred_300": self.pred_300,
            "percentile_60": self.percentile_60,
            "percentile_300": self.percentile_300,
            "model_used": self.model_used,
            # Trade
            "is_trade": self.is_trade,
            "trade_direction": self.trade_direction,
            "trade_price": self.trade_price,
            "trade_time": self.trade_time,
        }


async def broadcast_pipeline_state(symbol: str, state: dict) -> None:
    """Broadcast pipeline state to all connected clients"""
    pipeline_states[symbol] = state
    
    message = json.dumps({
        "type": "pipeline_state",
        "data": state
    })
    
    # Copy set to avoid "Set changed size during iteration" error
    clients = list(connected_clients)
    for client in clients:
        try:
            await client.send_text(message)
        except Exception:
            connected_clients.discard(client)


async def broadcast_trade(trade: dict) -> None:
    """Broadcast new trade to all connected clients"""
    trade_history.insert(0, trade)
    if len(trade_history) > MAX_TRADE_HISTORY:
        trade_history.pop()
    
    message = json.dumps({
        "type": "new_trade",
        "data": trade
    })
    
    # Copy set to avoid "Set changed size during iteration" error
    clients = list(connected_clients)
    for client in clients:
        try:
            await client.send_text(message)
        except Exception:
            connected_clients.discard(client)


async def broadcast_position(position: dict) -> None:
    """Broadcast new position (Stage 6) to all connected clients"""
    position_history.insert(0, position)
    if len(position_history) > MAX_POSITION_HISTORY:
        position_history.pop()
    
    message = json.dumps({
        "type": "new_position",
        "data": position
    })
    
    clients = list(connected_clients)
    for client in clients:
        try:
            await client.send_text(message)
        except Exception:
            connected_clients.discard(client)


async def broadcast_open_positions(positions: List[dict]) -> None:
    """Broadcast open positions with live PnL data to all connected clients"""
    message = json.dumps({
        "type": "open_positions",
        "data": positions
    })
    
    clients = list(connected_clients)
    for client in clients:
        try:
            await client.send_text(message)
        except Exception:
            connected_clients.discard(client)


async def broadcast_closed_trades(trades: List[dict]) -> None:
    """Broadcast closed trades to all connected clients"""
    message = json.dumps({
        "type": "closed_trades",
        "data": trades
    })
    
    clients = list(connected_clients)
    for client in clients:
        try:
            await client.send_text(message)
        except Exception:
            connected_clients.discard(client)


async def broadcast_rejection(rejection: dict) -> None:
    """Broadcast Stage 6 rejection to all connected clients"""
    rejection_history.insert(0, rejection)
    if len(rejection_history) > MAX_REJECTION_HISTORY:
        rejection_history.pop()
    
    message = json.dumps({
        "type": "new_rejection",
        "data": rejection
    })
    
    clients = list(connected_clients)
    for client in clients:
        try:
            await client.send_text(message)
        except Exception:
            connected_clients.discard(client)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info("global_dashboard_client_connected", total=len(connected_clients))
    
    # Send current states
    if pipeline_states:
        await websocket.send_text(json.dumps({
            "type": "all_states",
            "data": pipeline_states
        }))
    
    # Send trade history
    if trade_history:
        await websocket.send_text(json.dumps({
            "type": "trade_history",
            "data": trade_history
        }))
    
    # Send position history
    if position_history:
        await websocket.send_text(json.dumps({
            "type": "position_history",
            "data": position_history
        }))
    
    # Send rejection history
    if rejection_history:
        await websocket.send_text(json.dumps({
            "type": "rejection_history",
            "data": rejection_history
        }))
    
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info("global_dashboard_client_disconnected", total=len(connected_clients))


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the global dashboard HTML"""
    return DASHBOARD_HTML


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hydra Global Pipeline Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 10px;
        }
        .header {
            text-align: center;
            margin-bottom: 15px;
            padding: 12px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 10px;
            border: 1px solid #2a2a4a;
        }
        .header h1 {
            font-size: 1.6rem;
            background: linear-gradient(90deg, #00d4ff, #7c3aed, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        .header .status { font-size: 0.85rem; color: #888; }
        .status.connected { color: #00ff88; }
        .status.disconnected { color: #ff4444; }
        
        .main-container {
            display: flex;
            gap: 15px;
            max-height: calc(100vh - 100px);
        }
        
        .pairs-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            overflow-y: auto;
            max-height: calc(100vh - 100px);
            align-content: start;
            flex: 1;
            min-width: 0;
        }
        
        .sidebar {
            width: 300px;
            flex-shrink: 0;
        }
        
        .pair-card {
            background: #12121a;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #2a2a4a;
            transition: border-color 0.3s ease;
            font-size: 0.8rem;
            overflow: hidden;
        }
        .pair-card:hover { border-color: #4a4a6a; }
        .pair-card.has-trade {
            border-color: #00ff88;
            box-shadow: 0 0 15px rgba(0, 255, 136, 0.2);
        }
        
        .pair-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid #2a2a4a;
        }
        .symbol {
            font-size: 1.2rem;
            font-weight: 700;
            color: #fff;
        }
        .price {
            font-size: 1.1rem;
            font-weight: 600;
            color: #00d4ff;
        }
        
        .stages-minimal {
            display: flex;
            gap: 6px;
            margin: 8px 0;
            justify-content: center;
        }
        .stage-dot {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.65rem;
            cursor: default;
        }
        .stage-dot.pending { background: #333; color: #666; }
        .stage-dot.ok { background: #00ff88; color: #000; }
        .stage-dot.fail { background: #ff4444; color: #fff; }
        .stage-dot.active { background: #ffaa00; color: #000; }
        
        .signal-compact {
            font-size: 0.75rem;
            color: #00ff88;
            text-align: center;
            padding: 4px;
            background: rgba(0,255,136,0.1);
            border-radius: 4px;
            margin: 4px 0;
        }
        .veto-compact {
            font-size: 0.7rem;
            color: #ff4444;
            text-align: center;
            padding: 3px;
        }
        
        .stages {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .stage {
            display: flex;
            align-items: center;
            padding: 8px 10px;
            background: #1a1a2a;
            border-radius: 6px;
            font-size: 0.85rem;
        }
        .stage-num {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.75rem;
            margin-right: 10px;
            flex-shrink: 0;
        }
        .stage-num.pending { background: #333; color: #666; }
        .stage-num.ok { background: #00ff88; color: #000; }
        .stage-num.fail { background: #ff4444; color: #fff; }
        .stage-num.active { background: #ffaa00; color: #000; animation: pulse 1s infinite; }
        
        .stage-content {
            flex: 1;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .stage-label {
            color: #888;
        }
        .stage-value {
            font-weight: 600;
            color: #fff;
        }
        .stage-value.pass { color: #00ff88; }
        .stage-value.fail { color: #ff4444; }
        .stage-value.warn { color: #ffaa00; }
        
        .stage-detail {
            font-size: 0.75rem;
            color: #666;
            margin-top: 2px;
        }
        
        .regime-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .regime-CHOP { background: linear-gradient(135deg, #ff4444, #cc0000); color: white; }
        .regime-COMPRESSION { background: linear-gradient(135deg, #ffaa00, #ff8800); color: #000; }
        .regime-EXPANSION { background: linear-gradient(135deg, #00ff88, #00cc66); color: #000; }
        .regime-UNKNOWN { background: #333; color: #666; }
        
        .direction-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
        }
        .direction-LONG { background: #00ff88; color: #000; }
        .direction-SHORT { background: #ff4444; color: #fff; }
        .direction-NONE { background: #333; color: #666; }
        
        /* Holding state */
        .holding-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.7rem;
            text-transform: uppercase;
            background: linear-gradient(135deg, #7c3aed, #5b21b6);
            color: #fff;
            margin-left: 6px;
            animation: pulse 2s infinite;
        }
        .pair-card.is-holding {
            border-color: #7c3aed;
            box-shadow: 0 0 15px rgba(124, 58, 237, 0.3);
        }
        
        /* Trade Section */
        .trade-section {
            background: #12121a;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #2a2a4a;
            max-height: calc(100vh - 130px);
            overflow-y: auto;
        }
        .trade-section h2 {
            font-size: 1rem;
            color: #00ff88;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #2a2a4a;
        }
        .trade-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .trade-item {
            padding: 10px;
            background: #1a1a2a;
            border-radius: 6px;
            border-left: 3px solid #00ff88;
        }
        .trade-item.short {
            border-left-color: #ff4444;
        }
        .trade-item .trade-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .trade-item .trade-symbol {
            font-weight: 700;
            color: #fff;
        }
        .trade-item .trade-time {
            font-size: 0.75rem;
            color: #666;
        }
        .trade-item .trade-details {
            display: flex;
            gap: 10px;
            font-size: 0.8rem;
        }
        .trade-item .trade-price {
            color: #00d4ff;
        }
        .trade-item .trade-pct {
            color: #888;
        }
        
        .no-trades {
            text-align: center;
            padding: 30px;
            color: #666;
        }
        
        /* Position Section */
        .position-item {
            padding: 12px;
            background: #1a1a2a;
            border-radius: 8px;
            border-left: 4px solid #00ff88;
            margin-bottom: 12px;
        }
        .position-item.short {
            border-left-color: #ff4444;
        }
        .position-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }
        .position-symbol {
            font-size: 1.1rem;
            font-weight: 700;
            color: #fff;
        }
        .position-signal {
            font-size: 0.7rem;
            color: #00d4ff;
            background: rgba(0, 212, 255, 0.1);
            padding: 2px 8px;
            border-radius: 4px;
        }
        .position-time {
            font-size: 0.7rem;
            color: #666;
        }
        .tranche-container {
            display: flex;
            gap: 10px;
            margin-top: 8px;
        }
        .tranche {
            flex: 1;
            background: #0a0a0f;
            border-radius: 6px;
            padding: 8px;
        }
        .tranche-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            padding-bottom: 4px;
            border-bottom: 1px solid #2a2a4a;
        }
        .tranche-name {
            font-weight: 700;
            font-size: 0.8rem;
        }
        .tranche-name.a { color: #00d4ff; }
        .tranche-name.b { color: #7c3aed; }
        .tranche-size {
            font-size: 0.75rem;
            color: #888;
        }
        .tranche-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            margin: 3px 0;
        }
        .tranche-label {
            color: #666;
        }
        .tranche-value {
            color: #fff;
            font-weight: 500;
        }
        .tranche-value.stop { color: #ff4444; }
        .tranche-value.tp { color: #00ff88; }
        .tranche-value.entry { color: #00d4ff; }
        
        /* Rejection Section */
        .rejection-item {
            padding: 10px;
            background: rgba(255, 68, 68, 0.1);
            border-radius: 6px;
            border-left: 3px solid #ff4444;
            margin-bottom: 8px;
        }
        .rejection-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        .rejection-symbol {
            font-weight: 700;
            color: #ff4444;
        }
        .rejection-reason {
            font-size: 0.75rem;
            color: #ff8888;
        }
        
        /* Metrics row */
        .metrics-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 6px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #2a2a4a;
        }
        .metric {
            text-align: center;
            padding: 5px;
            background: #0a0a0f;
            border-radius: 4px;
        }
        .metric-label {
            font-size: 0.65rem;
            color: #666;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 0.8rem;
            font-weight: 600;
            color: #fff;
        }
        .metric-value.positive { color: #00ff88; }
        .metric-value.negative { color: #ff4444; }
        .metric-value.neutral { color: #ffaa00; }
        
        .section-title {
            font-size: 0.7rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 8px 0 4px 0;
            padding-top: 8px;
            border-top: 1px solid #2a2a4a;
        }
        
        .signal-info {
            background: rgba(0, 255, 136, 0.1);
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            margin: 8px 0;
            color: #00ff88;
        }
        .veto-info {
            background: rgba(255, 68, 68, 0.1);
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            margin: 8px 0;
            color: #ff4444;
        }
        
        .indicator {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.6rem;
            font-weight: 700;
            margin-left: 6px;
            animation: pulse 1s infinite;
        }
        .indicator.cascade { background: #ff4444; color: #fff; }
        .indicator.exhaust { background: #00ff88; color: #000; }
        
        .liq-bar {
            height: 6px;
            background: #ff4444;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 6px;
        }
        .liq-bar-inner {
            height: 100%;
            transition: width 0.3s ease;
        }
        .liq-bar-inner.liq-long { background: #00ff88; }
        
        .trade-active {
            margin-top: 10px;
            padding: 10px;
            background: rgba(0, 255, 136, 0.15);
            border: 1px solid #00ff88;
            border-radius: 6px;
            text-align: center;
        }
        .trade-active-label {
            font-size: 0.75rem;
            color: #00ff88;
            margin-bottom: 4px;
        }
        .trade-active-value {
            font-size: 1rem;
            font-weight: 700;
            color: #fff;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .no-data {
            text-align: center;
            padding: 60px 20px;
            color: #666;
            grid-column: 1 / -1;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #1a1a2a;
        }
        ::-webkit-scrollbar-thumb {
            background: #4a4a6a;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>HYDRA Global Pipeline Dashboard</h1>
        <div class="status disconnected" id="status">Connecting...</div>
    </div>
    
    <div class="main-container">
        <div class="pairs-grid" id="pairs-grid">
            <div class="no-data">
                <h2>Waiting for data...</h2>
                <p>Pipeline states will appear here once the system is running</p>
            </div>
        </div>
        
        <div class="trade-section sidebar">
            <!-- Account Status -->
            <div id="account-status" style="background: linear-gradient(135deg, #1a1a2a, #2a2a4a); border-radius: 8px; padding: 12px; margin-bottom: 15px; border: 1px solid #3a3a5a;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #888; font-size: 0.75rem; text-transform: uppercase;">Account</span>
                    <span id="account-mode" style="background: #7c3aed; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.65rem;">TEST</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #888; font-size: 0.8rem;">Equity</span>
                    <span id="account-equity" style="color: #00ff88; font-weight: 700; font-size: 1.1rem;">$1,000.00</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #888; font-size: 0.75rem;">Margin Available</span>
                    <span id="account-margin" style="color: #00d4ff; font-size: 0.9rem;">$1,000.00</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #888; font-size: 0.75rem;">Unrealized PnL</span>
                    <span id="account-unrealized" style="color: #888; font-size: 0.9rem;">$0.00</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #888; font-size: 0.75rem;">Realized PnL</span>
                    <span id="account-realized" style="color: #888; font-size: 0.9rem;">$0.00</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #888; font-size: 0.75rem;">Total R</span>
                    <span id="account-total-r" style="color: #888; font-size: 0.9rem;">0.00R</span>
                </div>
            </div>
            
            <h2>📊 OPEN POSITIONS</h2>
            <div id="position-list">
                <div class="no-trades">No positions yet</div>
            </div>
            
            <h2 style="margin-top: 20px; color: #ff4444;">⚠️ REJECTIONS</h2>
            <div id="rejection-list">
                <div class="no-trades" style="padding: 15px;">No rejections</div>
            </div>
            
            <h2 style="margin-top: 20px; color: #7c3aed;">📜 CLOSED TRADES</h2>
            <div id="closed-trades-list" style="max-height: 300px; overflow-y: auto;">
                <div class="no-trades" style="padding: 15px;">No closed trades</div>
            </div>
        </div>
    </div>

    <script>
        const pairsGrid = document.getElementById('pairs-grid');
        const positionList = document.getElementById('position-list');
        const rejectionList = document.getElementById('rejection-list');
        const closedTradesList = document.getElementById('closed-trades-list');
        const status = document.getElementById('status');
        const accountEquity = document.getElementById('account-equity');
        const accountMargin = document.getElementById('account-margin');
        const accountUnrealized = document.getElementById('account-unrealized');
        const accountRealized = document.getElementById('account-realized');
        const accountTotalR = document.getElementById('account-total-r');
        const states = {};
        const positions = [];
        let lastAccountState = {};
        const rejections = [];
        let closedTrades = [];
        
        function formatNumber(n, decimals = 2) {
            if (n === undefined || n === null) return '-';
            if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
            if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
            return n.toFixed(decimals);
        }
        
        function formatPrice(p) {
            if (!p) return '-';
            if (p >= 1000) return p.toFixed(2);
            if (p >= 1) return p.toFixed(4);
            return p.toFixed(6);
        }
        
        function getValueClass(value, threshold = 0) {
            if (value > threshold) return 'positive';
            if (value < -threshold) return 'negative';
            return '';
        }
        
        function getStageStatus(ok, fail, active) {
            if (active) return 'active';
            if (ok) return 'ok';
            if (fail) return 'fail';
            return 'pending';
        }
        
        function renderPairCard(data) {
            const symbol = data.symbol;
            const hasTrade = data.is_trade;
            const d = data; // shorthand
            
            // Determine stage statuses
            const s1Status = d.stage1_ok ? 'ok' : 'pending';
            const s2Status = d.regime === 'CHOP' ? 'fail' : (d.regime !== 'UNKNOWN' ? 'ok' : 'pending');
            const s3Status = d.stage3_veto ? 'fail' : (d.signal_fired ? 'ok' : 'pending');
            const s4Status = (d.stage4_reason || '').includes('REJECTED') || (d.stage4_reason || '').includes('bad') ? 'fail' : (d.stage4_pass ? 'ok' : 'pending');
            const s45Status = (d.stage45_reason || '').includes('not_confirmed') ? 'fail' : (d.stage45_pass ? 'ok' : 'pending');
            const s5Status = d.percentile_300 < 80 && d.percentile_300 > 0 ? 'fail' : (d.stage5_pass ? 'ok' : 'pending');
            const s6Status = d.stage6_rejection ? 'fail' : (d.stage6_pass ? 'ok' : 'pending');
            
            // Liquidation bar
            const longUsd = d.liq_long_usd_30s || 0;
            const shortUsd = d.liq_short_usd_30s || 0;
            const totalLiq = longUsd + shortUsd;
            const longPct = totalLiq > 0 ? (longUsd / totalLiq * 100) : 50;
            
            // Cascade/exhaustion indicators
            const cascadeHtml = d.liq_cascade_active ? '<span class="indicator cascade">CASCADE</span>' : '';
            const exhaustHtml = d.liq_exhaustion ? '<span class="indicator exhaust">EXHAUST</span>' : '';
            const holdingHtml = d.is_holding ? '<span class="holding-badge">HOLDING</span>' : '';
            
            // Card classes
            const cardClasses = ['pair-card'];
            if (hasTrade) cardClasses.push('has-trade');
            if (d.is_holding) cardClasses.push('is-holding');
            
            return `
                <div class="${cardClasses.join(' ')}" id="card-${symbol}">
                    <div class="pair-header">
                        <div>
                            <span class="symbol">${symbol}</span>
                            <span class="regime-badge regime-${d.regime}">${d.regime}</span>
                            ${holdingHtml}${cascadeHtml}${exhaustHtml}
                        </div>
                        <span class="price">$${formatPrice(d.price)}</span>
                    </div>
                    
                    <!-- PIPELINE STAGES - MINIMAL -->
                    <div class="stages-minimal">
                        <span class="stage-dot ${s1Status}" title="Data">1</span>
                        <span class="stage-dot ${s2Status}" title="Regime">2</span>
                        <span class="stage-dot ${s3Status}" title="Signal">3</span>
                        <span class="stage-dot ${s4Status}" title="Structure">4</span>
                        <span class="stage-dot ${s45Status}" title="Orderflow">4.5</span>
                        <span class="stage-dot ${s5Status}" title="ML">5</span>
                        <span class="stage-dot ${s6Status}" title="Position">6</span>
                    </div>
                    
                    ${d.signal_fired ? `<div class="signal-compact">${d.signal_direction}: ${(d.signal_reasons||[]).join(', ') || 'N/A'}</div>` : ''}
                    ${d.stage3_veto ? `<div class="veto-compact">${d.stage3_veto}</div>` : ''}
                    
                    ${hasTrade ? `
                    <div class="trade-active">
                        <div class="trade-active-label">🎯 TRADE</div>
                        <div class="trade-active-value">${d.trade_direction} @ $${formatPrice(d.trade_price)}</div>
                    </div>
                    ` : ''}
                </div>
            `;
        }
        
        function renderPositionItem(pos) {
            const isShort = pos.side === 'SHORT';
            const tranches = pos.positions || [];
            const trancheA = tranches.find(t => t.tranche === 'A') || {};
            const trancheB = tranches.find(t => t.tranche === 'B') || {};
            
            return `
                <div class="position-item ${isShort ? 'short' : ''}">
                    <div class="position-header">
                        <div>
                            <span class="position-symbol">${pos.symbol}</span>
                            <span class="direction-badge direction-${pos.side}">${pos.side}</span>
                        </div>
                        <span class="position-time">${pos.time}</span>
                    </div>
                    <div class="position-signal">Signal: ${pos.signal_name || 'N/A'}</div>
                    <div style="font-size: 0.75rem; color: #888; margin: 4px 0;">Entry: $${formatPrice(pos.entry_price)} | Risk: $${pos.total_risk}</div>
                    
                    <div class="tranche-container">
                        <div class="tranche">
                            <div class="tranche-header">
                                <span class="tranche-name a">Tranche A</span>
                                <span class="tranche-size">${trancheA.size || 0}</span>
                            </div>
                            <div class="tranche-row"><span class="tranche-label">Stop</span><span class="tranche-value stop">$${formatPrice(trancheA.stop)}</span></div>
                            <div class="tranche-row"><span class="tranche-label">TP (1R)</span><span class="tranche-value tp">$${formatPrice(trancheA.tp_a)}</span></div>
                            <div class="tranche-row"><span class="tranche-label">B/E</span><span class="tranche-value">$${formatPrice(trancheA.breakeven)}</span></div>
                            <div class="tranche-row"><span class="tranche-label">Risk</span><span class="tranche-value">$${trancheA.risk || 0}</span></div>
                        </div>
                        <div class="tranche">
                            <div class="tranche-header">
                                <span class="tranche-name b">Tranche B</span>
                                <span class="tranche-size">${trancheB.size || 0}</span>
                            </div>
                            <div class="tranche-row"><span class="tranche-label">Stop</span><span class="tranche-value stop">$${formatPrice(trancheB.stop)}</span></div>
                            <div class="tranche-row"><span class="tranche-label">TP1 (2R)</span><span class="tranche-value tp">$${formatPrice(trancheB.tp_b_partial)} (40%)</span></div>
                            <div class="tranche-row"><span class="tranche-label">TP2 (3R)</span><span class="tranche-value tp">$${formatPrice(trancheB.tp_b_runner)}</span></div>
                            <div class="tranche-row"><span class="tranche-label">Risk</span><span class="tranche-value">$${trancheB.risk || 0}</span></div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function renderRejectionItem(rej) {
            return `
                <div class="rejection-item">
                    <div class="rejection-header">
                        <span class="rejection-symbol">${rej.symbol}</span>
                        <span class="position-time">${rej.time}</span>
                    </div>
                    <div class="rejection-reason">${rej.reason}</div>
                    <div style="font-size: 0.7rem; color: #666; margin-top: 4px;">Signal: ${rej.signal_name || 'N/A'}</div>
                </div>
            `;
        }
        
        function updateGrid() {
            const symbols = Object.keys(states).sort();
            if (symbols.length === 0) return;
            
            pairsGrid.innerHTML = symbols.map(s => renderPairCard(states[s])).join('');
        }
        
        function updatePositions() {
            if (positions.length === 0) {
                positionList.innerHTML = '<div class="no-trades">No positions yet</div>';
            } else {
                positionList.innerHTML = positions.map(p => renderPositionItem(p)).join('');
            }
        }
        
        function updateRejections() {
            if (rejections.length === 0) {
                rejectionList.innerHTML = '<div class="no-trades" style="padding: 15px;">No rejections</div>';
            } else {
                rejectionList.innerHTML = rejections.map(r => renderRejectionItem(r)).join('');
            }
        }
        
        let openPositionsData = [];
        
        function updateOpenPositions(tranches) {
            openPositionsData = tranches;
            renderOpenPositions();
        }
        
        function renderOpenPositions() {
            if (openPositionsData.length === 0) {
                positionList.innerHTML = '<div class="no-trades">No open positions</div>';
                return;
            }
            
            // Group tranches by symbol
            const bySymbol = {};
            openPositionsData.forEach(t => {
                if (!bySymbol[t.symbol]) bySymbol[t.symbol] = [];
                bySymbol[t.symbol].push(t);
            });
            
            let html = '';
            Object.entries(bySymbol).forEach(([symbol, tranches]) => {
                const trancheA = tranches.find(t => t.tranche === 'A');
                const trancheB = tranches.find(t => t.tranche === 'B');
                const side = tranches[0].side;
                const isShort = side === 'SHORT';
                
                // Calculate totals
                const totalUnrealizedPnl = tranches.reduce((sum, t) => sum + (t.unrealized_pnl || 0), 0);
                const totalRealizedPnl = tranches.reduce((sum, t) => sum + (t.realized_pnl || 0), 0);
                const totalR = tranches.reduce((sum, t) => sum + (t.r_multiple || 0), 0);
                const currentPrice = tranches[0].current_price || 0;
                const entryPrice = tranches[0].entry_price || 0;
                
                const pnlColor = totalUnrealizedPnl >= 0 ? '#00ff88' : '#ff4444';
                const rColor = totalR >= 0 ? '#00ff88' : '#ff4444';
                
                html += `
                    <div class="position-item ${isShort ? 'short' : ''}">
                        <div class="position-header">
                            <div>
                                <span class="position-symbol">${symbol}</span>
                                <span class="direction-badge direction-${side}">${side}</span>
                            </div>
                            <span style="color: ${pnlColor}; font-weight: 700; font-size: 0.9rem;">${totalUnrealizedPnl >= 0 ? '+' : ''}$${totalUnrealizedPnl.toFixed(2)}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #888; margin: 4px 0;">
                            <span>Entry: $${formatPrice(entryPrice)}</span>
                            <span>Current: $${formatPrice(currentPrice)}</span>
                            <span style="color: ${rColor}; font-weight: 600;">${totalR >= 0 ? '+' : ''}${totalR.toFixed(2)}R</span>
                        </div>
                        <div class="position-signal">Signal: ${tranches[0].signal_name || 'N/A'}</div>
                        
                        <div class="tranche-container">
                            ${trancheA ? `
                            <div class="tranche">
                                <div class="tranche-header">
                                    <span class="tranche-name a">Tranche A</span>
                                    <span style="color: ${(trancheA.unrealized_pnl || 0) >= 0 ? '#00ff88' : '#ff4444'}; font-size: 0.7rem;">${(trancheA.unrealized_pnl || 0) >= 0 ? '+' : ''}$${(trancheA.unrealized_pnl || 0).toFixed(2)}</span>
                                </div>
                                <div class="tranche-row"><span class="tranche-label">Size</span><span class="tranche-value">${trancheA.size || 0}</span></div>
                                <div class="tranche-row"><span class="tranche-label">Stop</span><span class="tranche-value stop">$${formatPrice(trancheA.stop_loss)}</span></div>
                                <div class="tranche-row"><span class="tranche-label">TP</span><span class="tranche-value tp">$${formatPrice(trancheA.take_profit)}</span></div>
                                <div class="tranche-row"><span class="tranche-label">R</span><span class="tranche-value" style="color: ${(trancheA.r_multiple || 0) >= 0 ? '#00ff88' : '#ff4444'}">${(trancheA.r_multiple || 0) >= 0 ? '+' : ''}${(trancheA.r_multiple || 0).toFixed(2)}R</span></div>
                            </div>
                            ` : '<div class="tranche" style="opacity: 0.5;"><div class="tranche-header"><span class="tranche-name a">Tranche A</span><span style="color: #888;">CLOSED</span></div></div>'}
                            ${trancheB ? `
                            <div class="tranche">
                                <div class="tranche-header">
                                    <span class="tranche-name b">Tranche B ${trancheB.status === 'partial' ? '(PARTIAL)' : ''}</span>
                                    <span style="color: ${(trancheB.unrealized_pnl || 0) >= 0 ? '#00ff88' : '#ff4444'}; font-size: 0.7rem;">${(trancheB.unrealized_pnl || 0) >= 0 ? '+' : ''}$${(trancheB.unrealized_pnl || 0).toFixed(2)}</span>
                                </div>
                                <div class="tranche-row"><span class="tranche-label">Size</span><span class="tranche-value">${trancheB.size || 0}</span></div>
                                <div class="tranche-row"><span class="tranche-label">Stop</span><span class="tranche-value stop">$${formatPrice(trancheB.stop_loss)}</span></div>
                                <div class="tranche-row"><span class="tranche-label">TP (Runner)</span><span class="tranche-value tp">$${formatPrice(trancheB.tp_runner)}</span></div>
                                <div class="tranche-row"><span class="tranche-label">R</span><span class="tranche-value" style="color: ${(trancheB.r_multiple || 0) >= 0 ? '#00ff88' : '#ff4444'}">${(trancheB.r_multiple || 0) >= 0 ? '+' : ''}${(trancheB.r_multiple || 0).toFixed(2)}R</span></div>
                            </div>
                            ` : '<div class="tranche" style="opacity: 0.5;"><div class="tranche-header"><span class="tranche-name b">Tranche B</span><span style="color: #888;">CLOSED</span></div></div>'}
                        </div>
                    </div>
                `;
            });
            
            positionList.innerHTML = html;
        }
        
        function updateAccountState(data) {
            if (!data) return;
            
            const equity = data.account_equity || 1000;
            const margin = data.account_margin_available || 1000;
            const unrealized = data.account_unrealized_pnl || 0;
            const realized = data.account_realized_pnl || 0;
            const totalR = data.account_total_r || 0;
            
            accountEquity.textContent = '$' + equity.toFixed(2);
            accountEquity.style.color = equity >= 1000 ? '#00ff88' : '#ff4444';
            
            accountMargin.textContent = '$' + margin.toFixed(2);
            
            accountUnrealized.textContent = (unrealized >= 0 ? '+' : '') + '$' + unrealized.toFixed(2);
            accountUnrealized.style.color = unrealized >= 0 ? '#00ff88' : '#ff4444';
            
            accountRealized.textContent = (realized >= 0 ? '+' : '') + '$' + realized.toFixed(2);
            accountRealized.style.color = realized >= 0 ? '#00ff88' : '#ff4444';
            
            accountTotalR.textContent = (totalR >= 0 ? '+' : '') + totalR.toFixed(2) + 'R';
            accountTotalR.style.color = totalR >= 0 ? '#00ff88' : '#ff4444';
        }
        
        function connect() {
            const ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = () => {
                status.textContent = 'Connected';
                status.className = 'status connected';
            };
            
            ws.onclose = () => {
                status.textContent = 'Disconnected - Reconnecting...';
                status.className = 'status disconnected';
                setTimeout(connect, 2000);
            };
            
            ws.onerror = () => {
                ws.close();
            };
            
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                
                if (msg.type === 'pipeline_state') {
                    states[msg.data.symbol] = msg.data;
                    updateGrid();
                    updateAccountState(msg.data);
                } else if (msg.type === 'all_states') {
                    Object.assign(states, msg.data);
                    updateGrid();
                    // Update account from first symbol's data
                    const firstSymbol = Object.keys(msg.data)[0];
                    if (firstSymbol) updateAccountState(msg.data[firstSymbol]);
                } else if (msg.type === 'new_position') {
                    positions.unshift(msg.data);
                    if (positions.length > 50) positions.pop();
                    updatePositions();
                } else if (msg.type === 'position_history') {
                    positions.length = 0;
                    positions.push(...msg.data);
                    updatePositions();
                } else if (msg.type === 'new_rejection') {
                    rejections.unshift(msg.data);
                    if (rejections.length > 20) rejections.pop();
                    updateRejections();
                } else if (msg.type === 'rejection_history') {
                    rejections.length = 0;
                    rejections.push(...msg.data);
                    updateRejections();
                } else if (msg.type === 'open_positions') {
                    // Update open positions with live PnL data
                    updateOpenPositions(msg.data);
                } else if (msg.type === 'closed_trades') {
                    // Update closed trades list
                    closedTrades = msg.data;
                    renderClosedTrades();
                } else if (msg.type === 'new_trade') {
                    // Keep for backward compatibility
                } else if (msg.type === 'trade_history') {
                    // Keep for backward compatibility
                }
            };
        }
        
        function renderClosedTrades() {
            if (closedTrades.length === 0) {
                closedTradesList.innerHTML = '<div class="no-trades" style="padding: 15px;">No closed trades</div>';
                return;
            }
            
            let html = closedTrades.map(t => {
                const pnlColor = t.realized_pnl >= 0 ? '#00ff88' : '#ff4444';
                const rColor = t.r_multiple >= 0 ? '#00ff88' : '#ff4444';
                const isWin = t.realized_pnl >= 0;
                
                return `
                    <div style="padding: 8px; margin-bottom: 6px; background: ${isWin ? 'rgba(0,255,136,0.1)' : 'rgba(255,68,68,0.1)'}; border-radius: 6px; border-left: 3px solid ${pnlColor};">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-weight: 600; font-size: 0.8rem;">${t.symbol}</span>
                            <span style="color: ${pnlColor}; font-weight: 700; font-size: 0.85rem;">${t.realized_pnl >= 0 ? '+' : ''}$${t.realized_pnl.toFixed(2)}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #888;">
                            <span>${t.side} ${t.tranche}</span>
                            <span style="color: ${rColor}">${t.r_multiple >= 0 ? '+' : ''}${t.r_multiple.toFixed(2)}R</span>
                        </div>
                        <div style="font-size: 0.65rem; color: #7c3aed; margin-top: 2px;">
                            Signal: ${t.signal_name || 'N/A'}
                        </div>
                        <div style="font-size: 0.65rem; color: #666; margin-top: 2px;">
                            ${t.close_reason} • ${t.closed_at ? t.closed_at.split('T')[1]?.split('.')[0] || '' : ''}
                        </div>
                    </div>
                `;
            }).join('');
            
            closedTradesList.innerHTML = html;
        }
        
        connect();
    </script>
</body>
</html>
"""


def run_dashboard(host: str = "0.0.0.0", port: int = 8888):
    """Run the global dashboard server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


async def start_dashboard_async(host: str = "0.0.0.0", port: int = 8888):
    """Start dashboard in async context"""
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
