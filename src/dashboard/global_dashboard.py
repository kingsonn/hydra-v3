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
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 15px;
            max-height: calc(100vh - 100px);
        }
        
        .pairs-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
            gap: 12px;
            overflow-y: auto;
            max-height: calc(100vh - 100px);
        }
        
        .pair-card {
            background: #12121a;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #2a2a4a;
            transition: all 0.3s ease;
            font-size: 0.8rem;
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
        
        <div class="trade-section">
            <h2>🎯 ACTIVE TRADES</h2>
            <div class="trade-list" id="trade-list">
                <div class="no-trades">No trades yet</div>
            </div>
        </div>
    </div>

    <script>
        const pairsGrid = document.getElementById('pairs-grid');
        const tradeList = document.getElementById('trade-list');
        const status = document.getElementById('status');
        const states = {};
        const trades = [];
        
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
            
            // Liquidation bar
            const longUsd = d.liq_long_usd_30s || 0;
            const shortUsd = d.liq_short_usd_30s || 0;
            const totalLiq = longUsd + shortUsd;
            const longPct = totalLiq > 0 ? (longUsd / totalLiq * 100) : 50;
            
            // Cascade/exhaustion indicators
            const cascadeHtml = d.liq_cascade_active ? '<span class="indicator cascade">CASCADE</span>' : '';
            const exhaustHtml = d.liq_exhaustion ? '<span class="indicator exhaust">EXHAUST</span>' : '';
            
            return `
                <div class="pair-card ${hasTrade ? 'has-trade' : ''}" id="card-${symbol}">
                    <div class="pair-header">
                        <div>
                            <span class="symbol">${symbol}</span>
                            <span class="regime-badge regime-${d.regime}">${d.regime}</span>
                            ${cascadeHtml}${exhaustHtml}
                        </div>
                        <span class="price">$${formatPrice(d.price)}</span>
                    </div>
                    
                    <!-- PIPELINE STAGES -->
                    <div class="stages">
                        <div class="stage"><div class="stage-num ${s1Status}">1</div><div class="stage-content"><span>Data</span><span class="stage-value ${d.stage1_ok?'pass':''}">${d.stage1_ok?'OK':'WAIT'}</span></div></div>
                        <div class="stage"><div class="stage-num ${s2Status}">2</div><div class="stage-content"><span>Regime</span><span class="stage-value">${d.regime}</span></div></div>
                        <div class="stage"><div class="stage-num ${s3Status}">3</div><div class="stage-content"><span>Signal</span><span class="stage-value ${d.signal_fired?'pass':''}">${d.signal_fired ? d.signal_direction : (d.stage3_veto ? 'BLOCKED' : 'WAIT')}</span></div></div>
                        <div class="stage"><div class="stage-num ${s4Status}">4</div><div class="stage-content"><span>Structure</span><span class="stage-value ${d.stage4_pass?'pass':(d.stage4_reason?'fail':'')}">${d.stage4_pass ? 'PASS' : (d.stage4_reason || 'WAIT')}</span></div></div>
                        <div class="stage"><div class="stage-num ${s45Status}">4.5</div><div class="stage-content"><span>Orderflow</span><span class="stage-value ${d.stage45_pass?'pass':''}">${d.stage45_pass ? 'OK' : (d.stage45_reason || 'WAIT')}</span></div></div>
                        <div class="stage"><div class="stage-num ${s5Status}">5</div><div class="stage-content"><span>ML pct_300</span><span class="stage-value ${d.percentile_300>=80?'pass':(d.percentile_300>0?'fail':'')}">${d.percentile_300 > 0 ? formatNumber(d.percentile_300,0)+'%' : 'WAIT'}</span></div></div>
                    </div>
                    
                    ${d.signal_fired ? `<div class="signal-info"><b>Signals:</b> ${(d.signal_reasons||[]).join(', ') || 'N/A'}</div>` : ''}
                    ${d.stage3_veto ? `<div class="veto-info">${d.stage3_veto}</div>` : ''}
                    
                    <!-- REGIME SCORES -->
                    <div class="section-title">Regime Scores</div>
                    <div class="metrics-row">
                        <div class="metric"><div class="metric-label">Expansion</div><div class="metric-value ${(d.regime_expansion_score||0)>=3?'positive':''}">${d.regime_expansion_score||0}/5</div></div>
                        <div class="metric"><div class="metric-label">Compression</div><div class="metric-value ${(d.regime_compression_score||0)>=3?'neutral':''}">${d.regime_compression_score||0}/5</div></div>
                        <div class="metric"><div class="metric-label">Chop</div><div class="metric-value ${(d.regime_chop_score||0)>0?'negative':''}">${(d.regime_chop_score||0)>0?'YES':'NO'}</div></div>
                        <div class="metric"><div class="metric-label">Confidence</div><div class="metric-value">${formatNumber((d.regime_confidence||0)*100,0)}%</div></div>
                        <div class="metric"><div class="metric-label">Time</div><div class="metric-value">${formatNumber(d.time_in_regime||0,0)}s</div></div>
                        <div class="metric"><div class="metric-label">Price Δ5m</div><div class="metric-value ${getValueClass((d.price_change_5m||0)*100)}">${formatNumber((d.price_change_5m||0)*100,2)}%</div></div>
                    </div>
                    
                    <!-- ORDER FLOW -->
                    <div class="section-title">Order Flow</div>
                    <div class="metrics-row">
                        <div class="metric"><div class="metric-label">MOI 250ms</div><div class="metric-value ${getValueClass(d.of_moi_250ms)}">${formatNumber(d.of_moi_250ms,3)}</div></div>
                        <div class="metric"><div class="metric-label">MOI 1s</div><div class="metric-value ${getValueClass(d.of_moi_1s)}">${formatNumber(d.of_moi_1s,3)}</div></div>
                        <div class="metric"><div class="metric-label">Δ Velocity</div><div class="metric-value ${getValueClass(d.of_delta_velocity)}">${formatNumber(d.of_delta_velocity,3)}</div></div>
                        <div class="metric"><div class="metric-label">Aggression</div><div class="metric-value">${formatNumber(d.of_aggression_persistence,2)}</div></div>
                        <div class="metric"><div class="metric-label">MOI Std</div><div class="metric-value">${formatNumber(d.of_moi_std,3)}</div></div>
                        <div class="metric"><div class="metric-label">Flip Rate</div><div class="metric-value">${formatNumber(d.of_moi_flip_rate,1)}/m</div></div>
                    </div>
                    
                    <!-- ABSORPTION -->
                    <div class="section-title">Absorption</div>
                    <div class="metrics-row">
                        <div class="metric"><div class="metric-label">Absorb Z</div><div class="metric-value ${getValueClass(d.abs_absorption_z,1)}">${formatNumber(d.abs_absorption_z,2)}</div></div>
                        <div class="metric"><div class="metric-label">Refill</div><div class="metric-value">${formatNumber(d.abs_refill_rate,2)}</div></div>
                        <div class="metric"><div class="metric-label">Sweep</div><div class="metric-value ${d.abs_liquidity_sweep?'negative':''}">${d.abs_liquidity_sweep?'YES':'NO'}</div></div>
                        <div class="metric"><div class="metric-label">Depth Imb</div><div class="metric-value ${getValueClass(d.abs_depth_imbalance)}">${formatNumber((d.abs_depth_imbalance||0)*100,1)}%</div></div>
                    </div>
                    
                    <!-- VOLATILITY -->
                    <div class="section-title">Volatility</div>
                    <div class="metrics-row">
                        <div class="metric"><div class="metric-label">ATR 5m</div><div class="metric-value">${formatNumber(d.vol_atr_5m,4)}</div></div>
                        <div class="metric"><div class="metric-label">ATR 1h</div><div class="metric-value">${formatNumber(d.vol_atr_1h,4)}</div></div>
                        <div class="metric"><div class="metric-label">Expansion</div><div class="metric-value">${formatNumber(d.vol_vol_expansion_ratio,2)}x</div></div>
                        <div class="metric"><div class="metric-label">Rank</div><div class="metric-value">${formatNumber(d.vol_vol_rank,0)}%</div></div>
                        <div class="metric"><div class="metric-label">Vol 5m</div><div class="metric-value">${formatNumber((d.vol_vol_5m||0)*100,2)}%</div></div>
                        <div class="metric"><div class="metric-label">Regime</div><div class="metric-value">${d.vol_vol_regime||'MID'}</div></div>
                    </div>
                    
                    <!-- STRUCTURE -->
                    <div class="section-title">Structure</div>
                    <div class="metrics-row">
                        <div class="metric"><div class="metric-label">POC</div><div class="metric-value">$${formatPrice(d.str_poc)}</div></div>
                        <div class="metric"><div class="metric-label">VAH</div><div class="metric-value">$${formatPrice(d.str_vah)}</div></div>
                        <div class="metric"><div class="metric-label">VAL</div><div class="metric-value">$${formatPrice(d.str_val)}</div></div>
                        <div class="metric"><div class="metric-label">Dist LVN</div><div class="metric-value">${formatNumber(d.str_dist_lvn,2)} ATR</div></div>
                        <div class="metric"><div class="metric-label">Dist POC</div><div class="metric-value">${formatNumber(d.str_dist_poc,2)} ATR</div></div>
                        <div class="metric"><div class="metric-label">VA Width</div><div class="metric-value">$${formatNumber(d.str_value_area_width,2)}</div></div>
                    </div>
                    
                    <!-- LIQUIDATIONS -->
                    <div class="section-title">Liquidations</div>
                    <div class="metrics-row">
                        <div class="metric"><div class="metric-label">Long 30s</div><div class="metric-value positive">$${formatNumber(longUsd,0)}</div></div>
                        <div class="metric"><div class="metric-label">Short 30s</div><div class="metric-value negative">$${formatNumber(shortUsd,0)}</div></div>
                        <div class="metric"><div class="metric-label">Imbal 30s</div><div class="metric-value ${getValueClass(d.liq_imbalance_30s)}">${formatNumber((d.liq_imbalance_30s||0)*100,0)}%</div></div>
                        <div class="metric"><div class="metric-label">Total 2m</div><div class="metric-value">$${formatNumber((d.liq_long_usd_2m||0)+(d.liq_short_usd_2m||0),0)}</div></div>
                        <div class="metric"><div class="metric-label">Total 5m</div><div class="metric-value">$${formatNumber((d.liq_long_usd_5m||0)+(d.liq_short_usd_5m||0),0)}</div></div>
                        <div class="metric"><div class="metric-label">Cascade</div><div class="metric-value ${d.liq_cascade_active?'negative':''}">${d.liq_cascade_active?'YES':'NO'}</div></div>
                    </div>
                    <div class="liq-bar"><div class="liq-bar-inner liq-long" style="width:${longPct}%"></div></div>
                    
                    <!-- DERIVATIVES -->
                    <div class="section-title">Derivatives</div>
                    <div class="metrics-row">
                        <div class="metric"><div class="metric-label">Fund Rate</div><div class="metric-value ${getValueClass((d.fund_rate||0)*10000)}">${formatNumber((d.fund_rate||0)*100,4)}%</div></div>
                        <div class="metric"><div class="metric-label">Fund Z</div><div class="metric-value ${getValueClass(d.fund_funding_z,1)}">${formatNumber(d.fund_funding_z,2)}</div></div>
                        <div class="metric"><div class="metric-label">Crowd</div><div class="metric-value">${d.fund_crowd_side||'NEUTRAL'}</div></div>
                        <div class="metric"><div class="metric-label">OI</div><div class="metric-value">${formatNumber(d.oi_oi,0)}</div></div>
                        <div class="metric"><div class="metric-label">OI Δ1m</div><div class="metric-value ${getValueClass((d.oi_oi_delta_1m||0)*100)}">${formatNumber((d.oi_oi_delta_1m||0)*100,2)}%</div></div>
                        <div class="metric"><div class="metric-label">OI Δ5m</div><div class="metric-value ${getValueClass((d.oi_oi_delta_5m||0)*100)}">${formatNumber((d.oi_oi_delta_5m||0)*100,2)}%</div></div>
                    </div>
                    
                    <!-- ML PREDICTION -->
                    ${d.percentile_300 > 0 ? `
                    <div class="section-title">ML Prediction</div>
                    <div class="metrics-row">
                        <div class="metric"><div class="metric-label">Pred 60</div><div class="metric-value">${formatNumber(d.pred_60,4)}</div></div>
                        <div class="metric"><div class="metric-label">Pred 300</div><div class="metric-value">${formatNumber(d.pred_300,4)}</div></div>
                        <div class="metric"><div class="metric-label">Pct 60</div><div class="metric-value">${formatNumber(d.percentile_60,0)}%</div></div>
                        <div class="metric"><div class="metric-label">Pct 300</div><div class="metric-value ${d.percentile_300>=80?'positive':'negative'}">${formatNumber(d.percentile_300,0)}%</div></div>
                    </div>
                    ` : ''}
                    
                    ${hasTrade ? `
                    <div class="trade-active">
                        <div class="trade-active-label">🎯 TRADE ACTIVE</div>
                        <div class="trade-active-value">${d.trade_direction} @ $${formatPrice(d.trade_price)}</div>
                    </div>
                    ` : ''}
                </div>
            `;
        }
        
        function renderTradeItem(trade) {
            const isShort = trade.direction === 'SHORT';
            return `
                <div class="trade-item ${isShort ? 'short' : ''}">
                    <div class="trade-header">
                        <span class="trade-symbol">${trade.symbol}</span>
                        <span class="trade-time">${trade.time}</span>
                    </div>
                    <div class="trade-details">
                        <span class="direction-badge direction-${trade.direction}">${trade.direction}</span>
                        <span class="trade-price">$${formatPrice(trade.price)}</span>
                        <span class="trade-pct">pct_300: ${formatNumber(trade.percentile_300, 0)}%</span>
                    </div>
                </div>
            `;
        }
        
        function updateGrid() {
            const symbols = Object.keys(states).sort();
            if (symbols.length === 0) return;
            
            pairsGrid.innerHTML = symbols.map(s => renderPairCard(states[s])).join('');
        }
        
        function updateTrades() {
            if (trades.length === 0) {
                tradeList.innerHTML = '<div class="no-trades">No trades yet</div>';
                return;
            }
            tradeList.innerHTML = trades.map(t => renderTradeItem(t)).join('');
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
                } else if (msg.type === 'all_states') {
                    Object.assign(states, msg.data);
                    updateGrid();
                } else if (msg.type === 'new_trade') {
                    trades.unshift(msg.data);
                    if (trades.length > 100) trades.pop();
                    updateTrades();
                } else if (msg.type === 'trade_history') {
                    trades.length = 0;
                    trades.push(...msg.data);
                    updateTrades();
                }
            };
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
