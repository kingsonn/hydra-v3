"""
Global Dashboard V3 - Hybrid Alpha System
==========================================

Dashboard for Stage 3 V3 hybrid signals pipeline:
- Stage 1: Data Ingestion (trades, orderbook, funding, OI, liquidations)
- Stage 2: Alpha State Variables (funding_z, OI changes, price changes, trend, ATR, liquidations)
- Stage 3: Bias, Regime, Signal Detection
- Stage 6: Position Sizing
- Stage 7: Trade Management

WebSocket-based real-time updates for all pairs through the V3 pipeline.
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

app = FastAPI(title="Hydra V3 Pipeline Dashboard")

# Store connected WebSocket clients
connected_clients: Set[WebSocket] = set()

# Pipeline state for all symbols
pipeline_states: Dict[str, dict] = {}

# Trade history
trade_history: List[dict] = []
MAX_TRADE_HISTORY = 100

# Position history
position_history: List[dict] = []
MAX_POSITION_HISTORY = 50

# Rejection history
rejection_history: List[dict] = []
MAX_REJECTION_HISTORY = 20


async def broadcast_pipeline_state(symbol: str, state: dict) -> None:
    """Broadcast pipeline state to all connected clients"""
    pipeline_states[symbol] = state
    
    message = json.dumps({
        "type": "pipeline_state",
        "data": state
    })
    
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
    
    clients = list(connected_clients)
    for client in clients:
        try:
            await client.send_text(message)
        except Exception:
            connected_clients.discard(client)


async def broadcast_position(position: dict) -> None:
    """Broadcast new position to all connected clients"""
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
    """Broadcast open positions with live PnL data"""
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
    """Broadcast closed trades"""
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
    """Broadcast rejection"""
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


async def broadcast_gating_state(symbol: str, gating_state: dict) -> None:
    """
    Broadcast 3-layer gating state for a symbol.
    
    gating_state should contain:
    - bias: {direction, strength, reason, funding_score, liquidation_score, oi_score, trend_score}
    - regime: {current, confidence, bars_in_regime, pending_regime, pending_bars}
    - entry: {pullback_active, price_vs_ema20_pct, state}
    """
    message = json.dumps({
        "type": "gating_state",
        "symbol": symbol,
        "data": gating_state
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
    logger.info("v3_dashboard_client_connected", total=len(connected_clients))
    
    # Send current states
    if pipeline_states:
        await websocket.send_text(json.dumps({
            "type": "all_states",
            "data": pipeline_states
        }))
    
    # Send histories
    if trade_history:
        await websocket.send_text(json.dumps({
            "type": "trade_history",
            "data": trade_history
        }))
    
    if position_history:
        await websocket.send_text(json.dumps({
            "type": "position_history",
            "data": position_history
        }))
    
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
        logger.info("v3_dashboard_client_disconnected", total=len(connected_clients))


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the V3 dashboard HTML"""
    return DASHBOARD_HTML_V3


DASHBOARD_HTML_V3 = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hydra V3 Pipeline Dashboard</title>
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
        .header .subtitle { font-size: 0.75rem; color: #7c3aed; margin-bottom: 5px; }
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
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            overflow-y: auto;
            max-height: calc(100vh - 100px);
            align-content: start;
            flex: 1;
            min-width: 0;
        }
        
        .sidebar {
            width: 320px;
            flex-shrink: 0;
            overflow-y: auto;
            max-height: calc(100vh - 100px);
        }
        
        .pair-card {
            background: #12121a;
            border-radius: 8px;
            padding: 12px;
            border: 1px solid #2a2a4a;
            transition: border-color 0.3s ease;
            font-size: 0.75rem;
        }
        .pair-card:hover { border-color: #4a4a6a; }
        .pair-card.has-signal { border-color: #00ff88; box-shadow: 0 0 15px rgba(0, 255, 136, 0.2); }
        .pair-card.is-holding { border-color: #7c3aed; box-shadow: 0 0 15px rgba(124, 58, 237, 0.3); }
        
        .pair-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #2a2a4a;
        }
        .symbol { font-size: 1.1rem; font-weight: 700; color: #fff; }
        .price { font-size: 1rem; font-weight: 600; color: #00d4ff; }
        
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.65rem;
            text-transform: uppercase;
            margin-left: 6px;
        }
        .regime-TRENDING_UP { background: linear-gradient(135deg, #00ff88, #00cc66); color: #000; }
        .regime-TRENDING_DOWN { background: linear-gradient(135deg, #ff4444, #cc0000); color: #fff; }
        .regime-EXPANSION { background: linear-gradient(135deg, #00d4ff, #0099cc); color: #000; }
        .regime-COMPRESSION { background: linear-gradient(135deg, #ffaa00, #ff8800); color: #000; }
        .regime-RANGING { background: linear-gradient(135deg, #888, #666); color: #fff; }
        .regime-CHOPPY { background: linear-gradient(135deg, #ff4444, #cc0000); color: #fff; }
        .regime-UNKNOWN { background: #333; color: #666; }
        
        .bias-LONG { background: #00ff88; color: #000; }
        .bias-SHORT { background: #ff4444; color: #fff; }
        .bias-NEUTRAL { background: #888; color: #fff; }
        
        .holding-badge { background: linear-gradient(135deg, #7c3aed, #5b21b6); color: #fff; animation: pulse 2s infinite; }
        .warmup-badge { background: #ffaa00; color: #000; }
        .cascade-badge { background: #ff4444; color: #fff; animation: pulse 1s infinite; }
        .exhaust-badge { background: #00ff88; color: #000; }
        
        /* Stage indicators */
        .stages-row {
            display: flex;
            gap: 4px;
            margin: 8px 0;
            justify-content: center;
        }
        .stage-dot {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.6rem;
        }
        .stage-dot.pending { background: #333; color: #666; }
        .stage-dot.ok { background: #00ff88; color: #000; }
        .stage-dot.fail { background: #ff4444; color: #fff; }
        .stage-dot.active { background: #ffaa00; color: #000; }
        
        /* Alpha State Section */
        .alpha-section {
            background: #0a0a0f;
            border-radius: 6px;
            padding: 8px;
            margin-top: 8px;
        }
        .alpha-section-title {
            font-size: 0.65rem;
            color: #7c3aed;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        .alpha-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 4px;
        }
        .alpha-item {
            text-align: center;
            padding: 4px;
            background: #1a1a2a;
            border-radius: 4px;
        }
        .alpha-label { font-size: 0.55rem; color: #666; text-transform: uppercase; }
        .alpha-value { font-size: 0.75rem; font-weight: 600; color: #fff; }
        .alpha-value.positive { color: #00ff88; }
        .alpha-value.negative { color: #ff4444; }
        .alpha-value.neutral { color: #ffaa00; }
        .alpha-value.extreme { color: #ff6b6b; animation: pulse 1s infinite; }
        
        /* Signal Section */
        .signal-section {
            background: rgba(0, 255, 136, 0.1);
            border: 1px solid #00ff88;
            border-radius: 6px;
            padding: 8px;
            margin-top: 8px;
            text-align: center;
        }
        .signal-type { font-size: 0.7rem; color: #00ff88; font-weight: 600; }
        .signal-direction { font-size: 1rem; font-weight: 700; }
        .signal-direction.LONG { color: #00ff88; }
        .signal-direction.SHORT { color: #ff4444; }
        .signal-details { font-size: 0.65rem; color: #888; margin-top: 4px; }
        
        /* Trade Section */
        .trade-section {
            background: #12121a;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #2a2a4a;
            margin-bottom: 15px;
        }
        .trade-section h2 {
            font-size: 0.9rem;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #2a2a4a;
        }
        .trade-item {
            padding: 10px;
            background: #1a1a2a;
            border-radius: 6px;
            margin-bottom: 8px;
            border-left: 3px solid #00ff88;
        }
        .trade-item.short { border-left-color: #ff4444; }
        
        /* Account Section */
        .account-section {
            background: linear-gradient(135deg, #1a1a2a, #2a2a4a);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 15px;
            border: 1px solid #3a3a5a;
        }
        .account-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            font-size: 0.8rem;
        }
        .account-label { color: #888; }
        .account-value { font-weight: 600; }
        
        .no-data {
            text-align: center;
            padding: 40px;
            color: #666;
            grid-column: 1 / -1;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #1a1a2a; }
        ::-webkit-scrollbar-thumb { background: #4a4a6a; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>HYDRA V3 Pipeline Dashboard</h1>
        <div class="subtitle">Stage 1 → Stage 2 → Stage 3 V3 → Stage 6 → Stage 7</div>
        <div class="status disconnected" id="status">Connecting...</div>
    </div>
    
    <div class="main-container">
        <div class="pairs-grid" id="pairs-grid">
            <div class="no-data">
                <h2>Waiting for data...</h2>
                <p>Pipeline states will appear here once the system is running</p>
            </div>
        </div>
        
        <div class="sidebar">
            <!-- Account Status -->
            <div class="account-section">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #888; font-size: 0.75rem; text-transform: uppercase;">Account</span>
                    <span id="account-mode" style="background: #7c3aed; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.65rem;">V3 TEST</span>
                </div>
                <div class="account-row">
                    <span class="account-label">Equity</span>
                    <span class="account-value" id="account-equity" style="color: #00ff88; font-size: 1.1rem;">$1,000.00</span>
                </div>
                <div class="account-row">
                    <span class="account-label">Unrealized PnL</span>
                    <span class="account-value" id="account-unrealized" style="color: #888;">$0.00</span>
                </div>
                <div class="account-row">
                    <span class="account-label">Realized PnL</span>
                    <span class="account-value" id="account-realized" style="color: #888;">$0.00</span>
                </div>
                <div class="account-row">
                    <span class="account-label">Total R</span>
                    <span class="account-value" id="account-total-r" style="color: #888;">0.00R</span>
                </div>
                <div class="account-row">
                    <span class="account-label">Win Rate</span>
                    <span class="account-value" id="account-win-rate" style="color: #888;">0%</span>
                </div>
                <div class="account-row">
                    <span class="account-label">Trades</span>
                    <span class="account-value" id="account-trades" style="color: #888;">0</span>
                </div>
                <div class="account-row">
                    <span class="account-label">Margin Available</span>
                    <span class="account-value" id="account-margin" style="color: #00d4ff;">$1,000.00</span>
                </div>
            </div>
            
            <div class="trade-section">
                <h2 style="color: #7c3aed;">📜 CLOSED TRADES</h2>
                <div id="closed-trades-list" style="max-height: 200px; overflow-y: auto;">
                    <div class="no-data" style="padding: 15px; font-size: 0.75rem;">No closed trades</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const pairsGrid = document.getElementById('pairs-grid');
        const closedTradesList = document.getElementById('closed-trades-list');
        const status = document.getElementById('status');
        const states = {};
        let closedTrades = [];
        
        function formatNumber(n, decimals = 2) {
            if (n === undefined || n === null || isNaN(n)) return '-';
            if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M';
            if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K';
            return n.toFixed(decimals);
        }
        
        function formatPrice(p) {
            if (!p || isNaN(p)) return '-';
            if (p >= 1000) return p.toFixed(2);
            if (p >= 1) return p.toFixed(4);
            return p.toFixed(6);
        }
        
        function formatPct(p) {
            if (p === undefined || p === null || isNaN(p)) return '-';
            return (p * 100).toFixed(2) + '%';
        }
        
        function getValueClass(value, threshold = 0) {
            if (value > threshold) return 'positive';
            if (value < -threshold) return 'negative';
            return '';
        }
        
        function renderPairCard(d) {
            const symbol = d.symbol;
            const cardClasses = ['pair-card'];
            if (d.signal_fired) cardClasses.push('has-signal');
            if (d.is_holding) cardClasses.push('is-holding');
            
            // Stage statuses
            const s1Status = d.stage1_ok ? 'ok' : 'pending';
            const s2Status = d.atr_short > 0 ? 'ok' : 'pending';
            const s3Status = d.signal_fired ? 'ok' : (d.regime === 'CHOPPY' ? 'fail' : 'pending');
            const s6Status = d.is_holding ? 'ok' : (d.position_rejection ? 'fail' : 'pending');
            const s7Status = d.has_open_trade ? 'active' : 'pending';
            
            // Badges
            let badges = '';
            if (d.in_warmup) badges += '<span class="badge warmup-badge">WARMUP ' + d.warmup_remaining + 's</span>';
            if (d.is_holding) badges += '<span class="badge holding-badge">HOLDING</span>';
            if (d.cascade_active) badges += '<span class="badge cascade-badge">CASCADE</span>';
            if (d.liq_exhaustion) badges += '<span class="badge exhaust-badge">EXHAUST</span>';
            
            // Signal section - only show if trade actually opened (not just signal fired)
            let signalHtml = '';
            if (d.signal_fired && d.trade_opened) {
                signalHtml = `
                    <div class="signal-section">
                        <div class="signal-type">${d.signal_type || 'SIGNAL'}</div>
                        <div class="signal-direction ${d.signal_direction}">${d.signal_direction}</div>
                        <div class="signal-details">
                            Stop: ${(d.signal_stop_pct || 0).toFixed(2)}% | Target: ${(d.signal_target_pct || 0).toFixed(2)}%
                        </div>
                    </div>
                `;
            } else if (d.signal_fired && !d.trade_opened && d.position_rejection) {
                // Show rejection reason if signal fired but order failed
                signalHtml = `
                    <div style="background: rgba(255,68,68,0.1); border: 1px solid #ff4444; border-radius: 6px; padding: 8px; margin-top: 8px; text-align: center;">
                        <div style="font-size: 0.7rem; color: #ff4444;">⚠️ SIGNAL REJECTED</div>
                        <div style="font-size: 0.65rem; color: #888;">${d.position_rejection}</div>
                    </div>
                `;
            }
            
            // Open trade section
            let tradeHtml = '';
            if (d.has_open_trade) {
                const pnlColor = (d.open_trade_pnl || 0) >= 0 ? '#00ff88' : '#ff4444';
                const signalName = d.open_trade_signal || 'SIGNAL';
                tradeHtml = `
                    <div style="background: rgba(124, 58, 237, 0.2); border: 1px solid #7c3aed; border-radius: 6px; padding: 8px; margin-top: 8px; text-align: center;">
                        <div style="font-size: 0.7rem; color: #7c3aed;">OPEN TRADE - ${signalName}</div>
                        <div style="font-size: 0.85rem; font-weight: 700; color: ${pnlColor};">
                            ${d.open_trade_side} @ $${formatPrice(d.open_trade_entry)} | PnL: $${(d.open_trade_pnl || 0).toFixed(2)} (${(d.open_trade_r || 0).toFixed(2)}R)
                        </div>
                        <div style="font-size: 0.65rem; color: #888; margin-top: 4px;">
                            Stop: $${formatPrice(d.open_trade_stop)} | Target: $${formatPrice(d.open_trade_target)}
                            ${d.breakeven_triggered ? ' | <span style="color: #00ff88;">B/E ✓</span>' : ''}
                            ${d.trail_1r_triggered ? ' | <span style="color: #00d4ff;">TRAIL ✓</span>' : ''}
                        </div>
                        <div style="font-size: 0.6rem; color: #666; margin-top: 3px;">
                            Size: ${formatNumber(d.open_trade_size || 0, 6)} | Notional: $${formatNumber(d.open_trade_notional || 0)} | Margin: $${formatNumber(d.open_trade_margin || 0)}
                        </div>
                    </div>
                `;
            }
            
            return `
                <div class="${cardClasses.join(' ')}" id="card-${symbol}">
                    <div class="pair-header">
                        <div>
                            <span class="symbol">${symbol}</span>
                            <span class="badge regime-${d.regime || 'UNKNOWN'}">${d.regime || 'UNKNOWN'}</span>
                            <span class="badge bias-${d.bias_direction || 'NEUTRAL'}">${d.bias_direction || 'N'}</span>
                            ${badges}
                        </div>
                        <span class="price">$${formatPrice(d.price)}</span>
                    </div>
                    
                    <!-- Pipeline Stages -->
                    <div class="stages-row">
                        <span class="stage-dot ${s1Status}" title="Stage 1: Data">1</span>
                        <span class="stage-dot ${s2Status}" title="Stage 2: Alpha">2</span>
                        <span class="stage-dot ${s3Status}" title="Stage 3: Signal">3</span>
                        <span class="stage-dot ${s6Status}" title="Stage 6: Position">6</span>
                        <span class="stage-dot ${s7Status}" title="Stage 7: Trade">7</span>
                    </div>
                    
                    ${signalHtml}
                    ${tradeHtml}
                    
                    <!-- 3-LAYER GATING STATUS -->
                    <div class="alpha-section" style="background: linear-gradient(135deg, #1a1a2e, #16213e); border: 1px solid #3a3a5a;">
                        <div class="alpha-section-title" style="color: #00d4ff;">🔒 3-LAYER GATING</div>
                        <div class="alpha-grid">
                            <div class="alpha-item" style="background: ${d.bias_direction === 'LONG' ? 'rgba(0,255,136,0.2)' : (d.bias_direction === 'SHORT' ? 'rgba(255,68,68,0.2)' : 'rgba(136,136,136,0.2)')}">
                                <div class="alpha-label">L1: BIAS</div>
                                <div class="alpha-value ${d.bias_direction === 'LONG' ? 'positive' : (d.bias_direction === 'SHORT' ? 'negative' : '')}">${d.bias_direction || 'NEUTRAL'}</div>
                                <div style="font-size: 0.55rem; color: #888;">str: ${formatNumber(d.bias_strength || 0, 2)}</div>
                            </div>
                            <div class="alpha-item" style="background: ${d.regime === 'TRENDING_UP' ? 'rgba(0,255,136,0.2)' : (d.regime === 'TRENDING_DOWN' ? 'rgba(255,68,68,0.2)' : (d.regime === 'CHOPPY' ? 'rgba(255,68,68,0.3)' : 'rgba(136,136,136,0.2)'))}">
                                <div class="alpha-label">L2: REGIME</div>
                                <div class="alpha-value" style="font-size: 0.65rem;">${d.regime || 'UNKNOWN'}</div>
                                <div style="font-size: 0.55rem; color: #888;">conf: ${formatNumber(d.regime_confidence || 0, 2)}</div>
                            </div>
                            <div class="alpha-item" style="background: ${d.entry_state === 'PULLBACK_READY' ? 'rgba(0,255,136,0.2)' : (d.entry_state === 'EXTENDED' ? 'rgba(255,170,0,0.2)' : 'rgba(136,136,136,0.2)')}">
                                <div class="alpha-label">L3: ENTRY</div>
                                <div class="alpha-value" style="font-size: 0.65rem; color: ${d.entry_state === 'PULLBACK_READY' ? '#00ff88' : (d.entry_state === 'EXTENDED' ? '#ffaa00' : '#888')}">${d.entry_state || 'WAITING'}</div>
                                <div style="font-size: 0.55rem; color: #888;">vs EMA: ${formatNumber(d.price_vs_ema20 || 0, 2)}%</div>
                            </div>
                        </div>
                        <div style="margin-top: 6px; font-size: 0.6rem; color: #666; text-align: center;">
                            ${d.bias_reason ? '💡 ' + d.bias_reason.substring(0, 40) : ''}
                        </div>
                    </div>
                    
                    <!-- Alpha State Variables -->
                    <div class="alpha-section">
                        <div class="alpha-section-title">📊 Funding & OI</div>
                        <div class="alpha-grid">
                            <div class="alpha-item">
                                <div class="alpha-label">Fund Z</div>
                                <div class="alpha-value ${Math.abs(d.funding_z || 0) > 2 ? 'extreme' : getValueClass(d.funding_z, 0.5)}">${formatNumber(d.funding_z, 2)}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">Fund Rate</div>
                                <div class="alpha-value ${getValueClass(d.funding_rate, 0)}">${formatNumber((d.funding_rate || 0) * 100, 4)}%</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">OI Δ 1h</div>
                                <div class="alpha-value ${getValueClass(d.oi_change_1h, 0)}">${formatPct(d.oi_change_1h)}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">OI Δ 4h</div>
                                <div class="alpha-value ${getValueClass(d.oi_change_4h, 0)}">${formatPct(d.oi_change_4h)}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">OI Δ 24h</div>
                                <div class="alpha-value ${getValueClass(d.oi_change_24h, 0)}">${formatPct(d.oi_change_24h)}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">Trades/s</div>
                                <div class="alpha-value">${formatNumber(d.trades_per_sec, 1)}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="alpha-section">
                        <div class="alpha-section-title">📈 Price & Trend</div>
                        <div class="alpha-grid">
                            <div class="alpha-item">
                                <div class="alpha-label">Δ 1h</div>
                                <div class="alpha-value ${getValueClass(d.price_change_1h, 0)}">${formatPct(d.price_change_1h)}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">Δ 4h</div>
                                <div class="alpha-value ${getValueClass(d.price_change_4h, 0)}">${formatPct(d.price_change_4h)}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">Δ 24h</div>
                                <div class="alpha-value ${getValueClass(d.price_change_24h, 0)}">${formatPct(d.price_change_24h)}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">Trend</div>
                                <div class="alpha-value ${d.trend_direction === 'LONG' ? 'positive' : (d.trend_direction === 'SHORT' ? 'negative' : '')}">${d.trend_direction || '-'}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">EMA20</div>
                                <div class="alpha-value">${formatNumber(d.ema_20, 0)}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">RSI</div>
                                <div class="alpha-value ${d.rsi_14 > 70 ? 'negative' : (d.rsi_14 < 30 ? 'positive' : '')}">${formatNumber(d.rsi_14, 1)}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="alpha-section">
                        <div class="alpha-section-title">⚡ Volatility & Liquidations</div>
                        <div class="alpha-grid">
                            <div class="alpha-item">
                                <div class="alpha-label">ATR 5</div>
                                <div class="alpha-value">${formatNumber(d.atr_short, 2)}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">ATR 20</div>
                                <div class="alpha-value">${formatNumber(d.atr_long, 2)}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">Vol Exp</div>
                                <div class="alpha-value ${d.vol_expansion_ratio > 1.5 ? 'neutral' : ''}">${formatNumber(d.vol_expansion_ratio, 2)}x</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">Liq 1h</div>
                                <div class="alpha-value">${formatNumber(d.liq_total_1h, 0)}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">Liq Imb 1h</div>
                                <div class="alpha-value ${getValueClass(d.liq_imbalance_1h, 0.1)}">${formatNumber(d.liq_imbalance_1h, 2)}</div>
                            </div>
                            <div class="alpha-item">
                                <div class="alpha-label">Liq Imb 4h</div>
                                <div class="alpha-value ${getValueClass(d.liq_imbalance_4h, 0.1)}">${formatNumber(d.liq_imbalance_4h, 2)}</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function renderPositionItem(pos) {
            const isShort = pos.side === 'SHORT';
            const pnlColor = (pos.unrealized_pnl || 0) >= 0 ? '#00ff88' : '#ff4444';
            
            return `
                <div class="trade-item ${isShort ? 'short' : ''}">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-weight: 700;">${pos.symbol}</span>
                        <span style="color: ${pnlColor}; font-weight: 600;">${(pos.unrealized_pnl || 0) >= 0 ? '+' : ''}$${(pos.unrealized_pnl || 0).toFixed(2)}</span>
                    </div>
                    <div style="font-size: 0.7rem; color: #888;">
                        ${pos.side} @ $${formatPrice(pos.entry_price)} | ${(pos.r_multiple || 0).toFixed(2)}R
                    </div>
                </div>
            `;
        }
        
        function renderRejectionItem(rej) {
            return `
                <div style="padding: 8px; background: rgba(255, 68, 68, 0.1); border-left: 2px solid #ff4444; margin-bottom: 6px; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 600; color: #ff4444; font-size: 0.75rem;">${rej.symbol}</span>
                        <span style="font-size: 0.65rem; color: #666;">${rej.time || ''}</span>
                    </div>
                    <div style="font-size: 0.7rem; color: #ff8888;">${rej.reason || 'Unknown'}</div>
                </div>
            `;
        }
        
        function renderClosedTrade(t) {
            const pnlColor = t.realized_pnl >= 0 ? '#00ff88' : '#ff4444';
            const signalName = t.signal_type || t.signal_name || '';
            return `
                <div style="padding: 8px; background: ${t.realized_pnl >= 0 ? 'rgba(0,255,136,0.1)' : 'rgba(255,68,68,0.1)'}; border-left: 2px solid ${pnlColor}; margin-bottom: 6px; border-radius: 4px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-weight: 600; font-size: 0.75rem;">${t.symbol}</span>
                        <span style="color: ${pnlColor}; font-weight: 600; font-size: 0.75rem;">${t.realized_pnl >= 0 ? '+' : ''}$${t.realized_pnl.toFixed(2)} (${(t.r_multiple || 0).toFixed(2)}R)</span>
                    </div>
                    <div style="font-size: 0.65rem; color: #888;">${t.side} | ${signalName} | ${t.close_reason || ''}</div>
                </div>
            `;
        }
        
        function updateGrid() {
            const symbols = Object.keys(states).sort();
            if (symbols.length === 0) return;
            pairsGrid.innerHTML = symbols.map(s => renderPairCard(states[s])).join('');
        }
        
        function updatePositions() {
            if (openPositions.length === 0) {
                positionList.innerHTML = '<div class="no-data" style="padding: 20px;">No positions yet</div>';
            } else {
                positionList.innerHTML = openPositions.map(p => renderPositionItem(p)).join('');
            }
        }
        
        function updateRejections() {
            if (rejections.length === 0) {
                rejectionList.innerHTML = '<div class="no-data" style="padding: 15px; font-size: 0.75rem;">No rejections</div>';
            } else {
                rejectionList.innerHTML = rejections.map(r => renderRejectionItem(r)).join('');
            }
        }
        
        function updateClosedTrades() {
            if (closedTrades.length === 0) {
                closedTradesList.innerHTML = '<div class="no-data" style="padding: 15px; font-size: 0.75rem;">No closed trades</div>';
            } else {
                closedTradesList.innerHTML = closedTrades.map(t => renderClosedTrade(t)).join('');
            }
        }
        
        function updateAccount(d) {
            if (!d) return;
            const eq = document.getElementById('account-equity');
            const ur = document.getElementById('account-unrealized');
            const re = document.getElementById('account-realized');
            const tr = document.getElementById('account-total-r');
            const wr = document.getElementById('account-win-rate');
            const tc = document.getElementById('account-trades');
            const mg = document.getElementById('account-margin');
            
            if (d.account_equity !== undefined) {
                eq.textContent = '$' + d.account_equity.toFixed(2);
                eq.style.color = d.account_equity >= 1000 ? '#00ff88' : '#ff4444';
            }
            if (d.account_unrealized_pnl !== undefined) {
                ur.textContent = (d.account_unrealized_pnl >= 0 ? '+' : '') + '$' + d.account_unrealized_pnl.toFixed(2);
                ur.style.color = d.account_unrealized_pnl >= 0 ? '#00ff88' : '#ff4444';
            }
            if (d.account_realized_pnl !== undefined) {
                re.textContent = (d.account_realized_pnl >= 0 ? '+' : '') + '$' + d.account_realized_pnl.toFixed(2);
                re.style.color = d.account_realized_pnl >= 0 ? '#00ff88' : '#ff4444';
            }
            if (d.account_total_r !== undefined) {
                tr.textContent = (d.account_total_r >= 0 ? '+' : '') + d.account_total_r.toFixed(2) + 'R';
                tr.style.color = d.account_total_r >= 0 ? '#00ff88' : '#ff4444';
            }
            if (d.account_win_rate !== undefined) {
                wr.textContent = d.account_win_rate.toFixed(1) + '%';
            }
            if (d.account_trade_count !== undefined) {
                tc.textContent = d.account_trade_count;
            }
            if (d.account_margin_available !== undefined && mg) {
                mg.textContent = '$' + d.account_margin_available.toFixed(2);
            }
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
            
            ws.onerror = () => { ws.close(); };
            
            ws.onmessage = (event) => {
                const msg = JSON.parse(event.data);
                
                if (msg.type === 'pipeline_state') {
                    states[msg.data.symbol] = msg.data;
                    updateGrid();
                    updateAccount(msg.data);
                } else if (msg.type === 'all_states') {
                    Object.assign(states, msg.data);
                    updateGrid();
                    const firstSymbol = Object.keys(msg.data)[0];
                    if (firstSymbol) updateAccount(msg.data[firstSymbol]);
                } else if (msg.type === 'open_positions') {
                    openPositions = msg.data;
                    updatePositions();
                } else if (msg.type === 'new_rejection') {
                    rejections.unshift(msg.data);
                    if (rejections.length > 20) rejections.pop();
                    updateRejections();
                } else if (msg.type === 'rejection_history') {
                    rejections = msg.data;
                    updateRejections();
                } else if (msg.type === 'closed_trades') {
                    closedTrades = msg.data;
                    updateClosedTrades();
                }
            };
        }
        
        connect();
    </script>
</body>
</html>
"""


def run_dashboard(host: str = "0.0.0.0", port: int = 8889):
    """Run the V3 dashboard server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


async def start_dashboard_async(host: str = "0.0.0.0", port: int = 8889):
    """Start dashboard in async context"""
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
