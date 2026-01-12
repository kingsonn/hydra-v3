"""
Real-time Dashboard for Stage 2 Market State
WebSocket-based UI showing all pairs with regime classification
"""
import asyncio
import json
import time
from typing import Dict, List, Optional, Set
from pathlib import Path
import structlog

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.stage2.models import MarketState, Regime

logger = structlog.get_logger(__name__)

app = FastAPI(title="Hydra Stage 2 Dashboard")

# Store connected WebSocket clients
connected_clients: Set[WebSocket] = set()

# Latest market states
latest_states: Dict[str, dict] = {}


async def broadcast_state(state: MarketState) -> None:
    """Broadcast market state to all connected clients"""
    state_dict = state.to_flat_dict()
    latest_states[state.symbol] = state_dict
    
    message = json.dumps({
        "type": "market_state",
        "data": state_dict
    })
    
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)
    
    # Clean up disconnected clients
    for client in disconnected:
        connected_clients.discard(client)


async def broadcast_all_states() -> None:
    """Broadcast all current states to newly connected client"""
    message = json.dumps({
        "type": "all_states",
        "data": latest_states
    })
    
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info("dashboard_client_connected", total=len(connected_clients))
    
    # Send current states
    if latest_states:
        await websocket.send_text(json.dumps({
            "type": "all_states",
            "data": latest_states
        }))
    
    try:
        while True:
            # Keep connection alive, handle any client messages
            data = await websocket.receive_text()
            # Could handle client commands here
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info("dashboard_client_disconnected", total=len(connected_clients))


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the dashboard HTML"""
    return DASHBOARD_HTML


# Dashboard HTML with embedded JS
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hydra Stage 2 Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            border: 1px solid #2a2a4a;
        }
        .header h1 {
            font-size: 2rem;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        .header .status {
            font-size: 0.9rem;
            color: #888;
        }
        .status.connected { color: #00ff88; }
        .status.disconnected { color: #ff4444; }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 20px;
        }
        
        .card {
            background: #12121a;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #2a2a4a;
            transition: all 0.3s ease;
        }
        .card:hover {
            border-color: #4a4a6a;
            transform: translateY(-2px);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #2a2a4a;
        }
        .symbol {
            font-size: 1.4rem;
            font-weight: 700;
            color: #fff;
        }
        .price {
            font-size: 1.2rem;
            font-weight: 600;
            color: #00d4ff;
        }
        
        .regime-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .regime-CHOP {
            background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
            color: white;
        }
        .regime-COMPRESSION {
            background: linear-gradient(135deg, #ffaa00 0%, #ff8800 100%);
            color: #000;
        }
        .regime-EXPANSION {
            background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
            color: #000;
        }
        
        .metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }
        .metric {
            background: #1a1a2a;
            padding: 10px 12px;
            border-radius: 8px;
        }
        .metric-label {
            font-size: 0.75rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1rem;
            font-weight: 600;
            color: #fff;
        }
        .metric-value.positive { color: #00ff88; }
        .metric-value.negative { color: #ff4444; }
        .metric-value.neutral { color: #ffaa00; }
        
        .section-title {
            font-size: 0.8rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 15px 0 10px 0;
            padding-top: 15px;
            border-top: 1px solid #2a2a4a;
        }
        
        .liq-bar {
            height: 8px;
            background: #2a2a4a;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        .liq-bar-inner {
            height: 100%;
            transition: width 0.3s ease;
        }
        .liq-long { background: linear-gradient(90deg, #00ff88, #00cc66); }
        .liq-short { background: linear-gradient(90deg, #ff4444, #cc0000); }
        
        .cascade-indicator {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 8px;
        }
        .cascade-active {
            background: #ff4444;
            color: white;
            animation: pulse 1s infinite;
        }
        .exhaustion-active {
            background: #00ff88;
            color: #000;
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        .no-data {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        .no-data h2 {
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>HYDRA Stage 2 Dashboard</h1>
        <div class="status disconnected" id="status">Connecting...</div>
    </div>
    
    <div class="grid" id="grid">
        <div class="no-data">
            <h2>Waiting for data...</h2>
            <p>Market states will appear here once Stage 2 is running</p>
        </div>
    </div>

    <script>
        const grid = document.getElementById('grid');
        const status = document.getElementById('status');
        const states = {};
        
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
            return 'neutral';
        }
        
        function renderCard(data) {
            const symbol = data.symbol;
            const regime = data.regime || 'COMPRESSION';
            
            const cascadeHtml = data.liq_cascade_active ? 
                '<span class="cascade-indicator cascade-active">CASCADE</span>' : '';
            const exhaustionHtml = data.liq_exhaustion ? 
                '<span class="cascade-indicator exhaustion-active">EXHAUSTION</span>' : '';
            
            // Liquidation bar (30s window)
            const longUsd = data.liq_long_usd_30s || 0;
            const shortUsd = data.liq_short_usd_30s || 0;
            const totalLiq = longUsd + shortUsd;
            const longPct = totalLiq > 0 ? (longUsd / totalLiq * 100) : 50;
            
            // Regime scores
            const expScore = data.regime_expansion_score || 0;
            const compScore = data.regime_compression_score || 0;
            const chopActive = data.regime_chop_score > 0;
            const confidence = data.regime_confidence || 0;
            
            return `
                <div class="card" id="card-${symbol}">
                    <div class="card-header">
                        <div>
                            <span class="symbol">${symbol}</span>
                            <span class="regime-badge regime-${regime}">${regime}</span>
                            ${cascadeHtml}${exhaustionHtml}
                        </div>
                        <span class="price">$${formatPrice(data.price)}</span>
                    </div>
                    
                    <!-- KEY METRICS -->
                    <div class="section-title">Key Metrics</div>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Price Δ 5m</div>
                            <div class="metric-value ${getValueClass(data.price_change_5m)}">${formatNumber((data.price_change_5m || 0) * 100, 2)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Time in Regime</div>
                            <div class="metric-value">${formatNumber(data.time_in_regime || 0, 0)}s</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Funding Z</div>
                            <div class="metric-value ${getValueClass(data.fund_funding_z, 1)}">${formatNumber(data.fund_funding_z, 2)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">OI Δ 5m</div>
                            <div class="metric-value ${getValueClass(data.oi_oi_delta_5m * 100)}">${formatNumber(data.oi_oi_delta_5m * 100, 2)}%</div>
                        </div>
                    </div>
                    
                    <!-- REGIME SCORES -->
                    <div class="section-title">Regime Scores</div>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Expansion</div>
                            <div class="metric-value ${expScore >= 3 ? 'positive' : ''}">${expScore}/5</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Compression</div>
                            <div class="metric-value ${compScore >= 3 ? 'neutral' : ''}">${compScore}/5</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Chop Active</div>
                            <div class="metric-value ${chopActive ? 'negative' : ''}">${chopActive ? 'YES' : 'NO'}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value">${formatNumber(confidence * 100, 0)}%</div>
                        </div>
                    </div>
                    
                    <!-- ORDER FLOW -->
                    <div class="section-title">Order Flow</div>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">MOI 250ms</div>
                            <div class="metric-value ${getValueClass(data.of_moi_250ms)}">${formatNumber(data.of_moi_250ms, 3)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">MOI 1s</div>
                            <div class="metric-value ${getValueClass(data.of_moi_1s)}">${formatNumber(data.of_moi_1s, 3)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Delta Velocity</div>
                            <div class="metric-value ${getValueClass(data.of_delta_velocity)}">${formatNumber(data.of_delta_velocity, 3)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Aggression</div>
                            <div class="metric-value">${formatNumber(data.of_aggression_persistence, 2)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">MOI Std</div>
                            <div class="metric-value">${formatNumber(data.of_moi_std, 3)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Flip Rate</div>
                            <div class="metric-value">${formatNumber(data.of_moi_flip_rate, 1)}/min</div>
                        </div>
                    </div>
                    
                    <!-- ABSORPTION -->
                    <div class="section-title">Absorption</div>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Absorption Z</div>
                            <div class="metric-value ${getValueClass(data.abs_absorption_z, 1)}">${formatNumber(data.abs_absorption_z, 2)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Refill Rate</div>
                            <div class="metric-value">${formatNumber(data.abs_refill_rate, 2)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Liq Sweep</div>
                            <div class="metric-value ${data.abs_liquidity_sweep ? 'negative' : ''}">${data.abs_liquidity_sweep ? 'YES' : 'NO'}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Depth Imbal</div>
                            <div class="metric-value ${getValueClass(data.abs_depth_imbalance)}">${formatNumber(data.abs_depth_imbalance * 100, 1)}%</div>
                        </div>
                    </div>
                    
                    <!-- VOLATILITY -->
                    <div class="section-title">Volatility</div>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">ATR 5m</div>
                            <div class="metric-value">${formatNumber(data.vol_atr_5m, 4)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">ATR 1h</div>
                            <div class="metric-value">${formatNumber(data.vol_atr_1h, 4)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Vol Expansion</div>
                            <div class="metric-value">${formatNumber(data.vol_vol_expansion_ratio, 2)}x</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Vol Rank</div>
                            <div class="metric-value">${formatNumber(data.vol_vol_rank, 0)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Vol 5m</div>
                            <div class="metric-value">${formatNumber(data.vol_vol_5m * 100, 2)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Vol Regime</div>
                            <div class="metric-value">${data.vol_vol_regime || 'MID'}</div>
                        </div>
                    </div>
                    
                    <!-- STRUCTURE -->
                    <div class="section-title">Structure</div>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">POC</div>
                            <div class="metric-value">$${formatPrice(data.str_poc)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">VAH</div>
                            <div class="metric-value">$${formatPrice(data.str_vah)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">VAL</div>
                            <div class="metric-value">$${formatPrice(data.str_val)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">LVN</div>
                            <div class="metric-value">$${data.str_lvns && data.str_lvns.length > 0 ? formatPrice(data.str_lvns[0]) : 'N/A'}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">VA Width</div>
                            <div class="metric-value">$${formatNumber(data.str_value_area_width, 2)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Dist POC</div>
                            <div class="metric-value">${formatNumber(data.str_dist_poc, 2)} ATR</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Dist LVN</div>
                            <div class="metric-value">${formatNumber(data.str_dist_lvn, 2)} ATR</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Width Ratio</div>
                            <div class="metric-value">${formatNumber(data.str_value_width_ratio, 2)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Time Inside</div>
                            <div class="metric-value">${formatNumber(data.str_time_inside_value_pct, 0)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Accept Outside</div>
                            <div class="metric-value ${data.str_acceptance_outside_value ? 'positive' : ''}">${data.str_acceptance_outside_value ? 'YES' : 'NO'}</div>
                        </div>
                    </div>
                    
                    <!-- LIQUIDATIONS -->
                    <div class="section-title">Liquidations</div>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Long 30s</div>
                            <div class="metric-value positive">$${formatNumber(longUsd, 0)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Short 30s</div>
                            <div class="metric-value negative">$${formatNumber(shortUsd, 0)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Imbal 30s</div>
                            <div class="metric-value ${getValueClass(data.liq_imbalance_30s)}">${formatNumber(data.liq_imbalance_30s * 100, 0)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Total 2m</div>
                            <div class="metric-value">$${formatNumber((data.liq_long_usd_2m||0) + (data.liq_short_usd_2m||0), 0)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Total 5m</div>
                            <div class="metric-value">$${formatNumber((data.liq_long_usd_5m||0) + (data.liq_short_usd_5m||0), 0)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Cascade</div>
                            <div class="metric-value ${data.liq_cascade_active ? 'negative' : ''}">${data.liq_cascade_active ? 'ACTIVE' : 'NO'}</div>
                        </div>
                    </div>
                    <div class="liq-bar">
                        <div class="liq-bar-inner liq-long" style="width: ${longPct}%"></div>
                    </div>
                    
                    <!-- DERIVATIVES -->
                    <div class="section-title">Derivatives</div>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Funding Rate</div>
                            <div class="metric-value ${getValueClass(data.fund_rate * 10000)}">${formatNumber(data.fund_rate * 100, 4)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Funding Z</div>
                            <div class="metric-value ${getValueClass(data.fund_funding_z, 1)}">${formatNumber(data.fund_funding_z, 2)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Crowd Side</div>
                            <div class="metric-value">${data.fund_crowd_side || 'NEUTRAL'}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Ann. Rate</div>
                            <div class="metric-value">${formatNumber(data.fund_annualized_pct, 1)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">OI</div>
                            <div class="metric-value">${formatNumber(data.oi_oi, 0)}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">OI Δ 1m</div>
                            <div class="metric-value ${getValueClass(data.oi_oi_delta_1m * 100)}">${formatNumber((data.oi_oi_delta_1m || 0) * 100, 2)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">OI Δ 5m</div>
                            <div class="metric-value ${getValueClass(data.oi_oi_delta_5m * 100)}">${formatNumber(data.oi_oi_delta_5m * 100, 2)}%</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Participation</div>
                            <div class="metric-value">${data.oi_participation_type || 'NEUTRAL'}</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function updateGrid() {
            const symbols = Object.keys(states).sort();
            if (symbols.length === 0) return;
            
            grid.innerHTML = symbols.map(s => renderCard(states[s])).join('');
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
                
                if (msg.type === 'market_state') {
                    states[msg.data.symbol] = msg.data;
                    updateGrid();
                } else if (msg.type === 'all_states') {
                    Object.assign(states, msg.data);
                    updateGrid();
                }
            };
        }
        
        connect();
    </script>
</body>
</html>
"""


def run_dashboard(host: str = "0.0.0.0", port: int = 8080):
    """Run the dashboard server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


async def start_dashboard_async(host: str = "0.0.0.0", port: int = 8080):
    """Start dashboard in async context"""
    import uvicorn
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
