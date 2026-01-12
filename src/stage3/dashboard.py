"""
Real-time Dashboard for Stage 3 Thesis Engine
WebSocket-based UI showing all pairs with regime + thesis
"""
import asyncio
import json
from typing import Dict, Set, Any
import structlog

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logger = structlog.get_logger(__name__)

app = FastAPI(title="Hydra Stage 3 Dashboard")

# Store connected WebSocket clients
connected_clients: Set[WebSocket] = set()

# Latest combined states (MarketState + Thesis)
latest_states: Dict[str, Dict[str, Any]] = {}


async def broadcast_combined_state(symbol: str, state: Dict[str, Any]) -> None:
    """Broadcast combined market state + thesis to all connected clients"""
    latest_states[symbol] = state
    
    message = json.dumps({
        "type": "combined_state",
        "data": state
    })
    
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)
    
    for client in disconnected:
        connected_clients.discard(client)


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
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info("dashboard_client_disconnected", total=len(connected_clients))


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the dashboard HTML"""
    return DASHBOARD_HTML


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hydra Stage 3 - Thesis Engine</title>
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
            background: linear-gradient(90deg, #00d4ff, #7c3aed, #ff6b6b);
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
            grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
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
        .card.thesis-allowed {
            border-color: #00ff88;
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
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
        
        .badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-left: 8px;
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
        
        /* THESIS SECTION */
        .thesis-section {
            background: #1a1a2a;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .thesis-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .thesis-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #fff;
        }
        .thesis-badge {
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 0.9rem;
            text-transform: uppercase;
        }
        .thesis-LONG {
            background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
            color: #000;
            animation: pulse-green 2s infinite;
        }
        .thesis-SHORT {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
            color: #fff;
            animation: pulse-red 2s infinite;
        }
        .thesis-NONE {
            background: #3a3a4a;
            color: #888;
        }
        .thesis-blocked {
            background: #2a2a3a;
            color: #666;
        }
        
        @keyframes pulse-green {
            0%, 100% { box-shadow: 0 0 10px rgba(0, 255, 136, 0.5); }
            50% { box-shadow: 0 0 25px rgba(0, 255, 136, 0.8); }
        }
        @keyframes pulse-red {
            0%, 100% { box-shadow: 0 0 10px rgba(255, 107, 107, 0.5); }
            50% { box-shadow: 0 0 25px rgba(255, 107, 107, 0.8); }
        }
        
        .thesis-details {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 12px;
        }
        .thesis-stat {
            text-align: center;
            padding: 8px;
            background: #12121a;
            border-radius: 6px;
        }
        .thesis-stat-label {
            font-size: 0.7rem;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .thesis-stat-value {
            font-size: 1rem;
            font-weight: 600;
            color: #fff;
        }
        
        .signals-list {
            background: #12121a;
            border-radius: 6px;
            padding: 10px;
        }
        .signals-title {
            font-size: 0.75rem;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .signal-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 10px;
            margin-bottom: 4px;
            background: #1a1a2a;
            border-radius: 4px;
            font-size: 0.85rem;
        }
        .signal-item:last-child {
            margin-bottom: 0;
        }
        .signal-name {
            color: #ccc;
        }
        .signal-confidence {
            font-weight: 600;
        }
        .signal-LONG { color: #00ff88; }
        .signal-SHORT { color: #ff6b6b; }
        
        .veto-reason {
            background: #2a1a1a;
            border: 1px solid #4a2a2a;
            border-radius: 6px;
            padding: 10px;
            margin-top: 10px;
            font-size: 0.85rem;
            color: #ff8888;
        }
        
        /* METRICS GRID */
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .metric {
            background: #1a1a2a;
            padding: 10px 12px;
            border-radius: 8px;
        }
        .metric-label {
            font-size: 0.7rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 0.95rem;
            font-weight: 600;
            color: #fff;
        }
        .metric-value.positive { color: #00ff88; }
        .metric-value.negative { color: #ff4444; }
        .metric-value.neutral { color: #ffaa00; }
        
        .section-title {
            font-size: 0.75rem;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 15px 0 10px 0;
            padding-top: 15px;
            border-top: 1px solid #2a2a4a;
        }
        
        .no-data {
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }
        
        /* Collapsible sections */
        .collapsible {
            cursor: pointer;
            user-select: none;
        }
        .collapsible::after {
            content: ' ▼';
            font-size: 0.7rem;
        }
        .collapsible.collapsed::after {
            content: ' ▶';
        }
        .collapsible-content {
            display: block;
        }
        .collapsible-content.hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>HYDRA Stage 3 - Thesis Engine</h1>
        <div class="status disconnected" id="status">Connecting...</div>
    </div>
    
    <div class="grid" id="grid">
        <div class="no-data">
            <h2>Waiting for data...</h2>
            <p>Thesis signals will appear here once Stage 3 is running</p>
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
        
        function formatPct(n) {
            if (n === undefined || n === null) return '-';
            return (n * 100).toFixed(2) + '%';
        }
        
        function getValueClass(value, threshold = 0) {
            if (value > threshold) return 'positive';
            if (value < -threshold) return 'negative';
            return 'neutral';
        }
        
        function parseSignals(signalsStr) {
            if (!signalsStr) return [];
            return signalsStr.split(', ').filter(s => s.length > 0);
        }
        
        function renderCard(data) {
            const symbol = data.symbol;
            const regime = data.regime || 'COMPRESSION';
            const thesisAllowed = data.thesis_allowed;
            const thesisDirection = data.thesis_direction || 'NONE';
            const thesisStrength = data.thesis_strength || 0;
            const vetoReason = data.thesis_veto_reason;
            
            const longSignals = parseSignals(data.thesis_long_signals);
            const shortSignals = parseSignals(data.thesis_short_signals);
            const allSignals = [...longSignals.map(s => ({name: s, dir: 'LONG'})), 
                               ...shortSignals.map(s => ({name: s, dir: 'SHORT'}))];
            
            const cardClass = thesisAllowed ? 'card thesis-allowed' : 'card';
            const thesisBadgeClass = thesisAllowed ? `thesis-badge thesis-${thesisDirection}` : 'thesis-badge thesis-blocked';
            const thesisBadgeText = thesisAllowed ? thesisDirection : 'BLOCKED';
            
            // Signals HTML
            let signalsHtml = '';
            if (allSignals.length > 0) {
                signalsHtml = `
                    <div class="signals-list">
                        <div class="signals-title">Active Signals</div>
                        ${allSignals.map(s => `
                            <div class="signal-item">
                                <span class="signal-name">${s.name}</span>
                                <span class="signal-confidence signal-${s.dir}">${s.dir}</span>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
            
            // Veto HTML
            const vetoHtml = vetoReason ? `<div class="veto-reason">⚠️ ${vetoReason}</div>` : '';
            
            return `
                <div class="${cardClass}" id="card-${symbol}">
                    <div class="card-header">
                        <div>
                            <span class="symbol">${symbol}</span>
                            <span class="badge regime-${regime}">${regime}</span>
                        </div>
                        <span class="price">$${formatPrice(data.price)}</span>
                    </div>
                    
                    <!-- THESIS SECTION -->
                    <div class="thesis-section">
                        <div class="thesis-header">
                            <span class="thesis-title">Thesis</span>
                            <span class="${thesisBadgeClass}">${thesisBadgeText}</span>
                        </div>
                        <div class="thesis-details">
                            <div class="thesis-stat">
                                <div class="thesis-stat-label">Strength</div>
                                <div class="thesis-stat-value">${formatNumber(thesisStrength, 2)}</div>
                            </div>
                            <div class="thesis-stat">
                                <div class="thesis-stat-label">Signals</div>
                                <div class="thesis-stat-value">${data.thesis_signal_count || 0}</div>
                            </div>
                            <div class="thesis-stat">
                                <div class="thesis-stat-label">Time in Regime</div>
                                <div class="thesis-stat-value">${formatNumber(data.thesis_time_in_regime || 0, 0)}s</div>
                            </div>
                        </div>
                        ${signalsHtml}
                        ${vetoHtml}
                    </div>
                    
                    <!-- THESIS INPUTS -->
                    <div class="section-title collapsible" onclick="toggleSection(this)">Thesis Inputs</div>
                    <div class="collapsible-content">
                        <div class="metrics">
                            <div class="metric">
                                <div class="metric-label">Funding Z</div>
                                <div class="metric-value ${getValueClass(data.fund_funding_z, 1)}">${formatNumber(data.fund_funding_z, 2)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">OI Δ 5m</div>
                                <div class="metric-value ${getValueClass(data.oi_oi_delta_5m * 100)}">${formatPct(data.oi_oi_delta_5m)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Price Δ 5m</div>
                                <div class="metric-value ${getValueClass(data.thesis_price_change_5m)}">${formatPct(data.thesis_price_change_5m || 0)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Liq Imbalance</div>
                                <div class="metric-value ${getValueClass(data.liq_imbalance_2m)}">${formatNumber((data.liq_imbalance_2m || 0) * 100, 0)}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Absorption Z</div>
                                <div class="metric-value ${getValueClass(data.abs_absorption_z, 1)}">${formatNumber(data.abs_absorption_z, 2)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Vol Rank</div>
                                <div class="metric-value">${formatNumber(data.vol_vol_rank, 0)}%</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- ORDER FLOW -->
                    <div class="section-title collapsible collapsed" onclick="toggleSection(this)">Order Flow</div>
                    <div class="collapsible-content hidden">
                        <div class="metrics">
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
                                <div class="metric-label">Flip Rate</div>
                                <div class="metric-value">${formatNumber(data.of_moi_flip_rate, 1)}/min</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- DERIVATIVES -->
                    <div class="section-title collapsible collapsed" onclick="toggleSection(this)">Derivatives</div>
                    <div class="collapsible-content hidden">
                        <div class="metrics">
                            <div class="metric">
                                <div class="metric-label">Funding Rate</div>
                                <div class="metric-value ${getValueClass(data.fund_rate * 10000)}">${formatNumber(data.fund_rate * 100, 4)}%</div>
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
                                <div class="metric-label">Participation</div>
                                <div class="metric-value">${data.oi_participation_type || 'NEUTRAL'}</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- LIQUIDATIONS -->
                    <div class="section-title collapsible collapsed" onclick="toggleSection(this)">Liquidations</div>
                    <div class="collapsible-content hidden">
                        <div class="metrics">
                            <div class="metric">
                                <div class="metric-label">Long 30s</div>
                                <div class="metric-value positive">$${formatNumber(data.liq_long_usd_30s || 0, 0)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Short 30s</div>
                                <div class="metric-value negative">$${formatNumber(data.liq_short_usd_30s || 0, 0)}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Cascade</div>
                                <div class="metric-value ${data.liq_cascade_active ? 'negative' : ''}">${data.liq_cascade_active ? 'ACTIVE' : 'NO'}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Exhaustion</div>
                                <div class="metric-value ${data.liq_exhaustion ? 'positive' : ''}">${data.liq_exhaustion ? 'YES' : 'NO'}</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- REGIME SCORES -->
                    <div class="section-title collapsible collapsed" onclick="toggleSection(this)">Regime Scores</div>
                    <div class="collapsible-content hidden">
                        <div class="metrics">
                            <div class="metric">
                                <div class="metric-label">Expansion</div>
                                <div class="metric-value ${(data.regime_expansion_score || 0) >= 3 ? 'positive' : ''}">${data.regime_expansion_score || 0}/5</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Compression</div>
                                <div class="metric-value ${(data.regime_compression_score || 0) >= 3 ? 'neutral' : ''}">${data.regime_compression_score || 0}/5</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Chop Active</div>
                                <div class="metric-value ${data.regime_chop_score > 0 ? 'negative' : ''}">${data.regime_chop_score > 0 ? 'YES' : 'NO'}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value">${formatNumber((data.regime_confidence || 0) * 100, 0)}%</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function toggleSection(element) {
            element.classList.toggle('collapsed');
            const content = element.nextElementSibling;
            content.classList.toggle('hidden');
        }
        
        function updateGrid() {
            const symbols = Object.keys(states).sort();
            if (symbols.length === 0) return;
            
            // Sort by thesis allowed first, then by symbol
            symbols.sort((a, b) => {
                const aAllowed = states[a].thesis_allowed ? 1 : 0;
                const bAllowed = states[b].thesis_allowed ? 1 : 0;
                if (bAllowed !== aAllowed) return bAllowed - aAllowed;
                return a.localeCompare(b);
            });
            
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
                
                if (msg.type === 'combined_state') {
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
