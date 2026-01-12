"""
ML Model Testing Dashboard - Real-time predictions from Stage 2 features

Loads 12 ML models (up/down × high/mid/low × 60/300) and displays
predictions for all 8 pairs continuously using Stage 2 computed features.

Features (15 total):
- 7 computed: MOI_250ms, MOI_1s, delta_velocity, AggressionPersistence, 
              absorption_z, dist_lvn, vol_5m
- 8 one-hot encoded pairs: ADAUSDT, BNBUSDT, BTCUSDT, DOGEUSDT, 
                           ETHUSDT, LTCUSDT, SOLUSDT, XRPUSDT
"""
import asyncio
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import structlog
import numpy as np
import pandas as pd

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from src.stage2.models import MarketState

logger = structlog.get_logger(__name__)

# FastAPI app
ml_app = FastAPI(title="Hydra ML Model Dashboard")

# WebSocket clients
ml_connected_clients: Set[WebSocket] = set()

# Latest states and predictions
ml_latest_states: Dict[str, dict] = {}
ml_latest_predictions: Dict[str, Dict[str, float]] = {}

# ML Models storage
ML_MODELS: Dict[str, Any] = {}

# Feature columns (must match training)
FEATURE_COLUMNS = [
    "MOI_250ms", "MOI_1s", "delta_velocity", "AggressionPersistence",
    "absorption_z", "dist_lvn", "vol_5m",
    "pair_ADAUSDT", "pair_BNBUSDT", "pair_BTCUSDT", "pair_DOGEUSDT",
    "pair_ETHUSDT", "pair_LTCUSDT", "pair_SOLUSDT", "pair_XRPUSDT"
]

PAIRS = ["ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", 
         "ETHUSDT", "LTCUSDT", "SOLUSDT", "XRPUSDT"]

DIRECTIONS = ["up", "down"]
REGIMES = ["high", "mid", "low"]
TIMES = [60, 300]


def load_ml_models(models_dir: str = "ml_models") -> Dict[str, Any]:
    """Load all 12 ML models from pickle files"""
    models_path = Path(models_dir)
    models = {}
    
    for direction in DIRECTIONS:
        for regime in REGIMES:
            for time_val in TIMES:
                model_name = f"models_{direction}_{regime}_{time_val}"
                model_file = models_path / f"{model_name}.pkl"
                
                if model_file.exists():
                    try:
                        with open(model_file, 'rb') as f:
                            models[model_name] = pickle.load(f)
                        logger.info("model_loaded", model=model_name)
                    except Exception as e:
                        logger.error("model_load_failed", model=model_name, error=str(e))
                else:
                    logger.warning("model_not_found", model=model_name, path=str(model_file))
    
    return models


def extract_features(state: MarketState) -> np.ndarray:
    """Extract the 7 computed features from MarketState"""
    return np.array([
        state.order_flow.moi_250ms,        # MOI_250ms
        state.order_flow.moi_1s,            # MOI_1s
        state.order_flow.delta_velocity,    # delta_velocity
        state.order_flow.aggression_persistence,  # AggressionPersistence
        state.absorption.absorption_z,      # absorption_z
        state.structure.dist_lvn,           # dist_lvn
        state.volatility.vol_5m,            # vol_5m
    ])


def create_feature_vector(state: MarketState) -> pd.DataFrame:
    """Create full 15-feature DataFrame with one-hot encoded pair"""
    # Get 7 computed features
    features = extract_features(state)
    
    # Create one-hot encoding for pair
    pair_one_hot = np.zeros(8)
    symbol = state.symbol
    if symbol in PAIRS:
        pair_idx = PAIRS.index(symbol)
        pair_one_hot[pair_idx] = 1.0
    
    # Combine: 7 features + 8 pair one-hot = 15 features
    full_features = np.concatenate([features, pair_one_hot])
    
    # Return as DataFrame with proper column names
    return pd.DataFrame([full_features], columns=FEATURE_COLUMNS)


def get_predictions(state: MarketState) -> Dict[str, float]:
    """Get predictions from all 12 model ensembles for a given state"""
    if not ML_MODELS:
        return {}
    
    feature_vector = create_feature_vector(state)
    predictions = {}
    
    for model_name, model_ensemble in ML_MODELS.items():
        try:
            # Models are stored as lists (ensemble of 5 LGBMRegressors)
            # Use the last model in the list as per user's testing approach
            if isinstance(model_ensemble, list):
                pred = model_ensemble[-1].predict(feature_vector)[0]
                predictions[model_name] = float(pred)
            elif hasattr(model_ensemble, 'predict_proba'):
                prob = model_ensemble.predict_proba(feature_vector)[0]
                predictions[model_name] = float(prob[1]) if len(prob) > 1 else float(prob[0])
            else:
                pred = model_ensemble.predict(feature_vector)[0]
                predictions[model_name] = float(pred)
        except Exception as e:
            logger.error("prediction_failed", model=model_name, error=str(e))
            predictions[model_name] = 0.0
    
    return predictions


def get_vol_regime(state: MarketState) -> str:
    """Get volatility regime (high/mid/low) from state"""
    vol_regime = state.volatility.vol_regime.lower()
    if vol_regime in ["high", "mid", "low"]:
        return vol_regime
    # Fallback based on vol_rank
    if state.volatility.vol_rank > 70:
        return "high"
    elif state.volatility.vol_rank < 30:
        return "low"
    return "mid"


async def ml_broadcast_state(state: MarketState) -> None:
    """Broadcast market state and predictions to all connected clients"""
    # Get predictions from all models
    predictions = get_predictions(state)
    
    # Get current volatility regime for highlighting relevant models
    vol_regime = get_vol_regime(state)
    
    # Extract features for display
    features = {
        "MOI_250ms": state.order_flow.moi_250ms,
        "MOI_1s": state.order_flow.moi_1s,
        "delta_velocity": state.order_flow.delta_velocity,
        "AggressionPersistence": state.order_flow.aggression_persistence,
        "absorption_z": state.absorption.absorption_z,
        "dist_lvn": state.structure.dist_lvn,
        "vol_5m": state.volatility.vol_5m,
    }
    
    state_dict = {
        "symbol": state.symbol,
        "price": state.price,
        "timestamp_ms": state.timestamp_ms,
        "vol_regime": vol_regime,
        "features": features,
        "predictions": predictions,
    }
    
    ml_latest_states[state.symbol] = state_dict
    ml_latest_predictions[state.symbol] = predictions
    
    message = json.dumps({
        "type": "ml_state",
        "data": state_dict
    })
    
    disconnected = set()
    for client in ml_connected_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)
    
    for client in disconnected:
        ml_connected_clients.discard(client)


@ml_app.on_event("startup")
async def startup_event():
    """Load ML models on startup"""
    global ML_MODELS
    ML_MODELS = load_ml_models()
    logger.info("ml_models_loaded", count=len(ML_MODELS))


@ml_app.websocket("/ws")
async def ml_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time ML predictions"""
    await websocket.accept()
    ml_connected_clients.add(websocket)
    logger.info("ml_dashboard_client_connected", total=len(ml_connected_clients))
    
    # Send current states
    if ml_latest_states:
        await websocket.send_text(json.dumps({
            "type": "all_states",
            "data": ml_latest_states
        }))
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ml_connected_clients.discard(websocket)
        logger.info("ml_dashboard_client_disconnected", total=len(ml_connected_clients))


@ml_app.get("/", response_class=HTMLResponse)
async def get_ml_dashboard():
    """Serve the ML dashboard HTML"""
    return ML_DASHBOARD_HTML


ML_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hydra ML Model Dashboard</title>
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
            margin-bottom: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            border: 1px solid #2a2a4a;
        }
        .header h1 {
            font-size: 1.8rem;
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
        
        .main-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }
        
        .pair-card {
            background: #12121a;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #2a2a4a;
        }
        
        .pair-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #2a2a4a;
        }
        .pair-symbol {
            font-size: 1.3rem;
            font-weight: 700;
            color: #fff;
        }
        .pair-price {
            font-size: 1.1rem;
            font-weight: 600;
            color: #00d4ff;
        }
        .vol-regime {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .vol-high { background: #ff4444; color: white; }
        .vol-mid { background: #ffaa00; color: #000; }
        .vol-low { background: #00cc66; color: #000; }
        
        .features-section {
            margin-bottom: 12px;
        }
        .section-title {
            font-size: 0.75rem;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .features-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
        }
        .feature {
            background: #1a1a2a;
            padding: 8px;
            border-radius: 6px;
            text-align: center;
        }
        .feature-name {
            font-size: 0.65rem;
            color: #888;
            margin-bottom: 2px;
        }
        .feature-value {
            font-size: 0.9rem;
            font-weight: 600;
            color: #fff;
        }
        .feature-value.positive { color: #00ff88; }
        .feature-value.negative { color: #ff4444; }
        
        .predictions-section {
            margin-top: 12px;
        }
        .predictions-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        .direction-block {
            background: #1a1a2a;
            border-radius: 8px;
            padding: 10px;
        }
        .direction-title {
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 8px;
            padding-bottom: 6px;
            border-bottom: 1px solid #2a2a4a;
        }
        .direction-up .direction-title { color: #00ff88; }
        .direction-down .direction-title { color: #ff4444; }
        
        .model-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 0;
            font-size: 0.8rem;
        }
        .model-row.highlight {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
            padding: 4px 6px;
            margin: 2px -6px;
        }
        .model-name {
            color: #888;
        }
        .model-prob {
            font-weight: 600;
            font-family: 'Monaco', 'Consolas', monospace;
        }
        .prob-high { color: #00ff88; }
        .prob-mid { color: #ffaa00; }
        .prob-low { color: #888; }
        
        .prob-bar {
            width: 60px;
            height: 6px;
            background: #2a2a4a;
            border-radius: 3px;
            overflow: hidden;
            margin-left: 8px;
        }
        .prob-bar-inner {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
        }
        .prob-bar-up { background: linear-gradient(90deg, #00cc66, #00ff88); }
        .prob-bar-down { background: linear-gradient(90deg, #cc0000, #ff4444); }
        
        .model-group {
            display: flex;
            align-items: center;
        }
        
        .no-data {
            text-align: center;
            padding: 60px 20px;
            color: #666;
            grid-column: 1 / -1;
        }
        
        .summary-section {
            margin-top: 20px;
            background: #12121a;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #2a2a4a;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            gap: 10px;
            margin-top: 10px;
        }
        .summary-cell {
            text-align: center;
            padding: 10px;
            background: #1a1a2a;
            border-radius: 8px;
        }
        .summary-symbol {
            font-size: 0.75rem;
            color: #888;
            margin-bottom: 4px;
        }
        .summary-signal {
            font-size: 1rem;
            font-weight: 700;
        }
        .signal-long { color: #00ff88; }
        .signal-short { color: #ff4444; }
        .signal-neutral { color: #888; }
    </style>
</head>
<body>
    <div class="header">
        <h1>HYDRA ML Model Testing Dashboard</h1>
        <div class="status disconnected" id="status">Connecting...</div>
    </div>
    
    <div class="main-grid" id="grid">
        <div class="no-data">
            <h2>Waiting for data...</h2>
            <p>ML predictions will appear once Stage 2 is running</p>
        </div>
    </div>
    
    <div class="summary-section">
        <div class="section-title">Signal Summary (Current Regime Models)</div>
        <div class="summary-grid" id="summary-grid"></div>
    </div>

    <script>
        const grid = document.getElementById('grid');
        const summaryGrid = document.getElementById('summary-grid');
        const status = document.getElementById('status');
        const states = {};
        
        const PAIRS = ['ADAUSDT', 'BNBUSDT', 'BTCUSDT', 'DOGEUSDT', 
                       'ETHUSDT', 'LTCUSDT', 'SOLUSDT', 'XRPUSDT'];
        const REGIMES = ['high', 'mid', 'low'];
        const TIMES = [60, 300];
        
        function formatNumber(n, decimals = 3) {
            if (n === undefined || n === null) return '-';
            return n.toFixed(decimals);
        }
        
        function formatPrice(p) {
            if (!p) return '-';
            if (p >= 1000) return p.toFixed(2);
            if (p >= 1) return p.toFixed(4);
            return p.toFixed(6);
        }
        
        function getProbClass(prob) {
            if (prob >= 0.6) return 'prob-high';
            if (prob >= 0.4) return 'prob-mid';
            return 'prob-low';
        }
        
        function getValueClass(value) {
            if (value > 0.01) return 'positive';
            if (value < -0.01) return 'negative';
            return '';
        }
        
        function renderPairCard(data) {
            const symbol = data.symbol;
            const volRegime = data.vol_regime || 'mid';
            const features = data.features || {};
            const predictions = data.predictions || {};
            
            // Features display
            const featureItems = [
                { name: 'MOI 250ms', value: features.MOI_250ms },
                { name: 'MOI 1s', value: features.MOI_1s },
                { name: 'Delta Vel', value: features.delta_velocity },
                { name: 'Aggression', value: features.AggressionPersistence },
                { name: 'Absorb Z', value: features.absorption_z },
                { name: 'Dist LVN', value: features.dist_lvn },
                { name: 'Vol 5m', value: features.vol_5m, decimals: 6 },
            ];
            
            const featuresHtml = featureItems.map(f => `
                <div class="feature">
                    <div class="feature-name">${f.name}</div>
                    <div class="feature-value ${getValueClass(f.value)}">${formatNumber(f.value, f.decimals || 3)}</div>
                </div>
            `).join('');
            
            // Model predictions grouped by direction
            function renderModelGroup(direction) {
                let html = '';
                for (const regime of REGIMES) {
                    for (const time of TIMES) {
                        const modelName = `models_${direction}_${regime}_${time}`;
                        const prob = predictions[modelName] || 0;
                        const isCurrentRegime = regime === volRegime;
                        const highlightClass = isCurrentRegime ? 'highlight' : '';
                        const barClass = direction === 'up' ? 'prob-bar-up' : 'prob-bar-down';
                        
                        html += `
                            <div class="model-row ${highlightClass}">
                                <span class="model-name">${regime.toUpperCase()} ${time}s</span>
                                <div class="model-group">
                                    <span class="model-prob ${getProbClass(prob)}">${(prob * 100).toFixed(1)}%</span>
                                    <div class="prob-bar">
                                        <div class="prob-bar-inner ${barClass}" style="width: ${prob * 100}%"></div>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                }
                return html;
            }
            
            return `
                <div class="pair-card" id="card-${symbol}">
                    <div class="pair-header">
                        <div>
                            <span class="pair-symbol">${symbol}</span>
                            <span class="vol-regime vol-${volRegime}">${volRegime}</span>
                        </div>
                        <span class="pair-price">$${formatPrice(data.price)}</span>
                    </div>
                    
                    <div class="features-section">
                        <div class="section-title">Input Features (7)</div>
                        <div class="features-grid">
                            ${featuresHtml}
                        </div>
                    </div>
                    
                    <div class="predictions-section">
                        <div class="section-title">Model Predictions (12 models)</div>
                        <div class="predictions-grid">
                            <div class="direction-block direction-up">
                                <div class="direction-title">↑ LONG Models</div>
                                ${renderModelGroup('up')}
                            </div>
                            <div class="direction-block direction-down">
                                <div class="direction-title">↓ SHORT Models</div>
                                ${renderModelGroup('down')}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        function renderSummary() {
            let html = '';
            for (const symbol of PAIRS) {
                const data = states[symbol];
                if (!data) {
                    html += `
                        <div class="summary-cell">
                            <div class="summary-symbol">${symbol}</div>
                            <div class="summary-signal signal-neutral">-</div>
                        </div>
                    `;
                    continue;
                }
                
                const volRegime = data.vol_regime || 'mid';
                const predictions = data.predictions || {};
                
                // Get predictions for current regime (average of 60s and 300s)
                const upModel60 = `models_up_${volRegime}_60`;
                const upModel300 = `models_up_${volRegime}_300`;
                const downModel60 = `models_down_${volRegime}_60`;
                const downModel300 = `models_down_${volRegime}_300`;
                
                const upProb = ((predictions[upModel60] || 0) + (predictions[upModel300] || 0)) / 2;
                const downProb = ((predictions[downModel60] || 0) + (predictions[downModel300] || 0)) / 2;
                
                let signalClass = 'signal-neutral';
                let signalText = '—';
                
                if (upProb > 0.6 && upProb > downProb) {
                    signalClass = 'signal-long';
                    signalText = 'LONG';
                } else if (downProb > 0.6 && downProb > upProb) {
                    signalClass = 'signal-short';
                    signalText = 'SHORT';
                }
                
                html += `
                    <div class="summary-cell">
                        <div class="summary-symbol">${symbol}</div>
                        <div class="summary-signal ${signalClass}">${signalText}</div>
                        <div style="font-size: 0.7rem; color: #666; margin-top: 4px;">
                            ↑${(upProb * 100).toFixed(0)}% ↓${(downProb * 100).toFixed(0)}%
                        </div>
                    </div>
                `;
            }
            summaryGrid.innerHTML = html;
        }
        
        function updateGrid() {
            const symbols = Object.keys(states).sort();
            if (symbols.length === 0) return;
            
            grid.innerHTML = symbols.map(s => renderPairCard(states[s])).join('');
            renderSummary();
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
                
                if (msg.type === 'ml_state') {
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


def run_ml_dashboard(host: str = "0.0.0.0", port: int = 8081):
    """Run the ML dashboard server"""
    import uvicorn
    uvicorn.run(ml_app, host=host, port=port)


async def start_ml_dashboard_async(host: str = "0.0.0.0", port: int = 8081):
    """Start ML dashboard in async context"""
    import uvicorn
    config = uvicorn.Config(ml_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
