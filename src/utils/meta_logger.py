"""
Meta Logger for ML Model Building

Logs signal events with features and probabilities to meta.json for future model improvement.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from threading import Lock
import structlog

logger = structlog.get_logger(__name__)

# Default meta file path
META_FILE = Path("meta.json")

# Thread lock for file operations
_file_lock = Lock()


def _load_meta() -> List[Dict[str, Any]]:
    """Load existing meta data from file"""
    if not META_FILE.exists():
        return []
    
    try:
        with open(META_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("meta_load_error", error=str(e))
        return []


def _save_meta(data: List[Dict[str, Any]]) -> None:
    """Save meta data to file"""
    try:
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.error("meta_save_error", error=str(e))


def log_signal_event(
    symbol: str,
    direction: str,
    signal_names: List[str],
    vol_regime: str,
    prob_60: float,
    prob_300: float,
    model_60: str,
    model_300: str,
    features: Dict[str, float],
    price: float,
    passed_gate: bool,
    risk_tier: Optional[str] = None,
) -> None:
    """
    Log a signal event to meta.json for future model building.
    
    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        direction: Signal direction (LONG/SHORT)
        signal_names: List of signal names that fired
        vol_regime: Volatility regime (low/mid/high)
        prob_60: Model probability for 60s horizon
        prob_300: Model probability for 300s horizon
        model_60: Model name used for 60s prediction
        model_300: Model name used for 300s prediction
        features: Feature dictionary used for prediction
        price: Entry price at signal time
        passed_gate: Whether signal passed the probability gate
        risk_tier: Risk tier assigned (max/med/min) if passed
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    event = {
        "timestamp": timestamp,
        "symbol": symbol,
        "direction": direction,
        "signals": signal_names,
        "vol_regime": vol_regime,
        "price": price,
        "prob_60": round(prob_60, 4),
        "prob_300": round(prob_300, 4),
        "model_60": model_60,
        "model_300": model_300,
        "passed_gate": passed_gate,
        "risk_tier": risk_tier,
        "features": {k: round(v, 6) if isinstance(v, float) else v for k, v in features.items()},
    }
    
    with _file_lock:
        data = _load_meta()
        data.append(event)
        _save_meta(data)
    
    logger.debug(
        "meta_signal_logged",
        symbol=symbol,
        direction=direction,
        prob_60=f"{prob_60:.1%}",
        prob_300=f"{prob_300:.1%}",
        passed=passed_gate,
    )


def get_meta_stats() -> Dict[str, Any]:
    """Get statistics from meta.json"""
    data = _load_meta()
    
    if not data:
        return {"count": 0}
    
    passed = sum(1 for e in data if e.get("passed_gate", False))
    
    return {
        "count": len(data),
        "passed": passed,
        "rejected": len(data) - passed,
        "pass_rate": passed / len(data) if data else 0,
        "symbols": list(set(e.get("symbol", "") for e in data)),
        "first_event": data[0].get("timestamp") if data else None,
        "last_event": data[-1].get("timestamp") if data else None,
    }


def clear_meta() -> None:
    """Clear all meta data (use with caution)"""
    with _file_lock:
        _save_meta([])
    logger.info("meta_cleared")
