"""
ML Model Training Script
========================

Train regime and bias detection models using historical data.

Usage:
    python -m src.ml.train_models --data-path data/training_data.parquet

Data Requirements:
- 1H OHLCV bars for each symbol
- Funding rate history
- OI history  
- Liquidation data
- Labels: regime and bias at each timestamp
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import structlog

from src.ml.regime_model import RegimeMLModel, RegimeFeatures, RegimeLabel
from src.ml.bias_model import BiasMLModel, BiasFeatures, BiasLabel

logger = structlog.get_logger(__name__)


def label_regime(
    bars: pd.DataFrame,
    lookforward_hours: int = 4,
    trend_threshold: float = 0.02,
    volatility_threshold: float = 0.03,
) -> pd.Series:
    """
    Label regime based on forward price action.
    
    Rules:
    - TRENDING_UP: Forward return > +threshold, low volatility
    - TRENDING_DOWN: Forward return < -threshold, low volatility
    - RANGING: Abs forward return < threshold/2, low volatility
    - VOLATILE: High volatility regardless of direction
    - BREAKOUT: Move > threshold after period of low volatility
    """
    labels = []
    
    for i in range(len(bars) - lookforward_hours):
        current_close = bars.iloc[i]['close']
        future_closes = bars.iloc[i+1:i+lookforward_hours+1]['close']
        future_highs = bars.iloc[i+1:i+lookforward_hours+1]['high']
        future_lows = bars.iloc[i+1:i+lookforward_hours+1]['low']
        
        forward_return = (future_closes.iloc[-1] - current_close) / current_close
        max_return = (future_highs.max() - current_close) / current_close
        min_return = (future_lows.min() - current_close) / current_close
        range_pct = (future_highs.max() - future_lows.min()) / current_close
        
        # Check past volatility
        if i >= 24:
            past_closes = bars.iloc[i-24:i]['close']
            past_volatility = past_closes.pct_change().std()
        else:
            past_volatility = 0.01
        
        # Label
        if range_pct > volatility_threshold:
            label = RegimeLabel.VOLATILE
        elif past_volatility < 0.005 and abs(forward_return) > trend_threshold:
            label = RegimeLabel.BREAKOUT
        elif forward_return > trend_threshold:
            label = RegimeLabel.TRENDING_UP
        elif forward_return < -trend_threshold:
            label = RegimeLabel.TRENDING_DOWN
        else:
            label = RegimeLabel.RANGING
        
        labels.append(label.value)
    
    # Pad end with last label
    labels.extend([labels[-1]] * lookforward_hours)
    
    return pd.Series(labels, index=bars.index)


def label_bias(
    bars: pd.DataFrame,
    lookforward_hours: int = 4,
    bullish_threshold: float = 0.01,
    bearish_threshold: float = -0.01,
) -> pd.Series:
    """
    Label bias based on forward returns.
    
    Rules:
    - BULLISH: Forward return > +1%
    - BEARISH: Forward return < -1%
    - NEUTRAL: Between thresholds
    """
    labels = []
    
    for i in range(len(bars) - lookforward_hours):
        current_close = bars.iloc[i]['close']
        future_close = bars.iloc[i+lookforward_hours]['close']
        forward_return = (future_close - current_close) / current_close
        
        if forward_return > bullish_threshold:
            label = BiasLabel.BULLISH
        elif forward_return < bearish_threshold:
            label = BiasLabel.BEARISH
        else:
            label = BiasLabel.NEUTRAL
        
        labels.append(label.value)
    
    labels.extend([labels[-1]] * lookforward_hours)
    
    return pd.Series(labels, index=bars.index)


def prepare_regime_features(
    bars: pd.DataFrame,
    oi_data: Optional[pd.DataFrame] = None,
    funding_data: Optional[pd.DataFrame] = None,
    liq_data: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Prepare feature matrix for regime model.
    
    Args:
        bars: DataFrame with OHLCV columns
        oi_data: OI data aligned to bars
        funding_data: Funding data aligned to bars
        liq_data: Liquidation data aligned to bars
        
    Returns:
        Feature matrix (n_samples, n_features)
    """
    model = RegimeMLModel()
    features_list = []
    
    for i in range(50, len(bars)):
        # Get bar window
        bar_window = bars.iloc[i-50:i].to_dict('records')
        current_price = bars.iloc[i]['close']
        
        # Get auxiliary data if available
        oi_dict = None
        funding_dict = None
        liq_dict = None
        
        if oi_data is not None and i < len(oi_data):
            oi_dict = {
                'change_1h': oi_data.iloc[i].get('change_1h', 0),
                'change_4h': oi_data.iloc[i].get('change_4h', 0),
                'change_24h': oi_data.iloc[i].get('change_24h', 0),
            }
        
        if funding_data is not None and i < len(funding_data):
            funding_dict = {
                'rate': funding_data.iloc[i].get('rate', 0),
                'z_score': funding_data.iloc[i].get('z_score', 0),
            }
        
        if liq_data is not None and i < len(liq_data):
            liq_dict = {
                'imbalance_1h': liq_data.iloc[i].get('imbalance_1h', 0),
                'imbalance_4h': liq_data.iloc[i].get('imbalance_4h', 0),
                'cascade_active': liq_data.iloc[i].get('cascade_active', False),
            }
        
        features = model.extract_features(
            bars_1h=bar_window,
            current_price=current_price,
            oi_data=oi_dict,
            funding_data=funding_dict,
            liquidation_data=liq_dict,
        )
        
        features_list.append(features.to_array())
    
    return np.array(features_list)


def prepare_bias_features(
    bars: pd.DataFrame,
    oi_data: Optional[pd.DataFrame] = None,
    funding_history: Optional[List[Dict]] = None,
    liq_data: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Prepare feature matrix for bias model."""
    model = BiasMLModel()
    features_list = []
    
    for i in range(50, len(bars)):
        bar_window = bars.iloc[i-50:i].to_dict('records')
        current_price = bars.iloc[i]['close']
        
        # Get funding history window
        funding_hist = None
        if funding_history:
            funding_hist = funding_history[max(0, i-30):i]
        
        oi_hist = None
        if oi_data is not None:
            oi_hist = oi_data.iloc[max(0, i-168):i].to_dict('records')
        
        liq_dict = None
        if liq_data is not None and i < len(liq_data):
            liq_dict = liq_data.iloc[i].to_dict()
        
        features = model.extract_features(
            bars_1h=bar_window,
            current_price=current_price,
            funding_history=funding_hist,
            oi_history=oi_hist,
            liquidation_data=liq_dict,
        )
        
        features_list.append(features.to_array())
    
    return np.array(features_list)


def train_regime_model(
    bars: pd.DataFrame,
    oi_data: Optional[pd.DataFrame] = None,
    funding_data: Optional[pd.DataFrame] = None,
    liq_data: Optional[pd.DataFrame] = None,
    output_path: Path = Path("models/regime_model.pkl"),
) -> Dict:
    """
    Train regime detection model.
    
    Args:
        bars: OHLCV DataFrame
        oi_data: OI data
        funding_data: Funding data
        liq_data: Liquidation data
        output_path: Path to save model
        
    Returns:
        Training metrics
    """
    logger.info("preparing_regime_features", n_bars=len(bars))
    
    # Label data
    labels = label_regime(bars)
    
    # Extract features
    X = prepare_regime_features(bars, oi_data, funding_data, liq_data)
    y = labels.values[50:]  # Align with features
    
    # Ensure same length
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    
    logger.info("training_regime_model", n_samples=len(X), n_features=X.shape[1])
    
    # Train
    model = RegimeMLModel()
    metrics = model.train(X, y)
    
    # Save
    model.save(output_path)
    
    return metrics


def train_bias_model(
    bars: pd.DataFrame,
    oi_data: Optional[pd.DataFrame] = None,
    funding_history: Optional[List[Dict]] = None,
    liq_data: Optional[pd.DataFrame] = None,
    output_path: Path = Path("models/bias_model.pkl"),
) -> Dict:
    """
    Train bias detection model.
    
    Args:
        bars: OHLCV DataFrame
        oi_data: OI data
        funding_history: Funding history list
        liq_data: Liquidation data
        output_path: Path to save model
        
    Returns:
        Training metrics
    """
    logger.info("preparing_bias_features", n_bars=len(bars))
    
    # Label data
    labels = label_bias(bars)
    
    # Extract features
    X = prepare_bias_features(bars, oi_data, funding_history, liq_data)
    y = labels.values[50:]
    
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    
    logger.info("training_bias_model", n_samples=len(X), n_features=X.shape[1])
    
    # Train
    model = BiasMLModel()
    metrics = model.train(X, y)
    
    # Save
    model.save(output_path)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--data-path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory")
    parser.add_argument("--model", type=str, choices=["regime", "bias", "both"], default="both")
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if data_path.suffix == '.parquet':
        data = pd.read_parquet(data_path)
    elif data_path.suffix == '.csv':
        data = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    # Extract components
    bars = data[['open', 'high', 'low', 'close', 'volume']].copy()
    
    oi_data = None
    if 'oi' in data.columns:
        oi_data = data[['oi']].copy()
        oi_data['change_1h'] = oi_data['oi'].pct_change(1)
        oi_data['change_4h'] = oi_data['oi'].pct_change(4)
        oi_data['change_24h'] = oi_data['oi'].pct_change(24)
    
    funding_data = None
    if 'funding_rate' in data.columns:
        funding_data = data[['funding_rate']].copy()
        funding_data['rate'] = data['funding_rate']
        funding_data['z_score'] = (
            (funding_data['rate'] - funding_data['rate'].rolling(720).mean()) /
            funding_data['rate'].rolling(720).std()
        )
    
    liq_data = None
    if 'liq_long_1h' in data.columns:
        liq_data = data[[
            'liq_long_1h', 'liq_short_1h', 'liq_long_4h', 'liq_short_4h',
            'liq_long_24h', 'liq_short_24h'
        ]].copy()
        liq_data['imbalance_1h'] = (
            (liq_data['liq_long_1h'] - liq_data['liq_short_1h']) /
            (liq_data['liq_long_1h'] + liq_data['liq_short_1h'] + 1)
        )
        liq_data['imbalance_4h'] = (
            (liq_data['liq_long_4h'] - liq_data['liq_short_4h']) /
            (liq_data['liq_long_4h'] + liq_data['liq_short_4h'] + 1)
        )
    
    # Train models
    if args.model in ["regime", "both"]:
        regime_metrics = train_regime_model(
            bars, oi_data, funding_data, liq_data,
            output_path=output_dir / "regime_model.pkl"
        )
        print(f"\nRegime Model Metrics:\n{regime_metrics}")
    
    if args.model in ["bias", "both"]:
        funding_history = None
        if funding_data is not None:
            funding_history = funding_data.to_dict('records')
        
        bias_metrics = train_bias_model(
            bars, oi_data, funding_history, liq_data,
            output_path=output_dir / "bias_model.pkl"
        )
        print(f"\nBias Model Metrics:\n{bias_metrics}")
    
    print(f"\nModels saved to {output_dir}")


if __name__ == "__main__":
    main()
