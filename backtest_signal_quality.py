#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal Quality Backtest for Hydra V3 Models

Tests model performance at SIGNAL-LIKE conditions (simulating bot signals).
This is NOT a PnL backtest. It answers three questions:
1. Does higher model confidence → better outcomes? (probability buckets)
2. Does the model fail in specific contexts? (slicing by regime/hour/symbol/direction)
3. How stable is signal quality over time? (weekly analysis)

Signal conditions simulated (based on signals.py logic):
- High MOI_z with absorption (ILI-like)
- Flip rate compression (FRCB-like)
- Order flow dominance decay (OFDD-like)
- Queue reactive conditions
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import os
import gc
import json
import pickle
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import io
import zipfile

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

print("Loading ML libraries...", flush=True)
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss
print("Imports complete!", flush=True)

# ============================================================
# CONFIGURATION
# ============================================================

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "ADAUSDT"]
BACKTEST_DAYS = 24  # Days of unseen data
BAR_MS = 250
TP_SL_RATIO = 2.0
HORIZONS = [60, 300]
VOL_REGIMES = ["low", "mid", "high"]
MODELS_DIR = Path("models_v3")
FEATURE_COLS_PATH = Path("feature_columns_v3.json")
OUTPUT_DIR = Path("backtest_results")

# ============================================================
# DATA FETCHING (same as training)
# ============================================================

def fetch_aggtrades_day(symbol: str, date: datetime) -> Optional[pd.DataFrame]:
    date_str = date.strftime("%Y-%m-%d")
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(z.open(z.namelist()[0]))
        df["symbol"] = symbol
        return df
    except:
        return None

def fetch_all_data(pairs: List[str], days: int, end_offset_days: int = 25) -> pd.DataFrame:
    """Fetch historical data ending `end_offset_days` before today (unseen data)"""
    all_dfs = []
    end_date = datetime.now(timezone.utc) - timedelta(days=end_offset_days)
    
    for symbol in pairs:
        print(f"\nFetching {symbol} (unseen period)")
        symbol_dfs = []
        for i in tqdm(range(days)):
            date = end_date - timedelta(days=i)
            df = fetch_aggtrades_day(symbol, date)
            if df is not None:
                symbol_dfs.append(df)
        if symbol_dfs:
            all_dfs.append(pd.concat(symbol_dfs, ignore_index=True))
    
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df):,}")
    return df

# ============================================================
# BAR BUILDING & FEATURES (same as training)
# ============================================================

def compute_bars_fast(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Fast bar computation without slow lambdas"""
    print(f"    Building bars for {symbol}...", flush=True)
    df = df.sort_values("transact_time").reset_index(drop=True)
    df["ts_bar"] = (df["transact_time"] // BAR_MS) * BAR_MS
    df["signed_qty"] = np.where(df["is_buyer_maker"], -df["quantity"], df["quantity"])
    
    # Fast aggregation without lambdas
    bars = df.groupby("ts_bar").agg(
        price=("price", "last"),
        volume=("quantity", "sum"),
        delta=("signed_qty", "sum"),
        trades=("agg_trade_id", "count"),
        high=("price", "max"),
        low=("price", "min"),
    ).reset_index()
    bars["symbol"] = symbol
    print(f"    {len(bars):,} bars created", flush=True)
    return bars

def compute_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features with progress logging"""
    print(f"    Computing features...", flush=True)
    price = df["price"].values
    delta = df["delta"].values
    volume = df["volume"].values
    high = df["high"].values
    low = df["low"].values
    trades = df["trades"].values
    
    # Rolling windows
    df["cum_delta_1s"] = pd.Series(delta).rolling(4, min_periods=1).sum().values
    df["cum_delta_5s"] = pd.Series(delta).rolling(20, min_periods=1).sum().values
    df["cum_delta_30s"] = pd.Series(delta).rolling(120, min_periods=1).sum().values
    df["cum_delta_1m"] = pd.Series(delta).rolling(240, min_periods=1).sum().values
    df["cum_delta_5m"] = pd.Series(delta).rolling(1200, min_periods=1).sum().values
    
    vol_30s = pd.Series(volume).rolling(120, min_periods=1).sum().values
    vol_5m = pd.Series(volume).rolling(1200, min_periods=1).sum().values
    df["vol_ratio"] = np.where(vol_5m > 0, vol_30s / (vol_5m / 10 + 1e-9), 1.0)
    
    # MOI
    buy_pressure = np.maximum(delta, 0)
    sell_pressure = np.abs(np.minimum(delta, 0))
    total = buy_pressure + sell_pressure + 1e-9
    df["MOI_1s"] = (buy_pressure - sell_pressure) / total
    moi_mean = pd.Series(df["MOI_1s"]).rolling(240, min_periods=10).mean().values
    moi_std = pd.Series(df["MOI_1s"]).rolling(240, min_periods=10).std().values + 1e-6
    df["MOI_z"] = (df["MOI_1s"] - moi_mean) / moi_std
    
    # MOI flip rate
    moi_sign = np.sign(df["MOI_1s"].values)
    moi_diff = np.abs(np.diff(moi_sign, prepend=moi_sign[0]))
    df["MOI_flip_rate"] = pd.Series(moi_diff).rolling(240, min_periods=10).mean().values
    
    # Absorption
    price_change = np.abs(np.diff(price, prepend=price[0]))
    df["absorption"] = np.where(price_change > 0, volume / (price_change + 1e-9), 0)
    abs_mean = pd.Series(df["absorption"]).rolling(240, min_periods=10).mean().values
    abs_std = pd.Series(df["absorption"]).rolling(240, min_periods=10).std().values + 1e-6
    df["absorption_z"] = (df["absorption"] - abs_mean) / abs_std
    
    # Trade intensity
    df["trade_intensity"] = trades / (volume + 1e-9)
    ti_mean = pd.Series(df["trade_intensity"]).rolling(240, min_periods=10).mean().values
    ti_std = pd.Series(df["trade_intensity"]).rolling(240, min_periods=10).std().values + 1e-6
    df["trade_intensity_z"] = (df["trade_intensity"] - ti_mean) / ti_std
    
    # ATR
    tr = np.maximum(high - low, np.abs(high - np.roll(price, 1)), np.abs(low - np.roll(price, 1)))
    atr_5m = pd.Series(tr).rolling(1200, min_periods=10).mean().values
    atr_5m = np.where(np.isnan(atr_5m), price * 0.001, atr_5m)
    df["ATR_5m"] = atr_5m
    
    # Volatility regime
    atr_pct = atr_5m / (price + 1e-9)
    atr_rolling_pct = pd.Series(atr_pct).rolling(4800, min_periods=100).rank(pct=True).values
    df["vol_regime"] = np.where(atr_rolling_pct < 0.33, "low", np.where(atr_rolling_pct < 0.67, "mid", "high"))
    
    # Aggression persistence
    delta_sign = np.sign(delta)
    same_sign = (delta_sign == np.roll(delta_sign, 1)).astype(float)
    df["AggressionPersistence"] = pd.Series(same_sign).rolling(20, min_periods=1).mean().values
    
    # Delta velocity
    delta_fast = pd.Series(delta).rolling(4, min_periods=1).mean().values
    delta_slow = pd.Series(delta).rolling(20, min_periods=1).mean().values
    df["delta_velocity"] = delta_fast - delta_slow
    
    # Delta velocity z-score for signal detection
    dv_mean = pd.Series(df["delta_velocity"]).rolling(240, min_periods=10).mean().values
    dv_std = pd.Series(df["delta_velocity"]).rolling(240, min_periods=10).std().values + 1e-6
    df["delta_velocity_z"] = (df["delta_velocity"] - dv_mean) / dv_std
    
    # Price change 5m for signal detection
    df["price_change_5m"] = (price - np.roll(price, 1200)) / (np.roll(price, 1200) + 1e-9)
    df["price_change_5m"] = df["price_change_5m"].fillna(0)
    
    # Time features
    df["hour"] = (df["ts_bar"] // 3600000) % 24
    df["minute"] = (df["ts_bar"] // 60000) % 60
    df["day_of_week"] = pd.to_datetime(df["ts_bar"], unit="ms").dt.dayofweek
    
    # Date for weekly analysis
    df["date"] = pd.to_datetime(df["ts_bar"], unit="ms").dt.date
    df["week"] = pd.to_datetime(df["ts_bar"], unit="ms").dt.isocalendar().week
    
    print(f"    Features done", flush=True)
    return df

def add_cross_sectional_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """Simplified cross-sectional ranking - skip if too slow"""
    print("  Adding cross-sectional features (simplified)...", flush=True)
    df = df_all.copy()
    
    # Just add rank columns with simple rolling rank
    for feat in ["MOI_z", "vol_ratio", "absorption_z"]:
        if feat in df.columns:
            df[f"{feat}_rank"] = df.groupby("symbol")[feat].transform(
                lambda x: x.rolling(1000, min_periods=50).rank(pct=True)
            ).fillna(0.5)
    
    df["momentum_rank"] = df.get("MOI_z_rank", 0.5)
    print("  Cross-sectional features done.", flush=True)
    return df

# ============================================================
# SIGNAL DETECTION (Simulating bot signals from signals.py)
# ============================================================

def detect_signal_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect signal-like conditions based on signals.py logic.
    Since we don't have full ThesisState (funding, OI, liquidations),
    we simulate the order-flow based signals:
    
    1. ILI-like: High MOI_z + absorption + price stalling
    2. FRCB-like: Low flip rate + persistent aggression + compression
    3. OFDD-like: High MOI_z + decaying momentum + absorption
    4. Queue-reactive: High volume spike + MOI extreme
    """
    print("\nDetecting signal conditions (simulating bot signals)...", flush=True)
    
    signals = []
    
    for symbol in tqdm(PAIRS, desc="Signal detection"):
        sym_df = df[df["symbol"] == symbol].copy()
        if len(sym_df) < 1000:
            continue
        
        sym_df = sym_df.reset_index(drop=True)
        sym_df["orig_idx"] = sym_df.index
        
        # === SIGNAL 1: ILI-like (Inventory Lock) ===
        # High aggression + absorption + price not moving
        ili_mask = (
            (np.abs(sym_df["MOI_z"]) > 2.0) &  # Strong aggression
            (sym_df["absorption_z"] > 1.0) &   # Being absorbed
            (np.abs(sym_df["price_change_5m"]) < 0.0015) &  # Price stalling
            (sym_df["AggressionPersistence"] > 0.6)  # Persistent
        )
        
        # === SIGNAL 2: FRCB-like (Flip Rate Compression Break) ===
        # Low flip rate + persistent aggression + compressed vol
        frcb_mask = (
            (sym_df["MOI_flip_rate"] < 0.3) &  # Low flip rate
            (sym_df["AggressionPersistence"] > 0.65) &  # Persistent
            (np.abs(sym_df["MOI_z"]) > 1.0) &  # Some directional pressure
            (np.abs(sym_df["price_change_5m"]) < 0.002)  # Not already moved
        )
        
        # === SIGNAL 3: OFDD-like (Order Flow Dominance Decay) ===
        # High MOI_z but momentum decaying + absorption
        ofdd_mask = (
            (np.abs(sym_df["MOI_z"]) > 1.5) &  # Strong prior dominance
            (sym_df["delta_velocity_z"] * sym_df["MOI_z"] < 0) &  # Decaying
            (sym_df["absorption_z"] > 1.0) &  # Absorption present
            (np.abs(sym_df["price_change_5m"]) < 0.001)  # Price stalling
        )
        
        # === SIGNAL 4: Queue Reactive (volume spike + MOI extreme) ===
        queue_mask = (
            (sym_df["vol_ratio"] > 2.0) &  # Volume spike
            (np.abs(sym_df["MOI_z"]) > 1.5)  # Directional pressure
        )
        
        # Combine all signal masks
        combined_mask = ili_mask | frcb_mask | ofdd_mask | queue_mask
        
        # Get signal points
        signal_df = sym_df[combined_mask].copy()
        
        # Add signal type
        signal_df["signal_type"] = "mixed"
        signal_df.loc[ili_mask[combined_mask].values, "signal_type"] = "ILI"
        signal_df.loc[frcb_mask[combined_mask].values, "signal_type"] = "FRCB"
        signal_df.loc[ofdd_mask[combined_mask].values, "signal_type"] = "OFDD"
        signal_df.loc[queue_mask[combined_mask].values, "signal_type"] = "Queue"
        
        # Determine direction based on MOI
        signal_df["signal_direction"] = np.where(signal_df["MOI_z"] > 0, "up", "down")
        
        # For fade signals (ILI, OFDD), flip direction
        fade_mask = signal_df["signal_type"].isin(["ILI", "OFDD"])
        signal_df.loc[fade_mask, "signal_direction"] = np.where(
            signal_df.loc[fade_mask, "MOI_z"] > 0, "down", "up"
        )
        
        signal_df["bar_idx"] = signal_df["orig_idx"]
        signals.append(signal_df)
    
    if not signals:
        return pd.DataFrame()
    
    df_signals = pd.concat(signals, ignore_index=True)
    
    # Sample if too many signals
    for sym in PAIRS:
        sym_signals = df_signals[df_signals["symbol"] == sym]
        if len(sym_signals) > 20000:
            drop_idx = sym_signals.sample(len(sym_signals) - 20000, random_state=42).index
            df_signals = df_signals.drop(drop_idx)
    
    print(f"\nSignal summary:")
    for sig_type in ["ILI", "FRCB", "OFDD", "Queue"]:
        count = len(df_signals[df_signals["signal_type"] == sig_type])
        print(f"  {sig_type}: {count:,}")
    print(f"  Total: {len(df_signals):,}")
    
    return df_signals

def create_labels_for_signals(df_bars: pd.DataFrame, df_signals: pd.DataFrame, horizon_sec: int, tp_sl_ratio: float = 2.0) -> pd.DataFrame:
    """Create labels for signal points - uses signal direction"""
    print(f"\nLabeling {horizon_sec}s horizon...", flush=True)
    HORIZON = int(horizon_sec * 1000 / BAR_MS)
    results = []
    
    for symbol in tqdm(PAIRS, desc=f"Labeling {horizon_sec}s"):
        bars_sym = df_bars[df_bars["symbol"] == symbol].reset_index(drop=True)
        sig_sym = df_signals[df_signals["symbol"] == symbol].copy()
        
        if len(bars_sym) < HORIZON + 10 or len(sig_sym) == 0:
            continue
        
        prices = bars_sym["price"].values
        atrs = bars_sym["ATR_5m"].values
        n_bars = len(bars_sym)
        
        # Use bar_idx directly as position
        for _, row in sig_sym.iterrows():
            pos = int(row["bar_idx"])
            if pos + HORIZON >= n_bars:
                continue
            
            entry, atr = prices[pos], atrs[pos]
            if np.isnan(atr) or atr <= 0:
                continue
            
            sl_dist, tp_dist = atr, tp_sl_ratio * atr
            future = prices[pos+1:pos+HORIZON+1]
            direction = row["signal_direction"]
            
            # Label based on signal direction
            if direction == "up":
                tp_hit = np.where(future >= entry + tp_dist)[0]
                sl_hit = np.where(future <= entry - sl_dist)[0]
            else:
                tp_hit = np.where(future <= entry - tp_dist)[0]
                sl_hit = np.where(future >= entry + sl_dist)[0]
            
            tp_first = tp_hit[0] if len(tp_hit) else HORIZON + 1
            sl_first = sl_hit[0] if len(sl_hit) else HORIZON + 1
            
            # Only count if resolved
            if tp_first < HORIZON or sl_first < HORIZON:
                # Get feature columns for model prediction
                feature_data = {}
                for col in bars_sym.columns:
                    if col not in ["symbol", "bar_idx", "date", "week", "ts_bar", "orig_idx"]:
                        try:
                            feature_data[col] = bars_sym.iloc[pos][col]
                        except:
                            pass
                
                results.append({
                    "symbol": symbol,
                    "direction": direction,
                    "vol_regime": row.get("vol_regime", "mid"),
                    "hour": int(row.get("hour", 0)),
                    "date": row.get("date"),
                    "week": int(row.get("week", 0)),
                    "signal_type": row.get("signal_type", "unknown"),
                    "label": 1.0 if tp_first < sl_first else 0.0,
                    "horizon": horizon_sec,
                    **feature_data
                })
    
    df_results = pd.DataFrame(results)
    print(f"  Created {len(df_results):,} labeled samples")
    return df_results

# ============================================================
# MODEL LOADING & PREDICTION
# ============================================================

def load_models() -> Tuple[Dict, Dict, List[str]]:
    """Load only the 12 correct models (direction_regime_horizon format)"""
    models = {}
    calibrators = {}
    
    # Only load models with correct naming: models_{up/down}_{low/mid/high}_{60/300}.pkl
    expected_models = []
    for direction in ["up", "down"]:
        for regime in VOL_REGIMES:
            for horizon in HORIZONS:
                expected_models.append(f"models_{direction}_{regime}_{horizon}")
    
    print(f"Looking for {len(expected_models)} models: {expected_models[:3]}...")
    
    for model_name in expected_models:
        pkl_path = MODELS_DIR / f"{model_name}.pkl"
        if not pkl_path.exists():
            print(f"  WARNING: Missing {model_name}.pkl")
            continue
        
        try:
            with open(pkl_path, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict) and "models" in loaded:
                    models[model_name] = loaded["models"]
                    if loaded.get("calibrator"):
                        calibrators[model_name] = loaded["calibrator"]
                else:
                    models[model_name] = loaded if isinstance(loaded, list) else [loaded]
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")
    
    # Load feature columns
    feature_cols = []
    if FEATURE_COLS_PATH.exists():
        with open(FEATURE_COLS_PATH) as f:
            feature_cols = json.load(f)
    
    print(f"Loaded {len(models)} models, {len(calibrators)} calibrators, {len(feature_cols)} features")
    for m in sorted(models.keys()):
        print(f"  - {m}")
    return models, calibrators, feature_cols

def predict_batch(models: Dict, calibrators: Dict, feature_cols: List[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Add predictions to dataframe.
    Each signal uses ONLY its specific model based on direction + vol_regime + horizon.
    e.g., a 'long' signal in 'low' vol_regime uses models_up_low_60 and models_up_low_300
    """
    df = df.copy()
    df["pred_prob"] = np.nan  # Use NaN to identify unpredicted rows
    df["model_used"] = ""  # Track which model was used
    
    prediction_counts = {}
    
    for horizon in HORIZONS:
        for direction in ["up", "down"]:
            for regime in VOL_REGIMES:
                model_name = f"models_{direction}_{regime}_{horizon}"
                
                if model_name not in models:
                    print(f"  WARNING: Model {model_name} not found")
                    continue
                
                # Filter: direction + regime + horizon must ALL match
                mask = (
                    (df["direction"] == direction) & 
                    (df["vol_regime"].str.lower() == regime) & 
                    (df["horizon"] == horizon)
                )
                n_samples = mask.sum()
                if n_samples == 0:
                    continue
                
                # Get features
                feat_cols_available = [c for c in feature_cols if c in df.columns]
                X = df.loc[mask, feat_cols_available].fillna(0).values
                
                # Pad if needed
                if X.shape[1] < len(feature_cols):
                    X = np.hstack([X, np.zeros((X.shape[0], len(feature_cols) - X.shape[1]))])
                
                X_df = pd.DataFrame(X, columns=feature_cols)
                
                # Ensemble predict
                preds = []
                for m in models[model_name]:
                    try:
                        preds.append(m.predict_proba(X_df)[:, 1])
                    except Exception as e:
                        continue
                
                if preds:
                    raw_prob = np.mean(preds, axis=0)
                    
                    # Apply calibration if available
                    if model_name in calibrators:
                        try:
                            prob = calibrators[model_name].predict(raw_prob)
                        except:
                            prob = raw_prob
                    else:
                        prob = raw_prob
                    
                    df.loc[mask, "pred_prob"] = prob
                    df.loc[mask, "model_used"] = model_name
                    prediction_counts[model_name] = n_samples
    
    # Report prediction distribution
    print("\nPrediction counts by model:")
    for model_name in sorted(prediction_counts.keys()):
        print(f"  {model_name}: {prediction_counts[model_name]:,}")
    
    # Check for unpredicted rows
    unpredicted = df["pred_prob"].isna().sum()
    if unpredicted > 0:
        print(f"  WARNING: {unpredicted} signals had no matching model")
        df["pred_prob"] = df["pred_prob"].fillna(0.5)  # Default to 0.5 for unpredicted
    
    return df

# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def analyze_probability_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Question 1: Does higher confidence → better outcomes?"""
    print("\n" + "="*60)
    print("1. PROBABILITY BUCKET ANALYSIS")
    print("="*60)
    
    results = []
    
    for horizon in HORIZONS:
        df_h = df[df["horizon"] == horizon].copy()
        if len(df_h) == 0:
            continue
        
        # Calculate percentiles
        df_h["percentile"] = df_h["pred_prob"].rank(pct=True) * 100
        
        # Define buckets
        buckets = [
            ("Top 5%", df_h["percentile"] >= 95),
            ("Top 10%", (df_h["percentile"] >= 90) & (df_h["percentile"] < 95)),
            ("Top 20%", (df_h["percentile"] >= 80) & (df_h["percentile"] < 90)),
            ("Middle 50%", (df_h["percentile"] >= 30) & (df_h["percentile"] < 80)),
            ("Bottom 30%", df_h["percentile"] < 30),
        ]
        
        print(f"\n--- Horizon: {horizon}s ---")
        print(f"{'Bucket':<15} {'Count':>8} {'Win Rate':>10} {'Avg Prob':>10}")
        print("-" * 45)
        
        for name, mask in buckets:
            subset = df_h[mask]
            if len(subset) > 0:
                win_rate = subset["label"].mean() * 100
                avg_prob = subset["pred_prob"].mean() * 100
                count = len(subset)
                print(f"{name:<15} {count:>8,} {win_rate:>9.1f}% {avg_prob:>9.1f}%")
                results.append({
                    "horizon": horizon,
                    "bucket": name,
                    "count": count,
                    "win_rate": win_rate,
                    "avg_prob": avg_prob,
                })
        
        # Overall AUC
        if len(df_h) > 100:
            try:
                auc = roc_auc_score(df_h["label"], df_h["pred_prob"])
                brier = brier_score_loss(df_h["label"], df_h["pred_prob"])
                print(f"\nOverall AUC: {auc:.4f}")
                print(f"Brier Score: {brier:.4f}")
            except:
                pass
    
    return pd.DataFrame(results)

def analyze_context_slices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Question 2: Does the model fail in specific contexts?"""
    print("\n" + "="*60)
    print("2. CONTEXT SLICE ANALYSIS")
    print("="*60)
    
    results = {}
    
    for horizon in HORIZONS:
        df_h = df[df["horizon"] == horizon].copy()
        if len(df_h) == 0:
            continue
        
        print(f"\n{'='*40}")
        print(f"HORIZON: {horizon}s")
        print(f"{'='*40}")
        
        # By vol_regime
        print("\n--- By Volatility Regime ---")
        print(f"{'Regime':<10} {'Count':>8} {'Win Rate':>10} {'Avg Prob':>10} {'AUC':>8}")
        print("-" * 50)
        regime_results = []
        for regime in ["low", "mid", "high"]:
            subset = df_h[df_h["vol_regime"].str.lower() == regime]
            if len(subset) > 50:
                wr = subset["label"].mean() * 100
                ap = subset["pred_prob"].mean() * 100
                try:
                    auc = roc_auc_score(subset["label"], subset["pred_prob"])
                except:
                    auc = 0.5
                print(f"{regime:<10} {len(subset):>8,} {wr:>9.1f}% {ap:>9.1f}% {auc:>7.3f}")
                regime_results.append({"regime": regime, "count": len(subset), "win_rate": wr, "auc": auc})
        results[f"regime_{horizon}"] = pd.DataFrame(regime_results)
        
        # By direction
        print("\n--- By Direction ---")
        print(f"{'Direction':<10} {'Count':>8} {'Win Rate':>10} {'Avg Prob':>10} {'AUC':>8}")
        print("-" * 50)
        dir_results = []
        for direction in ["up", "down"]:
            subset = df_h[df_h["direction"] == direction]
            if len(subset) > 50:
                wr = subset["label"].mean() * 100
                ap = subset["pred_prob"].mean() * 100
                try:
                    auc = roc_auc_score(subset["label"], subset["pred_prob"])
                except:
                    auc = 0.5
                print(f"{direction:<10} {len(subset):>8,} {wr:>9.1f}% {ap:>9.1f}% {auc:>7.3f}")
                dir_results.append({"direction": direction, "count": len(subset), "win_rate": wr, "auc": auc})
        results[f"direction_{horizon}"] = pd.DataFrame(dir_results)
        
        # By signal type (if available)
        if "signal_type" in df_h.columns:
            print("\n--- By Signal Type ---")
            print(f"{'Signal':<10} {'Count':>8} {'Win Rate':>10} {'AUC':>8}")
            print("-" * 40)
            sig_results = []
            for sig_type in ["ILI", "FRCB", "OFDD", "Queue"]:
                subset = df_h[df_h["signal_type"] == sig_type]
                if len(subset) > 30:
                    wr = subset["label"].mean() * 100
                    try:
                        auc = roc_auc_score(subset["label"], subset["pred_prob"])
                    except:
                        auc = 0.5
                    print(f"{sig_type:<10} {len(subset):>8,} {wr:>9.1f}% {auc:>7.3f}")
                    sig_results.append({"signal_type": sig_type, "count": len(subset), "win_rate": wr, "auc": auc})
            results[f"signal_type_{horizon}"] = pd.DataFrame(sig_results)
        
        # By symbol
        print("\n--- By Symbol ---")
        print(f"{'Symbol':<12} {'Count':>8} {'Win Rate':>10} {'AUC':>8}")
        print("-" * 42)
        symbol_results = []
        for symbol in PAIRS:
            subset = df_h[df_h["symbol"] == symbol]
            if len(subset) > 50:
                wr = subset["label"].mean() * 100
                try:
                    auc = roc_auc_score(subset["label"], subset["pred_prob"])
                except:
                    auc = 0.5
                print(f"{symbol:<12} {len(subset):>8,} {wr:>9.1f}% {auc:>7.3f}")
                symbol_results.append({"symbol": symbol, "count": len(subset), "win_rate": wr, "auc": auc})
        results[f"symbol_{horizon}"] = pd.DataFrame(symbol_results)
        
        # By hour
        print("\n--- By Hour (worst 5 hours) ---")
        print(f"{'Hour':<6} {'Count':>8} {'Win Rate':>10} {'AUC':>8}")
        print("-" * 36)
        hour_results = []
        for hour in range(24):
            subset = df_h[df_h["hour"] == hour]
            if len(subset) > 30:
                wr = subset["label"].mean() * 100
                try:
                    auc = roc_auc_score(subset["label"], subset["pred_prob"])
                except:
                    auc = 0.5
                hour_results.append({"hour": hour, "count": len(subset), "win_rate": wr, "auc": auc})
        
        hour_df = pd.DataFrame(hour_results).sort_values("auc")
        for _, row in hour_df.head(5).iterrows():
            print(f"{int(row['hour']):>4}   {int(row['count']):>8,} {row['win_rate']:>9.1f}% {row['auc']:>7.3f}")
        results[f"hour_{horizon}"] = hour_df
    
    return results

def analyze_weekly_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Question 3: How stable is signal quality over time?"""
    print("\n" + "="*60)
    print("3. WEEKLY STABILITY ANALYSIS")
    print("="*60)
    
    results = []
    
    for horizon in HORIZONS:
        df_h = df[df["horizon"] == horizon].copy()
        if len(df_h) == 0:
            continue
        
        print(f"\n--- Horizon: {horizon}s ---")
        print(f"{'Week':<8} {'Count':>8} {'Win Rate':>10} {'AUC':>8} {'Status':<12}")
        print("-" * 50)
        
        weeks = sorted(df_h["week"].unique())
        week_aucs = []
        
        for week in weeks:
            subset = df_h[df_h["week"] == week]
            if len(subset) > 100:
                wr = subset["label"].mean() * 100
                try:
                    auc = roc_auc_score(subset["label"], subset["pred_prob"])
                except:
                    auc = 0.5
                
                week_aucs.append(auc)
                
                # Determine status
                if auc >= 0.60:
                    status = "✓ Good"
                elif auc >= 0.55:
                    status = "~ OK"
                else:
                    status = "✗ Poor"
                
                print(f"Week {week:<4} {len(subset):>8,} {wr:>9.1f}% {auc:>7.3f} {status:<12}")
                results.append({
                    "horizon": horizon,
                    "week": week,
                    "count": len(subset),
                    "win_rate": wr,
                    "auc": auc,
                })
        
        # Trend analysis
        if len(week_aucs) >= 3:
            first_half = np.mean(week_aucs[:len(week_aucs)//2])
            second_half = np.mean(week_aucs[len(week_aucs)//2:])
            trend = second_half - first_half
            
            print(f"\nTrend: First half AUC={first_half:.3f}, Second half AUC={second_half:.3f}")
            if trend < -0.05:
                print("⚠️  WARNING: Performance declining over time (potential edge decay)")
            elif trend > 0.05:
                print("✓ Performance improving over time")
            else:
                print("~ Performance stable")
    
    return pd.DataFrame(results)

def generate_summary_report(df: pd.DataFrame, bucket_results: pd.DataFrame) -> str:
    """Generate a summary report"""
    report = []
    report.append("\n" + "="*60)
    report.append("BACKTEST SUMMARY REPORT")
    report.append("="*60)
    
    report.append(f"\nData period: {df['date'].min()} to {df['date'].max()}")
    report.append(f"Total signals analyzed: {len(df):,}")
    
    for horizon in HORIZONS:
        df_h = df[df["horizon"] == horizon]
        if len(df_h) == 0:
            continue
        
        report.append(f"\n--- {horizon}s Horizon ---")
        
        # Overall metrics
        try:
            auc = roc_auc_score(df_h["label"], df_h["pred_prob"])
            report.append(f"Overall AUC: {auc:.4f}")
        except:
            pass
        
        # Check monotonicity
        bucket_h = bucket_results[bucket_results["horizon"] == horizon]
        if len(bucket_h) > 0:
            top_wr = bucket_h[bucket_h["bucket"] == "Top 5%"]["win_rate"].values
            bot_wr = bucket_h[bucket_h["bucket"] == "Bottom 30%"]["win_rate"].values
            
            if len(top_wr) > 0 and len(bot_wr) > 0:
                spread = top_wr[0] - bot_wr[0]
                report.append(f"Win rate spread (Top 5% - Bottom 30%): {spread:.1f}%")
                
                if spread > 10:
                    report.append("✓ GOOD: Strong monotonic relationship")
                elif spread > 5:
                    report.append("~ OK: Moderate relationship")
                else:
                    report.append("✗ WEAK: Little differentiation between buckets")
    
    return "\n".join(report)

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=24, help="Days of unseen data to backtest")
    parser.add_argument("--offset", type=int, default=25, help="Days before today to end backtest period")
    args = parser.parse_args()
    
    print("="*60)
    print("HYDRA V3 SIGNAL QUALITY BACKTEST")
    print("="*60)
    print(f"Backtesting {args.days} days of UNSEEN data")
    print(f"Period ends {args.offset} days before today (before training data)")
    
    # Load models
    models, calibrators, feature_cols = load_models()
    if not models:
        print("ERROR: No models found. Run training first.")
        return
    
    # Fetch unseen data
    df = fetch_all_data(PAIRS, args.days, args.offset)
    
    # Build bars and features per symbol
    print("\n" + "="*60)
    print("BUILDING BARS AND FEATURES")
    print("="*60)
    all_bars = []
    for symbol in PAIRS:
        print(f"\nProcessing {symbol}...", flush=True)
        sym_df = df[df["symbol"] == symbol].copy()
        if len(sym_df) == 0:
            print(f"  No data for {symbol}")
            continue
        bars = compute_bars_fast(sym_df, symbol)
        bars = compute_features_fast(bars)
        all_bars.append(bars)
        del sym_df; gc.collect()
    
    df_bars = pd.concat(all_bars, ignore_index=True)
    print(f"\nTotal bars: {len(df_bars):,}")
    del df, all_bars; gc.collect()
    
    # Cross-sectional features (simplified for speed)
    df_bars = add_cross_sectional_features(df_bars)
    
    # Detect signal conditions (simulating bot signals)
    df_signals = detect_signal_conditions(df_bars)
    
    if len(df_signals) == 0:
        print("ERROR: No signals detected")
        return
    
    # Create labels for both horizons
    all_results = []
    for horizon in HORIZONS:
        df_labels = create_labels_for_signals(df_bars, df_signals, horizon)
        all_results.append(df_labels)
    
    df_results = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal signals to analyze: {len(df_results):,}")
    
    # Add predictions
    print("\nRunning model predictions...")
    df_results = predict_batch(models, calibrators, feature_cols, df_results)
    
    # Filter to signals with predictions
    df_results = df_results[df_results["pred_prob"] > 0]
    print(f"Signals with predictions: {len(df_results):,}")
    
    # Run analyses
    bucket_results = analyze_probability_buckets(df_results)
    context_results = analyze_context_slices(df_results)
    weekly_results = analyze_weekly_stability(df_results)
    
    # Generate summary
    summary = generate_summary_report(df_results, bucket_results)
    print(summary)
    
    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    df_results.to_csv(OUTPUT_DIR / "backtest_signals.csv", index=False)
    bucket_results.to_csv(OUTPUT_DIR / "bucket_analysis.csv", index=False)
    weekly_results.to_csv(OUTPUT_DIR / "weekly_analysis.csv", index=False)
    
    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"\n✓ Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
