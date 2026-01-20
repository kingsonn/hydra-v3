#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hydra V4.1 ML Pipeline - PARAMETERIZED FOR EDGE

V4.1 Changes (from V4.0):
- R = 0.6 × ATR_1h + fees (more stable than ATR_5m)
- Horizon = 900s (15 minutes) instead of 300s
- ATR_1h computed from 14400 bars (1 hour at 250ms)

Core principles (unchanged from V4.0):
1. Model answers ONLY: P(+2R before -1R | Stage 3 signal at t with direction d)
2. NO signal detection features as model inputs
3. NO circular labeling (direction is INPUT, not derived from features)
4. Binary labels only: 1=TP first, 0=SL first, DROP unresolved
5. Single model per direction/regime (no separate filter model)

The model is a GATE that filters Stage 3 signals.
It does NOT generate signals or choose direction.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import os
import gc
import json
import pickle
import argparse
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import io
import zipfile

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "ADAUSDT"]
DAYS = 30
BAR_MS = 250
FEE_PCT = 0.0008  # 0.08% round-trip (0.04% each way)
TP_RATIO = 2.0    # Target = 2R
SL_RATIO = 1.0    # Stop = 1R
HORIZON_SEC = 900  # 15 minutes - V4.1: extended from 300s for better resolution
ATR_MULT = 0.6    # V4.1: R = 0.6 × ATR_1h + fees (more stable than ATR_5m)
VOL_REGIMES = ["low", "mid", "high"]
N_SPLITS = 5
OUTPUT_DIR = "models_v4.1"
MODEL_VERSION = "v4.1"

# Features that DO NOT leak signal detection
# These predict OUTCOME given a signal, not whether a signal exists
OUTCOME_FEATURES = [
    # Volatility context (predicts how likely 2R move is)
    "vol_5m", "vol_ratio", "vol_rank", "ATR_1h_pct",
    # Market depth (predicts slippage and execution quality)
    "absorption_z", "price_impact_z",
    # Activity level (predicts liquidity)
    "trade_intensity", "trade_intensity_z",
    # Structure context (predicts support/resistance)
    "dist_poc_atr", "dist_lvn_atr",
    # Time context (predicts session behavior)
    "hour_sin", "hour_cos", "is_weekend",
    # Lagged order flow (1-minute lag to avoid signal leakage)
    "MOI_z_lag", "cum_delta_5m_lag", "delta_velocity_lag",
    # Regime duration (how stable is current state)
    "bars_in_regime",
]

# ============================================================
# DATA FETCHING (unchanged)
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

def fetch_symbol_data(symbol: str, days: int) -> pd.DataFrame:
    all_dfs = []
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)

    print(f"  Fetching {symbol} ({days} days)...")
    for i in tqdm(range(days), desc=f"  {symbol}"):
        day = start_date + timedelta(days=i)
        df_day = fetch_aggtrades_day(symbol, day)
        if df_day is not None:
            df_day["price"] = df_day["price"].astype("float32")
            df_day["quantity"] = df_day["quantity"].astype("float32")
            df_day["is_buyer_maker"] = df_day["is_buyer_maker"].astype("int8")
            all_dfs.append(df_day)

    if not all_dfs:
        return pd.DataFrame()
    
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.rename(columns={"quantity": "qty", "transact_time": "timestamp", "is_buyer_maker": "is_sell"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"  {symbol}: {len(df):,} trades loaded")
    return df

# ============================================================
# BAR + FEATURE COMPUTATION
# ============================================================

def compute_bars_vectorized(df_sym: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df_sym = df_sym.copy()
    df_sym["timestamp"] = pd.to_datetime(df_sym["timestamp"])
    df_sym = df_sym.sort_values("timestamp")
    df_sym["signed_qty"] = np.where(df_sym["is_sell"], -df_sym["qty"], df_sym["qty"])

    bars = df_sym.set_index("timestamp").resample(f"{BAR_MS}ms").agg({
        "price": ["first", "max", "min", "last"],
        "qty": "sum",
        "signed_qty": "sum",
        "is_sell": "count",
    })
    bars.columns = ["open", "high", "low", "close", "qty", "signed_qty", "trade_count"]
    bars = bars.dropna(subset=["close"])
    bars["close"] = bars["close"].ffill()
    bars = bars.reset_index()
    bars["symbol"] = symbol
    bars["price"] = bars["close"]
    return bars

def compute_features_for_outcome(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute features that predict OUTCOME, not signal detection.
    Key difference: Order flow features are LAGGED to avoid leakage.
    """
    df = bars.copy()
    signed_qty = df["signed_qty"].values
    qty = df["qty"].values
    price = df["close"].values
    trade_count = df["trade_count"].values
    high, low = df["high"].values, df["low"].values
    
    # ========== VOLATILITY FEATURES ==========
    df["ret"] = pd.Series(price).pct_change().fillna(0).values
    ret = df["ret"].values
    df["vol_1m"] = pd.Series(ret).rolling(240, min_periods=10).std().fillna(0.0001).values
    df["vol_5m"] = pd.Series(ret).rolling(1200, min_periods=10).std().fillna(0.0001).values
    df["vol_ratio"] = df["vol_1m"] / (df["vol_5m"] + 1e-8)
    df["vol_rank"] = pd.Series(df["vol_5m"]).rolling(2000, min_periods=100).rank(pct=True).fillna(0.5).values
    df["vol_regime"] = pd.cut(df["vol_rank"], bins=[-np.inf, 0.33, 0.67, np.inf], labels=["low", "mid", "high"])
    
    # ========== ATR (for R calculation) ==========
    prev_close = np.roll(price, 1)
    prev_close[0] = price[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    
    # ATR_5m (for reference/features)
    atr_5m = pd.Series(tr).rolling(1200, min_periods=10).mean().values
    atr_5m = np.where(np.isnan(atr_5m), price * 0.001, atr_5m)
    df["ATR_5m"] = atr_5m
    df["ATR_5m_pct"] = df["ATR_5m"] / price
    
    # ATR_1h (V4.1: more stable for R calculation)
    # 1 hour = 14400 bars at 250ms
    atr_1h = pd.Series(tr).rolling(14400, min_periods=100).mean().values
    atr_1h = np.where(np.isnan(atr_1h), atr_5m, atr_1h)  # fallback to ATR_5m if insufficient data
    df["ATR_1h"] = atr_1h
    df["ATR_1h_pct"] = df["ATR_1h"] / price
    
    # ========== ABSORPTION / PRICE IMPACT ==========
    price_change = np.abs(np.diff(np.concatenate([[price[0]], price]))) + 1e-8
    absorption_raw = qty / price_change
    abs_mean = pd.Series(absorption_raw).rolling(500, min_periods=10).mean().values
    abs_std = pd.Series(absorption_raw).rolling(500, min_periods=10).std().values + 1e-6
    df["absorption_z"] = (absorption_raw - abs_mean) / abs_std
    
    price_impact = price_change / (qty + 1e-6)
    pi_mean = pd.Series(price_impact).rolling(500, min_periods=10).mean().values
    pi_std = pd.Series(price_impact).rolling(500, min_periods=10).std().values + 1e-6
    df["price_impact_z"] = (price_impact - pi_mean) / pi_std
    
    # ========== TRADE INTENSITY ==========
    df["trade_intensity"] = pd.Series(trade_count).rolling(100, min_periods=10).mean().fillna(0).values
    tc_mean = pd.Series(trade_count).rolling(500, min_periods=10).mean().values
    tc_std = pd.Series(trade_count).rolling(500, min_periods=10).std().values + 1e-6
    df["trade_intensity_z"] = (trade_count - tc_mean) / tc_std
    
    # ========== STRUCTURE (POC/LVN) ==========
    BIN_SIZE, BLOCK = 10, 4800
    df["price_bin"] = (price / BIN_SIZE).round() * BIN_SIZE
    poc_prices = np.full(len(df), price.mean())
    lvn_prices = np.full(len(df), price.mean())
    for i in range(0, len(df), BLOCK):
        end = min(i + BLOCK, len(df))
        block_bins, block_qty = df["price_bin"].values[i:end], qty[i:end]
        if block_qty.sum() > 0:
            unique_bins = np.unique(block_bins)
            bin_vols = np.array([block_qty[block_bins == b].sum() for b in unique_bins])
            poc_prices[i:end] = unique_bins[np.argmax(bin_vols)]
            thresh = bin_vols.max() * 0.1
            valid_mask = bin_vols >= thresh
            if valid_mask.any():
                lvn_prices[i:end] = unique_bins[valid_mask][np.argmin(bin_vols[valid_mask])]
    df["dist_poc"] = np.abs(price - poc_prices)
    df["dist_lvn"] = np.abs(price - lvn_prices)
    df["dist_poc_atr"] = df["dist_poc"] / (df["ATR_5m"] + 1e-6)
    df["dist_lvn_atr"] = df["dist_lvn"] / (df["ATR_5m"] + 1e-6)
    
    # ========== TIME FEATURES ==========
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    
    # ========== LAGGED ORDER FLOW (CRITICAL: 1-minute lag to avoid signal leakage) ==========
    LAG_BARS = 240  # 1 minute = 240 bars at 250ms
    
    # Current order flow (for signal detection, NOT for model)
    moi_1s = pd.Series(signed_qty).rolling(4, min_periods=1).sum().values
    moi_std = pd.Series(moi_1s).rolling(100, min_periods=10).std().fillna(1.0).values + 1e-6
    moi_z = np.abs(moi_1s) / moi_std
    cum_delta_5m = pd.Series(signed_qty).rolling(1200, min_periods=1).sum().values
    delta_velocity = moi_1s - np.roll(moi_1s, 4)
    delta_velocity[:4] = 0
    
    # LAGGED versions for model (1-minute delay)
    df["MOI_z_lag"] = np.roll(moi_z, LAG_BARS)
    df["MOI_z_lag"][:LAG_BARS] = 0
    df["cum_delta_5m_lag"] = np.roll(cum_delta_5m, LAG_BARS)
    df["cum_delta_5m_lag"][:LAG_BARS] = 0
    df["delta_velocity_lag"] = np.roll(delta_velocity, LAG_BARS)
    df["delta_velocity_lag"][:LAG_BARS] = 0
    
    # Current values for signal detection (stored but NOT used in model)
    df["_MOI_z"] = moi_z
    df["_cum_delta_5m"] = cum_delta_5m
    df["_delta_velocity"] = delta_velocity
    df["_moi_1s"] = moi_1s
    
    # Additional signal detection features (NOT for model)
    df["_flip_rate"] = pd.Series(np.sign(moi_1s) != np.roll(np.sign(moi_1s), 1)).rolling(240).sum().fillna(0).values
    abs_moi = np.abs(moi_1s)
    df["_aggression"] = pd.Series(abs_moi).rolling(100, min_periods=10).mean().values / (pd.Series(abs_moi).rolling(100, min_periods=10).std().values + 1e-6)
    
    # Regime duration
    regime_change = (df["vol_regime"] != df["vol_regime"].shift(1)).astype(int)
    df["bars_in_regime"] = regime_change.groupby(regime_change.cumsum()).cumcount() + 1
    
    return df

# ============================================================
# SIGNAL SIMULATION (for labeling purposes only)
# ============================================================

def simulate_stage3_signals(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate Stage 3 signals for training data generation.
    These are CANDIDATES that the model will learn to filter.
    
    IMPORTANT: This uses CURRENT order flow features to detect signals,
    but the MODEL will only see LAGGED features to avoid leakage.
    """
    bars = bars.copy()
    
    # Use current (non-lagged) features for signal detection
    moi_z = bars["_MOI_z"].fillna(0)
    moi_1s = bars["_moi_1s"].fillna(0)
    delta_vel = bars["_delta_velocity"].fillna(0)
    flip_rate = bars["_flip_rate"].fillna(10)
    aggression = bars["_aggression"].fillna(0)
    absorption = bars["absorption_z"].fillna(0)
    vol_ratio = bars["vol_ratio"].fillna(1)
    
    # Initialize
    signal_direction = pd.Series([0] * len(bars), index=bars.index)
    
    # Simplified signal conditions (matching live Stage 3 logic)
    # LONG signals
    long_mask = (
        ((moi_z > 1.5) & (delta_vel > 0) & (flip_rate < 3)) |  # Momentum
        ((moi_z < -1.5) & (absorption > 1.0) & (delta_vel > 0)) |  # Trapped shorts
        ((vol_ratio < 0.8) & (moi_z > 1.0) & (aggression > 1.2))  # Compression break
    )
    signal_direction = signal_direction.where(~long_mask, 1)
    
    # SHORT signals
    short_mask = (
        ((moi_z < -1.5) & (delta_vel < 0) & (flip_rate < 3)) |  # Momentum
        ((moi_z > 1.5) & (absorption > 1.0) & (delta_vel < 0)) |  # Trapped longs
        ((vol_ratio < 0.8) & (moi_z < -1.0) & (aggression > 1.2))  # Compression break
    ) & (signal_direction == 0)  # Don't override LONG
    signal_direction = signal_direction.where(~short_mask, -1)
    
    bars["signal_direction"] = signal_direction
    
    return bars

# ============================================================
# CORRECT LABELING (OUTCOME-BASED)
# ============================================================

def create_outcome_labels(
    bars: pd.DataFrame,
    feature_cols: List[str],
    horizon_sec: int = HORIZON_SEC,
) -> Tuple[Dict, Dict, Dict]:
    """
    Create labels based on OUTCOME, not signal detection.
    
    For each simulated Stage 3 signal:
    - Label = 1 if TP (+2R) hit before SL (-1R)
    - Label = 0 if SL hit before TP
    - DROPPED if neither hit within horizon
    
    Direction is taken from the signal (INPUT), not derived from features.
    This eliminates circular labeling.
    """
    HORIZON = int(horizon_sec * 1000 / BAR_MS)
    
    X_dict = {f"{d}_{r}": [] for d in ["long", "short"] for r in VOL_REGIMES}
    y_dict = {f"{d}_{r}": [] for d in ["long", "short"] for r in VOL_REGIMES}
    w_dict = {f"{d}_{r}": [] for d in ["long", "short"] for r in VOL_REGIMES}
    
    stats = {"total": 0, "resolved": 0, "tp_wins": 0, "sl_wins": 0, "unresolved": 0}
    
    bars = bars.reset_index(drop=True)
    n_bars = len(bars)
    
    if n_bars < HORIZON + 100:
        return X_dict, y_dict, w_dict
    
    # Simulate Stage 3 signals
    bars = simulate_stage3_signals(bars)
    
    prices = bars["price"].values
    atrs_1h = bars["ATR_1h"].values  # V4.1: Use ATR_1h for R calculation
    
    # Get signal points
    signal_mask = bars["signal_direction"] != 0
    signal_indices = bars.loc[signal_mask].index.values
    
    print(f"    Found {len(signal_indices):,} signal candidates")
    
    # Sample if too many
    if len(signal_indices) > 50000:
        np.random.seed(42)
        signal_indices = np.random.choice(signal_indices, 50000, replace=False)
    
    valid_features = [c for c in feature_cols if c in bars.columns]
    
    for pos in signal_indices:
        if pos + HORIZON >= n_bars:
            continue
        
        entry = prices[pos]
        atr_1h = atrs_1h[pos]
        if np.isnan(atr_1h) or atr_1h <= 0:
            continue
        
        # Direction from signal (NOT from features)
        direction = int(bars.iloc[pos]["signal_direction"])
        dir_str = "long" if direction == 1 else "short"
        
        regime = bars.iloc[pos].get("vol_regime", "mid")
        if pd.isna(regime) or regime not in VOL_REGIMES:
            regime = "mid"
        regime = str(regime).lower()
        
        # V4.1: R = 0.6 × ATR_1h + fees (more stable than ATR_5m)
        R = (ATR_MULT * atr_1h) + (entry * FEE_PCT)
        TP_dist = TP_RATIO * R
        SL_dist = SL_RATIO * R
        
        future = prices[pos+1:pos+1+HORIZON]
        
        stats["total"] += 1
        
        # Check TP/SL hits based on direction
        if direction == 1:  # LONG
            tp_hit = np.where(future >= entry + TP_dist)[0]
            sl_hit = np.where(future <= entry - SL_dist)[0]
        else:  # SHORT
            tp_hit = np.where(future <= entry - TP_dist)[0]
            sl_hit = np.where(future >= entry + SL_dist)[0]
        
        tp_first = tp_hit[0] if len(tp_hit) else HORIZON + 1
        sl_first = sl_hit[0] if len(sl_hit) else HORIZON + 1
        
        # Drop unresolved
        if tp_first >= HORIZON and sl_first >= HORIZON:
            stats["unresolved"] += 1
            continue
        
        stats["resolved"] += 1
        
        # Binary label: 1 = TP first, 0 = SL first
        if tp_first < sl_first:
            y = 1.0
            stats["tp_wins"] += 1
        else:
            y = 0.0
            stats["sl_wins"] += 1
        
        # Features (using lagged order flow)
        features = bars.iloc[pos][valid_features].values.astype(np.float32)
        
        # Recency weight
        recency_weight = 0.5 + 0.5 * (pos / n_bars)
        
        key = f"{dir_str}_{regime}"
        X_dict[key].append(features)
        y_dict[key].append(y)
        w_dict[key].append(recency_weight)
    
    # Print stats
    if stats["total"] > 0:
        print(f"    Total: {stats['total']:,} | Resolved: {stats['resolved']:,} ({stats['resolved']/stats['total']*100:.1f}%)")
        if stats["resolved"] > 0:
            print(f"    TP wins: {stats['tp_wins']:,} ({stats['tp_wins']/stats['resolved']*100:.1f}%) | SL wins: {stats['sl_wins']:,}")
    
    # Convert to arrays
    for key in X_dict:
        X_dict[key] = np.array(X_dict[key], dtype=np.float32) if X_dict[key] else np.array([]).reshape(0, len(valid_features))
        y_dict[key] = np.array(y_dict[key], dtype=np.float32) if y_dict[key] else np.array([])
        w_dict[key] = np.array(w_dict[key], dtype=np.float32) if w_dict[key] else np.array([])
        if len(X_dict[key]) > 0:
            print(f"    {key}: {len(X_dict[key]):,} samples, {y_dict[key].mean()*100:.1f}% positive")
    
    return X_dict, y_dict, w_dict

# ============================================================
# MODEL TRAINING
# ============================================================

def get_lgb_params(use_gpu=False):
    """LightGBM parameters for binary classification"""
    return {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 1500,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.01,
        "subsample": 0.7,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "min_child_samples": 100,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "max_bin": 127,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "is_unbalance": True,
        "device": "gpu" if use_gpu else "cpu",
    }

def purged_cv_splits(n, n_splits=5, purge_pct=0.02):
    """Time-series CV with purge gap to prevent leakage"""
    fold_size = n // (n_splits + 1)
    purge = int(fold_size * purge_pct)
    for i in range(n_splits):
        train_end = fold_size * (i + 1) - purge
        val_start = fold_size * (i + 1) + purge
        val_end = min(fold_size * (i + 2), n)
        if val_start < val_end:
            yield np.arange(0, train_end), np.arange(val_start, val_end)

def train_model(X, y, name, feature_cols, weights=None, use_gpu=False):
    """Train ensemble model for P(TP|signal)"""
    if len(X) < 500:
        print(f"  {name}: Skipping - only {len(X)} samples")
        return None, None
    
    params = get_lgb_params(use_gpu)
    X_df = pd.DataFrame(X, columns=feature_cols[:X.shape[1]])
    
    models = []
    aucs = []
    win_rates = {0.50: [], 0.55: [], 0.60: []}
    
    print(f"\n  Training {name} ({len(X):,} samples, {y.mean()*100:.1f}% positive)")
    
    for fold, (tr, va) in enumerate(purged_cv_splits(len(X), N_SPLITS)):
        if len(va) < 50:
            continue
        
        model = lgb.LGBMClassifier(**params)
        
        fit_kwargs = {
            "X": X_df.iloc[tr],
            "y": y[tr],
            "eval_set": [(X_df.iloc[va], y[va])],
            "callbacks": [lgb.early_stopping(100, verbose=False)]
        }
        if weights is not None:
            fit_kwargs["sample_weight"] = weights[tr]
        
        model.fit(**fit_kwargs)
        preds = model.predict_proba(X_df.iloc[va])[:, 1]
        
        # AUC
        auc = roc_auc_score(y[va], preds) if len(np.unique(y[va])) > 1 else 0.5
        aucs.append(auc)
        
        # Win rate at thresholds
        for thresh in [0.50, 0.55, 0.60]:
            mask = preds >= thresh
            if mask.sum() > 10:
                wr = y[va][mask].mean()
                win_rates[thresh].append(wr)
        
        models.append(model)
        print(f"    Fold {fold}: AUC={auc:.4f}, n_train={len(tr)}, n_val={len(va)}")
    
    if not models:
        return None, None
    
    # Summary
    print(f"  Mean AUC: {np.mean(aucs):.4f}")
    for thresh, wrs in win_rates.items():
        if wrs:
            print(f"    Win rate @{thresh}: {np.mean(wrs)*100:.1f}%")
    
    # Calibrate on pooled validation predictions
    from sklearn.isotonic import IsotonicRegression
    cal_preds, cal_y = [], []
    for fold, (tr, va) in enumerate(purged_cv_splits(len(X), N_SPLITS)):
        if len(va) < 50 or fold >= len(models):
            continue
        preds = models[fold].predict_proba(X_df.iloc[va])[:, 1]
        cal_preds.extend(preds)
        cal_y.extend(y[va])
    
    calibrator = None
    if len(cal_preds) > 100:
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(cal_preds, cal_y)
    
    return models, calibrator

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["full", "quick"])
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    days = 5 if args.mode == "quick" else DAYS
    print(f"{'='*70}")
    print(f"HYDRA {MODEL_VERSION.upper()} ML - PARAMETERIZED FOR EDGE ({days} days)")
    print(f"{'='*70}")
    print(f"Model objective: P(+2R before -1R | Stage 3 signal)")
    print(f"R = {ATR_MULT} × ATR_1h + {FEE_PCT*100:.2f}% fees")
    print(f"Horizon: {HORIZON_SEC}s ({HORIZON_SEC//60} minutes)")
    print(f"{'='*70}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Use only outcome-predicting features (no signal detection leakage)
    feature_cols = [f for f in OUTCOME_FEATURES]
    
    # Save feature columns
    with open(f"{OUTPUT_DIR}/feature_columns_v4.json", "w") as f:
        json.dump(feature_cols, f)
    
    total_models = 0

    for symbol in PAIRS:
        print(f"\n{'#'*70}")
        print(f"# SYMBOL: {symbol}")
        print(f"{'#'*70}")
        
        # Fetch data
        sym_df = fetch_symbol_data(symbol, days)
        if len(sym_df) < 1000:
            print(f"  Skipping {symbol}: insufficient data")
            continue
        
        # Build bars
        print(f"  Building bars from {len(sym_df):,} trades...")
        bars = compute_bars_vectorized(sym_df, symbol)
        del sym_df; gc.collect()
        
        # Compute features
        print(f"  Computing features for {len(bars):,} bars...")
        bars = compute_features_for_outcome(bars)
        
        # Create labels
        print(f"  Creating outcome labels (horizon={HORIZON_SEC}s)...")
        X, y, w = create_outcome_labels(bars, feature_cols, HORIZON_SEC)
        
        # Create symbol output directory
        symbol_dir = f"{OUTPUT_DIR}/{symbol}"
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Train one model per direction/regime
        for key in X:
            if len(X[key]) < 500:
                continue
            
            models, calibrator = train_model(
                X[key], y[key], f"{symbol}_{key}",
                feature_cols, weights=w[key], use_gpu=args.gpu
            )
            
            if models:
                model_path = f"{symbol_dir}/outcome_{key}_{HORIZON_SEC}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump({"models": models, "calibrator": calibrator}, f)
                total_models += 1
                print(f"  Saved: {model_path}")
        
        del bars; gc.collect()

    # Save training summary
    summary = {
        "version": MODEL_VERSION,
        "objective": "P(+2R before -1R | Stage 3 signal)",
        "symbols": PAIRS,
        "features": feature_cols,
        "horizon_sec": HORIZON_SEC,
        "R_definition": f"{ATR_MULT} × ATR_1h + {FEE_PCT*100:.2f}% fees",
        "ATR_multiplier": ATR_MULT,
        "TP_ratio": TP_RATIO,
        "SL_ratio": SL_RATIO,
        "total_models": total_models,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "v4_1_changes": [
            "R = 0.6 × ATR_1h + fees (more stable than ATR_5m)",
            "Horizon = 900s (15 min) instead of 300s (5 min)",
            "ATR_1h computed from 14400 bars (1 hour)",
        ],
        "core_principles": [
            "Lagged order flow features (1-min lag) to prevent signal leakage",
            "Direction is INPUT from signal, not derived from features",
            "Binary labels: 1=TP first, 0=SL first, unresolved DROPPED",
            "Single model per direction/regime (no separate filter model)",
        ],
    }
    with open(f"{OUTPUT_DIR}/training_summary_v4.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE - {total_models} models saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
