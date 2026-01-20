#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hydra V3 Final - ML Training Pipeline
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print("=" * 70, flush=True)
print("HYDRA V3 FINAL - Loading imports...", flush=True)

import os
import gc
import json
import pickle
import argparse
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import io
import zipfile

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

print("Core imports done, loading ML libraries...", flush=True)

print("  Loading lightgbm...", flush=True)
import lightgbm as lgb
print("  Loading sklearn...", flush=True)
from sklearn.metrics import mean_absolute_error, mean_squared_error
print("All imports complete!", flush=True)

warnings.filterwarnings('ignore')

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ============================================================
# CONFIGURATION
# ============================================================

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "LTCUSDT", "ADAUSDT"]
DAYS = 30
BAR_MS = 250
FEE_PCT = 0.0004
TP_SL_RATIO = 2.0
HORIZONS = [60, 300]
VOL_REGIMES = ["low", "mid", "high"]
N_SPLITS = 5
OUTPUT_DIR = "models_v3"

# ============================================================
# DATA FETCHING
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
    """Fetch data for a single symbol to reduce memory usage."""
    all_dfs = []
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)

    print(f"  Fetching {symbol} ({days} days)...")
    for i in tqdm(range(days), desc=f"  {symbol}"):
        day = start_date + timedelta(days=i)
        df_day = fetch_aggtrades_day(symbol, day)
        if df_day is not None:
            # Immediately optimize dtypes to save memory
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
# VECTORIZED BAR + FEATURE COMPUTATION
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

def compute_features_vectorized(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    signed_qty = df["signed_qty"].values
    qty = df["qty"].values
    price = df["close"].values
    trade_count = df["trade_count"].values
    high, low = df["high"].values, df["low"].values

    # Order flow
    df["MOI_250ms"] = signed_qty
    df["MOI_1s"] = pd.Series(signed_qty).rolling(4, min_periods=1).sum().values
    df["MOI_5s"] = pd.Series(signed_qty).rolling(20, min_periods=1).sum().values
    df["MOI_20s"] = pd.Series(signed_qty).rolling(80, min_periods=1).sum().values

    moi_1s = df["MOI_1s"].values
    df["MOI_std"] = pd.Series(moi_1s).rolling(100, min_periods=10).std().fillna(1.0).values + 1e-6
    df["MOI_z"] = np.abs(moi_1s) / df["MOI_std"].values

    abs_moi = np.abs(moi_1s)
    df["AggressionPersistence"] = pd.Series(abs_moi).rolling(100, min_periods=10).mean().values / (pd.Series(abs_moi).rolling(100, min_periods=10).std().values + 1e-6)

    df["delta_velocity"] = moi_1s - np.roll(moi_1s, 4)
    df["delta_velocity"][:4] = 0
    df["delta_velocity_5s"] = moi_1s - np.roll(moi_1s, 20)
    df["delta_velocity_5s"][:20] = 0

    prev_moi_4 = np.roll(moi_1s, 4)
    prev_moi_20 = np.roll(moi_1s, 20)
    df["MOI_roc_1s"] = np.clip((moi_1s - prev_moi_4) / (np.abs(prev_moi_4) + 1e-6), -10, 10)
    df["MOI_roc_1s"][:4] = 0
    df["MOI_roc_5s"] = np.clip((moi_1s - prev_moi_20) / (np.abs(prev_moi_20) + 1e-6), -10, 10)
    df["MOI_roc_5s"][:20] = 0

    delta_vel = df["delta_velocity"].values
    df["MOI_acceleration"] = delta_vel - np.roll(delta_vel, 1)
    df["MOI_acceleration"][0] = 0

    moi_sign = np.sign(moi_1s)
    sign_change = (moi_sign != np.roll(moi_sign, 1)).astype(int)
    sign_change[0] = 0
    df["MOI_flip_rate"] = pd.Series(sign_change).rolling(240, min_periods=1).sum().values

    # Volatility
    df["ret"] = pd.Series(price).pct_change().fillna(0).values
    ret = df["ret"].values
    df["vol_1m"] = pd.Series(ret).rolling(240, min_periods=10).std().fillna(0.0001).values
    df["vol_5m"] = pd.Series(ret).rolling(1200, min_periods=10).std().fillna(0.0001).values
    df["vol_ratio"] = df["vol_1m"] / (df["vol_5m"] + 1e-8)
    df["vol_rank"] = pd.Series(df["vol_5m"]).rolling(2000, min_periods=100).rank(pct=True).fillna(0.5).values
    df["vol_regime"] = pd.cut(df["vol_rank"], bins=[-np.inf, 0.33, 0.67, np.inf], labels=["low", "mid", "high"])

    # ATR
    prev_close = np.roll(price, 1)
    prev_close[0] = price[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr_5m = pd.Series(tr).rolling(1200, min_periods=10).mean().values
    atr_5m = np.where(np.isnan(atr_5m), price * 0.001, atr_5m)
    df["ATR_5m"] = atr_5m
    df["ATR_5m_pct"] = df["ATR_5m"] / price

    # Absorption
    price_change = np.abs(np.diff(np.concatenate([[price[0]], price]))) + 1e-8
    absorption_raw = qty / price_change
    abs_mean = pd.Series(absorption_raw).rolling(500, min_periods=10).mean().values
    abs_std = pd.Series(absorption_raw).rolling(500, min_periods=10).std().values + 1e-6
    df["absorption_z"] = (absorption_raw - abs_mean) / abs_std
    price_impact = price_change / (qty + 1e-6)
    pi_mean = pd.Series(price_impact).rolling(500, min_periods=10).mean().values
    pi_std = pd.Series(price_impact).rolling(500, min_periods=10).std().values + 1e-6
    df["price_impact_z"] = (price_impact - pi_mean) / pi_std

    # Structure (LVN/POC) - simplified for speed
    BIN_SIZE, BLOCK = 10, 4800  # Larger blocks for speed
    df["price_bin"] = (price / BIN_SIZE).round() * BIN_SIZE
    n_blocks = len(df) // BLOCK + 1
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
    df["POC_price"] = poc_prices
    df["LVN_price"] = lvn_prices
    df["dist_poc"] = np.abs(price - df["POC_price"].values)
    df["dist_lvn"] = np.abs(price - df["LVN_price"].values)
    df["dist_poc_atr"] = df["dist_poc"] / (df["ATR_5m"] + 1e-6)
    df["dist_lvn_atr"] = df["dist_lvn"] / (df["ATR_5m"] + 1e-6)

    # Trade intensity
    df["trade_intensity"] = pd.Series(trade_count).rolling(100, min_periods=10).mean().fillna(0).values
    tc_mean = pd.Series(trade_count).rolling(500, min_periods=10).mean().values
    tc_std = pd.Series(trade_count).rolling(500, min_periods=10).std().values + 1e-6
    df["trade_intensity_z"] = (trade_count - tc_mean) / tc_std

    # Cumulative delta
    df["cum_delta_1m"] = pd.Series(signed_qty).rolling(240, min_periods=1).sum().values
    df["cum_delta_5m"] = pd.Series(signed_qty).rolling(1200, min_periods=1).sum().values

    # Time
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)

    return df

# ============================================================
# CROSS-SECTIONAL RANKING
# ============================================================

def add_cross_sectional_features(df_all: pd.DataFrame) -> pd.DataFrame:
    """Simplified cross-sectional ranking - uses symbol-level rolling ranks instead of timestamp groupby"""
    print("  Adding cross-sectional features (simplified)...", flush=True)
    df = df_all.copy()
    rank_features = ["MOI_z", "vol_ratio", "absorption_z", "AggressionPersistence", "MOI_flip_rate", "trade_intensity_z", "cum_delta_5m"]
    
    # Use rolling percentile rank within each symbol instead of cross-sectional (much faster)
    for feat in rank_features:
        if feat in df.columns:
            print(f"    Ranking {feat}...", flush=True)
            df[f"{feat}_rank"] = df.groupby("symbol")[feat].transform(
                lambda x: x.rolling(2000, min_periods=100).rank(pct=True)
            ).fillna(0.5)
    
    # MOI_z_relative: z-score within rolling window
    print("    Computing MOI_z_relative...", flush=True)
    df["MOI_z_relative"] = df.groupby("symbol")["MOI_z"].transform(
        lambda x: (x - x.rolling(2000, min_periods=100).mean()) / (x.rolling(2000, min_periods=100).std() + 1e-6)
    ).fillna(0)
    
    df["momentum_rank"] = df["cum_delta_5m_rank"]  # Reuse existing rank
    print("  Cross-sectional features done.", flush=True)
    return df

# ============================================================
# DECISION POINTS + LABELING (SIGNAL-BASED)
# ============================================================

# Per-symbol training: no cross-sectional features needed
FEATURE_COLS = [
    "MOI_250ms", "MOI_1s", "MOI_5s", "MOI_20s", "MOI_z", "delta_velocity", "delta_velocity_5s",
    "AggressionPersistence", "MOI_roc_1s", "MOI_roc_5s", "MOI_acceleration", "MOI_flip_rate",
    "absorption_z", "price_impact_z", "vol_1m", "vol_5m", "vol_ratio", "vol_rank", "ATR_5m_pct",
    "dist_lvn_atr", "dist_poc_atr", "dist_lvn", "dist_poc", "hour_sin", "hour_cos", "is_weekend",
    "trade_intensity", "trade_intensity_z", "cum_delta_1m", "cum_delta_5m",
]


def detect_signal_conditions_vectorized(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Detect signal conditions using available aggTrades features (VECTORIZED).
    Returns DataFrame with signal_type and signal_direction columns.
    
    Signals (adapted from signals.py using available data):
    1. ENTROPY_FLOW: Low flip rate + sustained aggression = predictable flow (trend)
    2. COMPRESSION_BREAK: Low vol + low flip rate + aggression = breakout (trend)
    3. DOMINANCE_DECAY: Strong MOI + decaying velocity + absorption = reversal (fade)
    4. ABSORPTION_FADE: High absorption + aggression + price stall = fade
    5. MOMENTUM_CONTINUATION: Strong momentum + acceleration = trend
    6. TRAPPED_LONGS: Strong buying + weak price + decelerating = longs trapped (SHORT)
    7. TRAPPED_SHORTS: Strong selling + strong price + decelerating = shorts trapped (LONG)
    8. LIQUIDATION_CASCADE: Sudden acceleration + extreme intensity = cascade (follow)
    9. TEMPORARY_IMPACT: Price moved + low current aggression = temporary (fade)
    10. INVENTORY_LOCK: Aggression absorbed at structure level = trapped (fade)
    11. STRUCTURE_REJECTION: At LVN/POC + absorption + price stall = rejection (fade)
    """
    bars = bars.copy()
    
    # Precompute needed values
    moi_z = bars["MOI_z"].fillna(0)
    moi_1s = bars["MOI_1s"].fillna(0)
    moi_5s = bars["MOI_5s"].fillna(0)
    moi_20s = bars["MOI_20s"].fillna(0)
    delta_vel = bars["delta_velocity"].fillna(0)
    delta_vel_5s = bars["delta_velocity_5s"].fillna(0)
    flip_rate = bars["MOI_flip_rate"].fillna(10)
    aggression = bars["AggressionPersistence"].fillna(0)
    absorption = bars["absorption_z"].fillna(0)
    vol_ratio = bars["vol_ratio"].fillna(1)
    price_5m = bars["price"].pct_change(1200).fillna(0)
    price_1m = bars["price"].pct_change(240).fillna(0)
    cum_delta = bars["cum_delta_5m"].fillna(0)
    cum_delta_1m = bars["cum_delta_1m"].fillna(0)
    trade_intensity = bars["trade_intensity_z"].fillna(0)
    dist_lvn = bars["dist_lvn_atr"].fillna(10)
    dist_poc = bars["dist_poc_atr"].fillna(10)
    moi_accel = bars["MOI_acceleration"].fillna(0)
    
    # Rolling statistics
    cum_delta_90 = bars["cum_delta_5m"].abs().rolling(4800, min_periods=100).quantile(0.9).fillna(1e9)
    intensity_90 = bars["trade_intensity_z"].rolling(4800, min_periods=100).quantile(0.9).fillna(1e9)
    moi_accel_90 = bars["MOI_acceleration"].abs().rolling(4800, min_periods=100).quantile(0.9).fillna(1e9)
    
    # Initialize
    signal_type = pd.Series([None] * len(bars), index=bars.index)
    signal_direction = pd.Series([0] * len(bars), index=bars.index)
    signal_confidence = pd.Series([0.0] * len(bars), index=bars.index)
    
    # === SIGNAL 1: ENTROPY FLOW (Trend Following) ===
    # Low flip rate = coordinated flow = edge
    mask1 = (flip_rate < 2.0) & (aggression > 1.3) & (moi_z.abs() > 1.5)
    signal_type = signal_type.where(~mask1, "entropy_flow")
    signal_direction = signal_direction.where(~mask1, np.where(moi_z > 0, 1, -1))
    signal_confidence = signal_confidence.where(~mask1, 0.72)
    
    # === SIGNAL 2: COMPRESSION BREAK ===
    mask2 = ((vol_ratio < 0.8) & (flip_rate < 2.5) & (aggression > 1.3) & 
             (price_5m.abs() < 0.002) & (moi_z.abs() > 1.0) & (delta_vel * moi_z > 0))
    mask2 = mask2 & signal_type.isna()
    signal_type = signal_type.where(~mask2, "compression_break")
    signal_direction = np.where(mask2, np.where(moi_z > 0, 1, -1), signal_direction)
    signal_confidence = signal_confidence.where(~mask2, 0.70)
    
    # === SIGNAL 3: DOMINANCE DECAY (Reversal) ===
    mask3 = ((moi_z.abs() > 1.5) & (delta_vel * moi_z < 0) & 
             (absorption > 1.0) & (price_5m.abs() < 0.001) & (flip_rate > 2.0))
    mask3 = mask3 & signal_type.isna()
    signal_type = signal_type.where(~mask3, "dominance_decay")
    signal_direction = np.where(mask3, np.where(moi_z > 0, -1, 1), signal_direction)  # Fade
    signal_confidence = signal_confidence.where(~mask3, 0.73)
    
    # === SIGNAL 4: ABSORPTION FADE ===
    mask4 = ((absorption > 1.5) & (moi_z.abs() > 1.2) & 
             (aggression > 1.0) & (price_5m.abs() < 0.0015))
    mask4 = mask4 & signal_type.isna()
    signal_type = signal_type.where(~mask4, "absorption_fade")
    signal_direction = np.where(mask4, np.where(moi_z > 0, -1, 1), signal_direction)  # Fade
    signal_confidence = signal_confidence.where(~mask4, 0.70)
    
    # === SIGNAL 5: MOMENTUM CONTINUATION ===
    mask5 = ((cum_delta.abs() > cum_delta_90) & (trade_intensity > 1.5) & 
             (delta_vel * cum_delta > 0))
    mask5 = mask5 & signal_type.isna()
    signal_type = signal_type.where(~mask5, "momentum_continuation")
    signal_direction = np.where(mask5, np.where(cum_delta > 0, 1, -1), signal_direction)
    signal_confidence = signal_confidence.where(~mask5, 0.68)
    
    # === SIGNAL 6: TRAPPED LONGS (Funding-Price Proxy) ===
    # Strong buying pressure (high MOI) + weak/flat price + decelerating = longs failing
    mask6 = ((moi_z > 1.5) & (price_5m < 0.0005) & (delta_vel < 0) & 
             (absorption > 0.5) & (aggression > 1.0))
    mask6 = mask6 & signal_type.isna()
    signal_type = signal_type.where(~mask6, "trapped_longs")
    signal_direction = np.where(mask6, -1, signal_direction)  # SHORT - fade trapped longs
    signal_confidence = signal_confidence.where(~mask6, 0.72)
    
    # === SIGNAL 7: TRAPPED SHORTS (Funding-Price Proxy) ===
    # Strong selling pressure (negative MOI) + price not dropping + decelerating = shorts failing
    mask7 = ((moi_z < -1.5) & (price_5m > -0.0005) & (delta_vel > 0) & 
             (absorption > 0.5) & (aggression > 1.0))
    mask7 = mask7 & signal_type.isna()
    signal_type = signal_type.where(~mask7, "trapped_shorts")
    signal_direction = np.where(mask7, 1, signal_direction)  # LONG - fade trapped shorts
    signal_confidence = signal_confidence.where(~mask7, 0.72)
    
    # === SIGNAL 8: LIQUIDATION CASCADE (Hawkes Proxy) ===
    # Sudden acceleration in order flow + extreme trade intensity = cascade in progress
    # Recent intensity >> medium intensity (acceleration pattern)
    moi_1s_intensity = moi_1s.abs()
    moi_5s_intensity = moi_5s.abs()
    is_accelerating = moi_1s_intensity > moi_5s_intensity * 1.5
    mask8 = ((moi_accel.abs() > moi_accel_90) & (trade_intensity > intensity_90) & 
             is_accelerating & (moi_z.abs() > 2.0))
    mask8 = mask8 & signal_type.isna()
    signal_type = signal_type.where(~mask8, "liquidation_cascade")
    # Follow the cascade direction (not fade)
    signal_direction = np.where(mask8, np.where(moi_z > 0, -1, 1), signal_direction)
    signal_confidence = signal_confidence.where(~mask8, 0.74)
    
    # === SIGNAL 9: TEMPORARY IMPACT (Kyle's Lambda Proxy) ===
    # Price moved significantly + current aggression low = impact was temporary, expect reversion
    mask9 = ((price_5m.abs() > 0.002) & (moi_z.abs() < 0.5) & 
             (aggression < 0.8) & (absorption < 1.0))
    mask9 = mask9 & signal_type.isna()
    signal_type = signal_type.where(~mask9, "temporary_impact")
    signal_direction = np.where(mask9, np.where(price_5m > 0, -1, 1), signal_direction)  # Fade the move
    signal_confidence = signal_confidence.where(~mask9, 0.65)
    
    # === SIGNAL 10: INVENTORY LOCK (ILI Proxy) ===
    # Aggression happening + absorption confirmed + price stall + at structure = trapped
    mask10 = ((moi_z.abs() > 2.0) & (aggression > 1.2) & (absorption > 1.0) & 
              (price_1m.abs() < 0.001) & ((dist_lvn < 0.5) | (dist_poc < 0.3)))
    mask10 = mask10 & signal_type.isna()
    signal_type = signal_type.where(~mask10, "inventory_lock")
    signal_direction = np.where(mask10, np.where(moi_z > 0, -1, 1), signal_direction)  # Fade aggressor
    signal_confidence = signal_confidence.where(~mask10, 0.72)
    
    # === SIGNAL 11: STRUCTURE REJECTION (Failed Acceptance Proxy) ===
    # At LVN or POC edge + absorption + price stall + moderate aggression = rejection
    at_structure = (dist_lvn < 0.3) | (dist_poc < 0.2)
    mask11 = (at_structure & (absorption > 1.2) & (price_5m.abs() < 0.001) & 
              (moi_z.abs() > 0.8) & (moi_z.abs() < 2.0) & (flip_rate < 4.0))
    mask11 = mask11 & signal_type.isna()
    signal_type = signal_type.where(~mask11, "structure_rejection")
    signal_direction = np.where(mask11, np.where(moi_z > 0, -1, 1), signal_direction)  # Fade
    signal_confidence = signal_confidence.where(~mask11, 0.70)
    
    bars["signal_type"] = signal_type
    bars["signal_direction"] = signal_direction
    bars["signal_confidence"] = signal_confidence
    
    return bars

def create_decision_points(df_bars: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    all_decisions = []
    for symbol in PAIRS:
        bars_sym = df_bars[df_bars["symbol"] == symbol].copy()
        bars_sym = bars_sym.dropna(subset=[c for c in feature_cols if c in bars_sym.columns])
        if len(bars_sym) < 1000:
            continue
        bars_sym["MOI_z_thresh"] = bars_sym["MOI_z"].rolling(2000, min_periods=100).quantile(0.75)
        has_rank = "MOI_z_rank" in bars_sym.columns
        mask = (bars_sym["MOI_z_rank"] > 0.7) if has_rank else (bars_sym["MOI_z"] > bars_sym["MOI_z_thresh"])
        mask |= (bars_sym["vol_ratio"] > 1.5) | (bars_sym["dist_lvn_atr"] < 0.3)
        df_dec = bars_sym.loc[mask].copy()
        df_dec["bar_idx"] = df_dec.index
        if len(df_dec) > len(bars_sym) * 0.3:
            df_dec = df_dec.sample(frac=0.3 * len(bars_sym) / len(df_dec), random_state=42)
        all_decisions.append(df_dec)
        print(f"  {symbol}: {len(df_dec):,} decision points")
    return pd.concat(all_decisions, ignore_index=True)

def create_realistic_labels(df_bars, df_decision, feature_cols, horizon_sec, tp_sl_ratio=2.0):
    """
    Continuous labeling with direction bias fix:
    - TP win: y = 1.0 - (tp_first / HORIZON)  -> fast TP = ~1.0, slow TP = ~0.0
    - SL loss: y = sl_first / HORIZON         -> fast SL = ~0.0, slow SL = ~1.0
    - Direction chosen by momentum bias (cum_delta_5m sign) to avoid conflicting labels
    """
    HORIZON = int(horizon_sec * 1000 / 250)
    X_dict = {f"{d}_{r}": [] for d in ["up", "down"] for r in VOL_REGIMES}
    y_dict = {f"{d}_{r}": [] for d in ["up", "down"] for r in VOL_REGIMES}
    w_dict = {f"{d}_{r}": [] for d in ["up", "down"] for r in VOL_REGIMES}  # Recency weights
    
    for symbol in tqdm(PAIRS, desc=f"Labeling {horizon_sec}s"):
        bars_sym = df_bars[df_bars["symbol"] == symbol].reset_index(drop=True)
        dec_sym = df_decision[df_decision["symbol"] == symbol].copy()
        if len(bars_sym) < HORIZON + 10 or len(dec_sym) == 0:
            continue
        
        prices = bars_sym["price"].values
        atrs = bars_sym["ATR_5m"].values
        cum_deltas = bars_sym["cum_delta_5m"].values if "cum_delta_5m" in bars_sym.columns else np.zeros(len(bars_sym))
        n_bars = len(bars_sym)
        
        # Sample to limit processing time
        if len(dec_sym) > 50000:
            dec_sym = dec_sym.sample(50000, random_state=42)
        
        valid_features = [c for c in feature_cols if c in dec_sym.columns]
        features_arr = dec_sym[valid_features].values.astype(np.float32)
        regimes = dec_sym["vol_regime"].fillna("mid").astype(str).str.lower().values
        bar_idxs = dec_sym["bar_idx"].values
        
        bar_idx_to_pos = {idx: i for i, idx in enumerate(bars_sym.index)}
        
        for i, (bar_idx, regime, features) in enumerate(zip(bar_idxs, regimes, features_arr)):
            if regime not in VOL_REGIMES:
                regime = "mid"
            pos = bar_idx_to_pos.get(bar_idx)
            if pos is None or pos + HORIZON >= n_bars:
                continue
            
            entry, atr = prices[pos], atrs[pos]
            if np.isnan(atr) or atr <= 0:
                continue
            
            sl_dist, tp_dist = atr, tp_sl_ratio * atr
            future = prices[pos+1:pos+HORIZON+1]
            
            # Recency weight: exponential decay, recent samples get higher weight
            recency_weight = 0.5 + 0.5 * (pos / n_bars)  # Range [0.5, 1.0]
            
            # Choose direction based on momentum bias (fixes Bug #2: conflicting labels)
            momentum = cum_deltas[pos] if pos < len(cum_deltas) else 0
            direction = "up" if momentum >= 0 else "down"
            
            if direction == "up":
                # LONG
                tp_hit = np.where(future >= entry + tp_dist)[0]
                sl_hit = np.where(future <= entry - sl_dist)[0]
            else:
                # SHORT
                tp_hit = np.where(future <= entry - tp_dist)[0]
                sl_hit = np.where(future >= entry + sl_dist)[0]
            
            tp_first = tp_hit[0] if len(tp_hit) else HORIZON + 1
            sl_first = sl_hit[0] if len(sl_hit) else HORIZON + 1
            
            # Only count if resolved within horizon
            if tp_first >= HORIZON and sl_first >= HORIZON:
                continue
            
            # Continuous label: fast TP -> 1.0, fast SL -> 0.0
            if tp_first < sl_first:
                y = 1.0 - (tp_first / HORIZON)  # Fast TP = ~1.0, slow TP = ~0.0
            else:
                y = sl_first / HORIZON  # Fast SL = ~0.0, slow SL = ~1.0
            
            key = f"{direction}_{regime}"
            X_dict[key].append(features)
            y_dict[key].append(y)
            w_dict[key].append(recency_weight)

    for key in X_dict:
        X_dict[key] = np.array(X_dict[key], dtype=np.float32) if X_dict[key] else np.array([]).reshape(0, len(feature_cols))
        y_dict[key] = np.array(y_dict[key], dtype=np.float32) if y_dict[key] else np.array([])
        w_dict[key] = np.array(w_dict[key], dtype=np.float32) if w_dict[key] else np.array([])
        print(f"  {key}: {len(X_dict[key]):,}")
    return X_dict, y_dict, w_dict

# ============================================================
# MODEL TRAINING
# ============================================================

def get_params(use_gpu=False, tuned=False):
    """Get LightGBM parameters optimized for noisy financial data"""
    if tuned:
        # Optimized for noisy data: more regularization, fewer trees, lower LR
        p = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 2000,
            "max_depth": 6,  # Shallower to reduce overfitting
            "num_leaves": 31,  # Much fewer leaves (2^5 - 1)
            "learning_rate": 0.008,  # Slower learning
            "subsample": 0.6,  # More dropout
            "subsample_freq": 1,  # Apply every iteration
            "colsample_bytree": 0.5,  # Use only half features per tree
            "min_child_samples": 100,  # Require more samples per leaf
            "min_child_weight": 0.01,  # Minimum sum of weights
            "reg_alpha": 1.0,  # Strong L1 regularization
            "reg_lambda": 2.0,  # Strong L2 regularization
            "max_bin": 127,  # Fewer bins for noise reduction
            "min_data_in_bin": 10,
            "feature_fraction_seed": 42,
            "bagging_seed": 42,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "is_unbalance": True,  # Handle class imbalance
        }
    else:
        p = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt", 
            "n_estimators": 1500, "max_depth": 8, "num_leaves": 127, 
            "learning_rate": 0.015, "subsample": 0.7, "colsample_bytree": 0.7, 
            "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 0.1,
            "random_state": 42, "n_jobs": -1, "verbose": -1
        }
    if use_gpu:
        p["device"] = "gpu"
        p["gpu_use_dp"] = False
    return p


def tune_hyperparameters(X, y, feature_cols, n_trials=30):
    """Use Optuna to find optimal hyperparameters for noisy data"""
    if not OPTUNA_AVAILABLE:
        print("  Optuna not available, using tuned defaults")
        return get_params(tuned=True)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X_df = pd.DataFrame(X, columns=feature_cols[:X.shape[1]])
    
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 1000,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        
        # Simple 3-fold CV for speed
        from sklearn.model_selection import cross_val_score
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X_df, y, cv=3, scoring="roc_auc", n_jobs=1)
        return scores.mean()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best = study.best_params
    best.update({
        "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
        "n_estimators": 2000, "random_state": 42, "n_jobs": -1, "verbose": -1
    })
    print(f"  Best AUC: {study.best_value:.4f}")
    return best


def select_features_by_importance(models, feature_cols, top_k=25):
    """Select top features by importance to reduce noise"""
    importances = np.zeros(len(feature_cols))
    for model in models:
        importances += model.feature_importances_
    importances /= len(models)
    
    indices = np.argsort(importances)[::-1][:top_k]
    selected = [feature_cols[i] for i in sorted(indices)]
    return selected, importances

def purged_splits(n, n_splits=5, purge_pct=0.01):
    fold_size = n // (n_splits + 1)
    purge = int(fold_size * purge_pct)
    for i in range(n_splits):
        yield np.arange(0, fold_size * (i + 1) - purge), np.arange(fold_size * (i + 1) + purge, min(fold_size * (i + 2), n))

def calibrate_probabilities(models, X_val, y_val):
    """Calibrate model predictions using isotonic regression (works for both classifier and regressor)"""
    from sklearn.isotonic import IsotonicRegression
    
    # Get average predictions from ensemble (handle both classifier and regressor)
    if hasattr(models[0], 'predict_proba'):
        preds = np.mean([m.predict_proba(X_val)[:, 1] for m in models], axis=0)
    else:
        preds = np.mean([m.predict(X_val) for m in models], axis=0)
    
    # Clip predictions to [0, 1] for regression outputs
    preds = np.clip(preds, 0, 1)
    
    # Fit isotonic regression for calibration
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(preds, y_val)
    
    return calibrator


def train_ensemble(X, y, name, feature_cols, weights=None, params=None, n_splits=5, tune=False, use_gpu=False):
    """
    Train ensemble with:
    - Binary classification (probability of TP hit)
    - Recency weighting via sample_weight
    - Calibration on ALL validation folds (not just last)
    """
    if len(X) < 500:
        return [], {}, None
    
    # Hyperparameter tuning
    if tune and params is None:
        print(f"  Tuning hyperparameters...")
        params = tune_hyperparameters(X, y, feature_cols)
    elif params is None:
        params = get_params(use_gpu=use_gpu, tuned=True)
    
    X_df = pd.DataFrame(X, columns=feature_cols[:X.shape[1]])
    models, metrics = [], {"aucs": [], "win_rates": []}
    print(f"\n{'='*50}\nTraining {name} ({len(X):,} samples, {y.mean()*100:.1f}% positive)\n{'='*50}")
    
    # Accumulate ALL validation data for calibration
    cal_X_list, cal_y_list = [], []
    
    for fold, (tr, va) in enumerate(purged_splits(len(X), n_splits)):
        model = lgb.LGBMClassifier(**params)
        
        # Apply recency weights if provided
        fit_kwargs = {
            "X": X_df.iloc[tr],
            "y": y[tr],
            "eval_set": [(X_df.iloc[va], y[va])],
            "callbacks": [lgb.early_stopping(150, verbose=False)]
        }
        if weights is not None:
            fit_kwargs["sample_weight"] = weights[tr]
        
        model.fit(**fit_kwargs)
        preds = model.predict_proba(X_df.iloc[va])[:, 1]
        
        # Metrics for classification
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y[va], preds) if len(np.unique(y[va])) > 1 else 0.5
        
        # Win rate at various thresholds
        for thresh in [0.50, 0.55, 0.60, 0.65]:
            mask = preds >= thresh
            if mask.sum() > 0:
                wr = y[va][mask].mean()
                metrics.setdefault(f"wr_{thresh}", []).append(wr)
        
        metrics["aucs"].append(auc)
        models.append(model)
        
        # Accumulate validation data for calibration
        cal_X_list.append(X_df.iloc[va])
        cal_y_list.append(y[va])
        
        print(f"  Fold {fold}: AUC={auc:.4f}")
    
    # Calibrate on ALL validation folds (pooled)
    cal_X_all = pd.concat(cal_X_list, ignore_index=True)
    cal_y_all = np.concatenate(cal_y_list)
    calibrator = calibrate_probabilities(models, cal_X_all, cal_y_all)
    
    print(f"\n{name} Mean AUC: {np.mean(metrics['aucs']):.4f}")
    
    # Print win rates at thresholds
    for thresh in [0.50, 0.55, 0.60, 0.65]:
        key = f"wr_{thresh}"
        if key in metrics and metrics[key]:
            print(f"  Win rate @{thresh}: {np.mean(metrics[key])*100:.1f}%")
    
    return models, metrics, calibrator

# ============================================================
# MAIN
# ============================================================

def create_labels_for_symbol(bars_sym, feature_cols, horizon_sec, tp_sl_ratio=2.0):
    """
    Create labels for a single symbol using SIGNAL-BASED candidate generation.
    
    FIXES APPLIED:
    1. Near-signal negatives: Add pos ± k (k=2,5,10) as no-trade to teach TIMING matters
    2. Both directions: For each signal, compute BOTH long and short outcomes
       - Model learns directional discrimination, not confirmation
    
    Two-stage approach:
    - Stage 1 (trade_filter): 0=NO_TRADE (bad timing), 1=TRADE (good timing)
    - Stage 2 (direction): For BOTH directions - learns which is better
    
    Returns X, y, weights dictionaries keyed by direction_regime.
    """
    HORIZON = int(horizon_sec * 1000 / 250)
    X_dict = {f"{d}_{r}": [] for d in ["up", "down"] for r in VOL_REGIMES}
    y_dict = {f"{d}_{r}": [] for d in ["up", "down"] for r in VOL_REGIMES}
    w_dict = {f"{d}_{r}": [] for d in ["up", "down"] for r in VOL_REGIMES}
    
    # No-trade samples (learn when NOT to trade - including near-signal negatives)
    X_notrade = {r: [] for r in VOL_REGIMES}
    y_notrade = {r: [] for r in VOL_REGIMES}
    w_notrade = {r: [] for r in VOL_REGIMES}
    
    # Signal statistics (all 11 signals)
    signal_stats = {s: {"total": 0, "long_wins": 0, "short_wins": 0} for s in 
                    ["entropy_flow", "compression_break", "dominance_decay", "absorption_fade", 
                     "momentum_continuation", "trapped_longs", "trapped_shorts", "liquidation_cascade",
                     "temporary_impact", "inventory_lock", "structure_rejection"]}
    
    # Near-signal offset for timing negatives
    NEAR_SIGNAL_OFFSETS = [2, 5, 10]
    
    bars_sym = bars_sym.reset_index(drop=True)
    if len(bars_sym) < HORIZON + 100:
        return X_dict, y_dict, w_dict, X_notrade, y_notrade, w_notrade
    
    # Detect signals - these are CANDIDATES, not guaranteed trades
    print(f"    Detecting signal candidates...", flush=True)
    bars_sym = detect_signal_conditions_vectorized(bars_sym)
    
    prices = bars_sym["price"].values
    atrs = bars_sym["ATR_5m"].values
    n_bars = len(bars_sym)
    
    # Get signal-based candidate points
    signal_mask = bars_sym["signal_type"].notna()
    dec_indices = bars_sym.loc[signal_mask].index.values
    
    print(f"    Found {len(dec_indices):,} signal candidates", flush=True)
    
    # Sample if too many
    if len(dec_indices) > 50000:
        np.random.seed(42)
        dec_indices = np.random.choice(dec_indices, 50000, replace=False)
    
    valid_features = [c for c in feature_cols if c in bars_sym.columns]
    
    # Track processed near-signal positions to avoid duplicates
    near_signal_positions = set()
    
    # Process signal candidates
    for pos in dec_indices:
        if pos + HORIZON >= n_bars:
            continue
        
        entry, atr = prices[pos], atrs[pos]
        if np.isnan(atr) or atr <= 0:
            continue
        
        regime = bars_sym.iloc[pos].get("vol_regime", "mid")
        if pd.isna(regime) or regime not in VOL_REGIMES:
            regime = "mid"
        regime = str(regime).lower()
        
        sl_dist, tp_dist = atr, tp_sl_ratio * atr
        future = prices[pos+1:pos+HORIZON+1]
        
        # Recency weight
        recency_weight = 0.5 + 0.5 * (pos / n_bars)
        
        signal_type = bars_sym.iloc[pos]["signal_type"]
        features = bars_sym.iloc[pos][valid_features].values.astype(np.float32)
        
        # ============================================================
        # FIX #2: Compute BOTH directions - model learns discrimination
        # ============================================================
        # LONG outcome
        long_tp_hit = np.where(future >= entry + tp_dist)[0]
        long_sl_hit = np.where(future <= entry - sl_dist)[0]
        long_tp_first = long_tp_hit[0] if len(long_tp_hit) else HORIZON + 1
        long_sl_first = long_sl_hit[0] if len(long_sl_hit) else HORIZON + 1
        
        # SHORT outcome
        short_tp_hit = np.where(future <= entry - tp_dist)[0]
        short_sl_hit = np.where(future >= entry + sl_dist)[0]
        short_tp_first = short_tp_hit[0] if len(short_tp_hit) else HORIZON + 1
        short_sl_first = short_sl_hit[0] if len(short_sl_hit) else HORIZON + 1
        
        # Track signal statistics
        if signal_type in signal_stats:
            signal_stats[signal_type]["total"] += 1
        
        # Check if either direction resolved
        long_resolved = long_tp_first < HORIZON or long_sl_first < HORIZON
        short_resolved = short_tp_first < HORIZON or short_sl_first < HORIZON
        
        if not long_resolved and not short_resolved:
            # Neither direction resolved - this is a no-trade (unresolved)
            X_notrade[regime].append(features)
            y_notrade[regime].append(0.0)
            w_notrade[regime].append(recency_weight)
            continue
        
        # LONG direction label: 1 if TP hit first, 0 if SL hit first
        if long_resolved:
            y_long = 1.0 if long_tp_first < long_sl_first else 0.0
            X_dict[f"up_{regime}"].append(features)
            y_dict[f"up_{regime}"].append(y_long)
            w_dict[f"up_{regime}"].append(recency_weight)
            if signal_type in signal_stats and y_long == 1.0:
                signal_stats[signal_type]["long_wins"] += 1
        
        # SHORT direction label: 1 if TP hit first, 0 if SL hit first
        if short_resolved:
            y_short = 1.0 if short_tp_first < short_sl_first else 0.0
            X_dict[f"down_{regime}"].append(features)
            y_dict[f"down_{regime}"].append(y_short)
            w_dict[f"down_{regime}"].append(recency_weight)
            if signal_type in signal_stats and y_short == 1.0:
                signal_stats[signal_type]["short_wins"] += 1
        
        # Signal point is a positive for filter (should trade)
        X_notrade[regime].append(features)
        y_notrade[regime].append(1.0)
        w_notrade[regime].append(recency_weight)
        
        # ============================================================
        # FIX #1: Add near-signal negatives to teach TIMING matters
        # ============================================================
        for offset in NEAR_SIGNAL_OFFSETS:
            for near_pos in [pos - offset, pos + offset]:
                # Skip if out of bounds or already processed
                if near_pos < 0 or near_pos + HORIZON >= n_bars:
                    continue
                if near_pos in near_signal_positions:
                    continue
                if near_pos in dec_indices:
                    continue  # Don't mark actual signals as negatives
                
                near_signal_positions.add(near_pos)
                
                near_atr = atrs[near_pos]
                if np.isnan(near_atr) or near_atr <= 0:
                    continue
                
                near_regime = bars_sym.iloc[near_pos].get("vol_regime", "mid")
                if pd.isna(near_regime) or near_regime not in VOL_REGIMES:
                    near_regime = "mid"
                near_regime = str(near_regime).lower()
                
                near_weight = 0.5 + 0.5 * (near_pos / n_bars)
                near_features = bars_sym.iloc[near_pos][valid_features].values.astype(np.float32)
                
                # Near-signal points are NEGATIVES for filter (wrong timing)
                X_notrade[near_regime].append(near_features)
                y_notrade[near_regime].append(0.0)  # 0 = don't trade (wrong timing)
                w_notrade[near_regime].append(near_weight * 0.8)  # Slightly lower weight
    
    # Print signal statistics
    print(f"    Signal stats (both directions):", flush=True)
    for sig, stats in signal_stats.items():
        if stats["total"] > 0:
            long_wr = stats["long_wins"] / stats["total"] * 100
            short_wr = stats["short_wins"] / stats["total"] * 100
            print(f"      {sig}: {stats['total']:,} | LONG WR: {long_wr:.1f}% | SHORT WR: {short_wr:.1f}%", flush=True)
    
    print(f"    Near-signal negatives added: {len(near_signal_positions):,}", flush=True)
    
    # Convert to arrays
    for key in X_dict:
        X_dict[key] = np.array(X_dict[key], dtype=np.float32) if X_dict[key] else np.array([]).reshape(0, len(valid_features))
        y_dict[key] = np.array(y_dict[key], dtype=np.float32) if y_dict[key] else np.array([])
        w_dict[key] = np.array(w_dict[key], dtype=np.float32) if w_dict[key] else np.array([])
    
    for regime in VOL_REGIMES:
        X_notrade[regime] = np.array(X_notrade[regime], dtype=np.float32) if X_notrade[regime] else np.array([]).reshape(0, len(valid_features))
        y_notrade[regime] = np.array(y_notrade[regime], dtype=np.float32) if y_notrade[regime] else np.array([])
        w_notrade[regime] = np.array(w_notrade[regime], dtype=np.float32) if w_notrade[regime] else np.array([])
    
    return X_dict, y_dict, w_dict, X_notrade, y_notrade, w_notrade


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["full", "quick"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    args = parser.parse_args()

    days = 3 if args.mode == "quick" else DAYS
    print(f"{'='*70}\nHYDRA V3 FINAL - PER-SYMBOL TRAINING ({days} days)\n{'='*70}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    feature_cols = FEATURE_COLS.copy()
    
    # Save feature columns
    with open("feature_columns_v3.json", "w") as f:
        json.dump(feature_cols, f)
    
    total_models = 0
    all_model_names = []

    # Train per-symbol models (one at a time to save memory)
    for symbol in PAIRS:
        print(f"\n{'#'*70}\n# SYMBOL: {symbol}\n{'#'*70}")
        
        # Fetch data for this symbol only
        sym_df = fetch_symbol_data(symbol, days)
        if len(sym_df) < 1000:
            print(f"  Skipping {symbol}: insufficient data")
            continue
        
        # Build bars for this symbol
        print(f"  Building bars from {len(sym_df):,} trades...")
        bars_sym = compute_bars_vectorized(sym_df, symbol)
        del sym_df; gc.collect()
        
        print(f"  Computing features for {len(bars_sym):,} bars...")
        bars_sym = compute_features_vectorized(bars_sym)
        
        # Create symbol output directory
        symbol_dir = f"{OUTPUT_DIR}/{symbol}"
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Train for each horizon
        for horizon in HORIZONS:
            print(f"\n  HORIZON: {horizon}s")
            X, y, w, X_nt, y_nt, w_nt = create_labels_for_symbol(bars_sym, feature_cols, horizon)
            
            # Train TRADE FILTER models (should we trade at all?)
            for regime in VOL_REGIMES:
                if len(X_nt[regime]) < 500:
                    print(f"    filter_{regime}: Skipping ({len(X_nt[regime])} samples)")
                    continue
                
                print(f"    filter_{regime}: {len(X_nt[regime]):,} samples")
                models, metrics, calibrator = train_ensemble(
                    X_nt[regime], y_nt[regime], f"{symbol}_filter_{regime}_{horizon}", feature_cols,
                    weights=w_nt[regime], tune=args.tune, use_gpu=args.gpu
                )
                
                if models:
                    model_name = f"filter_{regime}_{horizon}"
                    all_model_names.append(f"{symbol}/{model_name}")
                    total_models += 1
                    
                    with open(f"{symbol_dir}/models_{model_name}.pkl", "wb") as f:
                        pickle.dump({"models": models, "calibrator": calibrator}, f)
            
            # Train DIRECTION models (given we should trade, will TP hit?)
            for key in X:
                if len(X[key]) < 500:
                    print(f"    {key}: Skipping ({len(X[key])} samples)")
                    continue
                
                print(f"    {key}: {len(X[key]):,} samples")
                models, metrics, calibrator = train_ensemble(
                    X[key], y[key], f"{symbol}_{key}_{horizon}", feature_cols,
                    weights=w[key], tune=args.tune, use_gpu=args.gpu
                )
                
                if models:
                    model_name = f"{key}_{horizon}"
                    all_model_names.append(f"{symbol}/{model_name}")
                    total_models += 1
                    
                    with open(f"{symbol_dir}/models_{model_name}.pkl", "wb") as f:
                        pickle.dump({"models": models, "calibrator": calibrator}, f)
        
        del bars_sym; gc.collect()

    # Save training summary
    summary = {
        "training_type": "per_symbol_with_filter",
        "symbols": PAIRS,
        "models_per_symbol": 18,  # 6 direction + 6 filter per horizon × 2 horizons... actually 3 filter + 6 direction = 9 per horizon
        "total_models": total_models,
        "features": feature_cols,
        "horizons": HORIZONS,
        "regimes": VOL_REGIMES,
        "days": days,
        "model_types": {
            "filter": "Trade/No-trade classifier (should we enter?)",
            "direction": "TP probability classifier (will we win?)"
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(f"{OUTPUT_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE - {total_models} models saved to {OUTPUT_DIR}/")
    print(f"Structure:")
    print(f"  {OUTPUT_DIR}/<SYMBOL>/models_filter_<regime>_<horizon>.pkl  (trade/no-trade)")
    print(f"  {OUTPUT_DIR}/<SYMBOL>/models_<direction>_<regime>_<horizon>.pkl  (TP probability)")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
