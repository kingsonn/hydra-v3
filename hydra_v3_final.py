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
DAYS = 21
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

def fetch_all_data(pairs: List[str], days: int) -> pd.DataFrame:
    all_dfs = []
    end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)

    for symbol in pairs:
        print(f"\nFetching {symbol}")
        for i in tqdm(range(days)):
            day = start_date + timedelta(days=i)
            df_day = fetch_aggtrades_day(symbol, day)
            if df_day is not None:
                all_dfs.append(df_day)

    df = pd.concat(all_dfs, ignore_index=True)
    df = df.rename(columns={"price": "price", "quantity": "qty", "transact_time": "timestamp", "is_buyer_maker": "is_sell"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["price"] = df["price"].astype("float32")
    df["qty"] = df["qty"].astype("float32")
    df["is_sell"] = df["is_sell"].astype("int8")
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"\nTotal rows: {len(df):,}")
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
# DECISION POINTS + LABELING
# ============================================================

FEATURE_COLS = [
    "MOI_250ms", "MOI_1s", "MOI_5s", "MOI_20s", "MOI_z", "delta_velocity", "delta_velocity_5s",
    "AggressionPersistence", "MOI_roc_1s", "MOI_roc_5s", "MOI_acceleration", "MOI_flip_rate",
    "absorption_z", "price_impact_z", "vol_1m", "vol_5m", "vol_ratio", "vol_rank", "ATR_5m_pct",
    "dist_lvn_atr", "dist_poc_atr", "dist_lvn", "dist_poc", "hour_sin", "hour_cos", "is_weekend",
    "trade_intensity", "trade_intensity_z", "cum_delta_1m", "cum_delta_5m",
    "MOI_z_rank", "vol_ratio_rank", "absorption_z_rank", "AggressionPersistence_rank",
    "MOI_flip_rate_rank", "trade_intensity_z_rank", "cum_delta_5m_rank", "MOI_z_relative", "momentum_rank",
]

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
            df_dec = df_dec.sample(frac=0.3 * len(bars_sym) / len(df_dec))
        all_decisions.append(df_dec)
        print(f"  {symbol}: {len(df_dec):,} decision points")
    return pd.concat(all_decisions, ignore_index=True)

def create_realistic_labels(df_bars, df_decision, feature_cols, horizon_sec, tp_sl_ratio=2.0):
    """Vectorized labeling - much faster than row-by-row iteration"""
    HORIZON = int(horizon_sec * 1000 / 250)
    X_dict = {f"{d}_{r}": [] for d in ["up", "down"] for r in VOL_REGIMES}
    y_dict = {f"{d}_{r}": [] for d in ["up", "down"] for r in VOL_REGIMES}
    
    for symbol in tqdm(PAIRS, desc=f"Labeling {horizon_sec}s"):
        bars_sym = df_bars[df_bars["symbol"] == symbol].reset_index(drop=True)
        dec_sym = df_decision[df_decision["symbol"] == symbol].copy()
        if len(bars_sym) < HORIZON + 10 or len(dec_sym) == 0:
            continue
        
        prices = bars_sym["price"].values
        atrs = bars_sym["ATR_5m"].values
        n_bars = len(bars_sym)
        
        # Sample to limit processing time
        if len(dec_sym) > 50000:
            dec_sym = dec_sym.sample(50000, random_state=42)
        
        # Get decision point indices
        dec_indices = dec_sym.index.values
        valid_features = [c for c in feature_cols if c in dec_sym.columns]
        features_arr = dec_sym[valid_features].values.astype(np.float32)
        regimes = dec_sym["vol_regime"].fillna("mid").astype(str).str.lower().values
        bar_idxs = dec_sym["bar_idx"].values
        
        # Find bar positions
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
            
            # LONG
            tp_hit = np.where(future >= entry + tp_dist)[0]
            sl_hit = np.where(future <= entry - sl_dist)[0]
            tp_first = tp_hit[0] if len(tp_hit) else HORIZON + 1
            sl_first = sl_hit[0] if len(sl_hit) else HORIZON + 1
            if tp_first < HORIZON or sl_first < HORIZON:
                X_dict[f"up_{regime}"].append(features)
                y_dict[f"up_{regime}"].append(1.0 if tp_first < sl_first else 0.0)
            
            # SHORT
            tp_hit = np.where(future <= entry - tp_dist)[0]
            sl_hit = np.where(future >= entry + sl_dist)[0]
            tp_first = tp_hit[0] if len(tp_hit) else HORIZON + 1
            sl_first = sl_hit[0] if len(sl_hit) else HORIZON + 1
            if tp_first < HORIZON or sl_first < HORIZON:
                X_dict[f"down_{regime}"].append(features)
                y_dict[f"down_{regime}"].append(1.0 if tp_first < sl_first else 0.0)

    for key in X_dict:
        X_dict[key] = np.array(X_dict[key], dtype=np.float32) if X_dict[key] else np.array([]).reshape(0, len(feature_cols))
        y_dict[key] = np.array(y_dict[key], dtype=np.float32) if y_dict[key] else np.array([])
        print(f"  {key}: {len(X_dict[key]):,}")
    return X_dict, y_dict

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
    """Calibrate model probabilities using isotonic regression"""
    from sklearn.isotonic import IsotonicRegression
    
    # Get average predictions from ensemble
    preds = np.mean([m.predict_proba(X_val)[:, 1] for m in models], axis=0)
    
    # Fit isotonic regression for calibration
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(preds, y_val)
    
    return calibrator


def train_ensemble(X, y, name, feature_cols, params=None, n_splits=5, tune=False, use_gpu=False):
    if len(X) < 500:
        return [], {}, None
    
    # Hyperparameter tuning
    if tune and params is None:
        print(f"  Tuning hyperparameters...")
        params = tune_hyperparameters(X, y, feature_cols)
    elif params is None:
        params = get_params(use_gpu=use_gpu, tuned=True)  # Use tuned defaults
    
    X_df = pd.DataFrame(X, columns=feature_cols[:X.shape[1]])
    models, metrics = [], {"maes": [], "aucs": [], "win_rates": []}
    print(f"\n{'='*50}\nTraining {name} ({len(X):,} samples)\n{'='*50}")
    
    # Track calibration data from last fold
    last_val_X, last_val_y = None, None
    
    for fold, (tr, va) in enumerate(purged_splits(len(X), n_splits)):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_df.iloc[tr], y[tr], 
            eval_set=[(X_df.iloc[va], y[va])], 
            callbacks=[lgb.early_stopping(150, verbose=False)]
        )
        preds = model.predict_proba(X_df.iloc[va])[:, 1]
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y[va], preds) if len(np.unique(y[va])) > 1 else 0.5
        
        # Calculate win rate at different thresholds
        for thresh in [0.55, 0.60, 0.65]:
            mask = preds >= thresh
            if mask.sum() > 0:
                wr = y[va][mask].mean()
                metrics.setdefault(f"wr_{thresh}", []).append(wr)
        
        metrics["aucs"].append(auc)
        models.append(model)
        last_val_X, last_val_y = X_df.iloc[va], y[va]
        print(f"  Fold {fold}: AUC={auc:.4f}")
    
    # Calibrate on last validation fold
    calibrator = calibrate_probabilities(models, last_val_X, last_val_y) if last_val_X is not None else None
    
    print(f"\n{name} Mean AUC: {np.mean(metrics['aucs']):.4f}")
    
    # Print win rates at thresholds
    for thresh in [0.55, 0.60, 0.65]:
        key = f"wr_{thresh}"
        if key in metrics and metrics[key]:
            print(f"  Win rate @{thresh}: {np.mean(metrics[key])*100:.1f}%")
    
    return models, metrics, calibrator

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full", choices=["full", "quick"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    args = parser.parse_args()

    days = 3 if args.mode == "quick" else DAYS
    print(f"{'='*70}\nHYDRA V3 FINAL - {args.mode.upper()} MODE ({days} days)\n{'='*70}")

    # Fetch data
    df = fetch_all_data(PAIRS, days)

    # Build bars
    all_bars = []
    for symbol in PAIRS:
        print(f"\nProcessing {symbol}...")
        sym_df = df[df["symbol"] == symbol].copy()
        print(f"  Building bars from {len(sym_df):,} trades...")
        bars = compute_bars_vectorized(sym_df, symbol)
        print(f"  Computing features for {len(bars):,} bars...")
        bars = compute_features_vectorized(bars)
        all_bars.append(bars)
        print(f"  Done.")
        del sym_df; gc.collect()
    df_bars = pd.concat(all_bars, ignore_index=True)
    del df, all_bars; gc.collect()

    # Cross-sectional
    df_bars = add_cross_sectional_features(df_bars)

    # Decision points
    feature_cols = [c for c in FEATURE_COLS if c in df_bars.columns]
    print(f"\nUsing {len(feature_cols)} features")
    df_decision = create_decision_points(df_bars, feature_cols)

    # Save features
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open("feature_columns_v3.json", "w") as f:
        json.dump(feature_cols, f)

    # Train models
    all_models = {}
    all_calibrators = {}
    for horizon in HORIZONS:
        print(f"\n{'='*50}\nHORIZON: {horizon}s\n{'='*50}")
        X, y = create_realistic_labels(df_bars, df_decision, feature_cols, horizon)

        for key in X:
            if len(X[key]) > 0:
                models, metrics, calibrator = train_ensemble(
                    X[key], y[key], f"{key}_{horizon}", feature_cols,
                    tune=args.tune, use_gpu=args.gpu
                )
                if models:
                    model_name = f"{key}_{horizon}"
                    all_models[model_name] = models
                    all_calibrators[model_name] = calibrator
                    
                    # Save models and calibrator together
                    with open(f"{OUTPUT_DIR}/models_{model_name}.pkl", "wb") as f:
                        pickle.dump({"models": models, "calibrator": calibrator}, f)

    # Save training summary
    summary = {
        "models": list(all_models.keys()),
        "features": feature_cols,
        "horizons": HORIZONS,
        "regimes": VOL_REGIMES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(f"{OUTPUT_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}\nTRAINING COMPLETE - {len(all_models)} models saved to {OUTPUT_DIR}/\n{'='*70}")

if __name__ == "__main__":
    main()
