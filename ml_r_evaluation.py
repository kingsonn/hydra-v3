"""
ML R-based Evaluation System

Evaluates ML signals using MFE/MAE in R (risk units) rather than fixed TP/SL.
This matches ML training methodology.
"""

import pandas as pd
import numpy as np
import time

HORIZON_SECONDS = 300  # 300s forward simulation
ATR_MULTIPLIER = 0.8   # Stop distance = 0.8 * ATR_5m


def compute_atr_5m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 5-minute ATR for each symbol.
    ATR = Average True Range over 5-minute windows.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['ATR_5m'] = np.nan
    
    print("Computing 5-minute ATR by symbol...")
    
    for symbol in df['symbol'].unique():
        symbol_mask = df['symbol'] == symbol
        symbol_df = df[symbol_mask].sort_values('timestamp').copy()
        
        # Create 5-min buckets
        symbol_df['ts_5min'] = symbol_df['timestamp'].dt.floor('5min')
        
        # Compute high, low, close per 5-min bucket
        ohlc = symbol_df.groupby('ts_5min')['price'].agg(['max', 'min', 'last']).reset_index()
        ohlc.columns = ['ts_5min', 'high', 'low', 'close']
        ohlc['prev_close'] = ohlc['close'].shift(1)
        
        # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
        ohlc['tr1'] = ohlc['high'] - ohlc['low']
        ohlc['tr2'] = (ohlc['high'] - ohlc['prev_close']).abs()
        ohlc['tr3'] = (ohlc['low'] - ohlc['prev_close']).abs()
        ohlc['true_range'] = ohlc[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # ATR = 14-period EMA of True Range (but use rolling mean for simplicity)
        ohlc['atr'] = ohlc['true_range'].rolling(window=14, min_periods=1).mean()
        
        # Map ATR back to each row
        atr_dict = ohlc.set_index('ts_5min')['atr'].to_dict()
        symbol_df['ATR_5m'] = symbol_df['ts_5min'].map(atr_dict)
        
        df.loc[symbol_mask, 'ATR_5m'] = symbol_df['ATR_5m'].values
        
        print(f"  {symbol}: ATR range [{symbol_df['ATR_5m'].min():.6f}, {symbol_df['ATR_5m'].max():.6f}]")
    
    return df


def compute_stop_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stop_price based on direction and ATR.
    LONG: stop_price = price - (0.8 * ATR_5m)
    SHORT: stop_price = price + (0.8 * ATR_5m)
    """
    df = df.copy()
    
    long_mask = df['Decision'] == 'LONG'
    short_mask = df['Decision'] == 'SHORT'
    
    df['stop_price'] = np.nan
    df.loc[long_mask, 'stop_price'] = df.loc[long_mask, 'price'] - (ATR_MULTIPLIER * df.loc[long_mask, 'ATR_5m'])
    df.loc[short_mask, 'stop_price'] = df.loc[short_mask, 'price'] + (ATR_MULTIPLIER * df.loc[short_mask, 'ATR_5m'])
    
    return df


def compute_r(df: pd.DataFrame) -> pd.DataFrame:
    """Compute R = abs(entry_price - stop_price)"""
    df = df.copy()
    df['R'] = (df['price'] - df['stop_price']).abs()
    return df


def simulate_forward_and_compute_mfe_mae(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each signal, simulate forward 300s and compute MFE_R and MAE_R.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialize columns
    df['MFE_R'] = np.nan
    df['MAE_R'] = np.nan
    df['hit_stop'] = False
    df['max_price_300s'] = np.nan
    df['min_price_300s'] = np.nan
    
    print("\nSimulating forward price paths (300s horizon)...")
    
    # Only process signals (rows with Decision = LONG or SHORT)
    signal_mask = df['Decision'].isin(['LONG', 'SHORT'])
    
    total_signals = signal_mask.sum()
    print(f"Processing {total_signals:,} signals...")
    
    start_time = time.time()
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].sort_values('timestamp').reset_index()
        orig_indices = symbol_df['index'].values
        
        timestamps = symbol_df['timestamp'].values
        prices = symbol_df['price'].values.astype(np.float64)
        decisions = symbol_df['Decision'].values
        r_values = symbol_df['R'].values
        
        # Convert timestamps to seconds for comparison
        ts_seconds = (timestamps.astype('datetime64[s]').astype(np.int64))
        
        processed = 0
        for i in range(len(symbol_df)):
            if decisions[i] not in ['LONG', 'SHORT']:
                continue
            
            entry_price = prices[i]
            entry_ts = ts_seconds[i]
            r_val = r_values[i]
            
            if pd.isna(r_val) or r_val == 0:
                continue
            
            # Find future prices within horizon
            horizon_end = entry_ts + HORIZON_SECONDS
            future_mask = (ts_seconds > entry_ts) & (ts_seconds <= horizon_end)
            future_prices = prices[future_mask]
            
            if len(future_prices) == 0:
                continue
            
            max_future = future_prices.max()
            min_future = future_prices.min()
            
            orig_idx = orig_indices[i]
            df.loc[orig_idx, 'max_price_300s'] = max_future
            df.loc[orig_idx, 'min_price_300s'] = min_future
            
            if decisions[i] == 'LONG':
                mfe = (max_future - entry_price) / r_val
                mae = (entry_price - min_future) / r_val
            else:  # SHORT
                mfe = (entry_price - min_future) / r_val
                mae = (max_future - entry_price) / r_val
            
            df.loc[orig_idx, 'MFE_R'] = mfe
            df.loc[orig_idx, 'MAE_R'] = mae
            df.loc[orig_idx, 'hit_stop'] = mae >= 1.0
            
            processed += 1
        
        print(f"  {symbol}: {processed} signals processed")
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s")
    
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute all evaluation metrics."""
    # Filter to signals with valid MFE/MAE
    signals = df[df['MFE_R'].notna() & df['MAE_R'].notna()].copy()
    
    if len(signals) == 0:
        return {}
    
    # Realized R (simple: MFE - MAE, clipped at -1 for stop hits)
    signals['MAE_R_clipped'] = signals['MAE_R'].clip(upper=1.0)
    signals['realized_R'] = signals['MFE_R'] - signals['MAE_R_clipped']
    
    # Alternative: if hit stop, realized = -1, else = MFE
    signals['realized_R_alt'] = np.where(
        signals['hit_stop'],
        -1.0,
        signals['MFE_R'].clip(upper=3.0)  # Cap MFE at 3R for outliers
    )
    
    metrics = {}
    
    # 1. Expected R per trade
    metrics['expected_R'] = signals['realized_R'].mean()
    metrics['expected_R_alt'] = signals['realized_R_alt'].mean()
    
    # 2. Profit Factor
    positive_r = signals[signals['realized_R'] > 0]['realized_R'].sum()
    negative_r = signals[signals['realized_R'] < 0]['realized_R'].sum()
    metrics['profit_factor'] = positive_r / abs(negative_r) if negative_r != 0 else np.inf
    
    # 3. Win Rate (MFE >= 1.0)
    metrics['win_rate'] = (signals['MFE_R'] >= 1.0).mean() * 100
    
    # 4. Hit Stop Rate
    metrics['stop_rate'] = signals['hit_stop'].mean() * 100
    
    # 5. Tail Contribution (top 20% by edge_rank)
    if 'Model_output_300' in signals.columns:
        # Calculate edge_rank as percentile within the dataset
        signals['edge_rank'] = signals['Model_output_300'].rank(pct=True)
        
        top_20_mask = signals['edge_rank'] >= 0.80
        top_20_signals = signals[top_20_mask]
        
        total_r = signals['realized_R'].sum()
        top_20_r = top_20_signals['realized_R'].sum()
        
        metrics['tail_pnl_pct'] = (top_20_r / total_r * 100) if total_r != 0 else 0
        metrics['top_20_count'] = len(top_20_signals)
        metrics['top_20_expected_R'] = top_20_signals['realized_R'].mean() if len(top_20_signals) > 0 else 0
    
    # Additional metrics
    metrics['total_signals'] = len(signals)
    metrics['avg_MFE_R'] = signals['MFE_R'].mean()
    metrics['avg_MAE_R'] = signals['MAE_R'].mean()
    metrics['total_R'] = signals['realized_R'].sum()
    
    return metrics


def main():
    print("="*70)
    print("ML R-BASED EVALUATION SYSTEM")
    print("="*70)
    
    print("\nLoading tradesdata_with_ml_predictions.csv...")
    df = pd.read_csv('tradesdata_with_ml_predictions.csv', low_memory=False)
    print(f"Loaded {len(df):,} rows")
    
    # Filter to signals only
    signals_df = df[df['Decision'].isin(['LONG', 'SHORT'])].copy()
    print(f"Signals: {len(signals_df):,}")
    
    # Step 1: Compute ATR_5m
    print("\n" + "-"*70)
    print("STEP 1: Computing 5-minute ATR")
    print("-"*70)
    df = compute_atr_5m(df)
    
    # Step 2: Compute stop_price
    print("\n" + "-"*70)
    print("STEP 2: Computing stop_price")
    print("-"*70)
    df = compute_stop_price(df)
    
    long_stops = df[df['Decision'] == 'LONG']['stop_price']
    short_stops = df[df['Decision'] == 'SHORT']['stop_price']
    print(f"  LONG stops: avg distance = {(df[df['Decision']=='LONG']['price'] - long_stops).mean():.6f}")
    print(f"  SHORT stops: avg distance = {(short_stops - df[df['Decision']=='SHORT']['price']).mean():.6f}")
    
    # Step 3: Compute R
    print("\n" + "-"*70)
    print("STEP 3: Computing R (risk unit)")
    print("-"*70)
    df = compute_r(df)
    
    r_stats = df[df['Decision'].isin(['LONG', 'SHORT'])]['R']
    print(f"  R statistics:")
    print(f"    Mean: {r_stats.mean():.6f}")
    print(f"    Median: {r_stats.median():.6f}")
    print(f"    Min: {r_stats.min():.6f}")
    print(f"    Max: {r_stats.max():.6f}")
    
    # Step 4: Simulate forward and compute MFE/MAE
    print("\n" + "-"*70)
    print("STEP 4: Simulating forward price paths (300s horizon)")
    print("-"*70)
    df = simulate_forward_and_compute_mfe_mae(df)
    
    # Step 5: Compute metrics
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    metrics = compute_metrics(df)
    
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-"*45)
    print(f"{'Total Signals':<30} {metrics.get('total_signals', 0):>15,}")
    print(f"{'Expected R (per trade)':<30} {metrics.get('expected_R', 0):>15.4f}")
    print(f"{'Expected R (alt method)':<30} {metrics.get('expected_R_alt', 0):>15.4f}")
    print(f"{'Profit Factor':<30} {metrics.get('profit_factor', 0):>15.2f}")
    print(f"{'Win Rate (MFE >= 1R)':<30} {metrics.get('win_rate', 0):>14.2f}%")
    print(f"{'Stop Hit Rate':<30} {metrics.get('stop_rate', 0):>14.2f}%")
    print(f"{'Avg MFE (in R)':<30} {metrics.get('avg_MFE_R', 0):>15.4f}")
    print(f"{'Avg MAE (in R)':<30} {metrics.get('avg_MAE_R', 0):>15.4f}")
    print(f"{'Total R (cumulative)':<30} {metrics.get('total_R', 0):>15.2f}")
    
    print("\n" + "-"*45)
    print("TAIL CONTRIBUTION (Top 20% by ML edge score):")
    print("-"*45)
    print(f"{'Top 20% Signal Count':<30} {metrics.get('top_20_count', 0):>15,}")
    print(f"{'Top 20% Expected R':<30} {metrics.get('top_20_expected_R', 0):>15.4f}")
    print(f"{'Top 20% PnL Contribution':<30} {metrics.get('tail_pnl_pct', 0):>14.2f}%")
    
    # By Direction
    print("\n" + "="*70)
    print("BY DIRECTION")
    print("="*70)
    
    for direction in ['LONG', 'SHORT']:
        dir_df = df[df['Decision'] == direction]
        dir_metrics = compute_metrics(dir_df)
        
        if dir_metrics:
            print(f"\n{direction}:")
            print(f"  Signals: {dir_metrics.get('total_signals', 0):,}")
            print(f"  Expected R: {dir_metrics.get('expected_R', 0):.4f}")
            print(f"  Profit Factor: {dir_metrics.get('profit_factor', 0):.2f}")
            print(f"  Win Rate: {dir_metrics.get('win_rate', 0):.2f}%")
            print(f"  Stop Hit Rate: {dir_metrics.get('stop_rate', 0):.2f}%")
    
    # By Symbol
    print("\n" + "="*70)
    print("BY SYMBOL")
    print("="*70)
    
    for symbol in sorted(df['symbol'].unique()):
        sym_df = df[df['symbol'] == symbol]
        sym_metrics = compute_metrics(sym_df)
        
        if sym_metrics and sym_metrics.get('total_signals', 0) > 0:
            print(f"\n{symbol}:")
            print(f"  Signals: {sym_metrics.get('total_signals', 0):,}")
            print(f"  Expected R: {sym_metrics.get('expected_R', 0):.4f}")
            print(f"  Profit Factor: {sym_metrics.get('profit_factor', 0):.2f}")
            print(f"  Win Rate: {sym_metrics.get('win_rate', 0):.2f}%")
    
    # Compare ML filters
    print("\n" + "="*70)
    print("ML FILTER COMPARISON")
    print("="*70)
    
    # All signals
    all_metrics = compute_metrics(df)
    print(f"\nAll Signals: {all_metrics.get('total_signals', 0):,} | Expected R: {all_metrics.get('expected_R', 0):.4f}")
    
    # Top 15% ML filter (ML_success_300 AND ML_success_60)
    if 'ML_success_300' in df.columns and 'ML_success_60' in df.columns:
        ml_15 = df[(df['ML_success_300'] == True) & (df['ML_success_60'] == True)]
        ml_15_metrics = compute_metrics(ml_15)
        if ml_15_metrics:
            print(f"ML Top 15% BOTH: {ml_15_metrics.get('total_signals', 0):,} | Expected R: {ml_15_metrics.get('expected_R', 0):.4f}")
    
    # Save results
    print("\n" + "="*70)
    output_file = 'tradesdata_r_evaluation.csv'
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df):,} rows with {len(df.columns)} columns")
    
    print("\nNew columns added:")
    print("  - ATR_5m (5-minute Average True Range)")
    print("  - stop_price (entry - 0.8*ATR for LONG, entry + 0.8*ATR for SHORT)")
    print("  - R (risk unit = abs(entry - stop))")
    print("  - MFE_R (Maximum Favorable Excursion in R)")
    print("  - MAE_R (Maximum Adverse Excursion in R)")
    print("  - hit_stop (MAE >= 1.0)")
    print("  - max_price_300s, min_price_300s (price extremes in 300s window)")


if __name__ == '__main__':
    main()
