"""
Compute top 20 percentile from random sample of max 1000 rows per model.
"""

import pandas as pd
import numpy as np
import time
import random

TP_PCT = 0.0004  # 0.04%
SL_PCT = 0.00015  # 0.015%
WINDOW_MS = 24 * 60 * 60 * 1000  # 24 hours
PERCENTILE_THRESHOLD = 80  # Top 20% means >= 80th percentile
MAX_SAMPLE_SIZE = 1000


def evaluate_long_signals(timestamps, prices, signal_indices):
    """Evaluate LONG signals."""
    n_signals = len(signal_indices)
    results = np.zeros(n_signals, dtype=np.int8)
    
    for i in range(n_signals):
        sig_idx = signal_indices[i]
        entry_price = prices[sig_idx]
        entry_ts = timestamps[sig_idx]
        
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)
        cutoff_ts = entry_ts + WINDOW_MS
        
        for j in range(sig_idx + 1, len(prices)):
            if timestamps[j] > cutoff_ts:
                break
            
            price = prices[j]
            
            if price <= sl_price:
                results[i] = 0
                break
            
            if price >= tp_price:
                results[i] = 1
                break
    
    return results


def evaluate_short_signals(timestamps, prices, signal_indices):
    """Evaluate SHORT signals."""
    n_signals = len(signal_indices)
    results = np.zeros(n_signals, dtype=np.int8)
    
    for i in range(n_signals):
        sig_idx = signal_indices[i]
        entry_price = prices[sig_idx]
        entry_ts = timestamps[sig_idx]
        
        tp_price = entry_price * (1 - TP_PCT)
        sl_price = entry_price * (1 + SL_PCT)
        cutoff_ts = entry_ts + WINDOW_MS
        
        for j in range(sig_idx + 1, len(prices)):
            if timestamps[j] > cutoff_ts:
                break
            
            price = prices[j]
            
            if price >= sl_price:
                results[i] = 0
                break
            
            if price <= tp_price:
                results[i] = 1
                break
    
    return results


def main():
    print("="*70)
    print("ML RANDOM SAMPLE PERCENTILE FILTERING")
    print(f"Top 20% from random sample of max {MAX_SAMPLE_SIZE} rows per model")
    print("="*70)
    
    print("\nLoading tradesdata_with_ml_predictions.csv...")
    df = pd.read_csv('tradesdata_with_ml_predictions.csv', low_memory=False)
    print(f"Loaded {len(df):,} rows")
    
    # ========== STEP 1: Create ML_success_300 with random sampling ==========
    print("\n" + "-"*70)
    print("STEP 1: Create ML_success_300 (top 20% from random sample)")
    print("-"*70)
    
    df['ML_success_300'] = False
    
    for model_name, group in df.groupby('Model_name_300'):
        n_rows = len(group)
        
        if n_rows > MAX_SAMPLE_SIZE:
            # Random sample
            sample_indices = random.sample(group.index.tolist(), MAX_SAMPLE_SIZE)
            sample_group = group.loc[sample_indices]
            print(f"  {model_name}: Sampled {MAX_SAMPLE_SIZE} from {n_rows} rows")
        else:
            sample_group = group
            print(f"  {model_name}: Using all {n_rows} rows")
        
        # Calculate threshold from sample
        threshold = sample_group['Model_output_300'].quantile(PERCENTILE_THRESHOLD / 100)
        
        # Apply to ALL rows in the group
        mask = (df['Model_name_300'] == model_name) & (df['Model_output_300'] >= threshold)
        df.loc[mask, 'ML_success_300'] = True
        
        n_selected = mask.sum()
        print(f"    Selected {n_selected:,} rows (threshold={threshold:.4f})")
    
    ml300_count = df['ML_success_300'].sum()
    print(f"\nTotal ML_success_300=True: {ml300_count:,} ({ml300_count/len(df)*100:.1f}%)")
    
    # ========== STEP 2: Run TP/SL test for ML_success_300=True ==========
    print("\n" + "-"*70)
    print("STEP 2: Evaluate TP/SL for ML_success_300=True")
    print("-"*70)
    
    df['Success_ML300'] = np.nan
    ml300_mask = df['ML_success_300'] == True
    
    print(f"Evaluating {ml300_mask.sum():,} signals...")
    start = time.time()
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].sort_values('timestamp').reset_index()
        orig_indices = symbol_df['index'].values
        
        ts_series = pd.to_datetime(symbol_df['timestamp'])
        timestamps = (ts_series.astype(np.int64) // 10**6).values
        prices = symbol_df['price'].values.astype(np.float64)
        
        symbol_mask = ml300_mask.loc[orig_indices].values
        
        # LONG
        long_cond = (symbol_df['Decision'].values == 'LONG') & symbol_mask
        long_indices = np.where(long_cond)[0].astype(np.int64)
        
        if len(long_indices) > 0:
            long_results = evaluate_long_signals(timestamps, prices, long_indices)
            for idx, res in zip(long_indices, long_results):
                df.loc[orig_indices[idx], 'Success_ML300'] = bool(res)
        
        # SHORT
        short_cond = (symbol_df['Decision'].values == 'SHORT') & symbol_mask
        short_indices = np.where(short_cond)[0].astype(np.int64)
        
        if len(short_indices) > 0:
            short_results = evaluate_short_signals(timestamps, prices, short_indices)
            for idx, res in zip(short_indices, short_results):
                df.loc[orig_indices[idx], 'Success_ML300'] = bool(res)
    
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s")
    
    ml300_signals = df[df['ML_success_300'] == True]
    ml300_success = ml300_signals['Success_ML300'].sum()
    ml300_rate = ml300_success / len(ml300_signals) * 100 if len(ml300_signals) > 0 else 0
    print(f"\nML_success_300 results:")
    print(f"  Total: {len(ml300_signals):,}")
    print(f"  TP/SL Success: {int(ml300_success):,}")
    print(f"  Win rate: {ml300_rate:.2f}%")
    
    # ========== STEP 3: Create ML_success_60 ==========
    print("\n" + "-"*70)
    print("STEP 3: Create ML_success_60 (top 20% from random sample)")
    print("-"*70)
    
    df['ML_success_60'] = False
    
    for model_name, group in df.groupby('Model_name_60'):
        n_rows = len(group)
        
        if n_rows > MAX_SAMPLE_SIZE:
            sample_indices = random.sample(group.index.tolist(), MAX_SAMPLE_SIZE)
            sample_group = group.loc[sample_indices]
            print(f"  {model_name}: Sampled {MAX_SAMPLE_SIZE} from {n_rows} rows")
        else:
            sample_group = group
            print(f"  {model_name}: Using all {n_rows} rows")
        
        threshold = sample_group['Model_output_60'].quantile(PERCENTILE_THRESHOLD / 100)
        
        mask = (df['Model_name_60'] == model_name) & (df['Model_output_60'] >= threshold)
        df.loc[mask, 'ML_success_60'] = True
        
        n_selected = mask.sum()
        print(f"    Selected {n_selected:,} rows (threshold={threshold:.4f})")
    
    ml60_count = df['ML_success_60'].sum()
    print(f"\nTotal ML_success_60=True: {ml60_count:,} ({ml60_count/len(df)*100:.1f}%)")
    
    # ========== STEP 4: Evaluate BOTH ==========
    print("\n" + "-"*70)
    print("STEP 4: Evaluate TP/SL for BOTH ML_success_300 AND ML_success_60")
    print("-"*70)
    
    both_mask = (df['ML_success_300'] == True) & (df['ML_success_60'] == True)
    both_count = both_mask.sum()
    print(f"Rows with BOTH=True: {both_count:,}")
    
    df['Success_Both'] = np.nan
    
    print(f"Evaluating {both_count:,} signals...")
    start = time.time()
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].sort_values('timestamp').reset_index()
        orig_indices = symbol_df['index'].values
        
        ts_series = pd.to_datetime(symbol_df['timestamp'])
        timestamps = (ts_series.astype(np.int64) // 10**6).values
        prices = symbol_df['price'].values.astype(np.float64)
        
        symbol_mask = both_mask.loc[orig_indices].values
        
        # LONG
        long_cond = (symbol_df['Decision'].values == 'LONG') & symbol_mask
        long_indices = np.where(long_cond)[0].astype(np.int64)
        
        if len(long_indices) > 0:
            long_results = evaluate_long_signals(timestamps, prices, long_indices)
            for idx, res in zip(long_indices, long_results):
                df.loc[orig_indices[idx], 'Success_Both'] = bool(res)
        
        # SHORT
        short_cond = (symbol_df['Decision'].values == 'SHORT') & symbol_mask
        short_indices = np.where(short_cond)[0].astype(np.int64)
        
        if len(short_indices) > 0:
            short_results = evaluate_short_signals(timestamps, prices, short_indices)
            for idx, res in zip(short_indices, short_results):
                df.loc[orig_indices[idx], 'Success_Both'] = bool(res)
    
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f}s")
    
    both_signals = df[both_mask]
    both_success = both_signals['Success_Both'].sum()
    both_rate = both_success / len(both_signals) * 100 if len(both_signals) > 0 else 0
    print(f"\nBOTH results:")
    print(f"  Total: {len(both_signals):,}")
    print(f"  TP/SL Success: {int(both_success):,}")
    print(f"  Win rate: {both_rate:.2f}%")
    
    # ========== COMPARISON ==========
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    # Baseline
    if 'Success_Entry' in df.columns:
        baseline_signals = df[df['Success_Entry'].notna()]
        baseline_success = baseline_signals['Success_Entry'].sum()
        baseline_rate = baseline_success / len(baseline_signals) * 100 if len(baseline_signals) > 0 else 0
    else:
        baseline_rate = 0
        baseline_signals = df
    
    print(f"\n{'Filter':<40} {'Signals':>10} {'Win Rate':>12}")
    print("-"*62)
    print(f"{'Baseline (Entry filter only)':<40} {len(baseline_signals):>10,} {baseline_rate:>11.2f}%")
    print(f"{'ML_success_300 (random sample)':<40} {len(ml300_signals):>10,} {ml300_rate:>11.2f}%")
    print(f"{'BOTH ML filters (random sample)':<40} {len(both_signals):>10,} {both_rate:>11.2f}%")
    
    # Improvement
    if baseline_rate > 0:
        ml300_improvement = ml300_rate - baseline_rate
        both_improvement = both_rate - baseline_rate
        print(f"\n{'Improvement over baseline:'}")
        print(f"  ML_success_300: {ml300_improvement:+.2f}%")
        print(f"  BOTH: {both_improvement:+.2f}%")
    
    # By direction
    print("\n" + "-"*70)
    print("BY DIRECTION (BOTH filter):")
    print("-"*70)
    
    for direction in ['LONG', 'SHORT']:
        dir_both = both_signals[both_signals['Decision'] == direction]
        if len(dir_both) > 0:
            dir_rate = dir_both['Success_Both'].mean() * 100
            print(f"  {direction}: {len(dir_both):,} signals, {dir_rate:.2f}% win rate")
    
    # Save
    print("\n" + "="*70)
    output_file = 'tradesdata_ml_random.csv'
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df):,} rows with {len(df.columns)} columns")


if __name__ == '__main__':
    main()
