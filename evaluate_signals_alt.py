"""
Evaluate signal success with alternative parameters:
TP = 0.07%, SL = 0.02%
"""

import pandas as pd
import numpy as np
import time

TP_PCT = 0.0007  # 0.07%
SL_PCT = 0.0002  # 0.02%
WINDOW_MS = 24 * 60 * 60 * 1000  # 24 hours in milliseconds


def evaluate_long_signals(
    timestamps: np.ndarray,
    prices: np.ndarray,
    signal_indices: np.ndarray,
) -> np.ndarray:
    """Evaluate LONG signals with new parameters."""
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


def evaluate_short_signals(
    timestamps: np.ndarray,
    prices: np.ndarray,
    signal_indices: np.ndarray,
) -> np.ndarray:
    """Evaluate SHORT signals with new parameters."""
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


def evaluate_symbol(df_symbol: pd.DataFrame) -> tuple:
    """Evaluate all signals for a single symbol."""
    df = df_symbol.sort_values('timestamp').reset_index(drop=True)
    
    ts_series = pd.to_datetime(df['timestamp'])
    timestamps = (ts_series.astype(np.int64) // 10**6).values
    prices = df['price'].values.astype(np.float64)
    
    long_mask = df['Decision'] == 'LONG'
    long_indices = np.where(long_mask)[0].astype(np.int64)
    
    long_success = 0
    long_total = len(long_indices)
    
    if long_total > 0:
        long_results = evaluate_long_signals(timestamps, prices, long_indices)
        long_success = long_results.sum()
    
    short_mask = df['Decision'] == 'SHORT'
    short_indices = np.where(short_mask)[0].astype(np.int64)
    
    short_success = 0
    short_total = len(short_indices)
    
    if short_total > 0:
        short_results = evaluate_short_signals(timestamps, prices, short_indices)
        short_success = short_results.sum()
    
    return long_total, long_success, short_total, short_success


def main():
    print("="*70)
    print("SIGNAL EVALUATION WITH NEW PARAMETERS")
    print("Take Profit: 0.07% | Stop Loss: 0.02%")
    print("="*70)
    
    print("\nLoading tradesdata_with_signals.csv...")
    df = pd.read_csv('tradesdata_with_signals.csv')
    print(f"Loaded {len(df)} rows")
    
    long_count = (df['Decision'] == 'LONG').sum()
    short_count = (df['Decision'] == 'SHORT').sum()
    print(f"\nSignals to evaluate:")
    print(f"  LONG: {long_count:,}")
    print(f"  SHORT: {short_count:,}")
    print(f"  Total: {(long_count + short_count):,}")
    
    print("\nEvaluating signals by symbol...")
    
    symbols = df['symbol'].unique()
    symbol_results = {}
    
    total_long = 0
    total_long_success = 0
    total_short = 0
    total_short_success = 0
    
    for symbol in symbols:
        print(f"  Processing {symbol}...", end=" ")
        start = time.time()
        
        df_symbol = df[df['symbol'] == symbol].copy()
        long_total, long_success, short_total, short_success = evaluate_symbol(df_symbol)
        
        symbol_results[symbol] = {
            'long_total': long_total,
            'long_success': long_success,
            'short_total': short_total,
            'short_success': short_success,
        }
        
        total_long += long_total
        total_long_success += long_success
        total_short += short_total
        total_short_success += short_success
        
        elapsed = time.time() - start
        print(f"{elapsed:.1f}s")
    
    print("\n" + "="*70)
    print("OVERALL RESULTS")
    print("="*70)
    
    if total_long > 0:
        long_rate = total_long_success / total_long * 100
        print(f"\nLONG signals:")
        print(f"  Total: {total_long:,}")
        print(f"  Success: {total_long_success:,}")
        print(f"  Win rate: {long_rate:.2f}%")
    
    if total_short > 0:
        short_rate = total_short_success / total_short * 100
        print(f"\nSHORT signals:")
        print(f"  Total: {total_short:,}")
        print(f"  Success: {total_short_success:,}")
        print(f"  Win rate: {short_rate:.2f}%")
    
    all_total = total_long + total_short
    all_success = total_long_success + total_short_success
    if all_total > 0:
        all_rate = all_success / all_total * 100
        print(f"\nALL signals:")
        print(f"  Total: {all_total:,}")
        print(f"  Success: {all_success:,}")
        print(f"  Win rate: {all_rate:.2f}%")
    
    print("\n" + "="*70)
    print("RESULTS BY SYMBOL")
    print("="*70)
    
    for symbol in sorted(symbols):
        res = symbol_results[symbol]
        sym_total = res['long_total'] + res['short_total']
        sym_success = res['long_success'] + res['short_success']
        
        if sym_total > 0:
            sym_rate = sym_success / sym_total * 100
            
            long_rate = res['long_success'] / res['long_total'] * 100 if res['long_total'] > 0 else 0
            short_rate = res['short_success'] / res['short_total'] * 100 if res['short_total'] > 0 else 0
            
            print(f"\n{symbol}:")
            print(f"  Total: {sym_total:,} signals | Win rate: {sym_rate:.2f}%")
            if res['long_total'] > 0:
                print(f"  LONG: {res['long_total']:,} signals | {long_rate:.2f}% win rate")
            if res['short_total'] > 0:
                print(f"  SHORT: {res['short_total']:,} signals | {short_rate:.2f}% win rate")
    
    print("\n" + "="*70)
    print("COMPARISON WITH PREVIOUS PARAMETERS")
    print("="*70)
    print("\nPrevious (TP=0.04%, SL=0.015%):")
    print("  LONG: 35.30% | SHORT: 31.99% | ALL: 34.83%")
    print(f"\nNew (TP=0.07%, SL=0.02%):")
    if total_long > 0:
        print(f"  LONG: {total_long_success / total_long * 100:.2f}%", end="")
    if total_short > 0:
        print(f" | SHORT: {total_short_success / total_short * 100:.2f}%", end="")
    if all_total > 0:
        print(f" | ALL: {all_success / all_total * 100:.2f}%")


if __name__ == '__main__':
    main()
