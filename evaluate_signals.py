"""
Evaluate signal success based on forward-looking price action.

For LONG signals:
- Success = True if price hits +0.04% (TP) before -0.015% (SL) within 24h
- Success = False otherwise

For SHORT signals:
- Success = True if price hits -0.04% (TP) before +0.015% (SL) within 24h
- Success = False otherwise
"""

import pandas as pd
import numpy as np
import time

TP_PCT = 0.0004  # 0.04%
SL_PCT = 0.00015  # 0.015%
WINDOW_MS = 24 * 60 * 60 * 1000  # 24 hours in milliseconds


def evaluate_long_signals(
    timestamps: np.ndarray,
    prices: np.ndarray,
    signal_indices: np.ndarray,
) -> np.ndarray:
    """
    Evaluate LONG signals.
    Returns array of success values (1=True, 0=False)
    """
    n_signals = len(signal_indices)
    results = np.zeros(n_signals, dtype=np.int8)
    
    for i in range(n_signals):
        sig_idx = signal_indices[i]
        entry_price = prices[sig_idx]
        entry_ts = timestamps[sig_idx]
        
        tp_price = entry_price * (1 + TP_PCT)
        sl_price = entry_price * (1 - SL_PCT)
        cutoff_ts = entry_ts + WINDOW_MS
        
        # Scan forward
        for j in range(sig_idx + 1, len(prices)):
            if timestamps[j] > cutoff_ts:
                break
            
            price = prices[j]
            
            # Check SL first (more conservative)
            if price <= sl_price:
                results[i] = 0  # Hit SL first = failure
                break
            
            # Check TP
            if price >= tp_price:
                results[i] = 1  # Hit TP = success
                break
    
    return results


def evaluate_short_signals(
    timestamps: np.ndarray,
    prices: np.ndarray,
    signal_indices: np.ndarray,
) -> np.ndarray:
    """
    Evaluate SHORT signals.
    Returns array of success values (1=True, 0=False)
    """
    n_signals = len(signal_indices)
    results = np.zeros(n_signals, dtype=np.int8)
    
    for i in range(n_signals):
        sig_idx = signal_indices[i]
        entry_price = prices[sig_idx]
        entry_ts = timestamps[sig_idx]
        
        # For SHORT: TP is price going DOWN, SL is price going UP
        tp_price = entry_price * (1 - TP_PCT)
        sl_price = entry_price * (1 + SL_PCT)
        cutoff_ts = entry_ts + WINDOW_MS
        
        # Scan forward
        for j in range(sig_idx + 1, len(prices)):
            if timestamps[j] > cutoff_ts:
                break
            
            price = prices[j]
            
            # Check SL first (price going up = bad for short)
            if price >= sl_price:
                results[i] = 0  # Hit SL first = failure
                break
            
            # Check TP (price going down = good for short)
            if price <= tp_price:
                results[i] = 1  # Hit TP = success
                break
    
    return results


def evaluate_symbol(df_symbol: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all signals for a single symbol."""
    df = df_symbol.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Convert timestamps to milliseconds
    ts_series = pd.to_datetime(df['timestamp'])
    timestamps = (ts_series.astype(np.int64) // 10**6).values
    prices = df['price'].values.astype(np.float64)
    
    # Initialize Success column
    df['Success'] = np.nan
    
    # Find LONG signal indices
    long_mask = df['Decision'] == 'LONG'
    long_indices = np.where(long_mask)[0].astype(np.int64)
    
    if len(long_indices) > 0:
        long_results = evaluate_long_signals(timestamps, prices, long_indices)
        df.loc[long_mask, 'Success'] = long_results.astype(bool)
    
    # Find SHORT signal indices
    short_mask = df['Decision'] == 'SHORT'
    short_indices = np.where(short_mask)[0].astype(np.int64)
    
    if len(short_indices) > 0:
        short_results = evaluate_short_signals(timestamps, prices, short_indices)
        df.loc[short_mask, 'Success'] = short_results.astype(bool)
    
    return df


def main():
    print("Loading tradesdata_with_signals.csv...")
    df = pd.read_csv('tradesdata_with_signals.csv')
    print(f"Loaded {len(df)} rows")
    
    # Count signals
    long_count = (df['Decision'] == 'LONG').sum()
    short_count = (df['Decision'] == 'SHORT').sum()
    print(f"\nSignals to evaluate:")
    print(f"  LONG: {long_count}")
    print(f"  SHORT: {short_count}")
    print(f"  Total: {long_count + short_count}")
    
    print("\nEvaluating signals by symbol...")
    results = []
    
    symbols = df['symbol'].unique()
    for symbol in symbols:
        print(f"  Processing {symbol}...", end=" ")
        start = time.time()
        
        df_symbol = df[df['symbol'] == symbol].copy()
        df_evaluated = evaluate_symbol(df_symbol)
        results.append(df_evaluated)
        
        # Stats for this symbol
        symbol_long = (df_evaluated['Decision'] == 'LONG').sum()
        symbol_short = (df_evaluated['Decision'] == 'SHORT').sum()
        symbol_success = df_evaluated['Success'].sum() if df_evaluated['Success'].notna().any() else 0
        
        elapsed = time.time() - start
        print(f"{elapsed:.1f}s - LONG: {symbol_long}, SHORT: {symbol_short}, Success: {int(symbol_success)}")
    
    print("\nCombining results...")
    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.sort_values('timestamp').reset_index(drop=True)
    
    # Overall statistics
    print("\n" + "="*60)
    print("SIGNAL SUCCESS STATISTICS")
    print("="*60)
    
    # LONG signals
    long_signals = final_df[final_df['Decision'] == 'LONG']
    if len(long_signals) > 0:
        long_success = long_signals['Success'].sum()
        long_total = len(long_signals)
        long_rate = long_success / long_total * 100 if long_total > 0 else 0
        print(f"\nLONG signals:")
        print(f"  Total: {long_total}")
        print(f"  Success: {int(long_success)}")
        print(f"  Win rate: {long_rate:.2f}%")
    
    # SHORT signals
    short_signals = final_df[final_df['Decision'] == 'SHORT']
    if len(short_signals) > 0:
        short_success = short_signals['Success'].sum()
        short_total = len(short_signals)
        short_rate = short_success / short_total * 100 if short_total > 0 else 0
        print(f"\nSHORT signals:")
        print(f"  Total: {short_total}")
        print(f"  Success: {int(short_success)}")
        print(f"  Win rate: {short_rate:.2f}%")
    
    # Combined
    all_signals = final_df[final_df['Decision'].isin(['LONG', 'SHORT'])]
    if len(all_signals) > 0:
        all_success = all_signals['Success'].sum()
        all_total = len(all_signals)
        all_rate = all_success / all_total * 100 if all_total > 0 else 0
        print(f"\nALL signals:")
        print(f"  Total: {all_total}")
        print(f"  Success: {int(all_success)}")
        print(f"  Win rate: {all_rate:.2f}%")
    
    # By symbol
    print("\n" + "-"*60)
    print("SUCCESS RATE BY SYMBOL")
    print("-"*60)
    
    for symbol in symbols:
        sym_df = final_df[(final_df['symbol'] == symbol) & (final_df['Decision'].isin(['LONG', 'SHORT']))]
        if len(sym_df) > 0:
            sym_success = sym_df['Success'].sum()
            sym_total = len(sym_df)
            sym_rate = sym_success / sym_total * 100 if sym_total > 0 else 0
            
            sym_long = sym_df[sym_df['Decision'] == 'LONG']
            sym_short = sym_df[sym_df['Decision'] == 'SHORT']
            
            long_rate = sym_long['Success'].mean() * 100 if len(sym_long) > 0 else 0
            short_rate = sym_short['Success'].mean() * 100 if len(sym_short) > 0 else 0
            
            print(f"\n{symbol}:")
            print(f"  Total signals: {sym_total}, Win rate: {sym_rate:.2f}%")
            print(f"  LONG: {len(sym_long)} signals, {long_rate:.2f}% win rate")
            print(f"  SHORT: {len(sym_short)} signals, {short_rate:.2f}% win rate")
    
    # Save
    print("\n" + "="*60)
    print("Saving to tradesdata_with_success.csv...")
    final_df.to_csv('tradesdata_with_success.csv', index=False)
    print(f"Saved {len(final_df)} rows with {len(final_df.columns)} columns")
    
    # Sample successful signals
    print("\nSample successful LONG signals:")
    sample_long = final_df[(final_df['Decision'] == 'LONG') & (final_df['Success'] == True)].head(10)
    print(sample_long[['timestamp', 'symbol', 'price', 'Signals', 'Total_Score', 'Decision', 'Success']])
    
    print("\nSample successful SHORT signals:")
    sample_short = final_df[(final_df['Decision'] == 'SHORT') & (final_df['Success'] == True)].head(10)
    print(sample_short[['timestamp', 'symbol', 'price', 'Signals', 'Total_Score', 'Decision', 'Success']])


if __name__ == '__main__':
    main()
