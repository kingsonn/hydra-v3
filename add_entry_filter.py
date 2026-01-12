"""
Add VAL_30, VAH_30, Entry columns and filter signals with Entry=True
Then evaluate success with TP=0.04%, SL=0.015%
"""

import pandas as pd
import numpy as np
import time

TP_PCT = 0.0004  # 0.04%
SL_PCT = 0.00015  # 0.015%
WINDOW_MS = 24 * 60 * 60 * 1000  # 24 hours in milliseconds
VALUE_AREA_PCT = 0.70  # 70% of volume for value area


def compute_value_area(prices: np.ndarray, volumes: np.ndarray, pct: float = 0.70):
    """
    Compute Value Area High and Low from price/volume data.
    Value Area = price range containing 70% of volume.
    """
    if len(prices) == 0 or len(volumes) == 0:
        return np.nan, np.nan
    
    # Create price bins
    price_min, price_max = prices.min(), prices.max()
    if price_min == price_max:
        return price_min, price_max
    
    # Use price_bin if available, otherwise create bins
    n_bins = min(50, len(np.unique(prices)))
    bins = np.linspace(price_min, price_max, n_bins + 1)
    
    # Assign prices to bins and sum volume per bin
    bin_indices = np.digitize(prices, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    volume_per_bin = np.zeros(n_bins)
    for i, vol in zip(bin_indices, volumes):
        volume_per_bin[i] += vol
    
    total_volume = volume_per_bin.sum()
    if total_volume == 0:
        return price_min, price_max
    
    target_volume = total_volume * pct
    
    # Find POC (Point of Control) - bin with max volume
    poc_bin = np.argmax(volume_per_bin)
    
    # Expand from POC until we reach target volume
    cumulative_volume = volume_per_bin[poc_bin]
    low_bin = poc_bin
    high_bin = poc_bin
    
    while cumulative_volume < target_volume:
        # Check which direction to expand
        can_go_lower = low_bin > 0
        can_go_higher = high_bin < n_bins - 1
        
        if not can_go_lower and not can_go_higher:
            break
        
        lower_vol = volume_per_bin[low_bin - 1] if can_go_lower else -1
        higher_vol = volume_per_bin[high_bin + 1] if can_go_higher else -1
        
        if lower_vol >= higher_vol and can_go_lower:
            low_bin -= 1
            cumulative_volume += volume_per_bin[low_bin]
        elif can_go_higher:
            high_bin += 1
            cumulative_volume += volume_per_bin[high_bin]
        elif can_go_lower:
            low_bin -= 1
            cumulative_volume += volume_per_bin[low_bin]
        else:
            break
    
    # Convert bins back to prices
    val = bins[low_bin]
    vah = bins[high_bin + 1] if high_bin + 1 < len(bins) else bins[high_bin]
    
    return val, vah


def compute_30min_value_areas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 30-minute VAL and VAH for each symbol."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create 30-min floor timestamp
    df['ts_30min'] = df['timestamp'].dt.floor('30min')
    
    # Initialize columns
    df['VAL_30'] = np.nan
    df['VAH_30'] = np.nan
    
    print("Computing 30-min Value Areas by symbol...")
    
    for symbol in df['symbol'].unique():
        print(f"  {symbol}...", end=" ")
        start = time.time()
        
        symbol_mask = df['symbol'] == symbol
        symbol_df = df[symbol_mask].copy()
        
        # Group by 30-min intervals
        grouped = symbol_df.groupby('ts_30min')
        
        val_dict = {}
        vah_dict = {}
        
        for ts_30min, group in grouped:
            prices = group['price'].values
            volumes = group['qty'].values
            
            val, vah = compute_value_area(prices, volumes)
            val_dict[ts_30min] = val
            vah_dict[ts_30min] = vah
        
        # Map back to original rows
        df.loc[symbol_mask, 'VAL_30'] = symbol_df['ts_30min'].map(val_dict).values
        df.loc[symbol_mask, 'VAH_30'] = symbol_df['ts_30min'].map(vah_dict).values
        
        elapsed = time.time() - start
        print(f"{elapsed:.1f}s ({len(grouped)} intervals)")
    
    # Drop temporary column
    df.drop(columns=['ts_30min'], inplace=True)
    
    return df


def add_entry_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Entry column based on:
    - LONG: dist_lvn < LVN_thresh OR price <= VAL_30
    - SHORT: dist_lvn < LVN_thresh OR price >= VAH_30
    """
    df = df.copy()
    df['Entry'] = False
    
    # LONG entry conditions
    long_mask = df['Decision'] == 'LONG'
    long_entry = (
        (df['dist_lvn'] < df['LVN_thresh']) & (df['price'] <= df['VAL_30'])
    )
    df.loc[long_mask & long_entry, 'Entry'] = True
    
    # SHORT entry conditions
    short_mask = df['Decision'] == 'SHORT'
    short_entry = (
        (df['dist_lvn'] < df['LVN_thresh']) &(df['price'] >= df['VAH_30'])
    )
    df.loc[short_mask & short_entry, 'Entry'] = True
    
    return df


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


def evaluate_symbol_entries(df_symbol: pd.DataFrame) -> pd.DataFrame:
    """Evaluate success for signals with Entry=True."""
    df = df_symbol.copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    ts_series = pd.to_datetime(df['timestamp'])
    timestamps = (ts_series.astype(np.int64) // 10**6).values
    prices = df['price'].values.astype(np.float64)
    
    # Initialize Success_Entry column
    df['Success_Entry'] = np.nan
    
    # LONG signals with Entry=True
    long_entry_mask = (df['Decision'] == 'LONG') & (df['Entry'] == True)
    long_indices = np.where(long_entry_mask)[0].astype(np.int64)
    
    if len(long_indices) > 0:
        long_results = evaluate_long_signals(timestamps, prices, long_indices)
        df.loc[long_entry_mask, 'Success_Entry'] = long_results.astype(bool)
    
    # SHORT signals with Entry=True
    short_entry_mask = (df['Decision'] == 'SHORT') & (df['Entry'] == True)
    short_indices = np.where(short_entry_mask)[0].astype(np.int64)
    
    if len(short_indices) > 0:
        short_results = evaluate_short_signals(timestamps, prices, short_indices)
        df.loc[short_entry_mask, 'Success_Entry'] = short_results.astype(bool)
    
    return df


def main():
    print("="*70)
    print("ADD VAL_30, VAH_30, ENTRY AND EVALUATE FILTERED SIGNALS")
    print("="*70)
    
    print("\nLoading tradesdata_with_success.csv...")
    df = pd.read_csv('tradesdata_with_success.csv', low_memory=False)
    print(f"Loaded {len(df):,} rows")
    
    # Step 1: Compute 30-min Value Areas
    print("\n" + "-"*70)
    df = compute_30min_value_areas(df)
    
    # Step 2: Add Entry column
    print("\n" + "-"*70)
    print("Adding Entry column...")
    df = add_entry_column(df)
    
    # Stats on Entry
    signals_mask = df['Decision'].isin(['LONG', 'SHORT'])
    signals_with_entry = (signals_mask & df['Entry']).sum()
    total_signals = signals_mask.sum()
    
    print(f"\nSignals with Entry=True: {signals_with_entry:,} / {total_signals:,} ({signals_with_entry/total_signals*100:.1f}%)")
    
    long_signals = (df['Decision'] == 'LONG').sum()
    long_entry = ((df['Decision'] == 'LONG') & df['Entry']).sum()
    print(f"  LONG with Entry: {long_entry:,} / {long_signals:,} ({long_entry/long_signals*100:.1f}%)")
    
    short_signals = (df['Decision'] == 'SHORT').sum()
    short_entry = ((df['Decision'] == 'SHORT') & df['Entry']).sum()
    print(f"  SHORT with Entry: {short_entry:,} / {short_signals:,} ({short_entry/short_signals*100:.1f}%)")
    
    # Step 3: Evaluate success for Entry=True signals
    print("\n" + "-"*70)
    print("Evaluating success for signals with Entry=True...")
    
    results = []
    for symbol in df['symbol'].unique():
        print(f"  Processing {symbol}...", end=" ")
        start = time.time()
        
        df_symbol = df[df['symbol'] == symbol].copy()
        df_evaluated = evaluate_symbol_entries(df_symbol)
        results.append(df_evaluated)
        
        elapsed = time.time() - start
        print(f"{elapsed:.1f}s")
    
    df = pd.concat(results, ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Step 4: Statistics
    print("\n" + "="*70)
    print("RESULTS FOR SIGNALS WITH ENTRY=TRUE")
    print("(TP=0.04%, SL=0.015%)")
    print("="*70)
    
    entry_signals = df[(df['Decision'].isin(['LONG', 'SHORT'])) & (df['Entry'] == True)]
    
    # Overall
    total_entry = len(entry_signals)
    success_entry = entry_signals['Success_Entry'].sum()
    win_rate = success_entry / total_entry * 100 if total_entry > 0 else 0
    
    print(f"\nOverall:")
    print(f"  Total signals with Entry: {total_entry:,}")
    print(f"  Successful: {int(success_entry):,}")
    print(f"  Win rate: {win_rate:.2f}%")
    
    # LONG
    long_entry_df = entry_signals[entry_signals['Decision'] == 'LONG']
    if len(long_entry_df) > 0:
        long_success = long_entry_df['Success_Entry'].sum()
        long_rate = long_success / len(long_entry_df) * 100
        print(f"\nLONG signals with Entry:")
        print(f"  Total: {len(long_entry_df):,}")
        print(f"  Successful: {int(long_success):,}")
        print(f"  Win rate: {long_rate:.2f}%")
    
    # SHORT
    short_entry_df = entry_signals[entry_signals['Decision'] == 'SHORT']
    if len(short_entry_df) > 0:
        short_success = short_entry_df['Success_Entry'].sum()
        short_rate = short_success / len(short_entry_df) * 100
        print(f"\nSHORT signals with Entry:")
        print(f"  Total: {len(short_entry_df):,}")
        print(f"  Successful: {int(short_success):,}")
        print(f"  Win rate: {short_rate:.2f}%")
    
    # By symbol
    print("\n" + "-"*70)
    print("BY SYMBOL:")
    print("-"*70)
    
    for symbol in sorted(df['symbol'].unique()):
        sym_entry = entry_signals[entry_signals['symbol'] == symbol]
        if len(sym_entry) > 0:
            sym_success = sym_entry['Success_Entry'].sum()
            sym_rate = sym_success / len(sym_entry) * 100
            
            sym_long = sym_entry[sym_entry['Decision'] == 'LONG']
            sym_short = sym_entry[sym_entry['Decision'] == 'SHORT']
            
            print(f"\n{symbol}: {len(sym_entry):,} signals | {sym_rate:.2f}% win rate")
            if len(sym_long) > 0:
                long_r = sym_long['Success_Entry'].mean() * 100
                print(f"  LONG: {len(sym_long):,} | {long_r:.2f}%")
            if len(sym_short) > 0:
                short_r = sym_short['Success_Entry'].mean() * 100
                print(f"  SHORT: {len(sym_short):,} | {short_r:.2f}%")
    
    # Comparison with all signals
    print("\n" + "="*70)
    print("COMPARISON: ENTRY FILTER vs ALL SIGNALS")
    print("="*70)
    
    all_signals = df[df['Decision'].isin(['LONG', 'SHORT'])]
    all_success = all_signals['Success'].sum()
    all_rate = all_success / len(all_signals) * 100 if len(all_signals) > 0 else 0
    
    print(f"\nAll signals (no Entry filter):")
    print(f"  Total: {len(all_signals):,} | Win rate: {all_rate:.2f}%")
    
    print(f"\nWith Entry filter:")
    print(f"  Total: {total_entry:,} | Win rate: {win_rate:.2f}%")
    
    improvement = win_rate - all_rate
    print(f"\nImprovement: {improvement:+.2f}%")
    
    # Step 5: Save filtered CSV
    print("\n" + "="*70)
    print("Saving filtered CSV (only rows with Signal + Entry)...")
    
    filtered_df = df[(df['Decision'].isin(['LONG', 'SHORT'])) & (df['Entry'] == True)].copy()
    filtered_df.to_csv('tradesdata_entry_filtered.csv', index=False)
    print(f"Saved {len(filtered_df):,} rows to tradesdata_entry_filtered.csv")
    
    # Also save full CSV with new columns
    print("\nSaving full CSV with VAL_30, VAH_30, Entry columns...")
    df.to_csv('tradesdata_with_success.csv', index=False)
    print(f"Updated tradesdata_with_success.csv ({len(df):,} rows, {len(df.columns)} columns)")
    
    # Sample
    print("\nSample filtered rows:")
    print(filtered_df[['timestamp', 'symbol', 'price', 'VAL_30', 'VAH_30', 'Decision', 'Entry', 'Success_Entry', 'Signals']].head(15))


if __name__ == '__main__':
    main()
