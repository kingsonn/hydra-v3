"""
Script to enrich tradesdata.csv with additional columns:
- regime: 'TEST' for all rows
- time_under_regime: 500 for all rows
- funding_z: z-score of funding rate at timestamp
- oi_delta_5m: 5-minute OI change at timestamp
- price_change_5m: 5-minute price change
- liq_long_usd: Long liquidations in USD from Coinalyze
- liq_short_usd: Short liquidations in USD from Coinalyze
- liqimbalance: (liq_long - liq_short) / (liq_long + liq_short)
"""

import pandas as pd
import numpy as np
import requests
import time
from typing import Dict, Tuple

COINALYZE_API_KEY = "d02ff8e4-16e7-44b1-bcb8-ef663a8de294"

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'LTCUSDT', 'ADAUSDT']

COINALYZE_SYMBOL_MAP = {
    'BTCUSDT': 'BTCUSDT_PERP.A',
    'ETHUSDT': 'ETHUSDT_PERP.A',
    'SOLUSDT': 'SOLUSDT_PERP.A',
    'BNBUSDT': 'BNBUSDT_PERP.A',
    'XRPUSDT': 'XRPUSDT_PERP.A',
    'DOGEUSDT': 'DOGEUSDT_PERP.A',
    'LTCUSDT': 'LTCUSDT_PERP.A',
    'ADAUSDT': 'ADAUSDT_PERP.A',
}


def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch funding rates from Binance for a symbol."""
    all_rates = []
    current_start = start_ms - 86400000 * 7  # Get 7 days prior for z-score calc
    
    while current_start < end_ms:
        try:
            r = requests.get(
                'https://fapi.binance.com/fapi/v1/fundingRate',
                params={
                    'symbol': symbol,
                    'startTime': current_start,
                    'endTime': end_ms,
                    'limit': 1000
                },
                timeout=30
            )
            r.raise_for_status()
            data = r.json()
            
            if not data:
                break
                
            all_rates.extend(data)
            current_start = data[-1]['fundingTime'] + 1
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Error fetching funding rates for {symbol}: {e}")
            break
    
    if not all_rates:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rates)
    df['fundingRate'] = df['fundingRate'].astype(float)
    df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
    
    # Compute z-score using rolling window of all prior rates
    mean_rate = df['fundingRate'].expanding().mean()
    std_rate = df['fundingRate'].expanding().std()
    df['funding_z'] = (df['fundingRate'] - mean_rate) / std_rate
    df['funding_z'] = df['funding_z'].fillna(0.0)
    
    return df[['timestamp', 'fundingRate', 'funding_z']].sort_values('timestamp')


def fetch_oi_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch OI history from Binance for a symbol (5m intervals)."""
    all_oi = []
    current_start = start_ms - 300000  # Start 5min earlier for delta calc
    max_iterations = 20  # Safety limit
    iteration = 0
    
    while current_start < end_ms and iteration < max_iterations:
        iteration += 1
        try:
            r = requests.get(
                'https://fapi.binance.com/futures/data/openInterestHist',
                params={
                    'symbol': symbol,
                    'period': '5m',
                    'startTime': current_start,
                    'endTime': end_ms,
                    'limit': 500
                },
                timeout=30
            )
            r.raise_for_status()
            data = r.json()
            
            if not data:
                break
            
            all_oi.extend(data)
            new_start = data[-1]['timestamp'] + 1
            
            if new_start <= current_start:
                break  # Prevent infinite loop
                
            current_start = new_start
            print(f"      OI batch {iteration}: {len(data)} records, total: {len(all_oi)}")
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching OI history for {symbol}: {e}")
            break
    
    if not all_oi:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_oi)
    df['sumOpenInterestValue'] = df['sumOpenInterestValue'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
    df['oi_delta_5m'] = df['sumOpenInterestValue'].diff().fillna(0.0)
    
    return df[['timestamp', 'sumOpenInterestValue', 'oi_delta_5m']]


def fetch_liquidations(symbol: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Fetch liquidation history from Coinalyze for a symbol (5min intervals)."""
    coinalyze_symbol = COINALYZE_SYMBOL_MAP.get(symbol)
    if not coinalyze_symbol:
        return pd.DataFrame()
    
    all_liq = []
    current_start = start_ts
    chunk_size = 86400 * 7  # 7 days per request
    
    while current_start < end_ts:
        chunk_end = min(current_start + chunk_size, end_ts)
        try:
            r = requests.get(
                'https://api.coinalyze.net/v1/liquidation-history',
                params={
                    'symbols': coinalyze_symbol,
                    'interval': '5min',
                    'from': current_start,
                    'to': chunk_end,
                    'api_key': COINALYZE_API_KEY
                }
            )
            r.raise_for_status()
            data = r.json()
            
            if data and data[0].get('history'):
                all_liq.extend(data[0]['history'])
            
            current_start = chunk_end
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error fetching liquidations for {symbol}: {e}")
            break
    
    if not all_liq:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_liq)
    df['timestamp'] = pd.to_datetime(df['t'], unit='s')
    df.rename(columns={'l': 'liq_long', 's': 'liq_short'}, inplace=True)
    return df[['timestamp', 'liq_long', 'liq_short']].sort_values('timestamp')


def compute_price_change_5m(df: pd.DataFrame) -> pd.Series:
    """Compute 5-minute price change using time-based lookup."""
    df = df.copy()
    df['ts'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Use time-based rolling: find price 5 minutes ago
    df.set_index('ts', inplace=True)
    df['price_5m_ago'] = df['price'].shift(freq='5min')
    df = df.reset_index()
    
    # Forward fill for 250ms resolution data
    df['price_5m_ago'] = df['price_5m_ago'].ffill()
    
    price_change = (df['price'] - df['price_5m_ago']) / df['price_5m_ago'] * 100
    return price_change.fillna(0.0)


def process_symbol_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Process data for a single symbol using vectorized merge_asof."""
    print(f"  Processing {symbol} ({len(df)} rows)...")
    
    # Ensure timestamp is datetime and sorted
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    ts_min = df['timestamp'].min()
    ts_max = df['timestamp'].max()
    
    start_ms = int(ts_min.timestamp() * 1000)
    end_ms = int(ts_max.timestamp() * 1000)
    start_ts = int(ts_min.timestamp())
    end_ts = int(ts_max.timestamp())
    
    # Fetch API data
    print(f"    Fetching funding rates...")
    funding_df = fetch_funding_rates(symbol, start_ms, end_ms)
    print(f"    Got {len(funding_df)} funding rate records")
    
    print(f"    Fetching OI history...")
    oi_df = fetch_oi_history(symbol, start_ms, end_ms)
    print(f"    Got {len(oi_df)} OI records")
    
    print(f"    Fetching liquidations...")
    liq_df = fetch_liquidations(symbol, start_ts, end_ts)
    print(f"    Got {len(liq_df)} liquidation records")
    
    # Merge funding_z using merge_asof (forward fill latest funding rate)
    print(f"    Merging funding_z...")
    if not funding_df.empty:
        merged = pd.merge_asof(
            df[['timestamp']],
            funding_df[['timestamp', 'funding_z']],
            on='timestamp',
            direction='backward'
        )
        df['funding_z'] = merged['funding_z'].fillna(0.0)
    else:
        df['funding_z'] = 0.0
    
    # Merge oi_delta_5m using merge_asof
    print(f"    Merging oi_delta_5m...")
    if not oi_df.empty:
        merged = pd.merge_asof(
            df[['timestamp']],
            oi_df[['timestamp', 'oi_delta_5m']],
            on='timestamp',
            direction='backward'
        )
        df['oi_delta_5m'] = merged['oi_delta_5m'].fillna(0.0)
    else:
        df['oi_delta_5m'] = 0.0
    
    # Compute price_change_5m
    print(f"    Computing price_change_5m...")
    df['price_change_5m'] = compute_price_change_5m(df)
    
    # Merge liquidations using merge_asof and convert to USD
    print(f"    Merging liquidation columns...")
    if not liq_df.empty:
        merged = pd.merge_asof(
            df[['timestamp', 'price']],
            liq_df[['timestamp', 'liq_long', 'liq_short']],
            on='timestamp',
            direction='backward'
        )
        df['liq_long_usd'] = (merged['liq_long'].fillna(0.0) * df['price'])
        df['liq_short_usd'] = (merged['liq_short'].fillna(0.0) * df['price'])
    else:
        df['liq_long_usd'] = 0.0
        df['liq_short_usd'] = 0.0
    
    # Compute liqimbalance
    total_liq = df['liq_long_usd'] + df['liq_short_usd']
    df['liqimbalance'] = np.where(
        total_liq > 0,
        (df['liq_long_usd'] - df['liq_short_usd']) / total_liq,
        0.0
    )
    
    return df


def main():
    print("Loading CSV file...")
    df = pd.read_csv('tradesdata.csv')
    print(f"Loaded {len(df)} rows")
    
    print("\nAdding static columns...")
    df['regime'] = 'TEST'
    df['time_under_regime'] = 500
    
    df['funding_z'] = 0.0
    df['oi_delta_5m'] = 0.0
    df['price_change_5m'] = 0.0
    df['liq_long_usd'] = 0.0
    df['liq_short_usd'] = 0.0
    df['liqimbalance'] = 0.0
    
    print("\nProcessing each symbol...")
    result_dfs = []
    
    for symbol in SYMBOLS:
        symbol_df = df[df['symbol'] == symbol].copy()
        if len(symbol_df) == 0:
            print(f"  Skipping {symbol} - no data")
            continue
        
        processed_df = process_symbol_data(symbol_df, symbol)
        result_dfs.append(processed_df)
    
    print("\nCombining results...")
    final_df = pd.concat(result_dfs, ignore_index=True)
    
    final_df = final_df.sort_values('timestamp').reset_index(drop=True)
    
    print("\nSaving enriched CSV...")
    final_df.to_csv('tradesdata_enriched.csv', index=False)
    print(f"Saved to tradesdata_enriched.csv ({len(final_df)} rows)")
    
    print("\nNew columns added:")
    new_cols = ['regime', 'time_under_regime', 'funding_z', 'oi_delta_5m', 
                'price_change_5m', 'liq_long_usd', 'liq_short_usd', 'liqimbalance']
    for col in new_cols:
        print(f"  - {col}: {final_df[col].dtype}")
    
    print("\nSample of new columns:")
    print(final_df[['timestamp', 'symbol'] + new_cols].head(10))


if __name__ == '__main__':
    main()
