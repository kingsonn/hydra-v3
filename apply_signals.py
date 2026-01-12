"""
Apply Stage 3 signals to enriched tradesdata
Creates: Signals, Total_Score, Decision columns
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

# Signal thresholds from stage3/processors/signals.py
MIN_PRICE_MOVE = 0.002  # 0.2%
MIN_SCORE_THRESHOLD = 0.6
MIN_ASYMMETRY = 0.35
EXTREME_FUNDING_Z = 2.5


def apply_signals_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all 5 Stage 3 signals using vectorized operations."""
    
    df = df.copy()
    
    # Initialize signal columns
    df['sig_funding_squeeze'] = ''
    df['sig_funding_squeeze_conf'] = 0.0
    df['sig_funding_squeeze_dir'] = ''
    
    df['sig_liq_exhaustion'] = ''
    df['sig_liq_exhaustion_conf'] = 0.0
    df['sig_liq_exhaustion_dir'] = ''
    
    df['sig_oi_divergence'] = ''
    df['sig_oi_divergence_conf'] = 0.0
    df['sig_oi_divergence_dir'] = ''
    
    df['sig_crowding_fade'] = ''
    df['sig_crowding_fade_conf'] = 0.0
    df['sig_crowding_fade_dir'] = ''
    
    df['sig_funding_carry'] = ''
    df['sig_funding_carry_conf'] = 0.0
    df['sig_funding_carry_dir'] = ''
    
    # ========== SIGNAL 1: FUNDING SQUEEZE ==========
    # Longs crowded & price stalling/falling -> SHORT
    mask_fs_short = (
        (df['regime'] != 'CHOP') &
        (df['funding_z'] > 1.2) &
        (df['oi_delta_5m'] > 0.01) &
        (df['price_change_5m'] <= 0)
    )
    df.loc[mask_fs_short, 'sig_funding_squeeze'] = 'Funding squeeze (crowded longs)'
    df.loc[mask_fs_short, 'sig_funding_squeeze_conf'] = 0.7
    df.loc[mask_fs_short, 'sig_funding_squeeze_dir'] = 'SHORT'
    
    # Shorts crowded & price stalling/rising -> LONG
    mask_fs_long = (
        (df['regime'] != 'CHOP') &
        (df['funding_z'] < -1.2) &
        (df['oi_delta_5m'] > 0.01) &
        (df['price_change_5m'] >= 0)
    )
    df.loc[mask_fs_long, 'sig_funding_squeeze'] = 'Funding squeeze (crowded shorts)'
    df.loc[mask_fs_long, 'sig_funding_squeeze_conf'] = 0.7
    df.loc[mask_fs_long, 'sig_funding_squeeze_dir'] = 'LONG'
    
    # ========== SIGNAL 2: LIQUIDATION EXHAUSTION ==========
    # Only in EXPANSION with absorption_z > 1.0
    # Heavy long liquidations -> LONG (bounce)
    mask_le_long = (
        (df['regime'] == 'EXPANSION') &
        (df['absorption_z'] > 1.0) &
        (df['liqimbalance'] > 0.6)
    )
    df.loc[mask_le_long, 'sig_liq_exhaustion'] = 'Long liquidation exhaustion'
    df.loc[mask_le_long, 'sig_liq_exhaustion_conf'] = 0.65
    df.loc[mask_le_long, 'sig_liq_exhaustion_dir'] = 'LONG'
    
    # Heavy short liquidations -> SHORT (fade)
    mask_le_short = (
        (df['regime'] == 'EXPANSION') &
        (df['absorption_z'] > 1.0) &
        (df['liqimbalance'] < -0.6)
    )
    df.loc[mask_le_short, 'sig_liq_exhaustion'] = 'Short liquidation exhaustion'
    df.loc[mask_le_short, 'sig_liq_exhaustion_conf'] = 0.65
    df.loc[mask_le_short, 'sig_liq_exhaustion_dir'] = 'SHORT'
    
    # ========== SIGNAL 3: OI DIVERGENCE ==========
    # Price up but OI down -> weak rally -> SHORT
    mask_oi_short = (
        (np.abs(df['price_change_5m']) >= MIN_PRICE_MOVE) &
        (df['price_change_5m'] > 0) &
        (df['oi_delta_5m'] < -0.01)
    )
    df.loc[mask_oi_short, 'sig_oi_divergence'] = 'OI divergence (weak rally)'
    df.loc[mask_oi_short, 'sig_oi_divergence_conf'] = 0.55
    df.loc[mask_oi_short, 'sig_oi_divergence_dir'] = 'SHORT'
    
    # Price down but OI down -> weak selloff -> LONG
    mask_oi_long = (
        (np.abs(df['price_change_5m']) >= MIN_PRICE_MOVE) &
        (df['price_change_5m'] < 0) &
        (df['oi_delta_5m'] < -0.01)
    )
    df.loc[mask_oi_long, 'sig_oi_divergence'] = 'OI divergence (weak selloff)'
    df.loc[mask_oi_long, 'sig_oi_divergence_conf'] = 0.55
    df.loc[mask_oi_long, 'sig_oi_divergence_dir'] = 'LONG'
    
    # ========== SIGNAL 4: CROWDING FADE ==========
    # Only if funding squeeze didn't fire (prevent double counting)
    has_funding_squeeze = df['sig_funding_squeeze'] != ''
    
    # Extremely crowded longs -> SHORT
    mask_cf_short = (
        ~has_funding_squeeze &
        (df['regime'] != 'CHOP') &
        ~((df['regime'] == 'EXPANSION') & (df['time_under_regime'] < 120)) &
        (df['funding_z'] > 1.5)
    )
    df.loc[mask_cf_short, 'sig_crowding_fade'] = 'Crowded longs'
    df.loc[mask_cf_short, 'sig_crowding_fade_conf'] = 0.6
    df.loc[mask_cf_short, 'sig_crowding_fade_dir'] = 'SHORT'
    
    # Extremely crowded shorts -> LONG
    mask_cf_long = (
        ~has_funding_squeeze &
        (df['regime'] != 'CHOP') &
        ~((df['regime'] == 'EXPANSION') & (df['time_under_regime'] < 120)) &
        (df['funding_z'] < -1.5)
    )
    df.loc[mask_cf_long, 'sig_crowding_fade'] = 'Crowded shorts'
    df.loc[mask_cf_long, 'sig_crowding_fade_conf'] = 0.6
    df.loc[mask_cf_long, 'sig_crowding_fade_dir'] = 'LONG'
    
    # ========== SIGNAL 5: FUNDING CARRY ==========
    # Only in COMPRESSION with abs(funding_z) > 0.8
    # Positive funding -> go short to earn
    mask_fc_short = (
        (df['regime'] == 'COMPRESSION') &
        (np.abs(df['funding_z']) >= 0.8) &
        (df['funding_z'] > 0)
    )
    df.loc[mask_fc_short, 'sig_funding_carry'] = 'Funding carry (range)'
    df.loc[mask_fc_short, 'sig_funding_carry_conf'] = 0.5
    df.loc[mask_fc_short, 'sig_funding_carry_dir'] = 'SHORT'
    
    # Negative funding -> go long to earn
    mask_fc_long = (
        (df['regime'] == 'COMPRESSION') &
        (np.abs(df['funding_z']) >= 0.8) &
        (df['funding_z'] < 0)
    )
    df.loc[mask_fc_long, 'sig_funding_carry'] = 'Funding carry (range)'
    df.loc[mask_fc_long, 'sig_funding_carry_conf'] = 0.5
    df.loc[mask_fc_long, 'sig_funding_carry_dir'] = 'LONG'
    
    # ========== AGGREGATE SIGNALS ==========
    
    # Collect all signals into a single column
    def build_signals_str(row):
        signals = []
        for sig in ['sig_funding_squeeze', 'sig_liq_exhaustion', 'sig_oi_divergence', 
                    'sig_crowding_fade', 'sig_funding_carry']:
            if row[sig]:
                signals.append(row[sig])
        return '; '.join(signals) if signals else 'None'
    
    # Calculate long and short scores
    df['long_score'] = 0.0
    df['short_score'] = 0.0
    
    for sig in ['sig_funding_squeeze', 'sig_liq_exhaustion', 'sig_oi_divergence', 
                'sig_crowding_fade', 'sig_funding_carry']:
        conf_col = sig + '_conf'
        dir_col = sig + '_dir'
        df['long_score'] += np.where(df[dir_col] == 'LONG', df[conf_col], 0)
        df['short_score'] += np.where(df[dir_col] == 'SHORT', df[conf_col], 0)
    
    # Total score is max of long/short
    df['Total_Score'] = np.maximum(df['long_score'], df['short_score'])
    
    # ========== APPLY VETO RULES ==========
    
    # CHOP regime = no trading
    is_chop = df['regime'] == 'CHOP'
    
    # Extreme funding in expansion = no trading
    is_extreme_expansion = (
        (np.abs(df['funding_z']) > EXTREME_FUNDING_Z) &
        (df['regime'] == 'EXPANSION')
    )
    
    # Score too low
    score_too_low = df['Total_Score'] < MIN_SCORE_THRESHOLD
    
    # Not enough asymmetry
    score_diff = np.abs(df['long_score'] - df['short_score'])
    low_asymmetry = score_diff < MIN_ASYMMETRY
    
    # ========== DECISION ==========
    
    # Start with direction based on scores
    df['Decision'] = np.where(
        df['long_score'] > df['short_score'],
        'LONG',
        np.where(df['short_score'] > df['long_score'], 'SHORT', 'NONE')
    )
    
    # Apply vetoes
    df.loc[is_chop, 'Decision'] = 'NONE'
    df.loc[is_extreme_expansion, 'Decision'] = 'NONE'
    df.loc[score_too_low, 'Decision'] = 'NONE'
    df.loc[low_asymmetry, 'Decision'] = 'NONE'
    
    # Build Signals column
    print("Building Signals column...")
    signal_cols = ['sig_funding_squeeze', 'sig_liq_exhaustion', 'sig_oi_divergence', 
                   'sig_crowding_fade', 'sig_funding_carry']
    
    # Vectorized approach to build signals string
    df['Signals'] = ''
    for col in signal_cols:
        df['Signals'] = df['Signals'] + np.where(
            df[col] != '', 
            np.where(df['Signals'] != '', '; ', '') + df[col],
            ''
        )
    df.loc[df['Signals'] == '', 'Signals'] = 'None'
    
    # Drop intermediate columns
    drop_cols = [
        'sig_funding_squeeze', 'sig_funding_squeeze_conf', 'sig_funding_squeeze_dir',
        'sig_liq_exhaustion', 'sig_liq_exhaustion_conf', 'sig_liq_exhaustion_dir',
        'sig_oi_divergence', 'sig_oi_divergence_conf', 'sig_oi_divergence_dir',
        'sig_crowding_fade', 'sig_crowding_fade_conf', 'sig_crowding_fade_dir',
        'sig_funding_carry', 'sig_funding_carry_conf', 'sig_funding_carry_dir',
        'long_score', 'short_score'
    ]
    df.drop(columns=drop_cols, inplace=True)
    
    return df


def main():
    print("Loading enriched CSV...")
    df = pd.read_csv('tradesdata_enriched.csv')
    print(f"Loaded {len(df)} rows")
    
    print("\nApplying Stage 3 signals...")
    df = apply_signals_vectorized(df)
    
    print("\nSignal statistics:")
    print(f"  Total rows: {len(df)}")
    print(f"  Rows with signals: {(df['Signals'] != 'None').sum()}")
    print(f"  LONG decisions: {(df['Decision'] == 'LONG').sum()}")
    print(f"  SHORT decisions: {(df['Decision'] == 'SHORT').sum()}")
    print(f"  NONE decisions: {(df['Decision'] == 'NONE').sum()}")
    
    print("\nScore distribution:")
    print(df['Total_Score'].describe())
    
    print("\nDecision distribution by symbol:")
    print(df.groupby('symbol')['Decision'].value_counts().unstack(fill_value=0))
    
    print("\nSample rows with signals:")
    sample_with_signals = df[df['Signals'] != 'None'].head(20)
    print(sample_with_signals[['timestamp', 'symbol', 'Signals', 'Total_Score', 'Decision']])
    
    print("\nSaving to tradesdata_with_signals.csv...")
    df.to_csv('tradesdata_with_signals.csv', index=False)
    print(f"Saved {len(df)} rows with {len(df.columns)} columns")
    
    print("\nFinal columns:")
    print(df.columns.tolist())


if __name__ == '__main__':
    main()
