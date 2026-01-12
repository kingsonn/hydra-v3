"""
Run ML models on tradesdata_entry_filtered.csv

For each row:
1. Match direction (LONG→up, SHORT→down) and vol_regime (HIGH→high, MID→mid, LOW→low)
2. Create feature vector: 7 features + 8 one-hot encoded symbol columns
3. Run through both 60 and 300 models
4. Add 4 columns: Model_name_60, Model_name_300, Model_output_60, Model_output_300
"""

import pandas as pd
import numpy as np
import pickle
import warnings
import time
from pathlib import Path

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Feature columns from feature_columns.json
FEATURE_COLUMNS = [
    "MOI_250ms", "MOI_1s", "delta_velocity", "AggressionPersistence", 
    "absorption_z", "dist_lvn", "vol_5m"
]

# One-hot encoded pair columns
PAIR_COLUMNS = [
    "pair_ADAUSDT", "pair_BNBUSDT", "pair_BTCUSDT", "pair_DOGEUSDT",
    "pair_ETHUSDT", "pair_LTCUSDT", "pair_SOLUSDT", "pair_XRPUSDT"
]

# Symbol list matching pair column order
SYMBOLS = ["ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT", "XRPUSDT"]


def load_models(models_dir: str = "ml_models") -> dict:
    """Load all ML models from pickle files."""
    models = {}
    models_path = Path(models_dir)
    
    print("Loading ML models...")
    for pkl_file in models_path.glob("*.pkl"):
        model_name = pkl_file.stem
        with open(pkl_file, 'rb') as f:
            model_ensemble = pickle.load(f)
        models[model_name] = model_ensemble
        n_models = len(model_ensemble) if isinstance(model_ensemble, list) else 1
        print(f"  {model_name}: {n_models} model(s)")
    
    return models


def get_model_key(direction: str, vol_regime: str) -> str:
    """Get base model key (without time suffix)."""
    dir_str = "up" if direction == "LONG" else "down"
    regime_str = vol_regime.lower() if isinstance(vol_regime, str) else "mid"
    if regime_str not in ["high", "mid", "low"]:
        regime_str = "mid"
    return f"models_{dir_str}_{regime_str}"


def process_by_group(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Process data by grouping on direction, vol_regime, symbol for batch predictions."""
    df = df.copy()
    
    # Initialize columns
    df['Model_name_60'] = ''
    df['Model_name_300'] = ''
    df['Model_output_60'] = np.nan
    df['Model_output_300'] = np.nan
    
    # Fill NaN in feature columns with 0
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Normalize vol_regime
    if 'vol_regime' in df.columns:
        df['vol_regime'] = df['vol_regime'].fillna('MID').str.upper()
    else:
        df['vol_regime'] = 'MID'
    
    print(f"\nProcessing {len(df):,} rows by group...")
    start_time = time.time()
    
    # Group by direction, vol_regime, symbol
    groups = df.groupby(['Decision', 'vol_regime', 'symbol'])
    total_groups = len(groups)
    
    processed_rows = 0
    for i, ((direction, regime, symbol), group) in enumerate(groups):
        group_indices = group.index
        n_rows = len(group_indices)
        
        # Get model names
        model_key = get_model_key(direction, regime)
        model_name_60 = f"{model_key}_60"
        model_name_300 = f"{model_key}_300"
        
        # Set model names
        df.loc[group_indices, 'Model_name_60'] = model_name_60
        df.loc[group_indices, 'Model_name_300'] = model_name_300
        
        # Build feature matrix for this group
        # 7 features
        feature_matrix = df.loc[group_indices, FEATURE_COLUMNS].values.astype(np.float64)
        
        # 8 one-hot columns for symbol
        one_hot = np.zeros((n_rows, len(SYMBOLS)), dtype=np.float64)
        symbol_idx = SYMBOLS.index(symbol) if symbol in SYMBOLS else -1
        if symbol_idx >= 0:
            one_hot[:, symbol_idx] = 1.0
        
        # Combine
        X = np.hstack([feature_matrix, one_hot])
        
        # Run model 60
        if model_name_60 in models:
            model = models[model_name_60]
            if isinstance(model, list):
                preds = model[-1].predict(X)
            else:
                preds = model.predict(X)
            df.loc[group_indices, 'Model_output_60'] = preds
        
        # Run model 300
        if model_name_300 in models:
            model = models[model_name_300]
            if isinstance(model, list):
                preds = model[-1].predict(X)
            else:
                preds = model.predict(X)
            df.loc[group_indices, 'Model_output_300'] = preds
        
        processed_rows += n_rows
        
        if (i + 1) % 10 == 0 or (i + 1) == total_groups:
            elapsed = time.time() - start_time
            print(f"  Group {i+1}/{total_groups} | {processed_rows:,} rows | {elapsed:.1f}s")
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s ({len(df)/elapsed:,.0f} rows/sec)")
    
    return df


def main():
    print("="*70)
    print("RUN ML MODELS ON ENTRY-FILTERED SIGNALS")
    print("="*70)
    
    # Load models
    models = load_models()
    print(f"\nLoaded {len(models)} models")
    
    # Load data
    print("\nLoading tradesdata_entry_filtered.csv...")
    df = pd.read_csv('tradesdata_entry_filtered.csv', low_memory=False)
    print(f"Loaded {len(df):,} rows")
    
    # Check required columns
    print("\nChecking required columns...")
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        print(f"  WARNING: Missing feature columns: {missing}")
    else:
        print("  All 7 feature columns present")
    
    if 'vol_regime' not in df.columns:
        print("  WARNING: vol_regime column missing, defaulting to 'MID'")
    
    # Show sample of feature values
    print("\nSample feature values:")
    print(df[FEATURE_COLUMNS].head(3))
    
    # Show direction and vol_regime distribution
    print("\nDirection distribution:")
    print(df['Decision'].value_counts())
    
    print("\nVol_regime distribution:")
    if 'vol_regime' in df.columns:
        print(df['vol_regime'].value_counts())
    
    # Process
    df = process_by_group(df, models)
    
    # Statistics
    print("\n" + "="*70)
    print("MODEL PREDICTION STATISTICS")
    print("="*70)
    
    print("\nModel usage by direction and regime:")
    model_counts = df['Model_name_60'].value_counts()
    for model_name, count in model_counts.items():
        print(f"  {model_name}: {count:,} rows")
    
    print("\nModel output statistics (60):")
    print(df['Model_output_60'].describe())
    
    print("\nModel output statistics (300):")
    print(df['Model_output_300'].describe())
    
    # Sample output
    print("\nSample rows with model predictions:")
    sample_cols = ['timestamp', 'symbol', 'Decision', 'vol_regime', 
                   'Model_name_60', 'Model_output_60', 'Model_name_300', 'Model_output_300']
    print(df[sample_cols].head(10))
    
    # Save
    print("\n" + "="*70)
    output_file = 'tradesdata_with_ml_predictions.csv'
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df):,} rows with {len(df.columns)} columns")
    
    print("\nNew columns added:")
    print("  - Model_name_60")
    print("  - Model_name_300")
    print("  - Model_output_60")
    print("  - Model_output_300")


if __name__ == '__main__':
    main()
