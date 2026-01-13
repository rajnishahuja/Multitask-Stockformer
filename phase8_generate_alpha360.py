#!/usr/bin/env python3
"""
Phase 8 Step 6: Generate Alpha360 Factors

Alpha360 = 6 raw OHLCV features × 60 days lag = 360 features
This is the original paper's approach - NO IC filtering, just raw lagged data.

Features per day: Open, High, Low, Close, Volume, VWAP
Lagged 1-60 days back for each feature.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Configuration
RAW_DATA_DIR = "data/NIFTY200_Subset10/raw"
OUTPUT_DIR = "data/NIFTY200_Subset10/Alpha360"
LOOKBACK_DAYS = 60

# Date range (same as Alpha158)
DATE_RANGE = ("2020-06-03", "2024-08-31")


def load_ohlcv_data(raw_dir: str) -> dict:
    """Load OHLCV data from raw CSV files."""
    stocks = {}
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
    csv_files = list(raw_path.glob("*.csv"))
    print(f"Found {len(csv_files)} stock files in {raw_dir}")
    
    for csv_file in tqdm(csv_files, desc="Loading OHLCV"):
        symbol = csv_file.stem
        try:
            df = pd.read_csv(csv_file, parse_dates=['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            
            # Rename if needed
            col_map = {
                'datetime': 'date',
                'open': 'open', 'high': 'high', 'low': 'low', 
                'close': 'close', 'volume': 'volume'
            }
            
            required = ['date', 'open', 'high', 'low', 'close', 'volume']
            if all(c in df.columns for c in required):
                stocks[symbol] = df
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
    
    return stocks


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute Volume-Weighted Average Price."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap


def generate_alpha360_features(stocks: dict, lookback: int = 60) -> pd.DataFrame:
    """
    Generate Alpha360 features: 6 raw features × lookback days.
    
    Output shape per day: (num_stocks, 360)
    """
    all_data = []
    feature_names = []
    
    # Generate feature names
    raw_features = ['open', 'high', 'low', 'close', 'volume', 'vwap']
    for feature in raw_features:
        for lag in range(1, lookback + 1):
            feature_names.append(f"{feature.upper()}_LAG{lag}")
    
    print(f"Generating {len(feature_names)} features...")
    
    for symbol, df in tqdm(stocks.items(), desc="Computing Alpha360"):
        df = df.copy()
        df['vwap'] = compute_vwap(df)
        
        # Normalize features (z-score per stock)
        for col in raw_features:
            if col in df.columns:
                mean_val = df[col].rolling(window=20, min_periods=5).mean()
                std_val = df[col].rolling(window=20, min_periods=5).std()
                df[f'{col}_norm'] = (df[col] - mean_val) / (std_val + 1e-8)
        
        # Generate lagged features
        for idx, row in df.iterrows():
            if idx < lookback:
                continue
            
            date = row['date']
            features = {'date': date, 'symbol': symbol}
            
            for feature in raw_features:
                norm_col = f'{feature}_norm'
                if norm_col in df.columns:
                    for lag in range(1, lookback + 1):
                        feat_name = f"{feature.upper()}_LAG{lag}"
                        features[feat_name] = df.loc[idx - lag, norm_col]
            
            all_data.append(features)
    
    result_df = pd.DataFrame(all_data)
    print(f"Generated features: {result_df.shape}")
    return result_df, feature_names


def save_factors_by_name(df: pd.DataFrame, feature_names: list, output_dir: str):
    """Save each factor as a separate CSV (matching Alpha158 format)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Pivot to date × stock format for each factor
    for feature in tqdm(feature_names, desc="Saving factors"):
        if feature not in df.columns:
            continue
        
        pivot = df.pivot(index='date', columns='symbol', values=feature)
        pivot.to_csv(output_path / f"{feature}.csv")
    
    print(f"Saved {len(feature_names)} factor files to {output_dir}")


def main():
    print("=" * 60)
    print("PHASE 8 STEP 6: Generate Alpha360 Factors")
    print("=" * 60)
    
    # Load raw OHLCV
    print(f"\n1. Loading raw data from {RAW_DATA_DIR}...")
    stocks = load_ohlcv_data(RAW_DATA_DIR)
    print(f"   Loaded {len(stocks)} stocks")
    
    # Generate Alpha360 features
    print(f"\n2. Generating Alpha360 features (6 × {LOOKBACK_DAYS} = {6 * LOOKBACK_DAYS})...")
    df, feature_names = generate_alpha360_features(stocks, lookback=LOOKBACK_DAYS)
    
    # Filter date range
    df['date'] = pd.to_datetime(df['date'])
    df = df[(df['date'] >= DATE_RANGE[0]) & (df['date'] <= DATE_RANGE[1])]
    print(f"   After date filter: {df.shape[0]} rows")
    
    # Save factors
    print(f"\n3. Saving to {OUTPUT_DIR}...")
    save_factors_by_name(df, feature_names, OUTPUT_DIR)
    
    # Also save a combined dataset for training
    combined_path = Path(OUTPUT_DIR) / "alpha360_combined.csv"
    df.to_csv(combined_path, index=False)
    print(f"   Combined dataset: {combined_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Stocks: {df['symbol'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Total rows: {len(df)}")
    print(f"\nNext: Update training script to use Alpha360 instead of Alpha158")


if __name__ == "__main__":
    main()
