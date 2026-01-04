#!/usr/bin/env python3
"""
Phase 4: Stockformer Input Data Preprocessing (NIFTY-200 Adaptation)

Follows the original Chinese implementation pattern:
- flow.npz['result']: shape (T, N) - 2D returns only (NO wavelet decomposition)
- Factor CSVs: loaded separately by dataset loader during training

This script generates 4 required files:
1. label.csv - Daily returns from raw OHLCV data
2. flow.npz - Returns data (same as label.csv, 2D format)
3. trend_indicator.npz - Binary up/down classification  
4. corr_adj.npy - Stock correlation matrix
5. 128_corr_struc2vec_adjgat.npy - Graph embeddings (reuses existing)

Reference: data_processing_script/stockformer_input_data_processing/
          Stockformer_data_preprocessing_script.py
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from datetime import datetime

# Configuration
BASE_DIR = Path('/home/ubuntu/rajnish/Multitask-Stockformer')
RAW_DATA_DIR = BASE_DIR / 'data/NIFTY200/raw'
FACTOR_DIR = BASE_DIR / 'data/NIFTY200/Alpha_158_2022-01-01_2024-08-31'
OUTPUT_DIR = BASE_DIR / 'data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31'

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("Phase 4: Stockformer Input Data Preprocessing")
print("NIFTY-200 Adaptation (Following Chinese Implementation)")
print("="*70)
print(f"\nConfiguration:")
print(f"  Raw data directory: {RAW_DATA_DIR}")
print(f"  Factor directory: {FACTOR_DIR}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"\nReference Implementation:")
print(f"  Chinese script: Stockformer_data_preprocessing_script.py")
print(f"  Key principle: flow.npz = returns ONLY (2D), NO wavelet here\n")


# ===========================================================================
# STEP 1: Generate Daily Returns from Raw OHLCV Data → label.csv
# ===========================================================================

print("\n" + "="*70)
print("STEP 1: Generating Daily Returns (label.csv)")
print("="*70)

# Load all raw stock data files
raw_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')])
print(f"\nFound {len(raw_files)} raw stock CSV files")

# Dictionary to store stock data
stock_data = {}

print("\nLoading raw data...")
for idx, csv_file in enumerate(raw_files):
    if (idx + 1) % 50 == 0:
        print(f"  Loaded {idx + 1}/{len(raw_files)} files...")
    
    symbol = csv_file.replace('.csv', '')
    df = pd.read_csv(RAW_DATA_DIR / csv_file, index_col='Date', parse_dates=True)
    stock_data[symbol] = df

print(f"✓ Loaded {len(stock_data)} stocks")

# Find stocks with complete date coverage (filter out recent IPOs)
print("\nAligning dates across all stocks...")

# Identify the most common (complete) date range
date_counts = {}
for symbol, df in stock_data.items():
    date_range = (df.index[0], df.index[-1])
    if date_range not in date_counts:
        date_counts[date_range] = 0
    date_counts[date_range] += 1

print(f"Date range distribution:")
for date_range, count in sorted(date_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {date_range[0].date()} to {date_range[1].date()}: {count} stocks")

# Use the most common date range (represents stocks with full historical coverage)
most_common_range = max(date_counts.items(), key=lambda x: x[1])[0]
print(f"\nUsing most common date range: {most_common_range[0].date()} to {most_common_range[1].date()}")

# Filter to stocks with complete date range (excludes recent IPOs)
aligned_stocks = {symbol: df for symbol, df in stock_data.items() 
                  if (df.index[0], df.index[-1]) == most_common_range}

excluded_stocks = [s for s in stock_data.keys() if s not in aligned_stocks]

print(f"✓ Stocks with complete date coverage: {len(aligned_stocks)}/{len(stock_data)}")
print(f"  Excluded (recent IPOs): {len(excluded_stocks)}")

if excluded_stocks:
    print(f"\nExcluded stocks (insufficient historical coverage):")
    for symbol in sorted(excluded_stocks):
        df = stock_data[symbol]
        print(f"  - {symbol:15s}: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")

# Use dates from aligned stocks
common_dates = sorted(list(aligned_stocks[list(aligned_stocks.keys())[0]].index))
num_dates = len(common_dates)
print(f"\n✓ Common trading dates: {num_dates}")
print(f"  Date range: {common_dates[0].date()} to {common_dates[-1].date()}")

# Create returns matrix: [time, stock]
symbols = sorted(aligned_stocks.keys())
num_stocks = len(symbols)

print(f"\nCreating returns matrix: ({num_dates}, {num_stocks})")

returns = np.zeros((num_dates, num_stocks))

for stock_idx, symbol in enumerate(symbols):
    df = aligned_stocks[symbol].loc[common_dates]
    close_prices = df['Close'].values
    
    # Calculate daily returns: (close_t - close_{t-1}) / close_{t-1}
    daily_returns = np.diff(close_prices) / close_prices[:-1]
    
    # First day has no previous close, set to 0
    returns[:, stock_idx] = np.concatenate([[0.0], daily_returns])

# Handle NaN/Inf values
nan_count = np.isnan(returns).sum()
inf_count = np.isinf(returns).sum()
if nan_count > 0:
    print(f"  Warning: Replacing {nan_count} NaN values with 0")
    returns = np.nan_to_num(returns, nan=0.0)
if inf_count > 0:
    print(f"  Warning: Replacing {inf_count} Inf values with 0")
    returns = np.nan_to_num(returns, posinf=0.0, neginf=0.0)

print(f"\n✓ Returns statistics:")
print(f"  Shape: {returns.shape}")
print(f"  Min: {returns.min():.6f}")
print(f"  Max: {returns.max():.6f}")
print(f"  Mean: {returns.mean():.6f}")
print(f"  Std: {returns.std():.6f}")

# Save as label.csv
label_df = pd.DataFrame(returns, index=common_dates, columns=symbols)
label_path = OUTPUT_DIR / 'label.csv'
label_df.to_csv(label_path)

print(f"\n✓ Saved: {label_path}")
print(f"  File size: {label_path.stat().st_size / 1024:.2f} KB")
print(f"  Shape: {label_df.shape}")


# ===========================================================================
# STEP 2: Generate flow.npz (Following Chinese Implementation)
# ===========================================================================

print("\n" + "="*70)
print("STEP 2: Generating flow.npz (Returns Data)")
print("="*70)
print("\nKey Principle: flow.npz = returns ONLY (2D), NO wavelet decomposition")
print("  Wavelet decomposition happens inside the model during training")
print("  Chinese script: np.savez('flow.npz', result=df.values)\n")

# Following Chinese implementation: direct save without wavelet
flow_data = label_df.values  # Shape: (660, 191)

# Save with 'result' key (matching Chinese implementation)
flow_output_path = OUTPUT_DIR / 'flow.npz'
np.savez_compressed(flow_output_path, result=flow_data)

file_size_mb = flow_output_path.stat().st_size / 1024 / 1024
print(f"✓ Saved: {flow_output_path}")
print(f"  File size: {file_size_mb:.2f} MB")
print(f"  Key: 'result'")
print(f"  Shape: {flow_data.shape} (2D: timesteps × stocks)")

# Verify
loaded = np.load(flow_output_path)
print(f"\nVerification:")
print(f"  Keys in file: {list(loaded.keys())}")
print(f"  result shape: {loaded['result'].shape}")
print(f"  ✓ Matches Chinese format: (T, N) 2D array")

# ===========================================================================
# STEP 3: Generate trend_indicator.npz
# ===========================================================================

print("\n" + "="*70)
print("STEP 3: Generating trend_indicator.npz")
print("="*70)
print("\nBinary classification: 1 if return > 0 (up), else 0 (down)\n")

# Following Chinese implementation
trend_indicator = (flow_data > 0).astype(np.int32)

print(f"Trend indicator statistics:")
print(f"  Shape: {trend_indicator.shape}")
print(f"  Up days (1): {(trend_indicator == 1).sum()} ({(trend_indicator == 1).mean()*100:.2f}%)")
print(f"  Down days (0): {(trend_indicator == 0).sum()} ({(trend_indicator == 0).mean()*100:.2f}%)")

# Save with 'result' key
trend_output_path = OUTPUT_DIR / 'trend_indicator.npz'
np.savez_compressed(trend_output_path, result=trend_indicator)

print(f"\n✓ Saved: {trend_output_path}")
print(f"  File size: {trend_output_path.stat().st_size / 1024:.2f} KB")
print(f"  Key: 'result'")
print(f"  Shape: {trend_indicator.shape}")

# ===========================================================================
# STEP 4: Generate corr_adj.npy
# ===========================================================================

print("\n" + "="*70)
print("STEP 4: Generating Correlation Adjacency Matrix")
print("="*70)
print("\nCalculating stock-stock correlation from returns data\n")

# Following Chinese implementation
# Handle zero variance columns
epsilon = 1e-10
df_for_corr = label_df.copy()
std_devs = np.std(df_for_corr, axis=0)
zero_variance_mask = std_devs < epsilon
num_zero_var = zero_variance_mask.sum()

if num_zero_var > 0:
    print(f"  Warning: {num_zero_var} stocks with near-zero variance")
    print(f"  Replacing with epsilon ({epsilon}) to avoid correlation issues")
    df_for_corr.loc[:, zero_variance_mask] = epsilon

# Calculate correlation matrix (stocks as variables, time as observations)
corr_matrix = np.corrcoef(df_for_corr, rowvar=False)  # Shape: (191, 191)

print(f"✓ Correlation matrix computed:")
print(f"  Shape: {corr_matrix.shape}")
print(f"  Diagonal (should be 1.0): min={np.diag(corr_matrix).min():.4f}, max={np.diag(corr_matrix).max():.4f}")
print(f"  Symmetric: {np.allclose(corr_matrix, corr_matrix.T)}")

# Save
corr_adj_path = OUTPUT_DIR / 'corr_adj.npy'
np.save(corr_adj_path, corr_matrix)

print(f"\n✓ Saved: {corr_adj_path}")
print(f"  File size: {corr_adj_path.stat().st_size / 1024:.2f} KB")

# ===========================================================================
# STEP 5: Filter Graph Embeddings to 185-Stock Universe
# ===========================================================================

print("\n" + "="*70)
print("STEP 5: Graph Embeddings (adjgat)")
print("="*70)

adjgat_path = OUTPUT_DIR / '128_corr_struc2vec_adjgat.npy'

if adjgat_path.exists():
    adjgat_orig = np.load(adjgat_path)
    print(f"\nOriginal adjgat shape: {adjgat_orig.shape}")
    
    # Check if filtering is needed
    if adjgat_orig.shape[0] == num_stocks:
        print(f"✓ adjgat already aligned with {num_stocks}-stock universe")
        adjgat = adjgat_orig
    elif adjgat_orig.shape[0] > num_stocks:
        print(f"\nFiltering adjgat from {adjgat_orig.shape[0]} to {num_stocks} stocks...")
        
        # Load original stock order from raw data directory
        raw_files = sorted([f.replace('.csv', '') for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')])
        print(f"  Original stock universe: {len(raw_files)} stocks")
        
        # Find indices of our 185 stocks in the original 191-stock list
        stock_indices = [raw_files.index(s) for s in symbols if s in raw_files]
        print(f"  Mapped {len(stock_indices)} stocks to original indices")
        
        # Filter embeddings
        adjgat = adjgat_orig[stock_indices, :]
        print(f"  Filtered shape: {adjgat.shape}")
        
        # Save filtered embeddings
        np.save(adjgat_path, adjgat)
        print(f"✓ Saved filtered adjgat: {adjgat.shape}")
    else:
        print(f"⚠ Warning: adjgat has fewer stocks ({adjgat_orig.shape[0]}) than expected ({num_stocks})")
        adjgat = adjgat_orig
    
    print(f"\n✓ Final adjgat embeddings: {adjgat_path}")
    print(f"  Shape: {adjgat.shape}")
    print(f"  File size: {adjgat_path.stat().st_size / 1024:.2f} KB")
else:
    print(f"\n⚠ adjgat file not found: {adjgat_path}")
    print(f"  You can generate it using node2vec or Struc2Vec")
    print(f"  For now, proceeding without it (can add later)")

# ===========================================================================
# STEP 6: Filter Factor CSVs to 185-Stock Universe
# ===========================================================================

print("\n" + "="*70)
print("STEP 6: Factor CSV Filtering")
print("="*70)
print("\nFiltering factor CSVs to match 185-stock universe...\n")

factor_files = sorted([f for f in os.listdir(FACTOR_DIR) 
                      if f.endswith('.csv') and 'ic_summary' not in f.lower()])

print(f"Found {len(factor_files)} factor CSV files")

# Filter each factor file
filtered_count = 0
skipped_count = 0
error_count = 0

for fname in factor_files:
    fpath = FACTOR_DIR / fname
    try:
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        
        # Skip files with wrong date range
        if df.shape[0] != num_dates:
            if skipped_count == 0:
                print(f"\nSkipping files with wrong date range:")
            print(f"  - {fname}: shape {df.shape} (expected {num_dates} days)")
            skipped_count += 1
            continue
        
        # Check if already filtered
        if df.shape[1] == num_stocks:
            filtered_count += 1
            continue
        
        # Filter to 185 stocks (only those in our final universe)
        common_stocks = [s for s in symbols if s in df.columns]
        
        if len(common_stocks) != num_stocks:
            missing = set(symbols) - set(df.columns)
            if error_count == 0:
                print(f"\nWarnings:")
            print(f"  - {fname}: missing {len(missing)} stocks from our universe")
            error_count += 1
        
        # Create filtered dataframe with exact column order from symbols list
        df_filtered = df[common_stocks].copy()
        df_filtered = df_filtered[symbols]  # Reorder to match
        
        # Save back
        df_filtered.to_csv(fpath)
        filtered_count += 1
        
        if filtered_count % 10 == 0:
            print(f"  Processed {filtered_count} files...")
            
    except Exception as e:
        print(f"  Error processing {fname}: {e}")
        error_count += 1

print(f"\n✓ Factor CSV filtering complete:")
print(f"  Filtered: {filtered_count} files")
print(f"  Skipped: {skipped_count} files (wrong date range)")
if error_count > 0:
    print(f"  Errors: {error_count} files")

# Verify alignment
print(f"\nVerifying alignment...")
sample_factor = pd.read_csv(FACTOR_DIR / factor_files[0], index_col=0)
print(f"  Sample factor shape: {sample_factor.shape}")
print(f"  Expected shape: ({num_dates}, {num_stocks})")
if sample_factor.shape == (num_dates, num_stocks):
    print(f"  ✓ Factors aligned correctly")
else:
    print(f"  ⚠ Alignment issue detected")

# ===========================================================================
# Summary
# ===========================================================================

print("\n" + "="*70)
print("PHASE 4 COMPLETE - Summary")
print("="*70)

print(f"\nGenerated Files:")
print(f"  1. {OUTPUT_DIR}/label.csv")
print(f"     - Shape: {label_df.shape}")
print(f"     - Content: Daily returns (close-to-close)")
print(f"     - Date range: {label_df.index[0].date()} to {label_df.index[-1].date()}")

print(f"\n  2. {OUTPUT_DIR}/flow.npz")
print(f"     - Key: 'result'")
print(f"     - Shape: {flow_data.shape} (2D: timesteps × stocks)")
print(f"     - Content: Returns data (NO wavelet decomposition)")
print(f"     - ✓ Matches Chinese implementation format")

print(f"\n  3. {OUTPUT_DIR}/trend_indicator.npz")
print(f"     - Key: 'result'")
print(f"     - Shape: {trend_indicator.shape}")
print(f"     - Content: Binary up/down classification")

print(f"\n  4. {OUTPUT_DIR}/corr_adj.npy")
print(f"     - Shape: {corr_matrix.shape}")
print(f"     - Content: Stock-stock correlation matrix")

if adjgat_path.exists():
    print(f"\n  5. {OUTPUT_DIR}/128_corr_struc2vec_adjgat.npy")
    print(f"     - Shape: {adjgat.shape}")
    print(f"     - Content: Graph embeddings")
    print(f"     - Status: Reused existing file")

print(f"\nFactor CSVs:")
print(f"  - Directory: {FACTOR_DIR}")
print(f"  - Aligned factors: {filtered_count} files with shape ({num_dates}, {num_stocks})")
print(f"  - Skipped: {skipped_count} files (wrong date range)")
print(f"  - These will be loaded by dataset loader during training")

print(f"\nNext Steps:")
print(f"  1. Verify outputs with verification script")
print(f"  2. Update Phase 5 config with correct paths")
print(f"  3. Proceed to model training")

print("\n" + "="*70)
print("✓ Phase 4 preprocessing complete!")
print("="*70)
