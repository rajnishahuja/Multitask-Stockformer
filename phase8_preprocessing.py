#!/usr/bin/env python3
"""
Phase 8: Stockformer Input Data Preprocessing

Generalized preprocessing script for any dataset configuration.
Generates required files for model training:
1. label.csv - Daily returns
2. flow.npz - Returns data (2D format for wavelet decomposition in model)
3. trend_indicator.npz - Binary up/down classification
4. corr_adj.npy - Stock correlation matrix
5. Graph embeddings (generated or identity matrix fallback)

Usage:
    python phase8_preprocessing.py --data-dir <path> --output-dir <path>

Example:
    python phase8_preprocessing.py \
        --data-dir ./data/NIFTY200_Subset10/raw \
        --output-dir ./data/NIFTY200_Subset10
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Force unbuffered output
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Phase 8: Stockformer Preprocessing')
    parser.add_argument('--data-dir', type=str, default='./data/NIFTY200_Subset10/raw',
                        help='Directory containing raw OHLCV CSV files')
    parser.add_argument('--factor-dir', type=str, default=None,
                        help='Directory containing Alpha158 factor CSVs (default: <output-dir>/Alpha158)')
    parser.add_argument('--output-dir', type=str, default='./data/NIFTY200_Subset10',
                        help='Output directory for preprocessed files')
    
    args = parser.parse_args()
    
    # Setup paths
    RAW_DATA_DIR = Path(args.data_dir)
    OUTPUT_DIR = Path(args.output_dir)
    FACTOR_DIR = Path(args.factor_dir) if args.factor_dir else OUTPUT_DIR / 'Alpha158'
    DATASET_DIR = OUTPUT_DIR / 'dataset'
    
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Phase 8: Stockformer Input Data Preprocessing")
    print("="*80)
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Factor directory: {FACTOR_DIR}")
    print(f"Output directory: {DATASET_DIR}")
    print("="*80 + "\n")
    
    # =========================================================================
    # STEP 1: Load Raw OHLCV Data and Generate Returns → label.csv
    # =========================================================================
    
    logger.info("STEP 1: Loading raw OHLCV data and generating returns...")
    
    raw_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')])
    logger.info(f"  Found {len(raw_files)} raw stock CSV files")
    
    stock_data = {}
    for idx, csv_file in enumerate(raw_files, 1):
        if idx % 20 == 0 or idx == 1:
            logger.info(f"  Loading file {idx}/{len(raw_files)}...")
        
        symbol = csv_file.replace('.csv', '')
        df = pd.read_csv(RAW_DATA_DIR / csv_file, index_col='Date', parse_dates=True)
        stock_data[symbol] = df
    
    logger.info(f"✓ Loaded {len(stock_data)} stocks")
    
    # Find common date range
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df.index)
    
    common_dates = sorted(all_dates)
    num_dates = len(common_dates)
    symbols = sorted(stock_data.keys())
    num_stocks = len(symbols)
    
    logger.info(f"  Date range: {common_dates[0].date()} to {common_dates[-1].date()}")
    logger.info(f"  Trading days: {num_dates}")
    logger.info(f"  Stocks: {num_stocks}")
    
    # Create returns matrix
    logger.info("\n  Creating returns matrix...")
    returns = np.zeros((num_dates, num_stocks))
    
    for stock_idx, symbol in enumerate(symbols):
        df = stock_data[symbol]
        # Align to common dates
        df_aligned = df.reindex(common_dates)
        close_prices = df_aligned['Close'].values
        
        # Calculate daily returns
        daily_returns = np.diff(close_prices) / (close_prices[:-1] + 1e-12)
        returns[:, stock_idx] = np.concatenate([[0.0], daily_returns])
    
    # Handle NaN/Inf values
    nan_count = np.isnan(returns).sum()
    inf_count = np.isinf(returns).sum()
    if nan_count > 0:
        logger.info(f"  Replacing {nan_count} NaN values with 0")
        returns = np.nan_to_num(returns, nan=0.0)
    if inf_count > 0:
        logger.info(f"  Replacing {inf_count} Inf values with 0")
        returns = np.nan_to_num(returns, posinf=0.0, neginf=0.0)
    
    # Save label.csv
    label_df = pd.DataFrame(returns, index=common_dates, columns=symbols)
    label_path = DATASET_DIR / 'label.csv'
    label_df.to_csv(label_path)
    
    logger.info(f"\n✓ Saved: {label_path}")
    logger.info(f"  Shape: {label_df.shape}")
    logger.info(f"  Returns stats: min={returns.min():.4f}, max={returns.max():.4f}, mean={returns.mean():.6f}")
    
    # =========================================================================
    # STEP 2: Generate flow.npz (Returns Data)
    # =========================================================================
    
    logger.info("\nSTEP 2: Generating flow.npz (returns data for wavelet)...")
    
    flow_data = label_df.values
    flow_output_path = DATASET_DIR / 'flow.npz'
    np.savez_compressed(flow_output_path, result=flow_data)
    
    logger.info(f"✓ Saved: {flow_output_path}")
    logger.info(f"  Shape: {flow_data.shape} (2D: timesteps × stocks)")
    logger.info(f"  Key: 'result'")
    
    # =========================================================================
    # STEP 3: Generate trend_indicator.npz
    # =========================================================================
    
    logger.info("\nSTEP 3: Generating trend_indicator.npz (binary up/down)...")
    
    trend_indicator = (flow_data > 0).astype(np.int32)
    trend_output_path = DATASET_DIR / 'trend_indicator.npz'
    np.savez_compressed(trend_output_path, result=trend_indicator)
    
    up_pct = (trend_indicator == 1).mean() * 100
    logger.info(f"✓ Saved: {trend_output_path}")
    logger.info(f"  Shape: {trend_indicator.shape}")
    logger.info(f"  Up days: {up_pct:.1f}%, Down days: {100-up_pct:.1f}%")
    
    # =========================================================================
    # STEP 4: Generate corr_adj.npy (Stock Correlation Matrix)
    # =========================================================================
    
    logger.info("\nSTEP 4: Generating corr_adj.npy (stock correlation matrix)...")
    
    # Handle zero variance columns
    epsilon = 1e-10
    df_for_corr = label_df.copy()
    std_devs = np.std(df_for_corr, axis=0)
    zero_variance_mask = std_devs < epsilon
    
    if zero_variance_mask.sum() > 0:
        logger.info(f"  Warning: {zero_variance_mask.sum()} stocks with near-zero variance")
        df_for_corr.loc[:, zero_variance_mask] = epsilon
    
    corr_matrix = np.corrcoef(df_for_corr, rowvar=False)
    
    # Handle NaN in correlation (can happen if still zero variance)
    if np.isnan(corr_matrix).any():
        logger.info(f"  Replacing NaN correlations with 0")
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    corr_adj_path = DATASET_DIR / 'corr_adj.npy'
    np.save(corr_adj_path, corr_matrix)
    
    logger.info(f"✓ Saved: {corr_adj_path}")
    logger.info(f"  Shape: {corr_matrix.shape}")
    
    # =========================================================================
    # STEP 5: Generate Graph Embeddings (Identity Matrix Fallback)
    # =========================================================================
    
    logger.info("\nSTEP 5: Generating graph embeddings (128-dim)...")
    
    adjgat_path = DATASET_DIR / '128_corr_struc2vec_adjgat.npy'
    
    # For now, use identity-like embeddings (each stock is unique)
    # This can be replaced with proper Struc2Vec embeddings later
    embedding_dim = 128
    adjgat = np.eye(num_stocks, embedding_dim)
    
    # Add some random noise to make embeddings more useful
    np.random.seed(42)
    adjgat += np.random.randn(num_stocks, embedding_dim) * 0.1
    
    # Normalize rows
    row_norms = np.linalg.norm(adjgat, axis=1, keepdims=True)
    adjgat = adjgat / (row_norms + 1e-12)
    
    np.save(adjgat_path, adjgat.astype(np.float32))
    
    logger.info(f"✓ Saved: {adjgat_path}")
    logger.info(f"  Shape: {adjgat.shape}")
    logger.info(f"  Note: Using identity + noise embeddings (can be replaced with Struc2Vec)")
    
    # =========================================================================
    # STEP 6: Save Stock List (instruments)
    # =========================================================================
    
    logger.info("\nSTEP 6: Saving stock/instrument list...")
    
    instruments_dir = OUTPUT_DIR / 'instruments'
    instruments_dir.mkdir(parents=True, exist_ok=True)
    instruments_path = instruments_dir / 'fno_stocks.txt'
    
    with open(instruments_path, 'w') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    
    logger.info(f"✓ Saved: {instruments_path}")
    logger.info(f"  Stocks: {num_stocks}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    
    print(f"\nGenerated Files:")
    print(f"  1. {label_path}")
    print(f"     - Shape: {label_df.shape}")
    print(f"     - Content: Daily returns (close-to-close)")
    
    print(f"\n  2. {flow_output_path}")
    print(f"     - Shape: {flow_data.shape}")
    print(f"     - Content: Returns data for wavelet decomposition")
    
    print(f"\n  3. {trend_output_path}")
    print(f"     - Shape: {trend_indicator.shape}")
    print(f"     - Content: Binary up/down classification")
    
    print(f"\n  4. {corr_adj_path}")
    print(f"     - Shape: {corr_matrix.shape}")
    print(f"     - Content: Stock-stock correlation matrix")
    
    print(f"\n  5. {adjgat_path}")
    print(f"     - Shape: {adjgat.shape}")
    print(f"     - Content: Graph embeddings (128-dim)")
    
    print(f"\n  6. {instruments_path}")
    print(f"     - Stocks: {num_stocks}")
    
    print(f"\nFactor CSVs:")
    if FACTOR_DIR.exists():
        factor_files = [f for f in os.listdir(FACTOR_DIR) if f.endswith('.csv') and 'ic_summary' not in f.lower()]
        print(f"  - Directory: {FACTOR_DIR}")
        print(f"  - Factor files: {len(factor_files)}")
    else:
        print(f"  - Not found at: {FACTOR_DIR}")
    
    print(f"\nNext Steps:")
    print(f"  1. Create config file for Subset 10")
    print(f"  2. Run model training")
    
    print("\n" + "="*80)
    print("✓ Preprocessing complete!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
