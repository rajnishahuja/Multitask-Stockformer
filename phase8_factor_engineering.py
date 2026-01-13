#!/usr/bin/env python3
"""
Phase 8: Alpha158 Factor Engineering Script

Generalized script for computing Alpha158 factors for any dataset/period.
Supports:
- Alpha158 factor computation (158 factors)
- Size + Sector neutralization (optional)
- IC-based factor filtering
- Standardization (Z-score)
- Label generation (daily returns)

Usage:
    python phase8_factor_engineering.py --data-dir <path> --output-dir <path>
    
Example:
    python phase8_factor_engineering.py \
        --data-dir ./data/NIFTY200_Subset10/raw \
        --output-dir ./data/NIFTY200_Subset10
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
import logging

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_IC_THRESHOLD = 0.02
DEFAULT_WINDOWS = [5, 10, 20, 30, 60]

# =============================================================================
# Logging Setup - with unbuffered output
# =============================================================================
import sys

# Force unbuffered stdout
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

# =============================================================================
# Alpha158 Factor Computation
# =============================================================================
def compute_alpha158_factors(ohlcv_data: Dict[str, pd.DataFrame], 
                             windows: List[int] = None) -> pd.DataFrame:
    """
    Compute all 158 Alpha factors for given OHLCV data.
    
    Args:
        ohlcv_data: Dict mapping symbol -> DataFrame with columns [Date, Open, High, Low, Close, Volume]
        windows: Rolling window sizes (default: [5, 10, 20, 30, 60])
    
    Returns:
        DataFrame with MultiIndex (date, symbol) and 158 feature columns
    """
    if windows is None:
        windows = DEFAULT_WINDOWS
    
    all_factors = []
    total_symbols = len(ohlcv_data)
    
    for idx, (symbol, df) in enumerate(ohlcv_data.items(), 1):
        if idx % 10 == 0 or idx == 1:
            logger.info(f"  Computing factors for symbol {idx}/{total_symbols}...")
        
        df = df.copy().sort_values('Date')
        df = df.set_index('Date')
        
        # Extract price and volume series
        open_price = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
        
        factors = {}
        
        # ========== KBAR Features (9 features) ==========
        factors['KMID'] = (close - open_price) / open_price
        factors['KLEN'] = (high - low) / open_price
        factors['KMID2'] = (close - open_price) / (high - low + 1e-12)
        factors['KUP'] = (high - np.maximum(open_price, close)) / open_price
        factors['KUP2'] = (high - np.maximum(open_price, close)) / (high - low + 1e-12)
        factors['KLOW'] = (np.minimum(open_price, close) - low) / open_price
        factors['KLOW2'] = (np.minimum(open_price, close) - low) / (high - low + 1e-12)
        factors['KSFT'] = (2 * close - high - low) / open_price
        factors['KSFT2'] = (2 * close - high - low) / (high - low + 1e-12)
        
        # ========== Price Features (4 features) ==========
        factors['OPEN0'] = open_price / close
        factors['HIGH0'] = high / close
        factors['LOW0'] = low / close
        vwap = (high + low + close) / 3
        factors['VWAP0'] = vwap / close
        
        # ========== Rolling Features (145 features) ==========
        
        # ROC - Rate of Change (5 features)
        for d in windows:
            factors[f'ROC{d}'] = close.shift(d) / close
        
        # MA - Moving Average (5 features)
        for d in windows:
            factors[f'MA{d}'] = close.rolling(d).mean() / close
        
        # STD - Standard Deviation (5 features)
        for d in windows:
            factors[f'STD{d}'] = close.rolling(d).std() / close
        
        # BETA - Slope/Trend (5 features)
        for d in windows:
            def slope(series):
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                y = series.values
                valid = ~np.isnan(y)
                if valid.sum() < 2:
                    return np.nan
                return np.polyfit(x[valid], y[valid], 1)[0]
            factors[f'BETA{d}'] = close.rolling(d).apply(slope, raw=False) / close
        
        # RSQR - R-squared (5 features)
        for d in windows:
            def rsquare(series):
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                y = series.values
                valid = ~np.isnan(y)
                if valid.sum() < 2:
                    return np.nan
                y_valid = y[valid]
                x_valid = x[valid]
                slope, intercept = np.polyfit(x_valid, y_valid, 1)
                y_pred = slope * x_valid + intercept
                ss_res = np.sum((y_valid - y_pred) ** 2)
                ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                return 1 - (ss_res / (ss_tot + 1e-12))
            factors[f'RSQR{d}'] = close.rolling(d).apply(rsquare, raw=False)
        
        # RESI - Residual (5 features)
        for d in windows:
            def residual(series):
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                y = series.values
                valid = ~np.isnan(y)
                if valid.sum() < 2:
                    return np.nan
                y_valid = y[valid]
                x_valid = x[valid]
                slope, intercept = np.polyfit(x_valid, y_valid, 1)
                y_pred = slope * x_valid[-1] + intercept
                return y_valid[-1] - y_pred
            factors[f'RESI{d}'] = close.rolling(d).apply(residual, raw=False) / close
        
        # MAX - Maximum High (5 features)
        for d in windows:
            factors[f'MAX{d}'] = high.rolling(d).max() / close
        
        # MIN - Minimum Low (5 features)
        for d in windows:
            factors[f'MIN{d}'] = low.rolling(d).min() / close
        
        # QTLU - 80% Quantile (5 features)
        for d in windows:
            factors[f'QTLU{d}'] = close.rolling(d).quantile(0.8) / close
        
        # QTLD - 20% Quantile (5 features)
        for d in windows:
            factors[f'QTLD{d}'] = close.rolling(d).quantile(0.2) / close
        
        # RANK - Percentile Rank (5 features)
        for d in windows:
            factors[f'RANK{d}'] = close.rolling(d).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        
        # RSV - Relative Strength Value (5 features)
        for d in windows:
            min_low = low.rolling(d).min()
            max_high = high.rolling(d).max()
            factors[f'RSV{d}'] = (close - min_low) / (max_high - min_low + 1e-12)
        
        # IMAX - Days since maximum (5 features)
        for d in windows:
            factors[f'IMAX{d}'] = high.rolling(d).apply(
                lambda x: (len(x) - 1 - x.argmax()) / d, raw=False)
        
        # IMIN - Days since minimum (5 features)
        for d in windows:
            factors[f'IMIN{d}'] = low.rolling(d).apply(
                lambda x: (len(x) - 1 - x.argmin()) / d, raw=False)
        
        # IMXD - IMAX - IMIN difference (5 features)
        for d in windows:
            idx_max = high.rolling(d).apply(lambda x: len(x) - 1 - x.argmax(), raw=False)
            idx_min = low.rolling(d).apply(lambda x: len(x) - 1 - x.argmin(), raw=False)
            factors[f'IMXD{d}'] = (idx_max - idx_min) / d
        
        # CORR - Correlation between close and log(volume) (5 features)
        for d in windows:
            log_vol = np.log(volume + 1)
            factors[f'CORR{d}'] = close.rolling(d).corr(log_vol)
        
        # CORD - Correlation between returns and volume changes (5 features)
        for d in windows:
            returns = close / close.shift(1)
            vol_change = np.log(volume / volume.shift(1) + 1)
            factors[f'CORD{d}'] = returns.rolling(d).corr(vol_change)
        
        # CNTP - Count of positive days (5 features)
        for d in windows:
            pos = (close > close.shift(1)).astype(float)
            factors[f'CNTP{d}'] = pos.rolling(d).mean()
        
        # CNTN - Count of negative days (5 features)
        for d in windows:
            neg = (close < close.shift(1)).astype(float)
            factors[f'CNTN{d}'] = neg.rolling(d).mean()
        
        # CNTD - Difference between positive and negative days (5 features)
        for d in windows:
            pos = (close > close.shift(1)).astype(float)
            neg = (close < close.shift(1)).astype(float)
            factors[f'CNTD{d}'] = pos.rolling(d).mean() - neg.rolling(d).mean()
        
        # SUMP - Sum of gains ratio (5 features)
        for d in windows:
            gain = np.maximum(close - close.shift(1), 0)
            abs_change = np.abs(close - close.shift(1))
            factors[f'SUMP{d}'] = gain.rolling(d).sum() / (abs_change.rolling(d).sum() + 1e-12)
        
        # SUMN - Sum of losses ratio (5 features)
        for d in windows:
            loss = np.maximum(close.shift(1) - close, 0)
            abs_change = np.abs(close - close.shift(1))
            factors[f'SUMN{d}'] = loss.rolling(d).sum() / (abs_change.rolling(d).sum() + 1e-12)
        
        # SUMD - Difference between gains and losses (5 features)
        for d in windows:
            gain = np.maximum(close - close.shift(1), 0)
            loss = np.maximum(close.shift(1) - close, 0)
            abs_change = np.abs(close - close.shift(1))
            factors[f'SUMD{d}'] = (gain.rolling(d).sum() - loss.rolling(d).sum()) / (abs_change.rolling(d).sum() + 1e-12)
        
        # VMA - Volume Moving Average (5 features)
        for d in windows:
            factors[f'VMA{d}'] = volume.rolling(d).mean() / (volume + 1e-12)
        
        # VSTD - Volume Standard Deviation (5 features)
        for d in windows:
            factors[f'VSTD{d}'] = volume.rolling(d).std() / (volume + 1e-12)
        
        # WVMA - Weighted Volume Moving Average (5 features)
        for d in windows:
            price_change = np.abs(close / close.shift(1) - 1) * volume
            factors[f'WVMA{d}'] = price_change.rolling(d).std() / (price_change.rolling(d).mean() + 1e-12)
        
        # VSUMP - Volume sum of gains ratio (5 features)
        for d in windows:
            vol_gain = np.maximum(volume - volume.shift(1), 0)
            abs_vol_change = np.abs(volume - volume.shift(1))
            factors[f'VSUMP{d}'] = vol_gain.rolling(d).sum() / (abs_vol_change.rolling(d).sum() + 1e-12)
        
        # VSUMN - Volume sum of losses ratio (5 features)
        for d in windows:
            vol_loss = np.maximum(volume.shift(1) - volume, 0)
            abs_vol_change = np.abs(volume - volume.shift(1))
            factors[f'VSUMN{d}'] = vol_loss.rolling(d).sum() / (abs_vol_change.rolling(d).sum() + 1e-12)
        
        # VSUMD - Volume difference between gains and losses (5 features)
        for d in windows:
            vol_gain = np.maximum(volume - volume.shift(1), 0)
            vol_loss = np.maximum(volume.shift(1) - volume, 0)
            abs_vol_change = np.abs(volume - volume.shift(1))
            factors[f'VSUMD{d}'] = (vol_gain.rolling(d).sum() - vol_loss.rolling(d).sum()) / (abs_vol_change.rolling(d).sum() + 1e-12)
        
        # Create DataFrame for this symbol
        factor_df = pd.DataFrame(factors)
        factor_df['symbol'] = symbol
        factor_df['date'] = factor_df.index
        all_factors.append(factor_df)
    
    # Combine all symbols
    result = pd.concat(all_factors, ignore_index=True)
    result = result.set_index(['date', 'symbol'])
    result = result.sort_index()
    
    return result


# =============================================================================
# Label Generation
# =============================================================================
def compute_labels(ohlcv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute daily returns (labels) for all stocks.
    
    Returns:
        DataFrame with index=Date, columns=symbols, values=daily returns
    """
    returns_dict = {}
    
    for symbol, df in ohlcv_data.items():
        df = df.copy().sort_values('Date')
        df = df.set_index('Date')
        returns_dict[symbol] = df['Close'].pct_change()
    
    label_df = pd.DataFrame(returns_dict)
    label_df = label_df.sort_index()
    
    return label_df


# =============================================================================
# Size Proxy Computation
# =============================================================================
def compute_size_proxy(ohlcv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute size proxy: log(close * 60d rolling mean volume)
    
    Returns:
        DataFrame with index=Date, columns=symbols, values=size proxy
    """
    size_dict = {}
    
    for symbol, df in ohlcv_data.items():
        df = df.copy().sort_values('Date')
        df = df.set_index('Date')
        vol_ma = df['Volume'].rolling(60, min_periods=1).mean()
        size_dict[symbol] = np.log(df['Close'] * vol_ma + 1)
    
    size_df = pd.DataFrame(size_dict)
    size_df = size_df.sort_index()
    
    return size_df


# =============================================================================
# IC Computation and Filtering
# =============================================================================
def compute_ic_and_filter(factors_df: pd.DataFrame, 
                          label_df: pd.DataFrame,
                          ic_threshold: float = 0.02) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Information Coefficient (IC) for each factor and filter.
    
    Returns:
        selected_factors: DataFrame of factors passing IC threshold
        ic_summary: DataFrame with IC values for all factors
    """
    ic_results = []
    selected_factors = {}
    
    feature_names = factors_df.columns.tolist()
    
    for i, feature_name in enumerate(feature_names, 1):
        if i % 10 == 0 or i == 1:
            logger.info(f"  Computing IC for factor {i}/{len(feature_names)}...")
        
        # Extract factor values
        factor_vals = factors_df[feature_name].unstack(level=1)
        
        # Align dates with labels
        common_dates = factor_vals.index.intersection(label_df.index)
        
        if len(common_dates) == 0:
            ic_results.append({'factor': feature_name, 'IC': np.nan, 'selected': False})
            continue
        
        # Compute daily IC (correlation between factor and next-day returns)
        ic_vals = []
        for date_idx, date in enumerate(common_dates[:-1]):
            next_date = common_dates[date_idx + 1]
            
            # Factor values at date, returns at next_date
            x = factor_vals.loc[date].dropna()
            y = label_df.loc[next_date].dropna()
            
            common_syms = x.index.intersection(y.index)
            if len(common_syms) < 10:
                continue
            
            x_aligned = x.loc[common_syms].values
            y_aligned = y.loc[common_syms].values
            
            mask = np.isfinite(x_aligned) & np.isfinite(y_aligned)
            if mask.sum() < 10:
                continue
            
            try:
                ic = np.corrcoef(x_aligned[mask], y_aligned[mask])[0, 1]
                if np.isfinite(ic):
                    ic_vals.append(ic)
            except:
                pass
        
        mean_ic = np.mean(ic_vals) if ic_vals else np.nan
        is_selected = not np.isnan(mean_ic) and abs(mean_ic) >= ic_threshold
        
        ic_results.append({
            'factor': feature_name,
            'IC': mean_ic,
            'selected': is_selected
        })
        
        if is_selected:
            selected_factors[feature_name] = factors_df[feature_name]
    
    ic_summary = pd.DataFrame(ic_results).sort_values('IC', key=abs, ascending=False)
    
    if selected_factors:
        selected_df = pd.concat(selected_factors, axis=1)
    else:
        selected_df = pd.DataFrame()
    
    return selected_df, ic_summary


# =============================================================================
# IC Ranking + Correlation Filtering (Phase 8 Improvement)
# =============================================================================
def filter_by_ic_rank_and_correlation(ic_summary: pd.DataFrame,
                                       factors_df: pd.DataFrame,
                                       top_pct: float = 0.20,
                                       corr_threshold: float = 0.80) -> Tuple[List[str], pd.DataFrame]:
    """
    Select factors using relative IC ranking + correlation filtering.
    
    Stage 1: Keep top N% factors by absolute IC value
    Stage 2: Remove correlated factors (keep higher IC one)
    
    Args:
        ic_summary: DataFrame with 'factor' and 'IC' columns
        factors_df: Original factors DataFrame (MultiIndex: date, symbol)
        top_pct: Top percentage to keep (default: 0.20 = top 20%)
        corr_threshold: Correlation threshold for filtering (default: 0.80)
    
    Returns:
        selected_names: List of selected factor names
        updated_ic_summary: IC summary with selection status
    """
    logger.info(f"  Stage 1: Selecting top {top_pct*100:.0f}% factors by |IC|...")
    
    # Remove factors with NaN IC
    valid_ic = ic_summary[ic_summary['IC'].notna()].copy()
    valid_ic['abs_IC'] = valid_ic['IC'].abs()
    valid_ic = valid_ic.sort_values('abs_IC', ascending=False)
    
    # Keep top N%
    n_keep = max(1, int(len(valid_ic) * top_pct))
    top_factors = valid_ic.head(n_keep)['factor'].tolist()
    logger.info(f"    → Selected {len(top_factors)} factors from {len(valid_ic)} valid factors")
    
    if len(top_factors) <= 1:
        logger.info(f"  Stage 2: Skipped (only {len(top_factors)} factor)")
        return top_factors, valid_ic
    
    # Stage 2: Correlation filtering
    logger.info(f"  Stage 2: Removing correlated factors (ρ > {corr_threshold})...")
    
    # Extract factor values for correlation calculation
    # Unstack to get (dates × symbols) for each factor, then compute cross-factor correlation
    factor_data = {}
    for f in top_factors:
        if f in factors_df.columns:
            # Flatten to 1D for correlation (across all dates and symbols)
            vals = factors_df[f].values
            factor_data[f] = vals
    
    if len(factor_data) < 2:
        return top_factors, valid_ic
    
    # Create DataFrame for correlation
    factor_matrix = pd.DataFrame(factor_data)
    corr_matrix = factor_matrix.corr()
    
    # Greedy removal of correlated factors
    selected = list(top_factors)  # Start with all top factors
    ic_lookup = valid_ic.set_index('factor')['abs_IC'].to_dict()
    
    removed = set()
    for i, f1 in enumerate(top_factors):
        if f1 in removed:
            continue
        for f2 in top_factors[i+1:]:
            if f2 in removed:
                continue
            if f1 in corr_matrix.columns and f2 in corr_matrix.columns:
                corr_val = abs(corr_matrix.loc[f1, f2])
                if corr_val > corr_threshold:
                    # Remove the one with lower IC
                    ic1 = ic_lookup.get(f1, 0)
                    ic2 = ic_lookup.get(f2, 0)
                    to_remove = f2 if ic1 >= ic2 else f1
                    removed.add(to_remove)
                    logger.info(f"    Removed {to_remove} (corr={corr_val:.3f} with {f1 if to_remove == f2 else f2})")
    
    selected = [f for f in selected if f not in removed]
    logger.info(f"    → {len(selected)} factors after correlation filter (removed {len(removed)})")
    
    # Update IC summary with selection status
    ic_summary_updated = ic_summary.copy()
    ic_summary_updated['selected'] = ic_summary_updated['factor'].isin(selected)
    
    return selected, ic_summary_updated


# =============================================================================
# Standardization (Z-score)
# =============================================================================
def standardize_factors(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize factors using cross-sectional Z-score per date.
    """
    standardized = factors_df.copy()
    
    for col in standardized.columns:
        # Group by date and z-score across symbols
        def zscore(x):
            return (x - x.mean()) / (x.std() + 1e-12)
        
        standardized[col] = factors_df[col].groupby(level=0).transform(zscore)
    
    return standardized


# =============================================================================
# Main Execution
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Phase 8: Alpha158 Factor Engineering')
    parser.add_argument('--data-dir', type=str, default='./data/NIFTY200_Subset10/raw',
                        help='Directory containing raw OHLCV CSV files')
    parser.add_argument('--output-dir', type=str, default='./data/NIFTY200_Subset10',
                        help='Output directory for factors and labels')
    parser.add_argument('--ic-threshold', type=float, default=0.02,
                        help='IC threshold for factor selection (legacy, use --use-ic-rank instead)')
    parser.add_argument('--no-ic-filter', action='store_true',
                        help='Skip IC filtering, use all 158 factors')
    parser.add_argument('--use-ic-rank', action='store_true',
                        help='Use relative IC ranking instead of fixed threshold (recommended)')
    parser.add_argument('--top-pct', type=float, default=0.20,
                        help='Top percentage of factors to keep by |IC| (default: 0.20 = top 20%%)')
    parser.add_argument('--corr-threshold', type=float, default=0.80,
                        help='Correlation threshold for filtering redundant factors (default: 0.80)')
    parser.add_argument('--skip-neutralization', action='store_true',
                        help='Skip size/sector neutralization')
    
    args = parser.parse_args()
    
    # Setup output directories
    alpha_dir = os.path.join(args.output_dir, 'Alpha158')
    dataset_dir = os.path.join(args.output_dir, 'dataset')
    os.makedirs(alpha_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("="*80)
    print("Phase 8: Alpha158 Factor Engineering")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.no_ic_filter:
        print("Filter mode: NONE (using all 158 factors)")
    elif args.use_ic_rank:
        print(f"Filter mode: IC RANK (top {args.top_pct*100:.0f}% + corr > {args.corr_threshold})")
    else:
        print(f"Filter mode: IC THRESHOLD (|IC| >= {args.ic_threshold})")
    print("="*80 + "\n")
    
    # Step 1: Load OHLCV data
    logger.info("Step 1: Loading raw OHLCV data...")
    csv_files = glob.glob(os.path.join(args.data_dir, '*.csv'))
    
    ohlcv_data = {}
    for csv_file in csv_files:
        symbol = os.path.basename(csv_file).replace('.csv', '')
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        ohlcv_data[symbol] = df
    
    logger.info(f"✓ Loaded {len(ohlcv_data)} stocks")
    
    # Get date range
    all_dates = set()
    for df in ohlcv_data.values():
        all_dates.update(df['Date'].dt.date)
    sorted_dates = sorted(all_dates)
    logger.info(f"  Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
    logger.info(f"  Trading days: {len(sorted_dates)}")
    
    # Step 2: Compute labels
    logger.info("\nStep 2: Computing labels (daily returns)...")
    label_df = compute_labels(ohlcv_data)
    label_path = os.path.join(dataset_dir, 'label.csv')
    label_df.to_csv(label_path)
    logger.info(f"✓ Saved labels to {label_path}")
    logger.info(f"  Shape: {label_df.shape}")
    
    # Step 3: Compute Alpha158 factors
    logger.info("\nStep 3: Computing Alpha158 factors...")
    start_time = time.time()
    alpha158 = compute_alpha158_factors(ohlcv_data)
    elapsed = time.time() - start_time
    logger.info(f"✓ Computed {alpha158.shape[1]} factors in {elapsed:.1f}s")
    logger.info(f"  Shape: {alpha158.shape}")
    
    # Step 4: Compute size proxy
    logger.info("\nStep 4: Computing size proxy...")
    size_proxy = compute_size_proxy(ohlcv_data)
    size_path = os.path.join(args.output_dir, 'size_proxy.csv')
    size_proxy.to_csv(size_path)
    logger.info(f"✓ Saved size proxy to {size_path}")
    
    # Step 5: IC filtering (or skip if --no-ic-filter)
    if args.no_ic_filter:
        logger.info("\nStep 5: Skipping IC filter (using ALL 158 factors)...")
        selected_factors = alpha158
        n_selected = alpha158.shape[1]
        ic_summary = None
        ic_path = None
        selected_names = list(alpha158.columns)
    elif args.use_ic_rank:
        # NEW: Relative IC ranking + correlation filter
        logger.info(f"\nStep 5: IC ranking + correlation filter (top {args.top_pct*100:.0f}%, corr > {args.corr_threshold})...")
        
        # First compute IC for all factors (without filtering)
        _, ic_summary = compute_ic_and_filter(alpha158, label_df, ic_threshold=0.0)
        
        # Apply ranking + correlation filter
        selected_names, ic_summary = filter_by_ic_rank_and_correlation(
            ic_summary, alpha158, 
            top_pct=args.top_pct,
            corr_threshold=args.corr_threshold
        )
        n_selected = len(selected_names)
        logger.info(f"✓ {n_selected} factors selected via ranking + correlation filter")
        
        # Extract selected factors
        if n_selected > 0:
            selected_factors = alpha158[selected_names]
        else:
            selected_factors = pd.DataFrame()
        
        # Save IC summary
        ic_path = os.path.join(alpha_dir, 'ic_summary.csv')
        ic_summary.to_csv(ic_path, index=False)
        logger.info(f"  IC summary saved to {ic_path}")
    else:
        # Legacy: Fixed IC threshold
        logger.info(f"\nStep 5: Computing IC and filtering (threshold: {args.ic_threshold})...")
        selected_factors, ic_summary = compute_ic_and_filter(
            alpha158, label_df, args.ic_threshold
        )
        n_selected = ic_summary['selected'].sum()
        selected_names = list(selected_factors.columns) if n_selected > 0 else []
        logger.info(f"✓ {n_selected}/{len(ic_summary)} factors passed IC filter")
        
        # Save IC summary
        ic_path = os.path.join(alpha_dir, 'ic_summary.csv')
        ic_summary.to_csv(ic_path, index=False)
        logger.info(f"  IC summary saved to {ic_path}")

    # Step 6: Standardize factors
    logger.info(f"\nStep 6: Standardizing {n_selected} factors...")
    if n_selected > 0:
        standardized_factors = standardize_factors(selected_factors)
        logger.info(f"  Standardized {n_selected} factors")
        
        # Save individual factor CSVs  
        logger.info("  Saving factor files...")
        for i, col in enumerate(standardized_factors.columns, 1):
            if i % 20 == 0 or i == 1:
                logger.info(f"    Saving factor {i}/{n_selected}: {col}")
            factor_pivot = standardized_factors[col].unstack(level=1)
            factor_path = os.path.join(alpha_dir, f'{col}.csv')
            factor_pivot.to_csv(factor_path)
        
        # Save selected factor names (use consistent list from Step 5)
        selected_path = os.path.join(alpha_dir, 'selected_factors.txt')
        with open(selected_path, 'w') as f:
            for name in selected_names:
                f.write(name + '\n')
        
        logger.info(f"✓ Saved {len(selected_names)} standardized factor files")
    else:
        logger.warning("No factors passed IC filter!")
        selected_names = []
    
    # Summary
    print("\n" + "="*80)
    print("FACTOR ENGINEERING COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  ✓ Raw factors computed: 158")
    print(f"  ✓ Factors after IC filter: {len(selected_names)}")
    print(f"  ✓ Trading days: {len(sorted_dates)}")
    print(f"  ✓ Stocks: {len(ohlcv_data)}")
    
    print(f"\nOutput files:")
    print(f"  - Labels: {label_path}")
    print(f"  - Size proxy: {size_path}")
    print(f"  - Factor files: {alpha_dir}/")
    print(f"  - IC summary: {ic_path}")
    print(f"  - Selected factors: {alpha_dir}/selected_factors.txt")
    
    if ic_summary is not None and n_selected > 0:
        print(f"\\nTop 10 factors by |IC|:")
        top10 = ic_summary[ic_summary['selected']].head(10)
        for _, row in top10.iterrows():
            print(f"  {row['factor']:12s} IC = {row['IC']:7.4f}")
    
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
