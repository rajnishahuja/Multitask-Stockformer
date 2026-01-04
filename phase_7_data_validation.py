"""
Phase 7 - Step 1: Data Validation for Backtest
Validates all required data before running backtest strategy.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('PHASE 7 - DATA VALIDATION FOR BACKTEST')
print('='*80)

# Paths
PREDICTIONS_FILE = 'output/Multitask_output_NIFTY200_Alpha158_MVP/regression/regression_pred_last_step.csv'
LABELS_FILE = 'output/Multitask_output_NIFTY200_Alpha158_MVP/regression/regression_label_last_step.csv'
PRICE_DATA_DIR = 'data/NIFTY200/raw'
STOCK_INFO_FILE = 'data/NIFTY200/stock_info_with_dummies.csv'
CONFIG_FILE = 'config/Multitask_NIFTY200_Alpha158.conf'
OUTPUT_DIR = 'output/Phase_7_Backtest_Results'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)

validation_report = []

def log(message, level='INFO'):
    """Log validation messages"""
    print(f"[{level}] {message}")
    validation_report.append(f"[{level}] {message}")

# ============================================================================
# 1. LOAD STOCK CODES AND COMPUTE TEST DATES
# ============================================================================
log('\n' + '='*80)
log('1. LOADING STOCK CODES AND DATE RANGE')
log('='*80)

# Load stock info to get stock codes
try:
    stock_info_df = pd.read_csv(STOCK_INFO_FILE)
    stock_codes = stock_info_df['symbol'].tolist()
    log(f"✓ Loaded {len(stock_codes)} stock codes from stock info file")
    log(f"  First 5: {stock_codes[:5]}")
    log(f"  Last 5: {stock_codes[-5:]}")
except Exception as e:
    log(f"✗ Error loading stock info: {e}", 'ERROR')
    exit(1)

# Load config to get date range
try:
    import configparser
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    
    T1 = int(config['data']['T1'])  # Lookback window
    T2 = int(config['data']['T2'])  # Prediction horizon
    train_ratio = float(config['data']['train_ratio'])
    val_ratio = float(config['data']['val_ratio'])
    
    log(f"✓ Config loaded: T1={T1}, T2={T2}, train_ratio={train_ratio}, val_ratio={val_ratio}")
    
    # Load a sample price file to get date range
    sample_stock = stock_codes[0]
    sample_df = pd.read_csv(f'{PRICE_DATA_DIR}/{sample_stock}.csv')
    sample_df['Date'] = pd.to_datetime(sample_df['Date'])
    sample_df = sample_df.sort_values('Date')
    
    all_dates = sample_df['Date'].dt.strftime('%Y-%m-%d').tolist()
    total_days = len(all_dates)
    
    # Calculate split indices (matching training logic)
    train_size = int(total_days * train_ratio)
    val_size = int(total_days * val_ratio)
    test_size = total_days - train_size - val_size
    
    # Important: Predictions start from T1 (lookback window) into the dataset
    # So actual prediction count = total_days - T1
    # Get test dates (last test_size days)
    test_start_idx = train_size + val_size
    test_dates = all_dates[test_start_idx:]
    
    # But predictions are available for all days after T1
    # So we need to get dates starting from T1
    pred_dates = all_dates[T1:]  # Predictions available from day T1 onwards
    
    log(f"✓ Date range calculated:")
    log(f"  Total days: {total_days}")
    log(f"  Train days: {train_size}")
    log(f"  Val days: {val_size}")
    log(f"  Test days: {test_size}")
    log(f"  Prediction days (from T1={T1}): {len(pred_dates)}")
    log(f"  Test period: {test_dates[0]} to {test_dates[-1]}")
    
except Exception as e:
    log(f"✗ Error loading config/dates: {e}", 'ERROR')
    exit(1)

# ============================================================================
# 2. LOAD AND VALIDATE PREDICTIONS
# ============================================================================
log('\n' + '='*80)
log('2. VALIDATING MODEL PREDICTIONS')
log('='*80)

try:
    # Load raw CSV without headers
    pred_raw = np.loadtxt(PREDICTIONS_FILE, delimiter=',')
    log(f"✓ Loaded predictions: {pred_raw.shape}")
    log(f"  Stocks: {pred_raw.shape[0]}")
    log(f"  Time steps: {pred_raw.shape[1]}")
    
    # IMPORTANT: The prediction file contains model outputs but doesn't include date/stock metadata
    # We'll use the last test_size columns as our test predictions
    # This assumes the predictions are in chronological order
    
    # Extract test period (last 83 days based on config)
    if pred_raw.shape[1] < test_size:
        log(f"✗ Not enough predictions! Have {pred_raw.shape[1]}, need {test_size}", 'ERROR')
        exit(1)
    
    # Take last test_size columns as test predictions
    test_pred_raw = pred_raw[:, -test_size:]
    
    # Create DataFrame with proper index and test_dates
    pred_df = pd.DataFrame(test_pred_raw, 
                          index=stock_codes[:test_pred_raw.shape[0]], 
                          columns=test_dates[:test_pred_raw.shape[1]])
    
    log(f"  Extracted test predictions: {pred_df.shape[0]} stocks × {pred_df.shape[1]} days")
    log(f"  Stock codes: {pred_df.index[:3].tolist()}...")
    log(f"  Date range: {pred_df.columns[0]} to {pred_df.columns[-1]}")
    
    # Check for NaN
    nan_count = pred_df.isna().sum().sum()
    if nan_count > 0:
        log(f"⚠ Found {nan_count} NaN values in predictions", 'WARNING')
        log(f"  NaN stocks: {pred_df.isna().any(axis=1).sum()}")
        log(f"  NaN dates: {pred_df.isna().any(axis=0).sum()}")
    else:
        log("✓ No NaN values in predictions")
    
    # Prediction statistics
    pred_flat = pred_df.values.flatten()
    pred_flat_clean = pred_flat[~np.isnan(pred_flat)]
    log(f"\nPrediction Statistics:")
    log(f"  Mean:   {pred_flat_clean.mean():.6f}")
    log(f"  Std:    {pred_flat_clean.std():.6f}")
    log(f"  Min:    {pred_flat_clean.min():.6f}")
    log(f"  Max:    {pred_flat_clean.max():.6f}")
    log(f"  Range:  {pred_flat_clean.max() - pred_flat_clean.min():.6f}")
    
except Exception as e:
    log(f"✗ Error loading predictions: {e}", 'ERROR')
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 3. LOAD AND VALIDATE LABELS (ACTUAL RETURNS)
# ============================================================================
log('\n' + '='*80)
log('3. VALIDATING ACTUAL RETURNS (LABELS)')
log('='*80)

try:
    # Load raw CSV without headers
    label_raw = np.loadtxt(LABELS_FILE, delimiter=',')
    log(f"✓ Loaded labels: {label_raw.shape}")
    
    # Extract test period (last test_size columns)
    test_label_raw = label_raw[:, -test_size:]
    
    # Create DataFrame with proper index and columns
    label_df = pd.DataFrame(test_label_raw, 
                           index=stock_codes[:test_label_raw.shape[0]], 
                           columns=test_dates[:test_label_raw.shape[1]])
    
    # Verify shape match with predictions
    if pred_df.shape != label_df.shape:
        log(f"✗ Shape mismatch! Predictions: {pred_df.shape}, Labels: {label_df.shape}", 'ERROR')
        exit(1)
    log("✓ Shapes match")
    
    # Check for NaN
    nan_count = label_df.isna().sum().sum()
    if nan_count > 0:
        log(f"⚠ Found {nan_count} NaN values in labels", 'WARNING')
    else:
        log("✓ No NaN values in labels")
    
    # Label statistics
    label_flat = label_df.values.flatten()
    label_flat_clean = label_flat[~np.isnan(label_flat)]
    log(f"\nActual Return Statistics:")
    log(f"  Mean:   {label_flat_clean.mean():.6f}")
    log(f"  Std:    {label_flat_clean.std():.6f}")
    log(f"  Min:    {label_flat_clean.min():.6f}")
    log(f"  Max:    {label_flat_clean.max():.6f}")
    log(f"  Range:  {label_flat_clean.max() - label_flat_clean.min():.6f}")
    
except Exception as e:
    log(f"✗ Error loading labels: {e}", 'ERROR')
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# 4. VALIDATE STOCK PRICE DATA
# ============================================================================
log('\n' + '='*80)
log('4. VALIDATING STOCK PRICE DATA')
log('='*80)

# Get stock codes from predictions (actual stocks in test set)
test_stocks = pred_df.index.tolist()
log(f"Total stocks in predictions: {len(test_stocks)}")

# Check price data availability
price_files_found = 0
price_files_missing = []

for stock in test_stocks:
    price_file = f"{PRICE_DATA_DIR}/{stock}.csv"
    if os.path.exists(price_file):
        price_files_found += 1
    else:
        price_files_missing.append(stock)

log(f"✓ Price files found: {price_files_found}/{len(test_stocks)}")

if price_files_missing:
    log(f"⚠ Missing price data for {len(price_files_missing)} stocks:", 'WARNING')
    for stock in price_files_missing[:10]:  # Show first 10
        log(f"    {stock}")
    if len(price_files_missing) > 10:
        log(f"    ... and {len(price_files_missing)-10} more")

# Sample a few price files to validate structure
log("\nValidating price data structure (sampling 5 stocks)...")
sample_stocks = test_stocks[:5]

for stock in sample_stocks:
    price_file = f"{PRICE_DATA_DIR}/{stock}.csv"
    if not os.path.exists(price_file):
        log(f"  ✗ {stock}: File not found", 'ERROR')
        continue
    
    try:
        df = pd.read_csv(price_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check required columns (case-insensitive)
        df.columns = [c.lower() for c in df.columns]
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            log(f"  ✗ {stock}: Missing columns {missing_cols}", 'ERROR')
        else:
            log(f"  ✓ {stock}: {len(df)} rows, date range: {df['date'].min().date()} to {df['date'].max().date()}")
            
            # Check if test dates are covered
            df_dates = df['date'].dt.strftime('%Y-%m-%d').tolist()
            covered = sum(1 for d in test_dates if d in df_dates)
            log(f"    Coverage: {covered}/{len(test_dates)} test dates ({covered/len(test_dates)*100:.1f}%)")
            
    except Exception as e:
        log(f"  ✗ {stock}: Error reading - {e}", 'ERROR')

# ============================================================================
# 4. DATE ALIGNMENT CHECK
# ============================================================================
log('\n' + '='*80)
log('4. CHECKING DATE ALIGNMENT')
log('='*80)

log(f"Test period dates: {len(test_dates)}")
log(f"  First date: {test_dates[0]}")
log(f"  Last date:  {test_dates[-1]}")

# Check date format
try:
    pd.to_datetime(test_dates)
    log("✓ Date format valid (can convert to datetime)")
except Exception as e:
    log(f"✗ Invalid date format: {e}", 'ERROR')

# Check for gaps in dates (weekends/holidays expected)
date_objs = pd.to_datetime(test_dates)
date_diffs = np.diff(date_objs.values).astype('timedelta64[D]').astype(int)
gaps = np.where(date_diffs > 5)[0]  # More than 5 days = potential long holiday

if len(gaps) > 0:
    log(f"⚠ Found {len(gaps)} gaps >5 days (holidays/long weekends):", 'WARNING')
    for idx in gaps[:5]:  # Show first 5
        log(f"    {test_dates[idx]} -> {test_dates[idx+1]} ({date_diffs[idx]} days)")
else:
    log("✓ No major gaps in date sequence")

if len(gaps) > 0:
    log(f"⚠ Found {len(gaps)} potential gaps in dates:", 'WARNING')
    for idx in gaps[:5]:  # Show first 5
        log(f"    {test_dates[idx]} -> {test_dates[idx+1]} ({date_diffs[idx]} days)")
else:
    log("✓ No major gaps in date sequence")

# ============================================================================
# 5. BENCHMARK DATA CHECK
# ============================================================================
log('\n' + '='*80)
log('5. CHECKING BENCHMARK DATA AVAILABILITY')
log('='*80)

benchmark_files = [
    'data/NIFTY200/benchmark/NIFTY50_TRI.csv',
    'data/NIFTY200/benchmark/NIFTY50.csv',
]

benchmark_found = False
for bf in benchmark_files:
    if os.path.exists(bf):
        log(f"✓ Found benchmark: {bf}")
        try:
            bench_df = pd.read_csv(bf)
            log(f"  Rows: {len(bench_df)}")
            log(f"  Columns: {list(bench_df.columns)}")
            benchmark_found = True
            break
        except Exception as e:
            log(f"  ✗ Error reading: {e}", 'ERROR')

if not benchmark_found:
    log("⚠ No benchmark data found - will need to fetch or skip benchmark comparison", 'WARNING')
    log("  Expected locations:")
    for bf in benchmark_files:
        log(f"    {bf}")

# ============================================================================
# 6. DATA QUALITY SUMMARY
# ============================================================================
log('\n' + '='*80)
log('6. DATA QUALITY SUMMARY')
log('='*80)

# Calculate coverage
valid_predictions = (~pred_df.isna()).sum().sum()
total_predictions = pred_df.shape[0] * pred_df.shape[1]
coverage = valid_predictions / total_predictions * 100

log(f"\nData Completeness:")
log(f"  Predictions coverage: {coverage:.2f}%")
log(f"  Price data coverage:  {price_files_found/len(test_stocks)*100:.2f}%")

# Check IC from previous analysis
ic_file = 'output/ranking_quality_analysis.csv'
if os.path.exists(ic_file):
    ic_df = pd.read_csv(ic_file)
    log(f"\nRanking Quality (from previous analysis):")
    log(f"  Mean IC:      {ic_df['IC'].mean():.4f}")
    log(f"  Mean Rank IC: {ic_df['Rank_IC'].mean():.4f}")
    log(f"  IC > 0:       {(ic_df['IC'] > 0).sum()}/{len(ic_df)} ({(ic_df['IC'] > 0).mean()*100:.1f}%)")

# ============================================================================
# 7. READINESS ASSESSMENT
# ============================================================================
log('\n' + '='*80)
log('7. BACKTEST READINESS ASSESSMENT')
log('='*80)

issues = []
warnings = []

# Critical checks
if pred_df.shape[0] < 10:
    issues.append("Too few stocks for TopK=10 strategy")
if pred_df.shape[1] < 20:
    issues.append("Too few time steps for meaningful backtest")
if price_files_found < len(test_stocks) * 0.8:
    issues.append(f"Missing price data for {len(test_stocks) - price_files_found} stocks")

# Warning checks
if pred_flat_clean.std() < 0.001:
    warnings.append("Predictions have very low variance (may indicate conservative model)")
if coverage < 95:
    warnings.append(f"Predictions coverage is {coverage:.1f}% (some NaN values)")
if not benchmark_found:
    warnings.append("Benchmark data not found (comparison limited)")

# Print assessment
if len(issues) > 0:
    log("\n❌ CRITICAL ISSUES FOUND:")
    for issue in issues:
        log(f"  • {issue}", 'ERROR')
    log("\n⚠ Cannot proceed with backtest until issues are resolved")
    ready = False
else:
    log("\n✅ NO CRITICAL ISSUES")
    ready = True

if len(warnings) > 0:
    log("\n⚠ WARNINGS:")
    for warning in warnings:
        log(f"  • {warning}", 'WARNING')

if ready:
    log("\n" + "="*80)
    log("✅ DATA VALIDATION COMPLETE - READY FOR BACKTEST")
    log("="*80)
else:
    log("\n" + "="*80)
    log("❌ DATA VALIDATION FAILED - RESOLVE ISSUES BEFORE PROCEEDING")
    log("="*80)

# ============================================================================
# 8. SAVE VALIDATION REPORT
# ============================================================================
report_file = f'{OUTPUT_DIR}/data_validation_report.txt'
with open(report_file, 'w') as f:
    f.write('\n'.join(validation_report))

log(f"\n📄 Validation report saved: {report_file}")

# Save key metrics for next steps
metrics = {
    'num_stocks': len(test_stocks),
    'num_dates': len(test_dates),
    'first_date': test_dates[0],
    'last_date': test_dates[-1],
    'price_files_available': price_files_found,
    'prediction_coverage': coverage,
    'ready_for_backtest': ready
}

import json
with open(f'{OUTPUT_DIR}/validation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

log(f"📊 Validation metrics saved: {OUTPUT_DIR}/validation_metrics.json")

print('\n' + '='*80)
print('VALIDATION COMPLETE')
print('='*80)

if ready:
    exit(0)
else:
    exit(1)
