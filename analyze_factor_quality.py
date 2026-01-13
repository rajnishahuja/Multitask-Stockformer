#!/usr/bin/env python3
"""
Factor Quality Analysis: Compute IC in TEST period (not training).
This is a diagnostic check to see if selected factors are predictive in the test period.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

# Paths
DATA_DIR = "/home/ubuntu/rajnish/Multitask-Stockformer/data/NIFTY200_Subset10/Alpha158"
LABEL_FILE = "/home/ubuntu/rajnish/Multitask-Stockformer/data/NIFTY200_Subset10/dataset/label.csv"

# Test period dates (from config)
TEST_START = "2022-09-29"
TEST_END = "2023-02-03"

# Selected factors
selected_factors = [
    "KUP2", "WVMA20", "WVMA30", "CORD5", "IMAX10", "CORR5", "RSQR10",
    "IMAX60", "KUP", "KLEN", "IMAX5", "RESI10", "MA5", "WVMA10",
    "RSQR20", "RSQR30", "VSTD20", "CNTP5", "RANK10", "RESI60",
    "MIN60", "STD60", "VSTD5", "MAX5", "VSTD10", "RSV10"
]

def load_factor(factor_name):
    """Load a factor file and return as DataFrame."""
    path = os.path.join(DATA_DIR, f"{factor_name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

def load_labels():
    """Load the labels (next-day returns)."""
    df = pd.read_csv(LABEL_FILE, index_col=0, parse_dates=True)
    return df

def compute_ic(factor_df, label_df, start_date, end_date):
    """
    Compute Information Coefficient (Spearman correlation) 
    between factor and next-day returns in the specified period.
    """
    # Filter to date range
    factor_test = factor_df.loc[start_date:end_date]
    label_test = label_df.loc[start_date:end_date]
    
    # Align indices
    common_dates = factor_test.index.intersection(label_test.index)
    common_cols = factor_test.columns.intersection(label_test.columns)
    
    factor_aligned = factor_test.loc[common_dates, common_cols]
    label_aligned = label_test.loc[common_dates, common_cols]
    
    # Compute daily IC and average
    daily_ics = []
    for date in common_dates:
        f = factor_aligned.loc[date].values
        l = label_aligned.loc[date].values
        
        # Remove NaN
        mask = ~(np.isnan(f) | np.isnan(l))
        if mask.sum() < 10:
            continue
        
        ic, _ = spearmanr(f[mask], l[mask])
        if not np.isnan(ic):
            daily_ics.append(ic)
    
    if len(daily_ics) == 0:
        return np.nan, 0
    
    return np.mean(daily_ics), len(daily_ics)


print("=" * 60)
print("FACTOR QUALITY ANALYSIS - TEST PERIOD")
print(f"Test Period: {TEST_START} to {TEST_END}")
print("=" * 60)

# Load labels
labels = load_labels()
print(f"\nLabels loaded: {labels.shape}")

# Compute IC for each selected factor
results = []
print(f"\nComputing IC for {len(selected_factors)} selected factors...\n")

for factor in selected_factors:
    factor_df = load_factor(factor)
    if factor_df is None:
        print(f"  {factor}: FILE NOT FOUND")
        continue
    
    ic, n_days = compute_ic(factor_df, labels, TEST_START, TEST_END)
    results.append({
        'factor': factor,
        'test_IC': ic,
        'n_days': n_days
    })
    print(f"  {factor}: IC = {ic:.4f} ({n_days} days)")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df['abs_IC'] = results_df['test_IC'].abs()
results_df = results_df.sort_values('abs_IC', ascending=False)

print(f"\nTop 10 factors by absolute IC in test period:")
print(results_df.head(10).to_string(index=False))

avg_abs_ic = results_df['abs_IC'].mean()
print(f"\nAverage absolute IC in test period: {avg_abs_ic:.4f}")

# Compare with training IC
print("\n" + "=" * 60)
print("COMPARISON: Training IC vs Test IC")
print("=" * 60)

# Load training IC
train_ic = pd.read_csv(os.path.join(DATA_DIR, "ic_summary.csv"))
train_ic = train_ic[train_ic['factor'].isin(selected_factors)]

# Merge
comparison = results_df.merge(train_ic[['factor', 'IC']], on='factor', how='left')
comparison = comparison.rename(columns={'IC': 'train_IC'})
comparison['IC_change'] = comparison['test_IC'] - comparison['train_IC']

print("\nFactor | Train IC | Test IC | Change")
print("-" * 50)
for _, row in comparison.iterrows():
    sign = "↑" if row['IC_change'] > 0 else "↓"
    print(f"{row['factor']:10} | {row['train_IC']:7.4f} | {row['test_IC']:7.4f} | {sign} {abs(row['IC_change']):.4f}")

# Correlation analysis
if len(comparison.dropna()) > 5:
    train_test_corr = comparison['train_IC'].corr(comparison['test_IC'])
    print(f"\nCorrelation between Train IC and Test IC: {train_test_corr:.4f}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
if avg_abs_ic < 0.01:
    print("⚠️  Average IC < 0.01: Factors have VERY WEAK predictive power in test period")
    print("   Recommendation: Try 360 factors or different factor selection")
elif avg_abs_ic < 0.02:
    print("⚠️  Average IC < 0.02: Factors have WEAK predictive power")
    print("   May explain why model struggles to rank stocks")
else:
    print("✅ Average IC >= 0.02: Factors have REASONABLE predictive power")
    print("   Model architecture or training may be the issue")

# Save results
output_path = os.path.join(DATA_DIR, "test_period_ic.csv")
comparison.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
