#!/usr/bin/env python3
"""
Phase 8: Factor Filter Script

Apply IC ranking + correlation filtering to existing factors without recomputing.
Reads ic_summary.csv and factor CSVs, outputs new selected_factors.txt.

Usage:
    python phase8_apply_factor_filter.py \
        --alpha-dir ./data/NIFTY200_Subset10/Alpha158 \
        --top-pct 0.20 \
        --corr-threshold 0.80
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_factor_data(alpha_dir: str, factor_names: List[str]) -> pd.DataFrame:
    """Load factor CSV files and combine into single DataFrame."""
    factor_data = {}
    
    for factor_name in factor_names:
        factor_path = os.path.join(alpha_dir, f'{factor_name}.csv')
        if os.path.exists(factor_path):
            df = pd.read_csv(factor_path, index_col=0)
            # Flatten to 1D array (all dates × all stocks)
            factor_data[factor_name] = df.values.flatten()
        else:
            logger.warning(f"Factor file not found: {factor_path}")
    
    return pd.DataFrame(factor_data)


def filter_by_ic_rank_and_correlation(ic_summary: pd.DataFrame,
                                       factor_matrix: pd.DataFrame,
                                       top_pct: float = 0.20,
                                       corr_threshold: float = 0.80) -> Tuple[List[str], pd.DataFrame]:
    """
    Select factors using relative IC ranking + correlation filtering.
    
    Stage 1: Keep top N% factors by absolute IC value
    Stage 2: Remove correlated factors (keep higher IC one)
    """
    logger.info(f"Stage 1: Selecting top {top_pct*100:.0f}% factors by |IC|...")
    
    # Remove factors with NaN IC
    valid_ic = ic_summary[ic_summary['IC'].notna()].copy()
    valid_ic['abs_IC'] = valid_ic['IC'].abs()
    valid_ic = valid_ic.sort_values('abs_IC', ascending=False)
    
    # Keep top N%
    n_keep = max(1, int(len(valid_ic) * top_pct))
    top_factors = valid_ic.head(n_keep)['factor'].tolist()
    logger.info(f"  → Selected {len(top_factors)} factors from {len(valid_ic)} valid factors")
    
    # Print top factors
    logger.info("  Top 10 factors by |IC|:")
    for i, row in valid_ic.head(10).iterrows():
        logger.info(f"    {row['factor']:12s} IC = {row['IC']:+.4f}")
    
    if len(top_factors) <= 1:
        logger.info(f"Stage 2: Skipped (only {len(top_factors)} factor)")
        return top_factors, valid_ic
    
    # Stage 2: Correlation filtering
    logger.info(f"Stage 2: Removing correlated factors (ρ > {corr_threshold})...")
    
    # Get factors that exist in factor_matrix
    available_factors = [f for f in top_factors if f in factor_matrix.columns]
    
    if len(available_factors) < 2:
        return top_factors, valid_ic
    
    # Compute correlation matrix
    corr_matrix = factor_matrix[available_factors].corr()
    
    # Greedy removal of correlated factors
    ic_lookup = valid_ic.set_index('factor')['abs_IC'].to_dict()
    
    removed = set()
    for i, f1 in enumerate(available_factors):
        if f1 in removed:
            continue
        for f2 in available_factors[i+1:]:
            if f2 in removed:
                continue
            corr_val = abs(corr_matrix.loc[f1, f2])
            if corr_val > corr_threshold:
                # Remove the one with lower IC
                ic1 = ic_lookup.get(f1, 0)
                ic2 = ic_lookup.get(f2, 0)
                to_remove = f2 if ic1 >= ic2 else f1
                removed.add(to_remove)
                logger.info(f"  Removed {to_remove} (corr={corr_val:.3f} with {f1 if to_remove == f2 else f2})")
    
    selected = [f for f in available_factors if f not in removed]
    logger.info(f"  → {len(selected)} factors after correlation filter (removed {len(removed)})")
    
    # Update IC summary with selection status
    ic_summary_updated = valid_ic.copy()
    ic_summary_updated['selected'] = ic_summary_updated['factor'].isin(selected)
    
    return selected, ic_summary_updated


def main():
    parser = argparse.ArgumentParser(description='Apply IC ranking + correlation filter to existing factors')
    parser.add_argument('--alpha-dir', type=str, default='./data/NIFTY200_Subset10/Alpha158',
                        help='Directory containing factor CSVs and ic_summary.csv')
    parser.add_argument('--top-pct', type=float, default=0.20,
                        help='Top percentage of factors to keep by |IC| (default: 0.20)')
    parser.add_argument('--corr-threshold', type=float, default=0.80,
                        help='Correlation threshold for filtering (default: 0.80)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as alpha-dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.alpha_dir
    
    print("=" * 80)
    print("Phase 8: Factor Filter (IC Ranking + Correlation)")
    print("=" * 80)
    print(f"Alpha directory: {args.alpha_dir}")
    print(f"Top percentage: {args.top_pct*100:.0f}%")
    print(f"Correlation threshold: {args.corr_threshold}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80 + "\n")
    
    # Step 1: Load existing IC summary
    ic_path = os.path.join(args.alpha_dir, 'ic_summary.csv')
    if not os.path.exists(ic_path):
        logger.error(f"IC summary not found: {ic_path}")
        logger.error("Run phase8_factor_engineering.py first to compute IC values.")
        return 1
    
    ic_summary = pd.read_csv(ic_path)
    logger.info(f"Loaded IC summary: {len(ic_summary)} factors")
    
    # Step 2: Load factor data for correlation calculation
    all_factors = ic_summary['factor'].tolist()
    logger.info(f"Loading {len(all_factors)} factor files for correlation analysis...")
    factor_matrix = load_factor_data(args.alpha_dir, all_factors)
    logger.info(f"Loaded factor matrix: {factor_matrix.shape}")
    
    # Step 3: Apply filtering
    selected_factors, updated_ic_summary = filter_by_ic_rank_and_correlation(
        ic_summary, factor_matrix,
        top_pct=args.top_pct,
        corr_threshold=args.corr_threshold
    )
    
    # Step 4: Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save updated IC summary
    ic_output_path = os.path.join(args.output_dir, 'ic_summary_filtered.csv')
    updated_ic_summary.to_csv(ic_output_path, index=False)
    logger.info(f"Saved updated IC summary: {ic_output_path}")
    
    # Save selected factors list
    selected_path = os.path.join(args.output_dir, 'selected_factors.txt')
    with open(selected_path, 'w') as f:
        for name in selected_factors:
            f.write(name + '\n')
    logger.info(f"Saved selected factors: {selected_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("FILTER COMPLETE")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  • Total factors: {len(ic_summary)}")
    print(f"  • After top {args.top_pct*100:.0f}% filter: {int(len(ic_summary) * args.top_pct)}")
    print(f"  • After correlation filter: {len(selected_factors)}")
    
    print(f"\nSelected factors ({len(selected_factors)}):")
    for i, factor in enumerate(selected_factors, 1):
        ic_val = updated_ic_summary[updated_ic_summary['factor'] == factor]['IC'].values[0]
        print(f"  {i:2d}. {factor:12s} IC = {ic_val:+.4f}")
    
    print(f"\nOutput files:")
    print(f"  • {selected_path}")
    print(f"  • {ic_output_path}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
