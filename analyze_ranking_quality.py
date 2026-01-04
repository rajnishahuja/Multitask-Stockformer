"""
Analyze ranking quality (IC, Rank IC) of model predictions.
This is what actually matters for trading, not just direction accuracy.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print('='*70)
print('RANKING QUALITY ANALYSIS: IC and Rank IC')
print('='*70)

# Load predictions and labels
output_dir = 'output/Multitask_output_2020-12-02_2023-08-02'
pred_file = f'{output_dir}/regression/regression_pred_with_index.csv'
label_file = f'{output_dir}/regression/regression_label_with_index.csv'

print(f'\n📂 Loading data from baseline model...')
pred_df = pd.read_csv(pred_file, index_col=0)
label_df = pd.read_csv(label_file, index_col=0)

print(f'  Predictions shape: {pred_df.shape}')
print(f'  Labels shape: {label_df.shape}')

# Calculate IC and Rank IC for each time step
ic_list = []
rank_ic_list = []
top10_accuracy = []
top20_accuracy = []

print(f'\n📊 Calculating metrics for each time step...')

for col in pred_df.columns:
    if col not in label_df.columns:
        continue
    
    pred = pred_df[col].values
    label = label_df[col].values
    
    # Remove NaN values
    mask = ~(np.isnan(pred) | np.isnan(label))
    pred_clean = pred[mask]
    label_clean = label[mask]
    
    if len(pred_clean) < 10:  # Need at least 10 stocks
        continue
    
    # IC: Pearson correlation between predictions and actual returns
    ic = np.corrcoef(pred_clean, label_clean)[0, 1]
    ic_list.append(ic)
    
    # Rank IC: Spearman correlation (rank-based)
    rank_ic, _ = stats.spearmanr(pred_clean, label_clean)
    rank_ic_list.append(rank_ic)
    
    # Top K accuracy: What % of top K predicted stocks actually went up?
    n_stocks = len(pred_clean)
    k10 = min(10, n_stocks // 10)  # Top 10 or 10%
    k20 = min(20, n_stocks // 5)   # Top 20 or 20%
    
    top10_indices = np.argsort(pred_clean)[-k10:]
    top20_indices = np.argsort(pred_clean)[-k20:]
    
    # Calculate accuracy: % that had positive returns
    top10_acc = (label_clean[top10_indices] > 0).mean()
    top20_acc = (label_clean[top20_indices] > 0).mean()
    
    top10_accuracy.append(top10_acc)
    top20_accuracy.append(top20_acc)

ic_array = np.array(ic_list)
rank_ic_array = np.array(rank_ic_list)
top10_array = np.array(top10_accuracy)
top20_array = np.array(top20_accuracy)

print('\n'+'='*70)
print('RESULTS: Information Coefficient (IC)')
print('='*70)
print(f'  Mean IC:      {np.mean(ic_array):.4f}')
print(f'  Median IC:    {np.median(ic_array):.4f}')
print(f'  Std IC:       {np.std(ic_array):.4f}')
print(f'  IC > 0:       {(ic_array > 0).sum()}/{len(ic_array)} ({(ic_array > 0).mean()*100:.1f}%)')
print(f'  |IC| > 0.03:  {(np.abs(ic_array) > 0.03).sum()}/{len(ic_array)} ({(np.abs(ic_array) > 0.03).mean()*100:.1f}%)')
print(f'  Min IC:       {np.min(ic_array):.4f}')
print(f'  Max IC:       {np.max(ic_array):.4f}')

print('\n' + '='*70)
print('RESULTS: Rank Information Coefficient (Rank IC)')
print('='*70)
print(f'  Mean Rank IC:      {np.mean(rank_ic_array):.4f}')
print(f'  Median Rank IC:    {np.median(rank_ic_array):.4f}')
print(f'  Std Rank IC:       {np.std(rank_ic_array):.4f}')
print(f'  Rank IC > 0:       {(rank_ic_array > 0).sum()}/{len(rank_ic_array)} ({(rank_ic_array > 0).mean()*100:.1f}%)')
print(f'  |Rank IC| > 0.03:  {(np.abs(rank_ic_array) > 0.03).sum()}/{len(rank_ic_array)} ({(np.abs(rank_ic_array) > 0.03).mean()*100:.1f}%)')
print(f'  Min Rank IC:       {np.min(rank_ic_array):.4f}')
print(f'  Max Rank IC:       {np.max(rank_ic_array):.4f}')

print('\n' + '='*70)
print('RESULTS: Top K Stock Selection Accuracy')
print('='*70)
print(f'  Mean Top 10 Accuracy:  {np.mean(top10_array)*100:.2f}%')
print(f'  Median Top 10 Acc:     {np.median(top10_array)*100:.2f}%')
print(f'  Top 10 Acc > 50%:      {(top10_array > 0.5).sum()}/{len(top10_array)} ({(top10_array > 0.5).mean()*100:.1f}%)')
print(f'  Top 10 Acc > 60%:      {(top10_array > 0.6).sum()}/{len(top10_array)} ({(top10_array > 0.6).mean()*100:.1f}%)')
print()
print(f'  Mean Top 20 Accuracy:  {np.mean(top20_array)*100:.2f}%')
print(f'  Median Top 20 Acc:     {np.median(top20_array)*100:.2f}%')
print(f'  Top 20 Acc > 50%:      {(top20_array > 0.5).sum()}/{len(top20_array)} ({(top20_array > 0.5).mean()*100:.1f}%)')

print('\n' + '='*70)
print('INTERPRETATION')
print('='*70)

mean_ic = np.mean(ic_array)
mean_rank_ic = np.mean(rank_ic_array)
mean_top10 = np.mean(top10_array)

print('\n📏 INDUSTRY BENCHMARKS:')
print('  IC Interpretation:')
print('    |IC| < 0.03:  Weak signal (not useful for trading)')
print('    |IC| 0.03-0.05: Moderate signal (usable)')
print('    |IC| 0.05-0.10: Strong signal (good for trading)')
print('    |IC| > 0.10:  Very strong signal (rare, excellent)')
print()
print('  Rank IC typically 0.02-0.05 lower than IC')
print('  Positive IC/Rank IC = correct ranking direction')
print()
print('  Top K Accuracy:')
print('    50%: Random (no skill)')
print('    55-60%: Slight edge (potentially profitable)')
print('    60-70%: Good edge (likely profitable)')
print('    >70%: Excellent (very profitable if sustained)')

print('\n📊 OUR MODEL ASSESSMENT:')

if mean_ic > 0.05:
    ic_assessment = '🟢 STRONG - Good ranking quality!'
elif mean_ic > 0.03:
    ic_assessment = '🟡 MODERATE - Usable for trading'
elif mean_ic > 0:
    ic_assessment = '🟠 WEAK - Limited trading value'
else:
    ic_assessment = '🔴 POOR - Not useful for trading'

print(f'  IC:       {mean_ic:.4f} - {ic_assessment}')

if mean_rank_ic > 0.04:
    rank_ic_assessment = '🟢 STRONG - Robust ranking!'
elif mean_rank_ic > 0.02:
    rank_ic_assessment = '🟡 MODERATE - Acceptable'
elif mean_rank_ic > 0:
    rank_ic_assessment = '🟠 WEAK - Marginal utility'
else:
    rank_ic_assessment = '🔴 POOR - No ranking skill'

print(f'  Rank IC:  {mean_rank_ic:.4f} - {rank_ic_assessment}')

if mean_top10 > 0.60:
    top10_assessment = '🟢 GOOD - Strong stock selection!'
elif mean_top10 > 0.55:
    top10_assessment = '🟡 MODERATE - Some edge'
elif mean_top10 > 0.50:
    top10_assessment = '🟠 WEAK - Minimal edge'
else:
    top10_assessment = '🔴 POOR - Worse than random'

print(f'  Top 10:   {mean_top10*100:.1f}% - {top10_assessment}')

print('\n🎯 CONCLUSION:')
if mean_ic > 0.03 and mean_rank_ic > 0.02:
    print('  ✅ Model shows USABLE ranking quality')
    print('  ✅ Proceed to backtest - likely to be profitable')
    print('  📈 Next step: Implement TopK-Dropout strategy')
elif mean_ic > 0.01:
    print('  ⚠️  Model shows WEAK ranking quality')
    print('  ⚠️  May still work with ensemble (14 windows)')
    print('  📊 Run backtest to confirm viability')
else:
    print('  ❌ Model shows POOR ranking quality')
    print('  ❌ Consider: More factors, different architecture, or longer training')
    print('  🔄 But still worth testing backtest to confirm')

print('\n📝 COMPARISON WITH PAPER:')
print('  Paper uses 360 factors, we use 22 (6% of theirs)')
print('  Paper achieves ~52-54% direction accuracy (same as us!)')
print('  Paper likely has IC ~0.03-0.05 range (industry standard)')
print(f'  Our IC: {mean_ic:.4f} - {"COMPARABLE" if abs(mean_ic) > 0.02 else "LOWER"} to expected')

print('\n' + '='*70)

# Save detailed results
results_df = pd.DataFrame({
    'IC': ic_array,
    'Rank_IC': rank_ic_array,
    'Top10_Accuracy': top10_array,
    'Top20_Accuracy': top20_array
})

output_file = 'output/ranking_quality_analysis.csv'
results_df.to_csv(output_file, index=False)
print(f'\n💾 Detailed results saved to: {output_file}')
print('='*70)
