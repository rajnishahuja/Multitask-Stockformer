"""
Phase 7 - Step 3: Visualization
Generate performance charts and analysis plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print('='*80)
print('PHASE 7 - GENERATING VISUALIZATIONS')
print('='*80)

# Load results
results_df = pd.read_csv('output/Phase_7_Backtest_Results/strategy_performance.csv')
results_df['date'] = pd.to_datetime(results_df['date'])

INITIAL_CAPITAL = results_df['portfolio_value'].iloc[0]

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# ============================================================================
# 1. Cumulative Returns
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])
cumulative_returns = (results_df['portfolio_value'] / INITIAL_CAPITAL - 1) * 100
ax1.plot(results_df['date'], cumulative_returns, linewidth=2, label='TopK-Dropout Strategy', color='#2E86AB')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('Cumulative Returns Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Annotate final return
final_return = cumulative_returns.iloc[-1]
ax1.annotate(f'{final_return:.2f}%', 
             xy=(results_df['date'].iloc[-1], final_return),
             xytext=(10, 10), textcoords='offset points',
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

print('✓ Generated: Cumulative Returns chart')

# ============================================================================
# 2. Drawdown
# ============================================================================
ax2 = fig.add_subplot(gs[1, 0])
cumulative_max = results_df['portfolio_value'].cummax()
drawdown = (results_df['portfolio_value'] - cumulative_max) / cumulative_max * 100
ax2.fill_between(results_df['date'], drawdown, 0, color='red', alpha=0.3)
ax2.plot(results_df['date'], drawdown, color='darkred', linewidth=1.5)
ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Drawdown (%)', fontsize=11)
ax2.grid(True, alpha=0.3)

# Annotate max drawdown
max_dd_idx = drawdown.idxmin()
max_dd_value = drawdown.iloc[max_dd_idx]
max_dd_date = results_df['date'].iloc[max_dd_idx]
ax2.annotate(f'Max: {max_dd_value:.2f}%',
             xy=(max_dd_date, max_dd_value),
             xytext=(10, -20), textcoords='offset points',
             fontsize=9, color='darkred',
             arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

print('✓ Generated: Drawdown chart')

# ============================================================================
# 3. Daily Returns Distribution
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])
daily_returns_pct = results_df['daily_return'] * 100
ax3.hist(daily_returns_pct, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax3.axvline(x=daily_returns_pct.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {daily_returns_pct.mean():.3f}%')
ax3.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
ax3.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Daily Return (%)', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

print('✓ Generated: Daily Returns Distribution')

# ============================================================================
# 4. Rolling Sharpe Ratio (20-day window)
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])
rolling_window = 20
if len(results_df) >= rolling_window:
    rolling_mean = results_df['daily_return'].rolling(window=rolling_window).mean()
    rolling_std = results_df['daily_return'].rolling(window=rolling_window).std()
    rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(252)).fillna(0)
    
    ax4.plot(results_df['date'], rolling_sharpe, color='purple', linewidth=2)
    ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax4.set_title(f'Rolling Sharpe Ratio ({rolling_window}-day window)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Date', fontsize=11)
    ax4.set_ylabel('Sharpe Ratio', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'Insufficient data for rolling calculation', 
             ha='center', va='center', fontsize=12)
    ax4.set_title(f'Rolling Sharpe Ratio ({rolling_window}-day window)', fontsize=14, fontweight='bold')

print('✓ Generated: Rolling Sharpe Ratio')

# ============================================================================
# 5. Monthly Performance Heatmap
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])

# Group by month and calculate returns
results_df['year_month'] = results_df['date'].dt.to_period('M')
monthly_returns = results_df.groupby('year_month')['daily_return'].apply(lambda x: (1 + x).prod() - 1) * 100

if len(monthly_returns) > 0:
    # Create heatmap data
    months = [str(m) for m in monthly_returns.index]
    values = monthly_returns.values.reshape(1, -1)
    
    im = ax5.imshow(values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    ax5.set_xticks(range(len(months)))
    ax5.set_xticklabels(months, rotation=45, ha='right', fontsize=9)
    ax5.set_yticks([0])
    ax5.set_yticklabels(['Returns'], fontsize=10)
    ax5.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(months)):
        text = ax5.text(i, 0, f'{values[0, i]:.1f}%',
                       ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax5, orientation='horizontal', pad=0.1)
    cbar.set_label('Return (%)', fontsize=9)
else:
    ax5.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
    ax5.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')

print('✓ Generated: Monthly Returns Heatmap')

# Save figure
plt.suptitle('TopK-Dropout Strategy Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.savefig('output/Phase_7_Backtest_Results/plots/performance_analysis.png', dpi=300, bbox_inches='tight')
print(f'\n✓ Saved: performance_analysis.png')
plt.close()

# ============================================================================
# Additional Plot: Portfolio Value vs Time
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(results_df['date'], results_df['portfolio_value'] / 1_000_000, linewidth=2.5, color='#2E86AB')
ax.fill_between(results_df['date'], 
                results_df['portfolio_value'] / 1_000_000, 
                INITIAL_CAPITAL / 1_000_000,
                alpha=0.3, color='#2E86AB')
ax.axhline(y=INITIAL_CAPITAL / 1_000_000, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
ax.set_title('Portfolio Value Over Time', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Portfolio Value (₹ Millions)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/Phase_7_Backtest_Results/plots/portfolio_value.png', dpi=300, bbox_inches='tight')
print('✓ Saved: portfolio_value.png')
plt.close()

# ============================================================================
# Performance Summary Stats
# ============================================================================
print('\n' + '='*80)
print('VISUALIZATION SUMMARY')
print('='*80)
print('Generated plots:')
print('  1. Cumulative Returns')
print('  2. Drawdown Analysis')
print('  3. Daily Returns Distribution')
print('  4. Rolling Sharpe Ratio')
print('  5. Monthly Returns Heatmap')
print('  6. Portfolio Value Over Time')
print(f'\nAll plots saved to: output/Phase_7_Backtest_Results/plots/')
