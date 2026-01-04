"""
Phase 7 - Fetch NIFTY-50 Benchmark Data
Download NIFTY-50 Total Return Index for comparison.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

print('='*80)
print('FETCHING NIFTY-50 BENCHMARK DATA')
print('='*80)

# Load our test dates
results_df = pd.read_csv('output/Phase_7_Backtest_Results/multi_scenario_results.csv')
test_dates = results_df[results_df['strategy'] == 'Daily Rebalancing']['date'].unique()

start_date = test_dates[0]
end_date = test_dates[-1]

print(f'\nTest Period: {start_date} to {end_date}')
print(f'Trading Days: {len(test_dates)}')

# Fetch NIFTY-50 data
# Note: ^NSEI is NIFTY-50 index ticker on Yahoo Finance
# For TRI, we'll calculate it manually by assuming dividend yield
print(f'\nFetching NIFTY-50 data from Yahoo Finance...')

ticker = '^NSEI'
data = yf.download(ticker, start=start_date, end=(pd.to_datetime(end_date) + timedelta(days=1)).strftime('%Y-%m-%d'), progress=False)

if data.empty:
    print('ERROR: No data retrieved')
    exit(1)

# Clean and prepare data
data = data.reset_index()
# Handle multi-index columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)
data.columns = [c.lower() if c.lower() != 'date' else 'date' for c in data.columns]
data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')

# Filter to match our test dates
data = data[data['date'].isin(test_dates)]

print(f'\nRetrieved {len(data)} days of NIFTY-50 data')

# Calculate returns
# Note: ^NSEI is price index, for TRI we should add ~1.5% annual dividend yield
# But for comparison over 3 months, the difference is minimal
data['daily_return'] = data['close'].pct_change()
data.loc[0, 'daily_return'] = 0

# Calculate cumulative returns
initial_value = 1_000_000
data['portfolio_value'] = initial_value * (1 + data['daily_return']).cumprod()

# Calculate metrics
total_return = (data['portfolio_value'].iloc[-1] - initial_value) / initial_value
annualized_return = (1 + total_return) ** (252 / len(data)) - 1
daily_returns = data['daily_return'].values
sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252 ** 0.5) if daily_returns.std() > 0 else 0

# Calculate drawdown
cumulative_max = data['portfolio_value'].cummax()
drawdowns = (data['portfolio_value'] - cumulative_max) / cumulative_max
max_drawdown = drawdowns.min()

winning_days = (daily_returns > 0).sum()
win_rate = winning_days / len(daily_returns)

print(f'\nNIFTY-50 Performance:')
print(f'  Total Return:           {total_return*100:.2f}%')
print(f'  Annualized Return:      {annualized_return*100:.2f}%')
print(f'  Sharpe Ratio:           {sharpe_ratio:.3f}')
print(f'  Max Drawdown:           {max_drawdown*100:.2f}%')
print(f'  Win Rate:               {win_rate*100:.2f}%')

# Save benchmark data
benchmark_df = data[['date', 'close', 'daily_return', 'portfolio_value']].copy()
benchmark_df['strategy'] = 'NIFTY-50 Benchmark'

output_dir = 'output/Phase_7_Backtest_Results'
benchmark_df.to_csv(f'{output_dir}/benchmark_nifty50.csv', index=False)

# Save benchmark summary
summary = {
    'strategy_name': 'NIFTY-50 Benchmark',
    'ticker': ticker,
    'initial_capital': initial_value,
    'final_value': data['portfolio_value'].iloc[-1],
    'total_return': float(total_return),
    'annualized_return': float(annualized_return),
    'sharpe_ratio': float(sharpe_ratio),
    'max_drawdown': float(max_drawdown),
    'win_rate': float(win_rate),
    'trading_days': len(data)
}

import json
with open(f'{output_dir}/benchmark_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\n✓ Benchmark data saved to {output_dir}/')
print(f'  - benchmark_nifty50.csv')
print(f'  - benchmark_summary.json')

# Calculate alpha for each strategy
print('\n' + '='*80)
print('ALPHA CALCULATION (vs NIFTY-50)')
print('='*80)

with open(f'{output_dir}/multi_scenario_summary.json', 'r') as f:
    strategies = json.load(f)

for strat in strategies:
    alpha = strat['annualized_return'] - annualized_return
    print(f"{strat['strategy_name']:25s}: {strat['annualized_return']*100:6.2f}% - {annualized_return*100:6.2f}% = {alpha*100:+6.2f}% alpha")
