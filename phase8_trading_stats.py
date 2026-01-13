#!/usr/bin/env python3
"""
Phase 8 Trading Statistics Analyzer
Calculates detailed trading metrics from the backtest results.
"""

import pandas as pd
import numpy as np
import os

# Load the backtest metrics
METRICS_FILE = 'output/Phase_8_Backtest_Results/phase8_backtest_metrics.csv'
df = pd.read_csv(METRICS_FILE)

print("=" * 80)
print("PHASE 8 DETAILED TRADING STATISTICS")
print("=" * 80)

# Basic metrics
initial_capital = 1_000_000
final_value = df['value'].iloc[-1]
total_return = (final_value - initial_capital) / initial_capital
trading_days = len(df)

print(f"\n1. SUMMARY")
print("-" * 40)
print(f"Period: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
print(f"Trading Days: {trading_days}")
print(f"Initial Capital: ₹{initial_capital:,.0f}")
print(f"Final Value: ₹{final_value:,.2f}")
print(f"Total Return: {total_return*100:.2f}%")

# Daily return statistics
daily_returns = df['return'].values[1:]  # Skip first day (0 return)
positive_days = daily_returns[daily_returns > 0]
negative_days = daily_returns[daily_returns < 0]

print(f"\n2. DAILY RETURN ANALYSIS")
print("-" * 40)
print(f"Total Days: {len(daily_returns)}")
print(f"Winning Days: {len(positive_days)} ({len(positive_days)/len(daily_returns)*100:.1f}%)")
print(f"Losing Days: {len(negative_days)} ({len(negative_days)/len(daily_returns)*100:.1f}%)")
print(f"Breakeven Days: {len(daily_returns) - len(positive_days) - len(negative_days)}")

print(f"\n3. WIN/LOSS METRICS")
print("-" * 40)
avg_win = np.mean(positive_days) * 100 if len(positive_days) > 0 else 0
avg_loss = np.mean(negative_days) * 100 if len(negative_days) > 0 else 0
max_win = np.max(positive_days) * 100 if len(positive_days) > 0 else 0
max_loss = np.min(negative_days) * 100 if len(negative_days) > 0 else 0

print(f"Average Daily Win: +{avg_win:.3f}%")
print(f"Average Daily Loss: {avg_loss:.3f}%")
print(f"Max Daily Win: +{max_win:.3f}%")
print(f"Max Daily Loss: {max_loss:.3f}%")

# Win/Loss Ratio
if avg_loss != 0:
    win_loss_ratio = abs(avg_win / avg_loss)
    print(f"Win/Loss Ratio: {win_loss_ratio:.2f}")
else:
    print("Win/Loss Ratio: N/A")

# Profit Factor
total_gains = np.sum(positive_days) if len(positive_days) > 0 else 0
total_losses = abs(np.sum(negative_days)) if len(negative_days) > 0 else 0
if total_losses > 0:
    profit_factor = total_gains / total_losses
    print(f"Profit Factor: {profit_factor:.2f}")
else:
    print("Profit Factor: N/A")

print(f"\n4. RISK METRICS")
print("-" * 40)
# Max Drawdown
cumulative_max = df['value'].cummax()
drawdown = (df['value'] - cumulative_max) / cumulative_max
max_dd = drawdown.min()
max_dd_date = df.loc[drawdown.idxmin(), 'date']
print(f"Max Drawdown: {max_dd*100:.2f}%")
print(f"Max Drawdown Date: {max_dd_date}")

# Sharpe Ratio (annualized)
sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
print(f"Sharpe Ratio: {sharpe:.3f}")

# Sortino Ratio (downside deviation)
downside_returns = daily_returns[daily_returns < 0]
downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
sortino = np.mean(daily_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
print(f"Sortino Ratio: {sortino:.3f}")

# Calmar Ratio
ann_return = (1 + total_return) ** (252/trading_days) - 1
calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
print(f"Calmar Ratio: {calmar:.3f}")

print(f"\n5. BENCHMARK COMPARISON")
print("-" * 40)
# NIFTY-50 approximate return for Sep-Nov 2022
# Sep 1: ~17542, Nov 30: ~18618
nifty_return = (18618 - 17542) / 17542
print(f"NIFTY-50 Return (approx): +{nifty_return*100:.2f}%")
print(f"Strategy Return: {total_return*100:.2f}%")
alpha = total_return - nifty_return
print(f"Alpha (Strategy - Benchmark): {alpha*100:.2f}%")

print(f"\n6. CONSECUTIVE STREAKS")
print("-" * 40)
# Calculate consecutive wins/losses
signs = np.sign(daily_returns)
current_streak = 0
max_win_streak = 0
max_loss_streak = 0
current_sign = 0

for s in signs:
    if s == current_sign:
        current_streak += 1
    else:
        if current_sign > 0:
            max_win_streak = max(max_win_streak, current_streak)
        elif current_sign < 0:
            max_loss_streak = max(max_loss_streak, current_streak)
        current_streak = 1
        current_sign = s

# Final streak
if current_sign > 0:
    max_win_streak = max(max_win_streak, current_streak)
elif current_sign < 0:
    max_loss_streak = max(max_loss_streak, current_streak)

print(f"Max Consecutive Winning Days: {max_win_streak}")
print(f"Max Consecutive Losing Days: {max_loss_streak}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
