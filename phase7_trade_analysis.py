#!/usr/bin/env python3
"""
Phase 7 Trade-Level Analysis
For comparison with Phase 8 results.
"""

import pandas as pd
import numpy as np
import os

# Configuration - Phase 7 Paths
PRED_FILE = 'output/Phase_7_Backtest_Results/test_predictions_corrected.csv'
PRICE_DIR = 'data/NIFTY200/raw'
K = 10
REBALANCE_FREQ = 5
TRANSACTION_FEE = 0.002

print("=" * 80)
print("PHASE 7 TRADE-LEVEL ANALYSIS (For Comparison)")
print("=" * 80)

# Load predictions
pred_df = pd.read_csv(PRED_FILE, index_col=0)
print(f"\nLoaded predictions: {pred_df.shape}")

# Load price data
price_data = {}
for stock in pred_df.index:
    file_path = os.path.join(PRICE_DIR, f"{stock}.csv")
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df.columns = [c.lower() for c in df.columns]
        df = df.set_index('date')
        price_data[stock] = df
    except:
        pass

print(f"Loaded price data for {len(price_data)} stocks")

# Track individual trades
all_trades = []
open_positions = {}

# Get valid dates
sample_stock = list(price_data.keys())[0]
dates = sorted([d for d in pred_df.columns if d in price_data[sample_stock].index])
print(f"Trading dates: {len(dates)} ({dates[0]} to {dates[-1]})")

def select_top_k(predictions, k=10):
    return predictions.nlargest(k).index.tolist()

# Simulate trading
holdings = {}
cash = 1_000_000

for day_idx, date in enumerate(dates):
    current_prices = {}
    for stock in pred_df.index:
        if stock in price_data and date in price_data[stock].index:
            current_prices[stock] = price_data[stock].loc[date, 'close']
    
    if day_idx % REBALANCE_FREQ == 0:
        preds = pred_df[date]
        preds = preds[preds.index.isin(current_prices.keys())]
        target_stocks = select_top_k(preds, K)
        
        # Close positions not in target
        for stock in list(holdings.keys()):
            if stock not in target_stocks and stock in current_prices:
                entry_info = open_positions.get(stock, {})
                exit_price = current_prices[stock]
                shares = holdings[stock]
                
                if entry_info:
                    entry_price = entry_info['entry_price']
                    entry_date = entry_info['entry_date']
                    gross_pnl = (exit_price - entry_price) * shares
                    cost = (entry_price * shares + exit_price * shares) * TRANSACTION_FEE
                    net_pnl = gross_pnl - cost
                    pnl_pct = ((exit_price / entry_price) - 1) * 100
                    
                    all_trades.append({
                        'stock': stock,
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'shares': shares,
                        'gross_pnl': gross_pnl,
                        'net_pnl': net_pnl,
                        'pnl_pct': pnl_pct,
                        'holding_days': (pd.to_datetime(date) - pd.to_datetime(entry_date)).days
                    })
                
                cash += shares * exit_price * (1 - TRANSACTION_FEE)
                del holdings[stock]
                if stock in open_positions:
                    del open_positions[stock]
        
        # Open new positions
        portfolio_value = sum(holdings.get(s, 0) * current_prices.get(s, 0) for s in holdings) + cash
        target_allocation = portfolio_value / K
        
        for stock in target_stocks:
            if stock not in holdings and stock in current_prices:
                price = current_prices[stock]
                shares = int(target_allocation / price)
                cost = shares * price * (1 + TRANSACTION_FEE)
                
                if cash >= cost and shares > 0:
                    cash -= cost
                    holdings[stock] = shares
                    open_positions[stock] = {
                        'entry_date': date,
                        'entry_price': price,
                        'shares': shares
                    }

# Close remaining positions
last_date = dates[-1]
for stock in list(holdings.keys()):
    if stock in current_prices:
        entry_info = open_positions.get(stock, {})
        if entry_info:
            exit_price = current_prices[stock]
            shares = holdings[stock]
            entry_price = entry_info['entry_price']
            entry_date = entry_info['entry_date']
            gross_pnl = (exit_price - entry_price) * shares
            cost = (entry_price * shares + exit_price * shares) * TRANSACTION_FEE
            net_pnl = gross_pnl - cost
            pnl_pct = ((exit_price / entry_price) - 1) * 100
            
            all_trades.append({
                'stock': stock,
                'entry_date': entry_date,
                'exit_date': last_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': shares,
                'gross_pnl': gross_pnl,
                'net_pnl': net_pnl,
                'pnl_pct': pnl_pct,
                'holding_days': (pd.to_datetime(last_date) - pd.to_datetime(entry_date)).days
            })

# Analyze trades
trades_df = pd.DataFrame(all_trades)

print(f"\n1. TRADE SUMMARY")
print("-" * 40)
print(f"Total Trades: {len(trades_df)}")

winning_trades = trades_df[trades_df['net_pnl'] > 0]
losing_trades = trades_df[trades_df['net_pnl'] < 0]

print(f"Winning Trades: {len(winning_trades)} ({len(winning_trades)/len(trades_df)*100:.1f}%)")
print(f"Losing Trades: {len(losing_trades)} ({len(losing_trades)/len(trades_df)*100:.1f}%)")

print(f"\n2. PROFIT/LOSS ANALYSIS")
print("-" * 40)
avg_win = winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0
avg_loss = losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0
avg_win_pct = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
avg_loss_pct = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0

print(f"Average Winning Trade: ₹{avg_win:,.0f} (+{avg_win_pct:.2f}%)")
print(f"Average Losing Trade: ₹{avg_loss:,.0f} ({avg_loss_pct:.2f}%)")

total_wins = winning_trades['net_pnl'].sum() if len(winning_trades) > 0 else 0
total_losses = abs(losing_trades['net_pnl'].sum()) if len(losing_trades) > 0 else 0
profit_factor = total_wins / total_losses if total_losses > 0 else 0
print(f"\nTotal Profit from Winners: ₹{total_wins:,.0f}")
print(f"Total Loss from Losers: ₹{total_losses:,.0f}")
print(f"Profit Factor: {profit_factor:.2f}")

expectancy = (len(winning_trades)/len(trades_df) * avg_win) + (len(losing_trades)/len(trades_df) * avg_loss)
print(f"Expectancy per Trade: ₹{expectancy:,.0f}")

print(f"\n3. HOLDING PERIOD")
print("-" * 40)
print(f"Average Holding Days: {trades_df['holding_days'].mean():.1f}")

trades_df.to_csv('output/Phase_7_Backtest_Results/trade_log.csv', index=False)
print(f"\n✓ Saved to output/Phase_7_Backtest_Results/trade_log.csv")

print("\n" + "=" * 80)
