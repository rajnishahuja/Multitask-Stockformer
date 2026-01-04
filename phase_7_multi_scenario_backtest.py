"""
Phase 7 - Multi-Scenario Backtest
Test Daily, Weekly, and Monthly rebalancing strategies.
"""

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

def run_backtest(rebalance_freq, dropout_freq, strategy_name):
    """Run backtest with specified rebalancing frequency."""
    
    print(f'\n{"="*80}')
    print(f'RUNNING: {strategy_name}')
    print(f'{"="*80}')
    
    # Configuration
    K = 10
    INITIAL_CAPITAL = 1_000_000
    TRANSACTION_FEE = 0.001229  # 0.1229% round-trip (Zerodha actual: 0.0189% buy + 0.1039% sell)
    DROPOUT_RATE = 0.20
    
    # Load data
    pred_df = pd.read_csv('output/Phase_7_Backtest_Results/test_predictions_corrected.csv', index_col=0)
    label_df = pd.read_csv('output/Phase_7_Backtest_Results/test_labels_corrected.csv', index_col=0)
    
    # Load price data
    price_data = {}
    for stock in pred_df.index:
        try:
            df = pd.read_csv(f'data/NIFTY200/raw/{stock}.csv')
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            df.columns = [c.lower() for c in df.columns]
            df = df.set_index('date')
            price_data[stock] = df
        except:
            pass
    
    # Helper functions
    def select_top_k(predictions, k=10, exclude_list=None):
        if exclude_list:
            predictions = predictions.drop(exclude_list, errors='ignore')
        return predictions.nlargest(k).index.tolist()
    
    def calculate_position_sizes(selected_stocks, capital, prices):
        weight_per_stock = 1.0 / len(selected_stocks)
        positions = {}
        for stock in selected_stocks:
            capital_allocated = capital * weight_per_stock
            price = prices[stock]
            shares = int(capital_allocated / price)
            positions[stock] = shares
        return positions
    
    def execute_trades(current_holdings, target_holdings, prices, cash, fee_rate):
        trade_log = []
        transaction_costs = 0
        
        # Sell
        for stock, shares in current_holdings.items():
            target_shares = target_holdings.get(stock, 0)
            if target_shares < shares:
                shares_to_sell = shares - target_shares
                price = prices[stock]
                proceeds = shares_to_sell * price
                cost = proceeds * fee_rate
                cash += proceeds - cost
                transaction_costs += cost
                trade_log.append({'stock': stock, 'action': 'SELL', 'shares': shares_to_sell})
        
        # Buy
        for stock, target_shares in target_holdings.items():
            current_shares = current_holdings.get(stock, 0)
            if target_shares > current_shares:
                shares_to_buy = target_shares - current_shares
                price = prices[stock]
                cost_without_fee = shares_to_buy * price
                cost_with_fee = cost_without_fee * (1 + fee_rate)
                
                if cash >= cost_with_fee:
                    cash -= cost_with_fee
                    transaction_costs += cost_without_fee * fee_rate
                    trade_log.append({'stock': stock, 'action': 'BUY', 'shares': shares_to_buy})
        
        # Update holdings
        new_holdings = current_holdings.copy()
        for trade in trade_log:
            stock = trade['stock']
            if trade['action'] == 'BUY':
                new_holdings[stock] = new_holdings.get(stock, 0) + trade['shares']
            else:
                new_holdings[stock] = new_holdings.get(stock, 0) - trade['shares']
                if new_holdings[stock] == 0:
                    del new_holdings[stock]
        
        return new_holdings, trade_log, transaction_costs, cash
    
    def calculate_portfolio_value(holdings, prices):
        total = 0
        for stock, shares in holdings.items():
            if stock in prices:
                total += shares * prices[stock]
        return total
    
    # Run backtest
    dates = pred_df.columns.tolist()
    portfolio_values = []
    daily_returns = []
    holdings = {}
    cash = INITIAL_CAPITAL
    dropout_list = []
    
    total_trades = 0
    total_transaction_costs = 0
    rebalance_count = 0
    
    for day_idx, date in enumerate(dates):
        # Get prices
        current_prices = {}
        for stock in pred_df.index:
            if stock in price_data and date in price_data[stock].index:
                current_prices[stock] = price_data[stock].loc[date, 'close']
        
        should_rebalance = (day_idx % rebalance_freq == 0)
        should_dropout = (day_idx > 0) and (day_idx % dropout_freq == 0)
        
        # Dropout mechanism
        if should_dropout and len(holdings) > 0:
            if day_idx >= dropout_freq:
                holding_returns = {}
                lookback_dates = dates[max(0, day_idx-dropout_freq):day_idx]
                for stock in holdings.keys():
                    if stock in label_df.index:
                        returns = label_df.loc[stock, lookback_dates].values
                        holding_returns[stock] = returns.mean()
                
                if len(holding_returns) > 0:
                    sorted_stocks = sorted(holding_returns.items(), key=lambda x: x[1])
                    n_to_drop = max(1, int(len(sorted_stocks) * DROPOUT_RATE))
                    dropout_list = [s[0] for s in sorted_stocks[:n_to_drop]]
        
        # Rebalance
        if should_rebalance:
            predictions = pred_df[date]
            selected_stocks = select_top_k(predictions, k=K, exclude_list=dropout_list)
            portfolio_value = calculate_portfolio_value(holdings, current_prices) + cash
            target_positions = calculate_position_sizes(selected_stocks, portfolio_value, current_prices)
            holdings, trades, costs, cash = execute_trades(holdings, target_positions, current_prices, cash, TRANSACTION_FEE)
            
            total_trades += len(trades)
            total_transaction_costs += costs
            rebalance_count += 1
            dropout_list = []
        
        # Calculate portfolio value
        portfolio_value = calculate_portfolio_value(holdings, current_prices) + cash
        portfolio_values.append(portfolio_value)
        
        if day_idx > 0:
            daily_ret = (portfolio_value - portfolio_values[day_idx-1]) / portfolio_values[day_idx-1]
            daily_returns.append(daily_ret)
        else:
            daily_returns.append(0.0)
    
    # Calculate metrics
    total_return = (portfolio_values[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    annualized_return = (1 + total_return) ** (252 / len(dates)) - 1
    daily_returns_array = np.array(daily_returns)
    sharpe_ratio = daily_returns_array.mean() / daily_returns_array.std() * np.sqrt(252) if daily_returns_array.std() > 0 else 0
    
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (np.array(portfolio_values) - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min()
    
    winning_days = (daily_returns_array > 0).sum()
    win_rate = winning_days / len(daily_returns_array) if len(daily_returns_array) > 0 else 0
    
    # Print results
    print(f'\nPerformance Metrics:')
    print(f'  Total Return:           {total_return*100:.2f}%')
    print(f'  Annualized Return:      {annualized_return*100:.2f}%')
    print(f'  Sharpe Ratio:           {sharpe_ratio:.3f}')
    print(f'  Max Drawdown:           {max_drawdown*100:.2f}%')
    print(f'  Win Rate:               {win_rate*100:.2f}%')
    print(f'  Rebalances:             {rebalance_count}')
    print(f'  Total Trades:           {total_trades}')
    print(f'  Transaction Costs:      ₹{total_transaction_costs:,.0f} ({total_transaction_costs/INITIAL_CAPITAL*100:.2f}%)')
    
    # Save results
    results_df = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values,
        'daily_return': daily_returns
    })
    
    summary = {
        'strategy_name': strategy_name,
        'rebalance_frequency': rebalance_freq,
        'initial_capital': INITIAL_CAPITAL,
        'final_value': portfolio_values[-1],
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'rebalance_count': rebalance_count,
        'transaction_costs': total_transaction_costs,
        'trading_days': len(dates)
    }
    
    return results_df, summary

# ============================================================================
# RUN ALL SCENARIOS
# ============================================================================

print('='*80)
print('MULTI-SCENARIO BACKTEST')
print('='*80)

scenarios = [
    (1, 2, 'Daily Rebalancing'),      # Rebalance every day, dropout every 2 days
    (5, 10, 'Weekly Rebalancing'),    # Rebalance every 5 days, dropout every 10 days
    (20, 20, 'Monthly Rebalancing')   # Rebalance every 20 days, dropout every 20 days
]

all_results = {}
all_summaries = []

for rebal_freq, drop_freq, name in scenarios:
    results_df, summary = run_backtest(rebal_freq, drop_freq, name)
    all_results[name] = results_df
    all_summaries.append(summary)

# Save combined results
combined_df = pd.DataFrame()
for name, df in all_results.items():
    df_copy = df.copy()
    df_copy['strategy'] = name
    combined_df = pd.concat([combined_df, df_copy], ignore_index=True)

combined_df.to_csv('output/Phase_7_Backtest_Results/multi_scenario_results.csv', index=False)

# Save summaries
with open('output/Phase_7_Backtest_Results/multi_scenario_summary.json', 'w') as f:
    json.dump(all_summaries, f, indent=2)

# Comparison table
print('\n' + '='*80)
print('STRATEGY COMPARISON')
print('='*80)
comparison_df = pd.DataFrame(all_summaries)
print(comparison_df[['strategy_name', 'total_return', 'annualized_return', 'sharpe_ratio', 
                     'max_drawdown', 'transaction_costs', 'rebalance_count']].to_string(index=False))

print(f'\n✓ Results saved to output/Phase_7_Backtest_Results/')
print(f'  - multi_scenario_results.csv')
print(f'  - multi_scenario_summary.json')
