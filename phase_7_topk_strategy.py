"""
Phase 7 - Step 2: TopK-Dropout Strategy Implementation
Implements portfolio strategy with weekly rebalancing and dropout mechanism.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('PHASE 7 - TOPK-DROPOUT STRATEGY BACKTEST')
print('='*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
K = 10  # Top K stocks to select
INITIAL_CAPITAL = 1_000_000  # ₹10 lakh
TRANSACTION_FEE = 0.002  # 0.2% per round-trip (Zerodha costs)
REBALANCE_FREQ = 5  # Every 5 trading days (weekly)
DROPOUT_RATE = 0.20  # Drop bottom 20%
DROPOUT_FREQ = 10  # Every 10 trading days (every 2 weeks)

# ============================================================================
# LOAD DATA
# ============================================================================
print('\n1. LOADING DATA')
print('-'*80)

# Load predictions and labels (185 stocks × 61 days)
pred_df = pd.read_csv('output/Phase_7_Backtest_Results/test_predictions_corrected.csv', index_col=0)
label_df = pd.read_csv('output/Phase_7_Backtest_Results/test_labels_corrected.csv', index_col=0)

print(f'Predictions: {pred_df.shape} (stocks × days)')
print(f'Labels (actual returns): {label_df.shape}')
print(f'Date range: {pred_df.columns[0]} to {pred_df.columns[-1]}')

# Load price data for each stock
price_data = {}
missing_stocks = []
for stock in pred_df.index:
    try:
        df = pd.read_csv(f'data/NIFTY200/raw/{stock}.csv')
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df.columns = [c.lower() for c in df.columns]
        df = df.set_index('date')
        price_data[stock] = df
    except:
        missing_stocks.append(stock)

if missing_stocks:
    print(f'⚠ Missing price data for {len(missing_stocks)} stocks: {missing_stocks[:5]}...')
    # Remove stocks without price data
    pred_df = pred_df.drop(missing_stocks, errors='ignore')
    label_df = label_df.drop(missing_stocks, errors='ignore')

print(f'✓ Loaded price data for {len(price_data)} stocks')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def select_top_k(predictions, k=10, exclude_list=None):
    """Select top K stocks by predicted return."""
    if exclude_list:
        predictions = predictions.drop(exclude_list, errors='ignore')
    return predictions.nlargest(k).index.tolist()

def calculate_position_sizes(selected_stocks, capital, prices, equal_weight=True):
    """Calculate position sizes (equal-weight or value-weight)."""
    if equal_weight:
        weight_per_stock = 1.0 / len(selected_stocks)
        positions = {}
        for stock in selected_stocks:
            capital_allocated = capital * weight_per_stock
            price = prices[stock]
            shares = int(capital_allocated / price)  # Integer shares
            positions[stock] = shares
        return positions
    else:
        # TODO: Implement value-weighted allocation
        pass

def execute_trades(current_holdings, target_holdings, prices, capital, fee_rate):
    """
    Execute trades to move from current to target portfolio.
    Returns: new_holdings, trade_log, transaction_costs, remaining_cash
    """
    trade_log = []
    transaction_costs = 0
    cash = capital
    
    # Sell stocks not in target or reduce positions
    for stock, shares in current_holdings.items():
        target_shares = target_holdings.get(stock, 0)
        if target_shares < shares:
            # Sell excess shares
            shares_to_sell = shares - target_shares
            price = prices[stock]
            proceeds = shares_to_sell * price
            cost = proceeds * fee_rate
            cash += proceeds - cost
            transaction_costs += cost
            trade_log.append({
                'stock': stock,
                'action': 'SELL',
                'shares': shares_to_sell,
                'price': price,
                'value': proceeds,
                'cost': cost
            })
    
    # Buy stocks in target or increase positions
    for stock, target_shares in target_holdings.items():
        current_shares = current_holdings.get(stock, 0)
        if target_shares > current_shares:
            # Buy additional shares
            shares_to_buy = target_shares - current_shares
            price = prices[stock]
            cost_without_fee = shares_to_buy * price
            cost_with_fee = cost_without_fee * (1 + fee_rate)
            
            if cash >= cost_with_fee:
                cash -= cost_with_fee
                transaction_costs += cost_without_fee * fee_rate
                trade_log.append({
                    'stock': stock,
                    'action': 'BUY',
                    'shares': shares_to_buy,
                    'price': price,
                    'value': cost_without_fee,
                    'cost': cost_without_fee * fee_rate
                })
            else:
                # Not enough cash - skip or use available cash
                print(f'⚠ Insufficient cash to buy {stock}: need ₹{cost_with_fee:.2f}, have ₹{cash:.2f}')
    
    # Update holdings
    new_holdings = current_holdings.copy()
    for trade in trade_log:
        stock = trade['stock']
        if trade['action'] == 'BUY':
            new_holdings[stock] = new_holdings.get(stock, 0) + trade['shares']
        else:  # SELL
            new_holdings[stock] = new_holdings.get(stock, 0) - trade['shares']
            if new_holdings[stock] == 0:
                del new_holdings[stock]
    
    return new_holdings, trade_log, transaction_costs, cash

def calculate_portfolio_value(holdings, prices):
    """Calculate total portfolio value."""
    total = 0
    for stock, shares in holdings.items():
        if stock in prices:
            total += shares * prices[stock]
    return total

# ============================================================================
# BACKTEST EXECUTION
# ============================================================================

print('\n2. RUNNING BACKTEST')
print('-'*80)
print(f'Strategy: TopK={K}, Rebalance every {REBALANCE_FREQ} days, Dropout {DROPOUT_RATE*100}% every {DROPOUT_FREQ} days')
print(f'Initial capital: ₹{INITIAL_CAPITAL:,.0f}')
print(f'Transaction costs: {TRANSACTION_FEE*100}% per round-trip')

# Initialize
dates = pred_df.columns.tolist()
portfolio_values = []
daily_returns = []
holdings = {}  # {stock: shares}
cash = INITIAL_CAPITAL
dropout_list = []  # Stocks to exclude temporarily

# Track metrics
total_trades = 0
total_transaction_costs = 0
rebalance_dates = []

for day_idx, date in enumerate(dates):
    # Get prices for this day
    current_prices = {}
    for stock in pred_df.index:
        if stock in price_data and date in price_data[stock].index:
            current_prices[stock] = price_data[stock].loc[date, 'close']
    
    # Check if we should rebalance
    should_rebalance = (day_idx % REBALANCE_FREQ == 0)
    should_dropout = (day_idx > 0) and (day_idx % DROPOUT_FREQ == 0)
    
    # Apply dropout mechanism
    if should_dropout and len(holdings) > 0:
        # Calculate returns of current holdings
        if day_idx >= DROPOUT_FREQ:
            holding_returns = {}
            lookback_dates = dates[max(0, day_idx-DROPOUT_FREQ):day_idx]
            for stock in holdings.keys():
                if stock in label_df.index:
                    returns = label_df.loc[stock, lookback_dates].values
                    holding_returns[stock] = returns.mean()
            
            # Drop bottom 20%
            if len(holding_returns) > 0:
                sorted_stocks = sorted(holding_returns.items(), key=lambda x: x[1])
                n_to_drop = max(1, int(len(sorted_stocks) * DROPOUT_RATE))
                dropout_list = [s[0] for s in sorted_stocks[:n_to_drop]]
                print(f'  Day {day_idx} ({date}): Dropout {n_to_drop} stocks: {dropout_list[:3]}...')
    
    # Rebalance portfolio
    if should_rebalance:
        # Get predictions for this day
        predictions = pred_df[date]
        
        # Select top K stocks (excluding dropout list)
        selected_stocks = select_top_k(predictions, k=K, exclude_list=dropout_list)
        
        # Calculate target positions
        portfolio_value = calculate_portfolio_value(holdings, current_prices) + cash
        target_positions = calculate_position_sizes(selected_stocks, portfolio_value, current_prices)
        
        # Execute trades
        holdings, trades, costs, cash = execute_trades(
            holdings, target_positions, current_prices, cash, TRANSACTION_FEE
        )
        
        total_trades += len(trades)
        total_transaction_costs += costs
        rebalance_dates.append(date)
        
        # Clear dropout list after rebalance
        dropout_list = []
        
        if day_idx % (REBALANCE_FREQ * 4) == 0:  # Print every ~4 weeks
            print(f'  Day {day_idx} ({date}): Rebalanced to {len(holdings)} stocks, value=₹{portfolio_value:,.0f}, trades={len(trades)}, costs=₹{costs:,.0f}')
    
    # Calculate daily portfolio value
    portfolio_value = calculate_portfolio_value(holdings, current_prices) + cash
    portfolio_values.append(portfolio_value)
    
    # Calculate daily return
    if day_idx > 0:
        daily_ret = (portfolio_value - portfolio_values[day_idx-1]) / portfolio_values[day_idx-1]
        daily_returns.append(daily_ret)
    else:
        daily_returns.append(0.0)

# ============================================================================
# RESULTS
# ============================================================================

print('\n3. BACKTEST RESULTS')
print('-'*80)

# Convert to DataFrame
results_df = pd.DataFrame({
    'date': dates,
    'portfolio_value': portfolio_values,
    'daily_return': daily_returns
})

# Calculate metrics
total_return = (portfolio_values[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
annualized_return = (1 + total_return) ** (252 / len(dates)) - 1
daily_returns_array = np.array(daily_returns)
sharpe_ratio = daily_returns_array.mean() / daily_returns_array.std() * np.sqrt(252) if daily_returns_array.std() > 0 else 0

# Maximum drawdown
cumulative_max = np.maximum.accumulate(portfolio_values)
drawdowns = (np.array(portfolio_values) - cumulative_max) / cumulative_max
max_drawdown = drawdowns.min()

# Win rate
winning_days = (daily_returns_array > 0).sum()
win_rate = winning_days / len(daily_returns_array) if len(daily_returns_array) > 0 else 0

print(f'\nPerformance Metrics:')
print(f'  Initial Capital:        ₹{INITIAL_CAPITAL:,.0f}')
print(f'  Final Value:            ₹{portfolio_values[-1]:,.0f}')
print(f'  Total Return:           {total_return*100:.2f}%')
print(f'  Annualized Return:      {annualized_return*100:.2f}%')
print(f'  Sharpe Ratio:           {sharpe_ratio:.3f}')
print(f'  Max Drawdown:           {max_drawdown*100:.2f}%')
print(f'  Win Rate:               {win_rate*100:.2f}%')
print(f'  Trading Days:           {len(dates)}')

print(f'\nTrading Statistics:')
print(f'  Total Trades:           {total_trades}')
print(f'  Rebalance Events:       {len(rebalance_dates)}')
print(f'  Transaction Costs:      ₹{total_transaction_costs:,.0f} ({total_transaction_costs/INITIAL_CAPITAL*100:.2f}%)')
print(f'  Turnover:               {total_trades / len(rebalance_dates):.1f} trades per rebalance')

# Save results
results_df.to_csv('output/Phase_7_Backtest_Results/strategy_performance.csv', index=False)
print(f'\n✓ Results saved to output/Phase_7_Backtest_Results/strategy_performance.csv')

# Save summary metrics
summary = {
    'initial_capital': INITIAL_CAPITAL,
    'final_value': portfolio_values[-1],
    'total_return': total_return,
    'annualized_return': annualized_return,
    'sharpe_ratio': sharpe_ratio,
    'max_drawdown': max_drawdown,
    'win_rate': win_rate,
    'total_trades': total_trades,
    'transaction_costs': total_transaction_costs,
    'trading_days': len(dates),
    'rebalance_frequency': REBALANCE_FREQ,
    'top_k': K,
    'dropout_rate': DROPOUT_RATE
}

import json
with open('output/Phase_7_Backtest_Results/strategy_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('✓ Summary saved to output/Phase_7_Backtest_Results/strategy_summary.json')

print('\n' + '='*80)
print('BACKTEST COMPLETE')
print('='*80)
