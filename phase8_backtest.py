
import pandas as pd
import numpy as np
import warnings
import os
import json

warnings.filterwarnings('ignore')

print('='*80)
print('PHASE 8 - BACKTEST (BEAR MARKET SUBSET 10)')
print('='*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
K = 10  # Top K stocks
INITIAL_CAPITAL = 1_000_000
TRANSACTION_FEE = 0.002
REBALANCE_FREQ = 5 # Weekly
DROPOUT_RATE = 0.20
DROPOUT_FREQ = 10

# Configuration - consolidated output folder
DATASET = 'NIFTY200_Subset10'
OUTPUT_DIR = f'output/{DATASET}'
PRED_FILE = f'{OUTPUT_DIR}/test_predictions.csv'
LABEL_FILE = f'{OUTPUT_DIR}/test_labels.csv'
PRICE_DIR = f'data/{DATASET}/raw'

# Symbol mapping (Must match phase8_yahoo_data_fetcher.py)
SYMBOL_MAPPING = {
    'AMARAJABAT': 'ARE&M',           
    'CADILAHC': 'ZYDUSLIFE',         
    'CENTURYTEX': 'ABREL',           
    'EQUITAS': 'EQUITASBNK',         
    'GMRINFRA': 'GMRAIRPORT',        
    'HDFC': 'HDFCBANK',              
    'IBULHSGFIN': 'IBULLSLTD',       
    'INFRATEL': 'INDUSTOWER',        
    'L&TFH': 'LTF',                  
    'MCDOWELL-N': 'UNITDSPR',        
    'MINDTREE': 'LTIM',              
    'MOTHERSUMI': 'MOTHERSON',       
    'NIITTECH': 'COFORGE',           
    'PEL': 'POONAWALLA',             
    'PVR': 'PVRINOX',                
    'SRTRANSFIN': 'SHRIRAMFIN',      
    'TATAMOTORS': 'TMPV',            
    'UJJIVAN': 'UJJIVANSFB',         
}

# ============================================================================
# LOAD DATA
# ============================================================================
print('\n1. LOADING DATA')
print('-'*80)

pred_df = pd.read_csv(PRED_FILE, index_col=0)
label_df = pd.read_csv(LABEL_FILE, index_col=0)

print(f'Predictions: {pred_df.shape} (stocks x days)')
if not pred_df.empty:
    print(f'Date range: {pred_df.columns[0]} to {pred_df.columns[-1]}')
else:
    print("Error: Prediction file is empty!")
    exit(1)

# Load price data
price_data = {}
missing_stocks = []
for stock in pred_df.index:
    # Map symbol to filename
    mapped_symbol = SYMBOL_MAPPING.get(stock, stock)
    file_path = os.path.join(PRICE_DIR, f"{mapped_symbol}.csv")
    
    try:
        df = pd.read_csv(file_path)
        # Ensure date format matches predictions (yyyy-mm-dd)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        # Rename cols to lower
        df.columns = [c.lower() for c in df.columns]
        df = df.set_index('date')
        
        # Yahho fetcher saves 'adj_close'. Use that? Or 'close'?
        # Phase 7 used 'close'. Adjusted is safer for returns.
        if 'adj_close' in df.columns:
            df['close'] = df['adj_close']
            
        price_data[stock] = df
    except Exception as e:
        # print(f"Error loading {stock}: {e}")
        missing_stocks.append(stock)

if missing_stocks:
    print(f'⚠ Missing price data for {len(missing_stocks)} stocks: {missing_stocks[:5]}...')
    pred_df = pred_df.drop(missing_stocks, errors='ignore')
    label_df = label_df.drop(missing_stocks, errors='ignore')

print(f'✓ Loaded price data for {len(price_data)} stocks')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def select_top_k(predictions, k=10, exclude_list=None):
    if exclude_list:
        predictions = predictions.drop(exclude_list, errors='ignore')
    return predictions.nlargest(k).index.tolist()

def calculate_position_sizes(selected_stocks, capital, prices):
    if not selected_stocks:
        return {}
    weight = 1.0 / len(selected_stocks)
    positions = {}
    for stock in selected_stocks:
        if stock in prices:
            allocation = capital * weight
            price = prices[stock]
            if price > 0:
                shares = int(allocation / price)
                positions[stock] = shares
    return positions

def execute_trades(current_holdings, target_holdings, prices, capital, fee_rate):
    """Execute trades and return ACTUAL holdings (not target holdings)."""
    trade_log = []
    transaction_costs = 0
    cash = capital
    
    # Sell first - reduce positions or exit completely
    for stock, shares in current_holdings.items():
        if stock not in prices: 
            continue
        target = target_holdings.get(stock, 0)
        if target < shares:
            sell_amt = shares - target
            price = prices[stock]
            proceeds = sell_amt * price
            cost = proceeds * fee_rate
            cash += proceeds - cost
            transaction_costs += cost
            trade_log.append({'stock': stock, 'action': 'SELL', 'shares': sell_amt, 'price': price, 'value': proceeds})
            
    # Buy - increase positions or enter new ones
    for stock, target in target_holdings.items():
        if stock not in prices: 
            continue
        current = current_holdings.get(stock, 0)
        if target > current:
            buy_amt = target - current
            price = prices[stock]
            cost_raw = buy_amt * price
            cost_total = cost_raw * (1 + fee_rate)
            if cash >= cost_total:
                cash -= cost_total
                transaction_costs += cost_raw * fee_rate
                trade_log.append({'stock': stock, 'action': 'BUY', 'shares': buy_amt, 'price': price, 'value': cost_raw})
                
    # FIXED: Update holdings based on ACTUAL trades, not target
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
    val = 0
    for stock, shares in holdings.items():
        if stock in prices:
            val += shares * prices[stock]
    return val

# ============================================================================
# BACKTEST EXECUTION
# ============================================================================
print('\n2. RUNNING BACKTEST')
print('-'*80)

dates = pred_df.columns.tolist()

# Filter dates common to predictions and at least one stock's price data
# Pick a reference stock (that exists) to check valid trading days
reference_stock = list(price_data.keys())[0] if price_data else None
if reference_stock:
    valid_price_dates = set(price_data[reference_stock].index)
    dates = [d for d in dates if d in valid_price_dates]
else:
    dates = []

print(f"Valid Backtest Dates: {len(dates)} (Filtered from {len(pred_df.columns)})")

dates.sort() # Ensure chronological
portfolio_values = []
daily_returns = []
holdings = {}
cash = INITIAL_CAPITAL
dropout_list = []

trade_count = 0
total_costs = 0

for day_idx, date in enumerate(dates):
    # Prices for today
    current_prices = {}
    for stock in pred_df.index:
        if stock in price_data and date in price_data[stock].index:
            current_prices[stock] = price_data[stock].loc[date, 'close']
            
    if day_idx == 0:
        print(f"DEBUG: Day 0 Date: {date}")
        print(f"DEBUG: loaded prices for {len(current_prices)} stocks")
        if len(current_prices) < 10:
             print(f"DEBUG: Sample price keys: {list(price_data[list(price_data.keys())[0]].index)[:5]}")

    # Portfolio Value Before Rebalance
    curr_val = calculate_portfolio_value(holdings, current_prices) + cash
    
    if day_idx == 0:
        print(f"DEBUG: Day 0 Date: {date} (Type: {type(date)})")
        sample_stock = list(current_prices.keys())[0] if current_prices else "None"
        print(f"DEBUG: Sample Price Key type: {type(list(price_data[sample_stock].index)[0]) if sample_stock != 'None' else 'N/A'}")

    
    # Rebalance Logic
    if day_idx % REBALANCE_FREQ == 0:
        # Dropout logic (simplified)
        preds = pred_df[date]
        selected = select_top_k(preds, K, dropout_list)
        
        target_pos = calculate_position_sizes(selected, curr_val, current_prices)
        holdings, trades, costs, cash = execute_trades(holdings, target_pos, current_prices, cash, TRANSACTION_FEE)
        
        trade_count += len(trades)
        total_costs += costs
        
    # Value After Rebalance
    new_val = calculate_portfolio_value(holdings, current_prices) + cash
    portfolio_values.append(new_val)
    
    ret = 0 if day_idx == 0 else (new_val - portfolio_values[day_idx-1]) / portfolio_values[day_idx-1]
    daily_returns.append(ret)

# ============================================================================
# RESULTS
# ============================================================================
print('\n3. RESULTS')
print('-'*80)

total_ret = (portfolio_values[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
annual_factor = 252 / len(dates) if len(dates) > 0 else 1
ann_ret = (1 + total_ret) ** annual_factor - 1
sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0

# Max Drawdown
cum_max = np.maximum.accumulate(portfolio_values)
drawdowns = (np.array(portfolio_values) - cum_max) / cum_max
max_dd = drawdowns.min()

print(f"Final Value: ₹{portfolio_values[-1]:,.2f}")
print(f"Total Return: {total_ret*100:.2f}%")
print(f"Annualized:   {ann_ret*100:.2f}%")
print(f"Sharpe Ratio: {sharpe:.3f}")
print(f"Max Drawdown: {max_dd*100:.2f}%")
print(f"Trades:       {trade_count}")

# Save
results = pd.DataFrame({'date': dates, 'value': portfolio_values, 'return': daily_returns})
results.to_csv(os.path.join(OUTPUT_DIR, 'phase8_backtest_metrics.csv'), index=False)
print(f"Saved metrics to {OUTPUT_DIR}")
