# Phase 7: Backtest Implementation Plan

## Overview
Implement TopK-Dropout backtesting strategy on NIFTY-200 stocks to evaluate trading performance of the trained Stockformer model. This phase validates whether the model's ranking quality (IC=0.0308, Rank IC=0.0123) translates to profitable trading.

## Goals (From nifty-adaptationchat.txt Phase 7)
1. **Primary**: Implement TopK-Dropout strategy (K=10, weekly rebalance, dropout 20% every 2 weeks)
2. **Secondary**: Calculate comprehensive metrics (Sharpe, Sortino, Calmar, IC, Rank IC, turnover)
3. **Tertiary**: Compare against NIFTY-50 TRI benchmark + equal-weight NIFTY-200
4. **Validation**: Test multiple rebalancing frequencies (daily, weekly, monthly) if computationally feasible
5. **Decision Point**: Determine if model ready for Phase 8 (ensemble) or needs iteration

## Prerequisites (Phase 6 Completed)
- ✅ Trained model: `cpt/NIFTY200/saved_model_Alpha158_MVP`
- ✅ Test predictions: `output/Multitask_output_NIFTY200_Alpha158_MVP/regression/`
- ✅ Training period: 2022-01-03 to 2024-08-30 (660 days)
- ✅ Data split: Train 75%, Val 12.5%, Test 12.5% (83 days test)
- ✅ Stocks: 185 NIFTY-200 stocks (post-filtering)
- ✅ Model performance: MAE 0.0159, Accuracy 52.02%, IC 0.0308

## Inputs

### 1. Model Predictions
**File**: `output/Multitask_output_NIFTY200_Alpha158_MVP/regression/regression_pred_last_step.csv`
- Format: CSV with stocks as rows, dates as columns
- Content: Predicted returns for each stock on each day
- Period: Test set only (83 days)
- Note: Last time step predictions (T+1 forecasts)

### 2. Actual Returns
**File**: `output/Multitask_output_NIFTY200_Alpha158_MVP/regression/regression_label_last_step.csv`
- Format: Same structure as predictions
- Content: Actual returns for validation
- Purpose: Calculate IC, accuracy, performance metrics

### 3. Stock Price Data
**Source**: NSE historical data (already collected in Phase 1)
**File**: `data/NIFTY200/stock_prices/`
- Required fields: Date, Stock Code, Close Price, Volume
- Purpose: Calculate actual portfolio returns with real prices
- Period: Must cover entire test period

### 4. Benchmark Data
**Source**: NSE NIFTY-50 Total Return Index (TRI)
**File**: `data/NIFTY200/benchmark/NIFTY50_TRI.csv` (to be created if missing)
- Required fields: Date, Close Price
- Purpose: Compare strategy vs buy-and-hold benchmark
- Alternative: Use NIFTY-50 price index if TRI unavailable

### 5. Trading Parameters
**Configuration File**: `config/Phase7_Backtest_NIFTY200.conf` (to be created)
```ini
[strategy]
name = TopK_Dropout
topk = 10              # Number of stocks to select
rebalance = weekly     # Rebalancing frequency (paper's approach)
weight_scheme = equal  # equal weight portfolio
dropout_rate = 0.20    # Drop bottom 20% every 2 weeks (per paper)

[costs]
# ZERODHA ACTUAL COSTS (2024):
# Buy: 0.053% (transaction 0.00325% + stamp 0.015% + GST + SEBI)
# Sell: 0.138% (STT 0.1% + transaction 0.00325% + GST + SEBI)
# Round-trip: 0.191%
transaction_fee = 0.002     # 0.2% per round-trip (conservative, includes slippage)
min_trade_value = 1000      # Minimum ₹1000 per trade

[risk]
initial_capital = 1000000   # ₹10 lakh starting capital
max_position = 0.15         # Max 15% per stock
min_position = 0.05         # Min 5% per stock (for K=10, ~10% each)

[dates]
start_date = 2024-06-01     # First day of test period
end_date = 2024-08-30       # Last day of test period
risk_free_rate = 0.06       # 6% annual (India)
```

## Outputs

### 1. Trading Log
**File**: `output/Phase7_Backtest_Results/trade_log.csv`

| Date       | Action | Stock_Code | Quantity | Price   | Value     | Transaction_Cost | Position_Weight |
|------------|--------|------------|----------|---------|-----------|------------------|-----------------|
| 2024-06-03 | BUY    | RELIANCE   | 50       | 2450.00 | 122500.00 | 61.25            | 0.1224          |
| 2024-06-03 | BUY    | TCS        | 30       | 3580.00 | 107400.00 | 53.70            | 0.1073          |
| ...        | ...    | ...        | ...      | ...     | ...       | ...              | ...             |

### 2. Daily Portfolio Metrics
**File**: `output/Phase7_Backtest_Results/daily_portfolio.csv`

| Date       | Portfolio_Value | Daily_Return | Cumulative_Return | Drawdown | Holdings_Count | Turnover |
|------------|-----------------|--------------|-------------------|----------|----------------|----------|
| 2024-06-03 | 1000000.00      | 0.0000       | 0.0000            | 0.0000   | 10             | 1.0000   |
| 2024-06-04 | 1002500.00      | 0.0025       | 0.0025            | 0.0000   | 10             | 0.0000   |
| ...        | ...             | ...          | ...               | ...      | ...            | ...      |

### 3. Performance Summary Report
**File**: `output/Phase7_Backtest_Results/performance_summary.txt`

```
================================================================================
BACKTEST PERFORMANCE SUMMARY
================================================================================

STRATEGY: TopK-Dropout (K=10, Weekly Rebalance)
PERIOD: 2024-06-01 to 2024-08-30 (83 trading days)
INITIAL CAPITAL: ₹1,000,000

RETURNS:
  Total Return:              12.5%
  Annualized Return:         45.2%
  CAGR:                      43.8%
  
  Benchmark (NIFTY-50):      8.3%
  Excess Return:             4.2%
  Alpha:                     3.8%

RISK METRICS:
  Volatility (Annual):       18.5%
  Sharpe Ratio:              2.15
  Sortino Ratio:             3.20
  Max Drawdown:              -8.5%
  Max Drawdown Duration:     12 days
  Calmar Ratio:              5.32

RANKING QUALITY:
  Mean IC:                   0.0308
  Mean Rank IC:              0.0123
  IC > 0 Rate:               59.6%
  IC Stability (Std):        0.1126

TRADING STATISTICS:
  Total Trades:              240
  Win Rate:                  54.2%
  Average Win:               2.8%
  Average Loss:              -1.9%
  Expectancy:                0.52%
  Profit Factor:             1.45
  
COSTS:
  Total Transaction Costs:   ₹12,500 (1.25% of initial capital)
  Average Turnover:          85.0%
  Trading Days per Year:     ~250

TOP 10 BEST TRADES:
  [Stock, Date, Return, P&L]
  
TOP 10 WORST TRADES:
  [Stock, Date, Return, P&L]
```

### 4. Visualization Plots
**Directory**: `output/Phase7_Backtest_Results/plots/`

1. **cumulative_returns.png**: Portfolio vs Benchmark cumulative returns
2. **drawdown_curve.png**: Underwater equity curve
3. **monthly_returns_heatmap.png**: Monthly return heatmap
4. **rolling_sharpe.png**: 30-day rolling Sharpe ratio
5. **ic_timeseries.png**: IC over time (by rebalance period)
6. **turnover_analysis.png**: Turnover rate by rebalance date
7. **top_contributors.png**: Top 10 stocks by contribution to returns
8. **sector_exposure.png**: Sector allocation over time

### 5. Stock Selection Analysis
**File**: `output/Phase7_Backtest_Results/stock_selection.csv`

| Stock_Code | Times_Selected | Win_Rate | Avg_Return | Total_PnL | Contribution |
|------------|----------------|----------|------------|-----------|--------------|
| RELIANCE   | 8              | 62.5%    | 2.3%       | 18,500    | 1.85%        |
| TCS        | 7              | 57.1%    | 1.8%       | 12,600    | 1.26%        |
| ...        | ...            | ...      | ...        | ...       | ...          |

## Implementation Steps

### Step 1: Data Preparation (Validation)
**Script**: `phase7_data_validation.py`

1. Load model predictions and labels
2. Verify data completeness (no NaN, correct dimensions)
3. Load stock price data for test period
4. Validate date alignment across all datasets
5. Calculate actual returns from price data
6. Load benchmark data (NIFTY-50 TRI)
7. Generate data quality report

**Output**: `output/Phase7_Backtest_Results/data_validation_report.txt`

**Success Criteria**:
- All dates aligned across predictions, labels, prices
- No missing data for top predicted stocks
- Price data covers full test period
- Benchmark data available

### Step 2: Strategy Implementation
**Script**: `phase7_topk_strategy.py`

**Core Functions**:

```python
def select_top_k_stocks(predictions_date, k=10):
    """
    Select top K stocks by predicted return.
    
    Args:
        predictions_date: Array of predictions for all stocks on one date
        k: Number of stocks to select
    
    Returns:
        list of (stock_index, predicted_return) tuples
    """
    
def rebalance_portfolio(current_holdings, target_holdings, prices, capital, fee_rate):
    """
    Generate trades to move from current to target portfolio.
    
    Returns:
        trades: List of (stock, action, quantity, price, cost) tuples
        new_capital: Remaining capital after trades
    """

def calculate_daily_returns(holdings, price_changes):
    """
    Calculate portfolio return for one day.
    
    Returns:
        daily_return: Portfolio return %
        new_holdings: Updated holdings after price changes
    """

def calculate_drawdown(cumulative_returns):
    """
    Calculate drawdown series and max drawdown.
    
    Returns:
        drawdown_series: Drawdown at each point in time
        max_drawdown: Maximum drawdown value
        max_dd_duration: Days from peak to recovery
    """

def calculate_sharpe_ratio(returns, risk_free_rate=0.06, periods_per_year=252):
    """
    Calculate annualized Sharpe ratio.
    """

def calculate_ic_by_period(predictions, actuals, rebalance_dates):
    """
    Calculate IC for each rebalancing period.
    
    Returns:
        ic_series: IC for each period
        rank_ic_series: Rank IC for each period
    """
```

**Execution Flow**:
1. Initialize portfolio with capital
2. For each rebalance date:
   - Load predictions for that date
   - Select top K stocks
   - Calculate target weights (equal weight)
   - Generate trades to rebalance
   - Apply transaction costs
3. For each trading day between rebalances:
   - Update holdings based on price changes
   - Calculate daily portfolio value and return
   - Track cumulative returns and drawdown
4. At end, calculate all performance metrics

### Step 3: Benchmark Comparison
**Script**: `phase7_benchmark_comparison.py`

1. Load NIFTY-50 TRI data for same period
2. Calculate benchmark returns (buy-and-hold)
3. Calculate excess returns (strategy - benchmark)
4. Calculate alpha and beta using regression
5. Perform statistical tests (t-test for excess returns)
6. Generate comparison plots

### Step 4: Results Analysis & Visualization
**Script**: `phase7_visualization.py`

1. Generate all plots listed in outputs
2. Create performance summary report
3. Analyze stock selection patterns
4. Identify best/worst trades
5. Sector exposure analysis (if sector data available)
6. Sensitivity analysis (vary K, rebalance frequency)

### Step 5: Decision Framework
**Script**: `phase7_decision_report.py`

Generate decision recommendation based on results:

**Scenario A: Good Performance (Sharpe > 1.0, Excess Return > 0)**
```
✅ RECOMMENDATION: Proceed to Phase 8 (Ensemble)
  - Model shows profitable trading signal
  - Expand to 14 rolling windows
  - Test ensemble aggregation methods
  - Prepare for production deployment
```

**Scenario B: Marginal Performance (Sharpe 0.5-1.0, Excess Return ≈ 0)**
```
⚠️  RECOMMENDATION: Iterate on Model (Phase 6.5)
  - Expand from 22 to 158 Alpha158 factors
  - Test different loss functions (correlation loss)
  - Adjust TopK (try K=15, K=20)
  - Test different rebalance frequencies
  - Re-run backtest after improvements
```

**Scenario C: Poor Performance (Sharpe < 0.5, Negative Excess Return)**
```
❌ RECOMMENDATION: Revisit Approach
  - Current model not suitable for trading
  - Consider: More factors (360 like paper), longer history, different architecture
  - Analyze failure mode: Is IC too low? High turnover? Bad market conditions?
  - May need fundamental architecture changes
```

## Validation Checklist

### Data Integrity
- [ ] All prediction dates exist in price data
- [ ] No NaN values in predictions for selected stocks
- [ ] Price data covers full test period without gaps
- [ ] Benchmark data aligned with test period
- [ ] Date formats consistent across all datasets

### Strategy Logic
- [ ] Top K selection matches manual calculation
- [ ] Portfolio weights sum to 1.0 (or close to 1.0 with cash)
- [ ] Transaction costs applied correctly
- [ ] Rebalancing triggers at correct dates
- [ ] Holdings update correctly between rebalances

### Performance Metrics
- [ ] Sharpe ratio calculation verified (annualized correctly)
- [ ] Drawdown calculation correct (matches manual check)
- [ ] IC calculation matches analyze_ranking_quality.py results
- [ ] Benchmark returns match NSE official data
- [ ] Total return reconciles with trade log

### Edge Cases
- [ ] Handle stocks with missing price data (skip or substitute)
- [ ] Handle insufficient capital for rebalance (partial execution)
- [ ] Handle stocks that delist during test period
- [ ] Handle extreme price movements (circuit limits)
- [ ] Handle first/last trading days (partial periods)

## Success Criteria

### Minimum Viable (Phase 7 Complete)
1. ✅ Backtest runs without errors
2. ✅ All metrics calculated and reported
3. ✅ Results align with expectations (IC, returns correlated)
4. ✅ Decision recommendation generated

### Good Result (Proceed to Phase 8)
1. ✅ Sharpe Ratio > 1.0
2. ✅ Positive excess return vs benchmark
3. ✅ Max drawdown < 20%
4. ✅ IC consistently positive across periods
5. ✅ Win rate > 50% for top K stocks

### Excellent Result (Production Ready)
1. ✅ Sharpe Ratio > 1.5
2. ✅ Excess return > 5% annualized
3. ✅ Max drawdown < 15%
4. ✅ IC > 0.03 with low variance
5. ✅ Turnover < 100% per rebalance
6. ✅ Robust across different K values

## Risk Considerations

### Market Risks
- **Test Period Bias**: 83 days may not capture full market cycle
- **Look-Ahead Bias**: Ensure predictions use only past data (already handled in training)
- **Survivorship Bias**: Using current NIFTY-200 constituents (acceptable for MVP)

### Implementation Risks
- **Price Data Quality**: NSE data may have gaps, need to handle
- **Transaction Costs**: 0.05% may be optimistic, test with 0.1% sensitivity
- **Slippage**: Market impact not modeled, may underestimate costs
- **Liquidity**: Assume all stocks liquid (valid for NIFTY-200)

### Statistical Risks
- **Small Sample**: 83 days is short for statistical significance
- **Overfitting**: Model may have fit test period by chance
- **Non-Stationarity**: Market regime may change post-test period

**Mitigation**: 
- Report confidence intervals for all metrics
- Perform bootstrap analysis if sample size allows
- Note limitations in decision report
- Plan for 14-window validation (Phase 8) to address sample size

## Timeline Estimate

| Task | Duration | Dependencies |
|------|----------|--------------|
| Data validation | 30 min | Predictions, prices, benchmark |
| Strategy implementation | 2 hours | Data validation complete |
| Benchmark comparison | 30 min | Strategy results |
| Visualization | 1 hour | All results available |
| Decision report | 30 min | Analysis complete |
| **Total** | **4-5 hours** | - |

## Files to Create (All Prefixed with phase7_)

1. `config/phase7_backtest_config.ini` - Strategy parameters
2. `phase7_data_validation.py` - Data prep and validation
3. `phase7_topk_strategy.py` - Core strategy implementation  
4. `phase7_benchmark_comparison.py` - Benchmark analysis
5. `phase7_visualization.py` - Plots and charts
6. `phase7_decision_report.py` - Final recommendation
7. `phase7_runner.py` - Master script to run all steps

**Note**: All new files prefixed with `phase7_` to avoid modifying GitHub project files.

## Dependencies

**Python Packages** (already installed):
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)
- scipy (statistical tests)

**New Packages** (if needed):
- `yfinance` or `nsepy` - for fetching NIFTY-50 benchmark if not available
- `tabulate` - for pretty-printing tables in reports

**Install command** (if needed):
```bash
pip install yfinance nsepy tabulate
```

## Next Steps After Phase 7

Based on backtest results, one of three paths:

**Path A: Success → Phase 8 (Ensemble)**
- Implement 14 rolling windows
- Aggregate predictions
- Full period backtest (2018-2024)
- Production deployment prep

**Path B: Marginal → Phase 6.5 (Iterate)**
- Expand to 158 factors
- Test different hyperparameters
- Alternative loss functions
- Re-run backtest

**Path C: Failure → Fundamental Revision**
- Revisit architecture
- Consider ensemble of different models (LSTM, GRU, etc.)
- Expand to 360 factors like paper
- Longer training periods

## Questions for Discussion Before Implementation

1. **Rebalance Frequency**: ✅ **APPROVED** - Weekly (paper's approach), test all three (daily/weekly/monthly) if computationally easy
2. **TopK Value**: Start with K=10 (paper's approach), can test sensitivity later
3. **Transaction Costs**: ✅ **CORRECTED** - Use 0.2% per round-trip (Zerodha actual costs)
4. **Benchmark**: NIFTY-50 TRI + equal-weight NIFTY-200 (as per nifty-adaptationchat.txt)
5. **Test Period Length**: Use all 83 test days (no further holdout needed)
6. **Dropout**: Implement 20% dropout every 2 weeks as per paper

---

**Status**: 📋 PLAN READY - Awaiting approval to proceed with implementation

**Last Updated**: January 4, 2026
