# Phase 8: Bear Market Testing - CORRECTED Results

**Date:** 2026-01-11 (Updated after bug fix)
**Test Period:** Sep 01, 2022 - Nov 30, 2022 (Subset 10 Test Split)
**Market Context:** Post-correction recovery rally (NIFTY +6.13%)

> [!CAUTION]
> **Previous results (+30.88% return) were incorrect due to a bug in the backtest script.**
> The `execute_trades()` function was returning target holdings instead of actual holdings, causing phantom gains. This has been fixed.

---

## Corrected Performance Summary

| Metric | Phase 8 (Corrected) | Phase 7 (Bull) | Benchmark (NIFTY 50) |
|:---|:---:|:---:|:---:|
| **Total Return** | **-1.71%** | +13.87% | +6.13% |
| **Alpha** | **-7.85%** | +18.38% | 0% |
| **Sharpe Ratio** | **-0.42** | 2.58 | ~1.1 |
| **Max Drawdown** | **-7.75%** | -6.87% | - |

---

## Trade-Level Statistics

| Metric | Value |
|:---|:---:|
| **Total Trades** | 114 |
| **Winning Trades** | 51 (44.7%) |
| **Losing Trades** | 63 (55.3%) |
| **Average Win** | ₹2,991 (+3.56%) |
| **Average Loss** | ₹-2,800 (-2.43%) |
| **Largest Win** | ₹12,287 (AMBUJACEM) |
| **Largest Loss** | ₹-22,450 (IBULHSGFIN) |
| **Profit Factor** | 0.86 |
| **Expectancy/Trade** | ₹-209 |
| **Avg Holding Days** | 7.8 |

---

## Key Observations

### 1. Model Underperformed in This Period
- The model lost 1.71% while NIFTY gained 6.13%
- Win rate of 44.7% is below random (50%)
- Profit Factor of 0.86 means losses exceed gains

### 2. Asymmetric Risk
- Average win (+3.56%) is larger than average loss (-2.43%)
- BUT a single catastrophic trade (IBULHSGFIN -22.6%) wiped out multiple winners
- The model picked a stock that crashed (IBUL -22% in one week)

### 3. Best Performers Were Sector-Specific
- Top wins: AMBUJACEM, SHRECEMENT (Cement sector rally)
- Top losses: IBULHSGFIN, HDFC, YESBANK (Financial stress)

---

## Comparison with Phase 7

| Aspect | Phase 7 (May-Jul 2024) | Phase 8 (Sep-Nov 2022) |
|:---|:---|:---|
| Market Regime | Strong bull market | Recovery after correction |
| Model Performance | Outperformed (+18% alpha) | Underperformed (-7.8% alpha) |
| Win Rate | 62.3% | 44.7% |
| Sharpe | 2.58 | -0.42 |

**Conclusion:** The model performs well in trending markets but struggles during regime transitions.

---

## Investigation Required

1. **Overfitting concern**: With 158 factors and only ~500 training days, the model may have overfit to noise
2. **Regime mismatch**: Training data (2020-2022) includes COVID crash patterns that may not apply to the test period
3. **Factor decay**: Some Alpha158 factors may have lost predictive power by the test period

---

## Files Generated

- `phase8_backtest.py` - Fixed backtest script
- `phase8_trade_analysis.py` - Trade-level statistics
- `output/Phase_8_Backtest_Results/trade_log.csv` - Detailed trade log
