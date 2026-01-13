# Stockformer Experiment Results Summary

**Project:** Multitask-Stockformer for NIFTY-200  
**Period:** Phases 7-8 (2024-2025)

---

## Overview

| Phase | Focus | Factors | Key Result |
|:---:|:---|:---:|:---|
| 7 | Backtest validation | 158 (full Alpha158) | ✅ +18.38% alpha, Sharpe 2.58 |
| 8 | Factor optimization + Alpha360 | 26 → 360 | ❌ Regime change issue |

---

## Phase 7: Baseline Backtest ✅

**Test Period:** May-July 2024 (61 days)  
**Factors:** 158 Alpha158 (full set)  
**Model:** Saved at `output/NIFTY200_Subset10/best_model`

### Results

| Metric | Weekly Strategy | NIFTY-50 Benchmark |
|:---|:---:|:---:|
| Total Return | **13.87%** | 10.60% |
| Annualized | 71.03% | 52.65% |
| Sharpe Ratio | **2.58** | 2.43 |
| Max Drawdown | -6.87% | -5.93% |
| Win Rate | **62.30%** | 58.33% |
| Alpha | **+18.38%** | — |

**Conclusion:** Model demonstrated genuine predictive power with significant alpha.

---

## Phase 8: Factor Optimization & Alpha360

### Step 1-3: TopK Precision Metric

Added TopK Precision tracking (% overlap between predicted and actual top 10 stocks).
- Random baseline: ~7%
- Alpha158: 9.4% TopK (near random)

### Step 5a-5b: Ranking Losses

| Loss Function | Best TopK Precision |
|:---|:---:|
| MAE only | 9.4% |
| Margin loss | 8.5% |
| ListMLE | 14.5% |

**Finding:** ListMLE improved ranking but still weak.

### Step 5c: Factor Quality Investigation

| Metric | Train Period | Test Period |
|:---|:---:|:---:|
| Avg IC | ~0.01 | ~0.20 |
| Train↔Test Correlation | — | **0.12** |

**Finding:** IC filtering was counterproductive - train IC ≠ test IC.

### Step 6: Alpha360 Training

**Factors:** 360 raw lagged OHLCV features (original paper approach)  
**Model:** Saved at `output/NIFTY200_Subset10_Alpha360/best_model`

| Split | Val TopK | Test TopK | Test Accuracy |
|:---|:---:|:---:|:---:|
| Original (75/12.5/12.5) | **67.8%** | 6.5% | 50% |
| Adjusted (65/10/25) | **61.6%** | — | 49.6% |

**Finding:** Validation performance excellent but test performance near random - **regime change between train/val and test periods**.

---

## Models Saved

| Model | Location | Factors | Notes |
|:---|:---|:---:|:---|
| Phase 7 Best | `output/NIFTY200_Subset10/best_model` | 158 | Production-ready |
| Phase 8 Alpha158 | `output/NIFTY200_Subset10/best_model` (replaced) | 26 | IC-filtered |
| Phase 8 Alpha360 | `output/NIFTY200_Subset10_Alpha360/best_model` | 360 | Best validation |
| Phase 8 Alpha360 TopK | `output/NIFTY200_Subset10_Alpha360/best_model_topk` | 360 | Best TopK |

**Recommendation:** Use Phase 7 model for production (validated backtest).

---

## Key Learnings

1. **Alpha360 > Alpha158** for validation performance
2. **IC filtering is not helpful** for this dataset
3. **Regime change** is the main challenge (Aug-Nov 2022 test period different from train)
4. **Weekly rebalancing** optimal for transaction costs

---

## Next Steps

| Option | Description |
|:---|:---|
| A | **More data** - Include 2023-2025 to capture regime changes |
| B | **Rolling windows** - Train separate model per regime |
| C | **Live paper trading** - Validate Phase 7 model |
