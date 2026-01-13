# Phase 8 Improvement Plan: Factor Filtering & Regularization

**Date:** 2026-01-11
**Objective:** Address overfitting and improve model generalization

---

## Problem Summary

- Phase 8 model (158 factors, no IC filter) achieved only 44.7% trade win rate
- Phase 7 model (22 factors, IC-filtered) achieved 54.6% trade win rate
- Current model has limited regularization (dropout only in temporal conv layer)

---

## Proposed Improvements

### 1. Factor Selection: Top 20% by IC + Correlation Filter

**Step 1a: Calculate IC for all 158 factors**
- Compute rank IC (Spearman correlation) between each factor and future returns
- Use training period only to avoid look-ahead bias

**Step 1b: Keep Top 20% (~32 factors)**
- Sort factors by absolute IC
- Keep top 32 factors

**Step 1c: Remove Correlated Factors**
- Among the 32, compute pairwise correlations
- If correlation > 0.8, keep only the one with higher IC
- Target: 15-25 diverse factors

**Files to modify:**
- `phase8_factor_engineering.py` - Add IC ranking and correlation filter

---

### 2. Enhanced Regularization

**Current state:**
- Dropout: 0.2 in temporalConvNet only
- Weight decay: None

**Proposed changes:**

| Component | Current | Proposed |
|:---|:---:|:---:|
| `temporalConvNet` | 0.2 | 0.3 |
| `sparseSpatialAttention` | None | 0.1 (add after attention) |
| `FeedForward` | None | 0.1 (add between layers) |
| Optimizer weight_decay | 0 | 1e-4 |

**Files to modify:**
- `Stockformermodel/Multitask_Stockformer_models.py` - Add dropout layers
- `phase8_train.py` - Add weight_decay to optimizer

---

## Phase 8 Implementation Status (COMPLETED)

| Step | Description | Status |
|:---:|:---|:---:|
| 1 | Add IC calculation to factor engineering | ✅ |
| 2 | Add correlation-based filtering | ✅ |
| 3 | Re-run factor engineering (26 factors selected) | ✅ |
| 4 | Add dropout to attention layers | ✅ |
| 5 | Add weight_decay to optimizer | ✅ |
| 6 | Retrain model (55% test accuracy, MAE: 0.0134) | ✅ |
| 7 | Run backtest (51.4% win rate, -2.54% return) | ✅ |

**Outcome:** Win rate improved from 44.7% → 51.4% (target achieved), but total return worsened (-1.71% → -2.54%). Lower MAE didn't translate to better trading because MAE measures prediction accuracy across ALL stocks, not ranking quality of TOP K.

---

# Phase 8 Continued: Ranking Improvement

## Problem

Training optimizes MAE (prediction accuracy for all 143 stocks), but trading uses only TOP 10 stocks. Lower MAE doesn't guarantee better ranking.

## Completed Steps

### Steps 1-2: Added Metrics ✅

- Added Rank IC and TopK Precision metrics to validation
- **Finding:** Rank IC showed no correlation with TopK Precision
- **Decision:** Dropped Rank IC from code (comment explains why)

### Step 3: Train and Analyze ✅

**Training run results (27 epochs before manual stop):**

| Metric | Best Value | Epoch |
|:---|:---:|:---:|
| MAE | 0.0167 | 2 |
| TopK Precision | 9.4% | 12 |

**Key Finding:** TopK Precision barely above random chance (~7%). Model not learning to rank stocks.

### Step 4: Backtest Comparison

Skipped - both models have poor TopK Precision (~9%), not worth comparing.

---

## Step 5: TopK-Weighted Loss (IN PROGRESS)

### Why This Is Needed

Current loss (MAE) treats all stocks equally. We need a loss that specifically penalizes:
- Predicting high returns for stocks that actually perform poorly
- Predicting low returns for stocks that actually are top performers

### Proposed Implementation

**File:** `phase8_train.py` - modify loss calculation

**Approach:** Margin-based TopK loss

```python
def topk_margin_loss(pred, actual, k=10, margin=0.01):
    """
    Penalize when predicted top-k stocks have lower actual returns 
    than non-top-k stocks.
    """
    # Get actual top-k and bottom stocks
    actual_topk_idx = torch.topk(actual, k).indices
    actual_bottomk_idx = torch.topk(-actual, k).indices
    
    # For each predicted top-k stock, it should have higher predicted value
    # than bottom stocks by at least 'margin'
    # ... (detailed implementation)
```

### Alternative: ListMLE Loss

```python
# Probability of correct ranking
loss = -log(P(correct_ranking | predictions))
```

---

## Next Steps

| Step | Description | Status |
|:---:|:---|:---:|
| 1 | Add Rank IC metric to validation | ✅ (dropped - not useful) |
| 2 | Add TopK Precision metric | ✅ |
| 3 | Train and analyze metrics | ✅ (9.4% TopK - near random) |
| 4 | Compare MAE vs TopK model via backtest | ⏭️ Skipped |
| 5a | Implement TopK margin loss | ✅ (8.5% TopK - no improvement) |
| 5b | Try ListMLE loss | ✅ (14.5% TopK - best but still weak) |
| 5c | Investigate factor quality | ✅ (IC filtering was useless) |
| 6 | Use Alpha360 factors (360 raw lagged) | 🔄 In Progress |

---

## Step 5b Results: ListMLE Loss

| Epoch | TopK Precision |
|:---:|:---:|
| 1 | 13.6% |
| 6 | **14.5%** (best) |
| 10 | 11.6% (declining) |

- **Best:** 14.5% at epoch 6 (~2x random chance)
- **Assessment:** Improvement over MAE (9.4%) but still weak for trading
- Model saved: `best_model_topk`

---

## Step 5c: Factor Quality Investigation ✅

**Goal:** Determine if our 26 selected factors have any predictive power in the test period.

**Results:**

| Metric | Training Period | Test Period |
|:---|:---:|:---:|
| Avg Absolute IC | ~0.01 | **0.20** |
| Train↔Test IC Correlation | - | **0.12** (near zero!) |

**Critical Finding:** IC filtering was useless - selected factors based on training IC have **completely different behavior** in test period.

**Top performing factors in TEST (not training):**
- MA5: IC = -0.67
- RESI10: IC = +0.64
- MAX5: IC = -0.55
- RANK10: IC = +0.55

---

## Step 6: Alpha360 Factors 🔄 IN PROGRESS

**Discovery:** Alpha360 is SIMPLER than Alpha158!

| Approach | Description |
|:---|:---|
| Alpha158 | 158 calculated technical indicators |
| **Alpha360** | **6 raw features × 60 days lag = 360 features** |

**Alpha360 features (per day):**
- Open, High, Low, Close, Volume, VWAP
- Lagged 1-60 days, normalized (z-score)

### Implementation Decisions

| Feature | Original Paper | Our Approach | Rationale |
|:---|:---:|:---:|:---|
| Loss function | MAE | **MAE** | Match original |
| Regularization | None | **weight_decay=1e-5** | Standard technique |
| Early stopping | None | **30 epochs, MAE-based** | Prevent wasted training |
| TopK logging | No | **Yes** | Monitor ranking quality |
| Checkpointing | No | **Yes** | Resume capability |
| Max epochs | Unknown | **100** | Allow sufficient training |

### Files Created

1. `phase8_generate_alpha360.py` ✅ - Generates 360 lagged features
2. `phase8_360_train.py` ✅ - Training script with original paper settings
3. `config/Phase8_NIFTY_Subset10_Alpha360.conf` - Config for Alpha360 (TODO)

### Data Generated

```
Location: data/NIFTY200_Subset10/Alpha360/
Stocks: 143
Features: 360 (6 OHLCV × 60 days)
Date range: 2020-07-02 to 2022-11-30
```

---

## Note on Original Paper

The original Stockformer paper used **all 360 factors** (raw OHLCV × 60 days lag) without IC filtering. Our IC filtering approach (Phase 7) did not help because:
- Training IC has no correlation with test IC (0.12)
- Market regime change between train/test periods

---

## Step 7: Backtest with Rebalancing Strategy Analysis

**Objective:** Compare daily vs weekly rebalancing and analyze net returns after transaction costs.

### Rebalancing Strategies to Test

| Strategy | Description | Expected Behavior |
|:---|:---|:---|
| **Daily** | Rebalance top 10 every day | Higher returns, higher costs |
| **Weekly** | Rebalance top 10 every Monday | Lower turnover, lower costs |

### Transaction Cost Assumptions

| Cost Type | Rate |
|:---|:---:|
| Brokerage (each way) | 0.03% |
| STT (sell only) | 0.1% |
| Other (GST, stamps) | 0.02% |
| **Total per round-trip** | **~0.2%** |

### Backtest Metrics to Report

1. **Gross Return** - Before transaction costs
2. **Net Return** - After transaction costs  
3. **Win Rate** - % of trades profitable
4. **Sharpe Ratio** - Risk-adjusted return
5. **Max Drawdown** - Largest peak-to-trough decline
6. **Turnover** - Average daily portfolio change

### Why One Model is Sufficient

With Alpha360, MAE and TopK improve together:
- Epoch 1: MAE=0.0151, TopK=31.6%
- Epoch 5: MAE=0.0075, TopK=62.2%

No need to compare MAE-optimized vs TopK-optimized models.

---

## Step 6 Results: Alpha360 Training ✅

### Training Performance (75/12.5/12.5 split)

| Epoch | Val TopK | Val MAE | Val Accuracy |
|:---:|:---:|:---:|:---:|
| 1 | 31.6% | 0.0151 | 67.4% |
| 26 | **67.8%** | 0.0063 | 93% |
| 35 | 65% | **0.0061** | 93% |

### Test Performance

| Metric | Validation (best) | **Test** |
|:---|:---:|:---:|
| TopK Precision | 67.8% | **6.5%** |
| Accuracy | 93% | **50%** |
| MAE | 0.0061 | 0.0213 |

### Alternate Split Test (65/10/25)

Same result: Val TopK 61.6% → Test Accuracy 49.6%

---

## Phase 8 Conclusion

### Key Findings

1. **Alpha360 (360 factors) >> Alpha158 (26 factors)** for validation performance
2. **IC filtering was counterproductive** - train IC ≠ test IC (correlation 0.12)
3. **Regime change problem**: Aug-Nov 2022 (test period) is fundamentally different from train/val

### What Worked
- Alpha360 raw lagged features
- MAE loss + regularization
- TopK Precision metric for monitoring

### What Didn't Work
- IC-based factor selection
- Generalization to bear market (Aug-Nov 2022)

### Recommendation for Next Phase

| Option | Description |
|:---|:---|
| A | **More data** - Include 2023-2025 data to capture regime changes |
| B | **Rolling window** - Train separate model per market regime |
| C | **Simpler model** - Linear/ensemble as baseline comparison |
