# Phase 8: Bear Market Testing Implementation Plan

**Date:** 2026-01-11  
**Objective:** Test Stockformer on bear market period (Subset 10: Aug-Nov 2022) using historical F&O universe  
**Target:** Validate model performance in non-bull market conditions

---

## Goals

1. **Primary**: Train and backtest model on bear market period (Subset 10)
2. **Secondary**: Implement historical F&O universe selection via Bhavcopy
3. **Tertiary**: Compare bear market alpha vs bull market alpha (current Phase 7 results)
4. **Validation**: Confirm model has predictive power across different market regimes

---

## Prerequisites (Phase 7 Complete)

- ✅ Current model trained on bull market period (May-Jul 2024): **+18.38% alpha, Sharpe 2.58**
- ✅ Pipeline established: Data → Factors → Training → Inference → Backtest
- ✅ Zerodha API access working

---

## Subset 10 Configuration

| Phase | Period | Trading Days (Est.) |
|-------|--------|---------------------|
| **Training** | Apr 2020 - Mar 2022 | ~500 days |
| **Validation** | Apr 2022 - Jul 2022 | ~80 days |
| **Test** | Aug 2022 - Nov 2022 | ~80 days |

**Market Context (Test Period)**:
- 🟡 Bear → Recovery phase
- NIFTY fell ~15% from Oct 2021 peak to Jun 2022
- Global rate hike pressure, FII outflows
- Elevated volatility compared to 2024 bull run

---

## Implementation Steps

### Step 1: Install F&O Bhavcopy Tools
**Package:** `jugaad-data` (recommended) or `bhavcopy`

```bash
pip install jugaad-data
# OR
pip install bhavcopy
```

**Purpose:** Download F&O bhavcopy to get historical F&O stock universe

### Step 2: Get Historical F&O Universe
**Script:** `phase8_get_fno_universe.py`

```python
from jugaad_data.nse import bhavcopy_fo_save
from datetime import date

# Download F&O bhavcopy for period start (Apr 2020)
bhavcopy_fo_save(date(2020, 4, 1), "./data/fno_bhavcopy_2020_04_01.csv")

# Extract unique stock symbols from FUTSTK instrument type
# This gives us the F&O universe as of Apr 2020
```

**Output:** `data/NIFTY200/fno_universe_2020_04.txt` (list of ~180 F&O eligible stocks)

**Success Criteria:**
- [ ] Downloaded bhavcopy file successfully
- [ ] Extracted 150-200 unique stock symbols
- [ ] Saved to instruments file

### Step 3: Download Historical Price Data
**Script:** Modify `Phase_2_Zerodha_Data_Fetcher.ipynb`

**Period:** 2020-04-01 to 2022-11-30 (~660 days)

**Input:** F&O universe from Step 2

**Output:** `data/NIFTY200_Subset10/raw/{SYMBOL}.csv`

**Success Criteria:**
- [ ] 80% data coverage per stock (same as original Phase 2)
- [ ] 150+ stocks with sufficient data
- [ ] No corporate action anomalies

### Step 4: Factor Engineering (Alpha158)
**Script:** Reuse `alpha158_pandas.py` and Phase 3 notebook

**Process:**
1. Calculate 158 factors for new period
2. Apply IC filtering (|IC| >= 0.02)
3. Apply size & sector neutralization
4. Save filtered factors to `data/NIFTY200_Subset10/Alpha_158_2020-04-01_2022-11-30/`

**Success Criteria:**
- [ ] 15-30 factors survive IC filtering (similar to Phase 3's 22)
- [ ] No look-ahead bias in factor calculation

### Step 5: Wavelet Transform & Graph Embeddings
**Script:** Reuse Phase 4 preprocessing

**Output:** `data/NIFTY200_Subset10/Stock_NIFTY_2020-04-01_2022-11-30/`
- `flow.npz` (wavelet decomposed)
- `trend_indicator.npz` (binary labels)
- `corr_adj.npy` (correlation matrix)
- `128_corr_struc2vec_adjgat.npy` (graph embeddings)

**Success Criteria:**
- [ ] All files generated without errors
- [ ] Dimensions match stock count

### Step 6: Create Subset 10 Config
**File:** `config/Multitask_NIFTY200_Subset10.conf`

Based on existing config, update paths and date range.

**Key Parameters (unchanged):**
- T1=20, T2=2
- train_ratio=0.75, val_ratio=0.125, test_ratio=0.125
- dims=128, layers=2, heads=1

### Step 7: Train Model
**Command:**
```bash
python MultiTask_Stockformer_train.py --config config/Multitask_NIFTY200_Subset10.conf --cuda 0
```

**Success Criteria:**
- [ ] Model converges (60-80 epochs)
- [ ] Val MAE < 0.020 (may be higher in volatile period)
- [ ] Classification accuracy > 50%

### Step 8: Run Inference
**Command:**
```bash
python run_inference.py --config config/Multitask_NIFTY200_Subset10.conf
```

**Output:** `output/Multitask_output_Subset10/`

### Step 9: Backtest (Phase 7 Scripts)
**Reuse:** `phase_7_topk_strategy.py`, `phase_7_visualization.py`

**Configuration:**
- TopK = 10
- Rebalance = Weekly
- Transaction costs = 0.2% round-trip (Zerodha)

**Success Criteria:**
- [ ] Backtest runs without errors
- [ ] All metrics calculated

### Step 10: Compare Results
**Create:** `Phase_8_Comparison_Report.md`

| Metric | Subset 10 (Bear) | Current (Bull) | Significance |
|--------|------------------|----------------|--------------|
| Total Return | ? | 13.87% | |
| Alpha vs NIFTY-50 | ? | +18.38% | |
| Sharpe Ratio | ? | 2.58 | |
| Max Drawdown | ? | -6.87% | |
| Win Rate | ? | 62.30% | |

**Success Criteria:**
- [ ] Alpha > 0% (model adds value above benchmark)
- [ ] Sharpe > 0.5 (reasonable risk-adjusted returns)
- [ ] Win Rate > 50% (better than random)

---

## Verification Checklist

### Data Integrity
- [ ] F&O universe downloaded from correct date (Apr 2020)
- [ ] No stocks from post-2020 IPOs in universe
- [ ] 80% data coverage enforced

### No Look-Ahead Bias  
- [ ] Factors computed only from past data
- [ ] F&O universe from period START, not end
- [ ] Benchmark data aligned correctly

### Reproducibility
- [ ] Random seed set (42)
- [ ] Same preprocessing as Phase 3-5
- [ ] Same model architecture

---

## Decision Framework

**Result A: Alpha > 5%, Sharpe > 1.0**
```
✅ STRONG SIGNAL
- Model works in both bull and bear markets
- Proceed to 14-window ensemble (Phase 9)
```

**Result B: Alpha 0-5%, Sharpe 0.5-1.0**
```
⚠️ MODERATE SIGNAL  
- Model has some predictive power in bear markets
- Consider: Expand factors, tune hyperparameters
```

**Result C: Alpha < 0%, Sharpe < 0.5**
```
❌ WEAK SIGNAL
- Model may be overfit to bull market patterns
- Investigate: Factor behavior, market regime detection
```

---

## Timeline Estimate

| Step | Duration | Dependencies |
|------|----------|--------------|
| 1. Install packages | 5 min | None |
| 2. Get F&O universe | 30 min | Step 1 |
| 3. Download data | 2-4 hours | Step 2, Zerodha token |
| 4. Factor engineering | 1-2 hours | Step 3 |
| 5. Preprocessing | 30 min | Step 4 |
| 6. Create config | 15 min | Step 5 |
| 7. Train model | 4-6 hours | Step 6 |
| 8. Inference | 15 min | Step 7 |
| 9. Backtest | 30 min | Step 8 |
| 10. Analysis | 1 hour | Step 9 |
| **Total** | **10-15 hours** | |

---

## Files to Create

| File | Purpose |
|------|---------|
| `phase8_get_fno_universe.py` | Download F&O bhavcopy, extract universe |
| `config/Multitask_NIFTY200_Subset10.conf` | Training configuration |
| `data/NIFTY200_Subset10/` | All subset data |
| `Phase_8_Comparison_Report.md` | Final results comparison |

---

**Status:** 📋 PLAN READY - Awaiting approval to proceed
