# Phase 9: Live Trading Window Implementation Plan

**Date:** 2026-01-13  
**Objective:** Train Stockformer on Subset 22 for live trading predictions  
**Key Change:** Use Alpha360 factors (from Phase 8 learnings)

---

## Subset 22 Configuration

| Phase | Period | Est. Trading Days |
|:------|:-------|:------------------|
| **Training** | Jun 2023 - May 2025 | ~500 days |
| **Validation** | Jun 2025 - Sep 2025 | ~80 days |
| **Test/Live** | Oct 2025 - Jan 2026 | ~70 days |

**Market Context:** 🟢 Bull market, pre-budget period

---

## Phase 8 Decisions Carried Forward

| Decision | Value | Rationale |
|:---------|:------|:----------|
| Factors | **Alpha360** (not Alpha158) | Better validation performance |
| Factor filtering | **None** | IC filtering was counterproductive |
| Loss function | **MAE** | Matches original paper |
| Regularization | **weight_decay=1e-5** | Standard technique |
| TopK logging | **Yes** | Monitor ranking quality |
| Checkpointing | **Yes** | Resume capability |

---

## Implementation Steps

### Step 1: Get F&O Universe (Jun 2023)
**Script:** `phase9_get_fno_universe.py`

```python
from jugaad_data.nse import bhavcopy_fo_save
from datetime import date

# Download F&O bhavcopy for period start (Jun 2023)
bhavcopy_fo_save(date(2023, 6, 1), "./data/fo01Jun2023bhav.csv")
# Extract FUTSTK symbols → fno_universe_2023_06.txt
```

**Output:** `data/NIFTY200_Subset22/fno_universe_2023_06.txt`

### Step 2: Download Price Data
**Script:** `phase9_zerodha_data_fetcher.py` (adapt from Phase 2)

**Period:** 2023-06-01 to 2026-01-10 (~650 days)  
**Source:** Zerodha (historical) + jugaad-data (recent)

**Output:** `data/NIFTY200_Subset22/raw/{SYMBOL}.csv`

### Step 3: Generate Alpha360 Factors
**Script:** `phase8_generate_alpha360.py` (reuse, update paths)

**Process:**
1. Load raw OHLCV from Step 2
2. Generate 360 lagged features (6 × 60 days)
3. Save to `data/NIFTY200_Subset22/Alpha360/`

### Step 4: Preprocessing
**Script:** `phase8_preprocessing.py` (reuse, update paths)

**Generate:**
- `flow.npz` - Returns data
- `trend_indicator.npz` - Binary up/down
- `corr_adj.npy` - Correlation matrix
- Graph embeddings

**Output:** `data/NIFTY200_Subset22/dataset/`

### Step 5: Create Config
**File:** `config/Phase9_NIFTY_Subset22.conf`

```ini
[file]
traffic = ./data/NIFTY200_Subset22/dataset/flow.npz
indicator = ./data/NIFTY200_Subset22/dataset/trend_indicator.npz
factor_dir = ./data/NIFTY200_Subset22/Alpha360
model = ./output/NIFTY200_Subset22/best_model

[data]
dataset = NIFTY200_Subset22
train_ratio = 0.75
val_ratio = 0.125
test_ratio = 0.125
```

### Step 6: Training
**Script:** `phase8_360_train.py` --config Phase9 config

**Settings:**
- Max epochs: 100
- Early stopping: 30 epochs (MAE-based)
- Interactive: Check every 10 epochs

### Step 7: Backtest & Live Trading
**Script:** `phase8_backtest.py` (adapt for Subset22)

**Strategies to test:**
- Daily rebalancing
- Weekly rebalancing
- Compare net returns after 0.2% transaction costs

---

## Directory Structure

```
data/NIFTY200_Subset22/
├── fno_universe_2023_06.txt     # Step 1
├── raw/                          # Step 2 (OHLCV CSVs)
├── Alpha360/                     # Step 3 (360 factor CSVs)
├── dataset/                      # Step 4 (flow.npz, etc.)
└── instruments/                  # Stock list

output/NIFTY200_Subset22/
├── best_model                    # Step 6
├── training.log
├── regression_pred.csv           # Step 7
└── backtest_results/
```

---

## Success Criteria

| Metric | Target | Phase 8 Result |
|:-------|:-------|:---------------|
| Val TopK Precision | >50% | 67.8% |
| Test TopK Precision | >20% | 6.5% (regime issue) |
| Test Accuracy | >52% | 50% |

**Key Hypothesis:** Subset 22 (bull market train → bull market test) should show better test generalization than Subset 10 (bull train → bear test).

---

## Timeline

| Step | Effort | Notes |
|:-----|:-------|:------|
| 1. F&O Universe | 30 min | jugaad-data |
| 2. Price Data | 1-2 hours | Zerodha API |
| 3. Alpha360 | 10 min | Reuse script |
| 4. Preprocessing | 10 min | Reuse script |
| 5. Config | 5 min | Copy + modify |
| 6. Training | 2-3 hours | 100 epochs max |
| 7. Backtest | 30 min | Analyze results |

**Total:** ~1 day
