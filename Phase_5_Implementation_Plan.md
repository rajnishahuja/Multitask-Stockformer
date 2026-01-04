# Phase 5: Model Configuration - Implementation Plan

**Date Created:** 2026-01-03  
**Current Status:** Phase 4 Complete → Ready for Phase 5  
**Objective:** Configure Multitask-Stockformer model for NIFTY-200 with 22 Alpha158 factors

---

## 📋 PHASE 5 OVERVIEW

### What Phase 5 Does:
Configure the Stockformer model and dataset loader to work with our NIFTY-200 preprocessed data (22 factors instead of original 360 factors, 191 stocks instead of 255).

### Inputs Required (from Phase 4):
```
data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/
├── flow.npz                           # Wavelet-decomposed factors [~335, 191, 22, 2]
├── trend_indicator.npz                # Binary up/down labels [660, 191]
├── corr_adj.npy                       # Correlation adjacency matrix [191, 191]
├── 128_corr_struc2vec_adjgat.npy     # Graph embeddings [191, 128]
└── label.csv                          # Daily returns [660, 191]

data/NIFTY200/Alpha_158_2022-01-01_2024-08-31/
├── selected_factors.txt               # List of 22 factor names
├── BETA20.csv                         # [660 dates × 191 stocks]
├── BETA60.csv
├── ... (20 more factor CSVs)
```

### Outputs Expected:
```
config/
└── Multitask_NIFTY200_Alpha158.conf   # Model configuration file

lib/
└── Multitask_Stockformer_utils.py     # Modified dataset loader (updated)

Verification:
- Config file points to correct data paths
- Dataset loader can load all files without errors
- Tensor shapes match model expectations
- Ready for training (Phase 6)
```

---

## 🔍 ALIGNMENT WITH ORIGINAL PROJECT

### What the Original Multitask-Stockformer Does:
1. **Dual-Frequency Architecture:** Separates price data into low/high frequency using wavelets
2. **Multi-Task Learning:** Simultaneously predicts:
   - **Regression:** Next-day returns (continuous values)
   - **Classification:** Up/down trend (binary)
3. **Graph Attention:** Uses stock correlation network + embeddings for spatial relationships
4. **Attention Mechanisms:** Temporal (self-attention) + Spatial (GAT) for feature fusion

### Where We Differ (Adaptations):
| Aspect | Original (CSI-300) | Our NIFTY-200 Adaptation | Reason |
|--------|-------------------|-------------------------|---------|
| **Stock Universe** | 255 stocks (CSI-300) | 191 stocks (NIFTY-200) | Market availability |
| **Factors** | 360 Alpha360 factors | 22 Alpha158 factors (IC filtered) | Quality over quantity - removed weak predictors |
| **Time Period** | 2021-06-04 to 2024-01-30 | 2022-01-01 to 2024-08-31 | Data availability |
| **Market** | Chinese A-shares | Indian equities | Different market |
| **Factor Engineering** | Qlib pipeline | Custom pandas + IC filtering | Same formulas, different execution |

### Where We Match (No Changes):
- **Architecture:** Same model structure (2 layers, 128 dims, 1 head)
- **Wavelet:** Same Symlet-2, level=1 decomposition
- **Training:** Same optimizer (Adam), same loss functions (MAE + CrossEntropy)
- **Strategy:** Same TopK selection approach
- **Splits:** Same 75%/12.5%/12.5% train/val/test ratio

---

## ✨ IMPROVEMENTS BEYOND ORIGINAL

### What We're Doing Better:

1. **IC-Based Factor Selection (NEW)**
   - **Original:** Used ALL 360 Alpha360 factors without filtering
   - **Ours:** Applied IC filtering (|IC| >= 0.02) → retained only 22/158 factors
   - **Benefit:** Removes noise, faster training, better generalization
   - **Evidence:** 13.9% survival rate aligns with quant research best practices

2. **Comprehensive Data Quality Control (NEW)**
   - **Original:** Assumed clean data from commercial providers
   - **Ours:** Implemented multi-stage validation pipeline:
     - Missing date checks
     - Corporate action verification (verified NMDC 1:2 bonus)
     - Price gap detection (>20% threshold)
     - Volume consistency checks
   - **Benefit:** Higher confidence in data integrity

3. **Historical Size Proxy (NEW)**
   - **Original:** Used current market cap (potential look-ahead bias)
   - **Ours:** log(close × rolling_volume_60d) as size proxy
   - **Benefit:** Avoids look-ahead bias, more realistic historical neutralization

4. **Transparent Factor Engineering (NEW)**
   - **Original:** Black-box preprocessing (uploaded pre-processed files)
   - **Ours:** Open-source pandas implementation with formula documentation
   - **Benefit:** Full reproducibility, easier debugging and customization

5. **Documented Adaptation Process (NEW)**
   - **Original:** Minimal documentation on data preparation
   - **Ours:** Comprehensive markdown docs at each phase
   - **Files:** 
     - `Alpha158_IC_Filtering_Analysis.md`
     - `nifty-adaptationchat.txt` (task tracker)
     - Phase notebooks with detailed comments

### What We're NOT Changing (Yet):
- Model architecture (layers=2, dims=128, heads=1)
- Hyperparameters (lr=0.001, batch_size=12)
- Loss functions (masked MAE + CrossEntropy)
- Strategy (TopK, equal weight, weekly rebalancing)

**Rationale:** Get MVP working first with proven architecture, then experiment with improvements in future phases.

---

## 📥 INPUTS NEEDED FOR PHASE 5

### From Phase 4 (Data Files):
✅ All files should exist in `data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/`

1. **flow.npz** (Wavelet coefficients)
   - Keys: `low_freq`, `high_freq`
   - Shape: `[~335 time steps, 191 stocks, 22 factors]` each
   - Purpose: Low/high frequency components for dual-frequency encoder

2. **trend_indicator.npz** (Binary labels)
   - Key: `trend`
   - Shape: `[660, 191]`
   - Purpose: Classification task targets (up=1, down=0)

3. **corr_adj.npy** (Adjacency matrix)
   - Shape: `[191, 191]`
   - Purpose: Stock correlation network for GAT layer
   - Sparse: Only correlations with |r| >= 0.3

4. **128_corr_struc2vec_adjgat.npy** (Graph embeddings)
   - Shape: `[191, 128]`
   - Purpose: Spatial positional encoding in attention mechanism

5. **label.csv** (Returns for regression)
   - Shape: `[660 dates, 191 stocks]`
   - Purpose: Regression task targets

### From Phase 3 (Factor Files):
✅ All files in `data/NIFTY200/Alpha_158_2022-01-01_2024-08-31/`

6. **22 Factor CSVs** (e.g., BETA20.csv, STD20.csv, etc.)
   - Shape: `[660 dates × 191 stocks]` per factor
   - Purpose: Additional features loaded separately by dataset loader

7. **selected_factors.txt**
   - Purpose: List of 22 factor names for loader to iterate

### Metadata:
- **num_stocks:** 191
- **num_factors:** 22 (not 360)
- **num_dates:** 660 (original), ~335 (after wavelet)
- **T1 (lookback):** 20 days
- **T2 (prediction):** 2 days ahead

---

## 📤 OUTPUTS EXPECTED FROM PHASE 5

### 1. Configuration File
**File:** `config/Multitask_NIFTY200_Alpha158.conf`

**Purpose:** Centralized settings for model, data, and training

**Structure:**
```ini
[file]
# Data paths
traffic = ./data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/flow.npz
indicator = ./data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/trend_indicator.npz
adj = ./data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/corr_adj.npy
adjgat = ./data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/128_corr_struc2vec_adjgat.npy

# Factor directory (NEW - not in original)
factor_dir = ./data/NIFTY200/Alpha_158_2022-01-01_2024-08-31

# Model save paths
model = ./cpt/NIFTY200/saved_model_Alpha158_MVP
log = ./log/NIFTY200/log_Alpha158_MVP

[data]
dataset = NIFTY200
T1 = 20           # Lookback window
T2 = 2            # Prediction horizon
train_ratio = 0.75
val_ratio = 0.125
test_ratio = 0.125

[train]
cuda = 0
max_epoch = 100
batch_size = 12
learning_rate = 0.001
seed = 42

[param]
layers = 2        # Encoder layers
heads = 1         # Attention heads
dims = 128        # Hidden dimensions
samples = 1       # Monte Carlo samples
wave = sym2       # Wavelet type
level = 1         # Wavelet decomposition level
```

### 2. Modified Dataset Loader
**File:** `lib/Multitask_Stockformer_utils.py`

**Changes Required:**
1. **Line 117:** Replace hardcoded Alpha_360 path with config parameter
2. **Line 143:** Update `self.infea` calculation for 22 factors instead of 360
3. Add dynamic stock count detection (191 instead of 255)

**Before:**
```python
path = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/Stock_CN_2021-06-04_2024-01-30/Alpha_360_2021-06-04_2024-01-30'
...
self.infea = bonus_all.shape[-1] + 2  # Hardcoded for 360 factors
```

**After:**
```python
path = args.factor_dir  # From config file
...
self.infea = bonus_all.shape[-1] + 2  # Works for any number of factors (22 in our case)
```

### 3. Verification Script
**File:** `verify_phase5_config.py` (NEW)

**Purpose:** Test configuration and data loading before training

**Tests:**
- ✅ All data files exist at specified paths
- ✅ Tensor shapes match expected dimensions
- ✅ No NaN/Inf values in loaded data
- ✅ Dataset can iterate through all samples
- ✅ Model input feature count matches data

**Expected Output:**
```
✓ Config file loaded successfully
✓ All 5 data files exist
✓ flow.npz: low_freq [335, 191, 22], high_freq [335, 191, 22]
✓ trend_indicator.npz: [660, 191]
✓ corr_adj.npy: [191, 191] (sparse: 92.3%)
✓ adjgat.npy: [191, 128]
✓ Factor CSVs: 22 files × [660, 191]
✓ Dataset loader: 641 train samples, 80 val samples, 80 test samples
✓ Feature dimension: 24 (22 factors + 2 temporal embeddings)
✓ Ready for training!
```

---

## 🛠️ IMPLEMENTATION TASKS

### Task 5.1: Review Model Architecture (RESEARCH)
**Status:** ⏳ NOT STARTED  
**Purpose:** Understand model components before configuration

**Files to Review:**
1. `Stockformermodel/Multitask_Stockformer_models.py` (main architecture)
2. `lib/graph_utils.py` (GAT implementation)

**Key Components to Understand:**
- **DecouplingFlowLayer:** How wavelet components are processed
- **DualFrequencyEncoder:** Low/high frequency encoding
- **FusionDecoder:** How features are combined for prediction
- **spatialAttention:** How graph embeddings (adjgat) are used
- **Multi-task heads:** Regression vs classification outputs

**Deliverable:** Notes on how input dimensions flow through model

---

### Task 5.2: Create Configuration File
**Status:** ⏳ NOT STARTED  
**File:** `config/Multitask_NIFTY200_Alpha158.conf`

**Steps:**
1. Copy `config/Multitask_Stock.conf` as template
2. Update `[file]` section:
   - Change paths from `Stock_CN_2021-06-04_2024-01-30` → `Stock_NIFTY_2022-01-01_2024-08-31`
   - Add `factor_dir` parameter (NEW)
3. Update `[data]` section:
   - Change `dataset = STOCK` → `dataset = NIFTY200`
   - Keep T1=20, T2=2, splits same
4. Update `[train]` section:
   - Change `seed = 1` → `seed = 42` (reproducibility)
5. Keep `[param]` section unchanged (proven hyperparameters)

**Validation:**
- Run ConfigParser to ensure valid INI format
- Check all paths resolve correctly

**Deliverable:** Working config file

---

### Task 5.3: Modify Dataset Loader
**Status:** ⏳ NOT STARTED  
**File:** `lib/Multitask_Stockformer_utils.py`

**Changes Required:**

#### Change 1: Dynamic Factor Path (Line 117)
**Current (Hardcoded):**
```python
path = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/Stock_CN_2021-06-04_2024-01-30/Alpha_360_2021-06-04_2024-01-30'
```

**Fixed (Config-driven):**
```python
path = args.factor_dir  # From config [file] section
```

#### Change 2: Dynamic Factor Count (Already OK)
**Current (Line 143):**
```python
self.infea = bonus_all.shape[-1] + 2  # Last dimension of bonus_all plus 2
```

**Status:** ✅ Already correct! This dynamically calculates:
- `bonus_all.shape[-1]` = 22 factors (from CSVs)
- `+ 2` = day-of-year and time-of-day embeddings
- `self.infea = 24` (correct for our data)

#### Change 3: Update Config Parser
**Add to `__init__` arguments:**
```python
def __init__(self, args, mode='train'):
    # Add attribute for factor directory
    self.factor_dir = args.factor_dir if hasattr(args, 'factor_dir') else None
```

**Validation:**
- Test with Phase 4 outputs
- Verify shapes: X[641, 20, 191], bonus_X[641, 20, 191, 22]
- Check no crashes during iteration

**Deliverable:** Modified `Multitask_Stockformer_utils.py`

---

### Task 5.4: Create Verification Script
**Status:** ⏳ NOT STARTED  
**File:** `verify_phase5_config.py` (NEW)

**Purpose:** Pre-flight checks before training

**Script Structure:**
```python
import configparser
import os
import numpy as np
import pandas as pd
from lib.Multitask_Stockformer_utils import StockDataset
import argparse

def verify_config(config_path):
    # 1. Load config
    # 2. Check all [file] paths exist
    # 3. Load each file and verify shapes
    # 4. Check for NaN/Inf
    # 5. Test dataset loader
    # 6. Verify feature dimensions
    # 7. Print summary report
```

**Tests:**
1. **File Existence:** All 5 data files + 22 factor CSVs exist
2. **Shape Validation:**
   - `flow.npz`: low_freq and high_freq both [~335, 191, 22]
   - `trend_indicator.npz`: [660, 191]
   - `corr_adj.npy`: [191, 191]
   - `adjgat.npy`: [191, 128]
   - Each factor CSV: [660, 191]
3. **Data Quality:**
   - No NaN in flow data (except masked values)
   - No Inf anywhere
   - corr_adj diagonal = 1.0
   - adjgat not all zeros
4. **Dataset Loader:**
   - Can instantiate train/val/test datasets
   - Correct number of samples per split
   - Can iterate through full epoch
   - Batch shapes correct
5. **Feature Dimension:**
   - `self.infea = 24` (22 factors + 2 temporal)
   - Matches model expected input

**Expected Output:**
```
=== Phase 5 Configuration Verification ===

[1] Config File
  ✓ Loaded: config/Multitask_NIFTY200_Alpha158.conf
  ✓ All sections present: [file], [data], [train], [param]

[2] Data Files
  ✓ flow.npz exists (14.2 MB)
    - low_freq: [335, 191, 22]
    - high_freq: [335, 191, 22]
  ✓ trend_indicator.npz exists (242 KB)
    - trend: [660, 191]
  ✓ corr_adj.npy exists (285 KB)
    - shape: [191, 191], sparsity: 92.3%
  ✓ 128_corr_struc2vec_adjgat.npy exists (190 KB)
    - shape: [191, 128]

[3] Factor Files
  ✓ Factor directory: data/NIFTY200/Alpha_158_2022-01-01_2024-08-31/
  ✓ Found 22 factor CSVs (expected 22)
  ✓ All shapes match: [660, 191]

[4] Data Quality
  ✓ No Inf values detected
  ✓ NaN handling: masked properly in loss functions
  ✓ corr_adj diagonal all 1.0
  ✓ adjgat has non-zero embeddings

[5] Dataset Loader
  ✓ Train dataset: 641 samples
  ✓ Val dataset: 80 samples
  ✓ Test dataset: 80 samples
  ✓ Feature dimension: 24 (22 factors + 2 temporal)
  ✓ Sample batch shape: X[12, 20, 191], bonus_X[12, 20, 191, 24]

[6] Model Readiness
  ✓ Input feature count: 24
  ✓ Stock dimension: 191
  ✓ Graph embedding dimension: 128
  ✓ All dimensions compatible with model architecture

=== VERIFICATION PASSED ===
Ready to proceed to Phase 6: Training
```

**Deliverable:** `verify_phase5_config.py` with comprehensive checks

---

### Task 5.5: Update Training Script (MINOR)
**Status:** ⏳ NOT STARTED  
**File:** `MultiTask_Stockformer_train.py`

**Changes Required:**
- Update argument parser to accept `factor_dir` from config
- Pass `factor_dir` to dataset loader

**Before:**
```python
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/Multitask_Stock.conf')
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()
```

**After (Add factor_dir):**
```python
# Load config
config = configparser.ConfigParser()
config.read(args.config)

# Parse [file] section
args.traffic_file = config['file']['traffic']
args.indicator_file = config['file']['indicator']
args.adj_file = config['file']['adj']
args.adjgat_file = config['file']['adjgat']
args.factor_dir = config['file']['factor_dir']  # NEW
```

**Deliverable:** Training script that reads factor_dir from config

---

### Task 5.6: Documentation Update
**Status:** ⏳ NOT STARTED  
**File:** `nifty-adaptationchat.txt`

**Update Phase 5 Section:**
```
## PHASE 5: Model Configuration

### Task 5.1: Review model architecture
**Status:** ✅ COMPLETED
**Notes:** [Key findings from architecture review]

### Task 5.2: Create configuration file
**Status:** ✅ COMPLETED
**File:** config/Multitask_NIFTY200_Alpha158.conf
**Changes:** Updated paths for NIFTY200, added factor_dir parameter

### Task 5.3: Modify dataset loader
**Status:** ✅ COMPLETED
**File:** lib/Multitask_Stockformer_utils.py (modified)
**Changes:** Line 117 - factor_dir from config instead of hardcoded

### Task 5.4: Create verification script
**Status:** ✅ COMPLETED
**File:** verify_phase5_config.py
**Result:** All checks passed, ready for training

### Task 5.5: Update training script
**Status:** ✅ COMPLETED
**Changes:** Added factor_dir argument parsing

### Task 5.6: Documentation
**Status:** ✅ COMPLETED
```

**Deliverable:** Updated task tracker

---

## 📊 PHASE 5 CHECKLIST

### Pre-Flight Checks (Before Starting):
- [ ] Phase 4 completed successfully
- [ ] All 5 required data files exist in `data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/`
- [ ] 22 factor CSVs exist in `data/NIFTY200/Alpha_158_2022-01-01_2024-08-31/`
- [ ] Git branch created for Phase 5 changes

### Task Completion:
- [ ] Task 5.1: Architecture review complete
- [ ] Task 5.2: Config file created and tested
- [ ] Task 5.3: Dataset loader modified and tested
- [ ] Task 5.4: Verification script created and passes
- [ ] Task 5.5: Training script updated
- [ ] Task 5.6: Documentation updated

### Final Validation:
- [ ] `verify_phase5_config.py` runs without errors
- [ ] All file paths resolve correctly
- [ ] Dataset loader can iterate through all splits
- [ ] Feature dimensions match model expectations (24 features)
- [ ] No hardcoded paths remain in code
- [ ] Git commit with descriptive message

### Ready for Phase 6 When:
- [ ] All checklist items ✅
- [ ] Verification script output = "VERIFICATION PASSED"
- [ ] Team review completed (if applicable)

---

## 🚀 EXECUTION ORDER

**Recommended Sequence:**
1. **Day 1:** Task 5.1 (Architecture review - 2 hours)
2. **Day 1:** Task 5.2 (Config file - 1 hour)
3. **Day 2:** Task 5.3 (Dataset loader - 2 hours)
4. **Day 2:** Task 5.4 (Verification script - 2 hours)
5. **Day 2:** Task 5.5 (Training script - 30 min)
6. **Day 2:** Task 5.6 (Documentation - 30 min)
7. **Day 2:** Run full verification and checklist

**Total Estimated Time:** 8 hours over 2 days

---

## ⚠️ CRITICAL NOTES

### Common Pitfalls to Avoid:
1. **Hardcoded Paths:** Always use config parameters, never hardcode paths
2. **Shape Mismatches:** Verify tensor shapes at each step (191 stocks, 22 factors)
3. **Missing factor_dir:** Training will crash if factor_dir not in config
4. **Wrong Data Files:** Ensure using Phase 4 outputs, not original Chinese data

### Debugging Tips:
- If dataset loader fails: Check `bonus_all` shape = [660, 191, 22]
- If model crashes: Verify `self.infea = 24` in dataset
- If config errors: Use ConfigParser to validate INI syntax
- If shapes wrong: Print dimensions at each loading step

### Success Criteria:
✅ Verification script passes all checks  
✅ Dataset loader runs without errors  
✅ Feature dimension = 24 (22 factors + 2 temporal)  
✅ Can instantiate model with loaded data  
✅ Ready to start training in Phase 6  

---

## 📝 DELIVERABLES SUMMARY

| # | File | Type | Purpose |
|---|------|------|---------|
| 1 | `config/Multitask_NIFTY200_Alpha158.conf` | Config | Model settings |
| 2 | `lib/Multitask_Stockformer_utils.py` | Modified | Dataset loader |
| 3 | `verify_phase5_config.py` | New Script | Pre-training validation |
| 4 | `Phase_5_Architecture_Notes.md` | Documentation | Model understanding |
| 5 | `nifty-adaptationchat.txt` | Updated | Task tracker |

---

## 🔗 LINKS TO RELATED PHASES

- **Previous Phase:** [Phase 4 - Data Preprocessing](Phase_4_Stockformer_Input_Preprocessing.ipynb)
- **Next Phase:** Phase 6 - Training Pipeline
- **Project Root:** [NIFTY Adaptation Tracker](nifty-adaptationchat.txt)

---

**End of Phase 5 Implementation Plan**
