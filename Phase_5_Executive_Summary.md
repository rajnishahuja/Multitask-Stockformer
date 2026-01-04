# Phase 5 Executive Summary - Questions & Answers

**Date:** 2026-01-03  
**Status:** Phase 4 Complete → Phase 5 Ready to Begin

---

## 1️⃣ PHASE 5 STEPS FROM ADAPTATION TEXT FILE

From `nifty-adaptationchat.txt` (lines 261-279), Phase 5 has **3 main tasks**:

### Task 5.1: Review Model Architecture ⏳ NOT STARTED
- **File:** `Stockformermodel/Multitask_Stockformer_models.py`
- **Purpose:** Understand Decoupling Flow Layer, Dual-Frequency Encoder, Fusion Decoder
- **Goal:** Know how input dimensions flow through model before configuration

### Task 5.2: Update Configuration ⏳ NOT STARTED
- **File:** `config/Multitask_NIFTY200_Alpha158.conf` (NEW)
- **Purpose:** Create config file with NIFTY-200 specific settings
- **Sections to Configure:**
  - `[file]`: Paths to flow.npz, trend_indicator.npz, corr_adj.npy, adjgat.npy, model, log
  - `[data]`: dataset=NIFTY200, T1=20, T2=2, train_ratio=0.75, val_ratio=0.125, test_ratio=0.125
  - `[train]`: cuda=0, max_epoch=100, batch_size=12, learning_rate=0.001, seed=42
  - `[param]`: layers=2, heads=1, dims=128, samples=1, wave=sym2, level=1

### Task 5.3: Modify Dataset Class ⏳ NOT STARTED
- **File:** `lib/Multitask_Stockformer_utils.py`
- **Critical Changes:**
  - **Line 117:** Replace hardcoded `Alpha_360` path → use config parameter `args.factor_dir`
  - **Line 143:** `self.infea` calculation (already dynamic, but verify = 24 for our case)
  - Adjust stock dimension from 255 → actual N (191 for NIFTY-200)

---

## 2️⃣ ALIGNMENT WITH ORIGINAL PROJECT

### What Matches (No Changes):
| Component | Original | Ours | Status |
|-----------|----------|------|--------|
| **Architecture** | 2 layers, 128 dims, 1 head | Same | ✅ Match |
| **Wavelet** | Symlet-2, level=1 | Same | ✅ Match |
| **Multi-task** | Regression + Classification | Same | ✅ Match |
| **Loss Functions** | Masked MAE + CrossEntropy | Same | ✅ Match |
| **Optimizer** | Adam (lr=0.001) | Same | ✅ Match |
| **Data Splits** | 75/12.5/12.5 | Same | ✅ Match |
| **Strategy** | TopK selection | Same | ✅ Match |

### What Differs (Adaptations):
| Component | Original (CSI-300) | Ours (NIFTY-200) | Reason |
|-----------|-------------------|------------------|---------|
| **Stock Universe** | 255 stocks | **191 stocks** | Market availability (NIFTY-200 constituents) |
| **Factors** | 360 Alpha360 | **22 Alpha158** (IC filtered) | Quality over quantity - removed weak predictors |
| **Time Period** | 2021-06-04 to 2024-01-30 | **2022-01-01 to 2024-08-31** | Data availability from Zerodha |
| **Market** | Chinese A-shares | **Indian equities** | Different market, different characteristics |
| **Factor Engineering** | Qlib dump (black box) | **Custom pandas + IC filtering** | Full transparency and reproducibility |
| **Data Source** | Commercial provider | **Zerodha API** | Different data infrastructure |

### Critical Insight:
The **core model architecture is identical** - we're only adapting the **data inputs** (fewer stocks, fewer factors). This is intentional to validate the model works on Indian markets before experimenting with architectural changes.

---

## 3️⃣ EXTRA IMPROVEMENTS BEYOND ORIGINAL

We're doing **5 major improvements** the original project didn't have:

### Improvement #1: IC-Based Factor Selection ⭐ MOST IMPORTANT
- **Original:** Used ALL 360 Alpha360 factors without filtering
- **Ours:** Applied Information Coefficient filtering (|IC| >= 0.02)
- **Result:** Retained only 22/158 factors (13.9% survival rate)
- **Benefits:**
  - Removes noisy/weak predictors
  - Faster training (22 vs 360 features)
  - Better generalization (less overfitting)
  - Aligns with quantitative research best practices
- **Evidence:** Top factors (STD20, KLEN, BETA60) have IC = 0.027-0.029

### Improvement #2: Comprehensive Data Quality Control
- **Original:** Assumed clean data from commercial providers (black box)
- **Ours:** Multi-stage validation pipeline:
  - Missing date detection (>20% threshold)
  - Corporate action verification (verified NMDC 1:2 bonus correctly adjusted)
  - Price gap detection (>20% threshold)
  - Zero volume checks (>10% threshold)
- **Outcome:** 95.5% success rate (191/200 stocks), only 1 suspicious gap (verified correct)
- **File:** `data_processing_script/nifty/zerodha_data_fetcher.ipynb` (26 cells)

### Improvement #3: Historical Size Proxy (Avoids Look-Ahead Bias)
- **Original:** Used current market cap for neutralization (potential look-ahead bias)
- **Ours:** `log(close × rolling_volume_60d)` as size proxy
- **Rationale:** Historical market cap data unavailable; current market cap inappropriate for 2022-2024 neutralization
- **Benefit:** Correlation ~0.8 with actual market cap, but no look-ahead bias
- **File:** `data/NIFTY200/historical_size_proxy.csv`

### Improvement #4: Transparent Factor Engineering
- **Original:** Black-box preprocessing (uploaded pre-processed files to Google Drive)
- **Ours:** Open-source pandas implementation with formula documentation
- **Files:**
  - `alpha158_pandas.py` (358 lines, all 158 formulas)
  - `Phase_3_Alpha158_NIFTY200.ipynb` (complete workflow)
  - `Alpha158_IC_Filtering_Analysis.md` (methodology documentation)
- **Benefit:** Full reproducibility, easier debugging, customizable

### Improvement #5: Comprehensive Documentation
- **Original:** Minimal documentation on data preparation, just code
- **Ours:** Detailed markdown docs at every phase:
  - `nifty-adaptationchat.txt` (task tracker, 1000+ lines)
  - `Alpha158_IC_Filtering_Analysis.md` (factor analysis)
  - `Phase_5_Implementation_Plan.md` (this plan)
  - Phase notebooks with extensive comments
- **Benefit:** Easier collaboration, knowledge transfer, and future extensions

---

## 4️⃣ INPUTS NEEDED FOR PHASE 5

### Required Files (All from Phase 4):

#### Group A: Preprocessed Data Files
**Directory:** `data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/`

1. **flow.npz** (Wavelet coefficients)
   - Size: ~14 MB
   - Keys: `low_freq`, `high_freq`
   - Shape: `[~335 time steps, 191 stocks, 22 factors]` each
   - Purpose: Dual-frequency encoder inputs (low/high frequency components)

2. **trend_indicator.npz** (Binary labels)
   - Size: ~242 KB
   - Key: `trend`
   - Shape: `[660, 191]`
   - Values: 1 (up day), 0 (down day)
   - Purpose: Classification task targets

3. **corr_adj.npy** (Adjacency matrix)
   - Size: ~285 KB
   - Shape: `[191, 191]`
   - Sparsity: ~92% (only |corr| >= 0.3 kept)
   - Purpose: Graph Attention Network (GAT) layer input

4. **128_corr_struc2vec_adjgat.npy** (Graph embeddings)
   - Size: ~190 KB
   - Shape: `[191, 128]`
   - Method: Node2vec on correlation graph
   - Purpose: Spatial positional encoding in attention mechanism

5. **label.csv** (Returns for regression)
   - Size: ~1-2 MB
   - Shape: `[660 dates, 191 stocks]`
   - Values: Daily returns (continuous)
   - Purpose: Regression task targets

#### Group B: Factor Files
**Directory:** `data/NIFTY200/Alpha_158_2022-01-01_2024-08-31/`

6. **22 Factor CSVs** (e.g., BETA20.csv, STD20.csv, KLEN.csv, etc.)
   - Size: ~1-2 MB each
   - Shape: `[660 dates × 191 stocks]` per factor
   - Purpose: Additional features loaded separately by dataset loader
   - List: BETA20, BETA60, CORR20, CORR60, CORD20, CORD60, CNTP20, CNTP60, 
           CNTN20, CNTN60, KLEN, KMID, KMID2, KSFT, KSFT2, ROC20, ROC60, 
           RESI60, STD20, STD60, QTLU20, QTLU60

7. **selected_factors.txt**
   - Size: ~1 KB
   - Content: List of 22 factor names (one per line)
   - Purpose: Dataset loader iterates through this list to load CSVs

#### Metadata Required:
- **num_stocks:** 191
- **num_factors:** 22 (not 360 like original)
- **num_dates:** 660 (original), ~335 (after wavelet decomposition)
- **T1 (lookback):** 20 days
- **T2 (prediction):** 2 days ahead
- **date_range:** 2022-01-03 to 2024-08-30

### Verification Command:
```bash
ls -lh data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/
# Should show: flow.npz, trend_indicator.npz, corr_adj.npy, 128_corr_struc2vec_adjgat.npy, label.csv

ls data/NIFTY200/Alpha_158_2022-01-01_2024-08-31/ | wc -l
# Should show: 24 files (22 CSVs + selected_factors.txt + ic_summary.csv)
```

---

## 5️⃣ OUTPUTS EXPECTED FROM PHASE 5

### Output #1: Configuration File ✅ PRIMARY DELIVERABLE
**File:** `config/Multitask_NIFTY200_Alpha158.conf`

**Purpose:** Centralized settings for model, data, and training

**Content Preview:**
```ini
[file]
traffic = ./data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/flow.npz
indicator = ./data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/trend_indicator.npz
adj = ./data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/corr_adj.npy
adjgat = ./data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/128_corr_struc2vec_adjgat.npy
factor_dir = ./data/NIFTY200/Alpha_158_2022-01-01_2024-08-31  # NEW PARAMETER
model = ./cpt/NIFTY200/saved_model_Alpha158_MVP
log = ./log/NIFTY200/log_Alpha158_MVP

[data]
dataset = NIFTY200
T1 = 20
T2 = 2
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
layers = 2
heads = 1
dims = 128
samples = 1
wave = sym2
level = 1
```

**Key Addition:** `factor_dir` parameter in `[file]` section (not in original config)

---

### Output #2: Modified Dataset Loader ✅ CRITICAL CODE CHANGE
**File:** `lib/Multitask_Stockformer_utils.py` (modified)

**Change #1: Line 117 (Factor Path)**
```python
# BEFORE (Hardcoded):
path = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/Stock_CN_2021-06-04_2024-01-30/Alpha_360_2021-06-04_2024-01-30'

# AFTER (Config-driven):
path = args.factor_dir  # From config [file] section
```

**Change #2: Line 143 (Feature Count)**
```python
# Already correct (dynamically calculates):
self.infea = bonus_all.shape[-1] + 2  
# Result: 22 (factors) + 2 (temporal embeddings) = 24
```

**Why This Matters:**
- Original code hardcoded path to 360 Alpha360 factors
- Our adaptation needs path to 22 Alpha158 factors
- Without this change, training will crash with "file not found" error

---

### Output #3: Verification Script ✅ NEW UTILITY
**File:** `verify_phase5_config.py` (NEW - not in original project)

**Purpose:** Pre-flight checks before training (prevent runtime errors)

**What It Tests:**
1. Config file loads without errors
2. All data files exist at specified paths
3. Tensor shapes match expected dimensions:
   - flow.npz: `[335, 191, 22]` × 2 (low + high freq)
   - trend_indicator.npz: `[660, 191]`
   - corr_adj.npy: `[191, 191]`
   - adjgat.npy: `[191, 128]`
4. No NaN/Inf values in critical data
5. Dataset loader can instantiate and iterate
6. Feature dimension = 24 (matches model input)

**Expected Output:**
```
=== Phase 5 Configuration Verification ===

✓ Config file loaded successfully
✓ All 5 data files exist
✓ flow.npz: low_freq [335, 191, 22], high_freq [335, 191, 22]
✓ trend_indicator.npz: [660, 191]
✓ corr_adj.npy: [191, 191] (sparsity: 92.3%)
✓ adjgat.npy: [191, 128]
✓ Factor CSVs: 22 files × [660, 191]
✓ Dataset loader: 641 train, 80 val, 80 test samples
✓ Feature dimension: 24 (22 factors + 2 temporal)

=== VERIFICATION PASSED ===
Ready for Phase 6: Training
```

---

### Output #4: Architecture Notes ✅ DOCUMENTATION
**File:** `Phase_5_Architecture_Notes.md` (NEW)

**Content:** Understanding of model components:
- DecouplingFlowLayer: How wavelet components are processed
- DualFrequencyEncoder: Low/high frequency feature extraction
- FusionDecoder: Multi-task prediction heads (regression + classification)
- spatialAttention: How graph embeddings (adjgat) are integrated
- Dimension flow: Input [24 features] → Hidden [128 dims] → Output [2 tasks]

**Purpose:** Document model understanding before training

---

### Output #5: Updated Task Tracker ✅ PROJECT MANAGEMENT
**File:** `nifty-adaptationchat.txt` (updated Phase 5 section)

**Updates:**
- Mark Phase 5 tasks as completed
- Add notes on any issues encountered
- Document any deviations from plan
- Update "Current Status Summary" section

---

## 6️⃣ IMPLEMENTATION PLAN FILE CREATED

### File Created:
✅ **`Phase_5_Implementation_Plan.md`** (29 KB, comprehensive)

### What It Contains:

#### Section 1: Overview
- Phase 5 objectives and scope
- Input/output summary
- Alignment with original project

#### Section 2: Detailed Task Breakdown
**Task 5.1:** Review Model Architecture
- Files to review
- Key components to understand
- Deliverable: Architecture notes

**Task 5.2:** Create Configuration File
- Step-by-step config creation
- What each section does
- Validation steps

**Task 5.3:** Modify Dataset Loader
- Specific line numbers to change
- Before/after code comparisons
- Testing procedures

**Task 5.4:** Create Verification Script
- Comprehensive test checklist
- Expected outputs
- Error handling

**Task 5.5:** Update Training Script
- Minor changes needed
- Argument parsing updates

**Task 5.6:** Documentation Updates
- Task tracker updates
- Git commit messages

#### Section 3: Execution Guide
- Recommended task order
- Time estimates (8 hours total)
- Common pitfalls to avoid
- Debugging tips

#### Section 4: Checklists
- Pre-flight checks
- Task completion checklist
- Final validation checklist
- "Ready for Phase 6" criteria

#### Section 5: Reference Materials
- Input/output specifications
- Comparison tables (original vs ours)
- Improvement documentation
- Links to related phases

---

## 📊 QUICK REFERENCE TABLE

| Question | Answer |
|----------|--------|
| **Phase 5 Steps** | 3 tasks: Review architecture, Create config, Modify dataset loader |
| **Alignment** | Core model identical; only data inputs differ (191 stocks, 22 factors) |
| **Improvements** | 5 major: IC filtering, data quality control, size proxy, transparency, docs |
| **Inputs Needed** | 5 data files + 22 factor CSVs (all from Phase 4) |
| **Outputs Expected** | Config file, modified loader, verification script, architecture notes |
| **Time Estimate** | 8 hours over 2 days |
| **Success Criteria** | Verification script passes all checks |

---

## 🎯 NEXT ACTIONS

1. ✅ **Read this document** - Understand Phase 5 scope and requirements
2. ✅ **Read `Phase_5_Implementation_Plan.md`** - Detailed task guide
3. ⏳ **Run pre-flight checks** - Verify Phase 4 outputs exist
4. ⏳ **Start Task 5.1** - Review model architecture (2 hours)
5. ⏳ **Continue through tasks** - Follow implementation plan sequentially
6. ⏳ **Run verification script** - Ensure everything works before Phase 6

---

**Questions Answered:**
1. ✅ Phase 5 steps listed from adaptation file
2. ✅ Alignment with original project documented
3. ✅ Extra improvements identified (5 major enhancements)
4. ✅ Inputs needed specified (5 data files + 22 CSVs)
5. ✅ Outputs expected defined (5 deliverables)
6. ✅ Implementation plan created in separate file

**Status:** Ready to begin Phase 5 implementation!
