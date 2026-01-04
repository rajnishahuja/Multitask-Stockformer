# Phase 4: Stockformer Input Data Preprocessing - Implementation Plan

## Executive Summary

**Goal**: Generate 4 required input files for Multitask-Stockformer model training:
1. `flow.npz` - Stock returns (2D: timesteps × stocks)
2. `trend_indicator.npz` - Binary up/down classification (2D: timesteps × stocks)
3. `corr_adj.npy` - Stock correlation adjacency matrix (2D: stocks × stocks)
4. `128_corr_struc2vec_adjgat.npy` - Graph embeddings (2D: stocks × 128)

**Critical Finding**: Our current implementation is **INCORRECT**. We created flow.npz with shape (331, 191, 22) containing factors, but it should be (T, 191) containing RETURNS only.

---

## 1. Reference: Chinese Implementation Analysis

### Input Files (Chinese)
- **label.csv**: (648, 255) - Daily returns for 255 stocks over 648 days
- **Alpha_360 factors**: 361 CSV files, each (648, 255) - Aligned with label.csv timesteps
- Date range: 2020-03-04 to 2022-11-02

### Output Files (Chinese)
1. **flow.npz**:
   - Key: `result`
   - Shape: **(648, 255)** - 2D array
   - Content: Stock returns (from label.csv)
   - Purpose: Target variable for prediction

2. **trend_indicator.npz**:
   - Key: `result`
   - Shape: **(648, 255)** - 2D array
   - Content: Binary (1 if return > 0, else 0)
   - Purpose: Classification labels

3. **corr_adj.npy**:
   - Shape: **(255, 255)** - 2D array
   - Content: Stock-stock correlation matrix
   - Computed from: label.csv returns (rowvar=False)

4. **128_corr_struc2vec_adjgat.npy**:
   - Shape: **(255, 128)** - 2D array
   - Content: Struc2Vec graph embeddings
   - Generated from: Correlation-based graph

### Chinese Preprocessing Logic
```python
# 1. Load returns
df = pd.read_csv('label.csv', index_col=0)
df.fillna(0, inplace=True)

# 2. Save as flow.npz (returns only, 2D)
np.savez('flow.npz', result=df.values)  # Shape: (648, 255)

# 3. Generate trend indicator
trend = (df.values > 0).astype(int)
np.savez('trend_indicator.npz', result=trend)

# 4. Compute correlation matrix
corr_matrix = np.corrcoef(df, rowvar=False)
np.save('corr_adj.npy', corr_matrix)

# 5. Generate graph embeddings via Struc2Vec
# (Creates edgelist → NetworkX graph → Struc2Vec → embeddings)
```

**Key Insight**: The Chinese script does **NOT** perform wavelet decomposition. It saves raw returns directly to flow.npz. The wavelet decomposition happens **inside the model during training**.

---

## 2. Current NIFTY-200 Implementation Issues

### What We Have (INCORRECT)
1. **flow.npz**:
   - Keys: `low_freq`, `high_freq`
   - Shape: **(331, 191, 22)** - 3D array ❌
   - Content: Wavelet-decomposed 22 FACTORS (not returns!)
   - Problem: Model expects 2D returns, not 3D factors

2. **trend_indicator.npz**:
   - Key: `trend`
   - Shape: **(95, 191)** - Partial data (Apr-Aug 2024 only) ❌
   - Should be: (660, 191) for full period

3. **corr_adj.npy**: ✓ (191, 191) - Correct
4. **128_corr_struc2vec_adjgat.npy**: ✓ (191, 128) - Correct

### Available Input Data
- **Raw stock data**: 191 CSV files in `data/NIFTY200/raw/`
  - Columns: Date, Open, High, Low, Close, Volume
  - Shape per file: (660, 6) - Full 2022-2024 period
  
- **Factor CSVs**: 31 files in `Alpha_158_2022-01-01_2024-08-31/`
  - 22 files with (660, 191) - Full period
  - 9 files with (95, 191) - Partial period (exclude these)

- **label.csv**: (95, 191) - Partial returns (Apr-Aug 2024 only)
  - Problem: Too short for training

---

## 3. Corrected Implementation Plan

### Task 4.1: Generate Daily Returns (label.csv)
**Input**: Raw stock CSVs in `data/NIFTY200/raw/` (660 days per stock)
**Output**: `label.csv` with shape (660, 191)

**Method**:
```python
# For each stock CSV:
# 1. Load CSV with Date, Open, High, Low, Close, Volume
# 2. Calculate daily return: (Close[t] - Close[t-1]) / Close[t-1]
# 3. Combine all stocks into single DataFrame (660, 191)
# 4. Save as label.csv

# Handle first row (no previous close):
#   - Option A: Drop first row → (659, 191)
#   - Option B: Set first row to 0 → (660, 191)
```

**Decision Point**: Use Close-to-Close returns (standard) or Open-to-Close returns?
- Recommendation: **Close-to-Close** (industry standard)

---

### Task 4.2: Generate flow.npz (Returns)
**Input**: `label.csv` (660, 191)
**Output**: `flow.npz` with key `result`, shape (660, 191)

**Method**:
```python
# Load label.csv
df = pd.read_csv('label.csv', index_col=0)
df.fillna(0, inplace=True)  # Fill NaN with 0

# Save as flow.npz (NO wavelet decomposition)
data = df.values  # Shape: (660, 191)
np.savez('flow.npz', result=data)
```

**Critical Note**: 
- Do **NOT** apply wavelet decomposition here
- Do **NOT** include factors in flow.npz
- flow.npz = pure returns, 2D array

---

### Task 4.3: Generate trend_indicator.npz
**Input**: `label.csv` (660, 191)
**Output**: `trend_indicator.npz` with key `result`, shape (660, 191)

**Method**:
```python
# Binary classification: 1 if return > 0, else 0
data = df.values  # From label.csv
trend = (data > 0).astype(np.int32)
np.savez('trend_indicator.npz', result=trend)
```

---

### Task 4.4: Generate corr_adj.npy
**Input**: `label.csv` (660, 191)
**Output**: `corr_adj.npy` with shape (191, 191)

**Method**:
```python
# Compute stock-stock correlation matrix
# Handle zero-variance columns
epsilon = 1e-10
std_devs = np.std(df, axis=0)
zero_variance_mask = std_devs < epsilon
df.loc[:, zero_variance_mask] = epsilon

# Correlation (stocks as variables, time as observations)
corr_matrix = np.corrcoef(df, rowvar=False)  # Shape: (191, 191)
np.save('corr_adj.npy', corr_matrix)
```

---

### Task 4.5: Generate 128_corr_struc2vec_adjgat.npy
**Input**: `corr_adj.npy` (191, 191)
**Output**: `128_corr_struc2vec_adjgat.npy` with shape (191, 128)

**Method**:
```python
# 1. Create edge list from correlation matrix
edge_list = []
for i in range(191):
    for j in range(i+1, 191):
        weight = corr_matrix[i, j]
        edge_list.append((i, j, weight))

# 2. Build NetworkX graph
G = nx.read_edgelist(
    'data.edgelist',
    create_using=nx.DiGraph(),
    nodetype=None,
    data=[('weight', float)]
)

# 3. Train Struc2Vec embeddings
# Option A: Use original ge.Struc2Vec library (Chinese implementation)
# Option B: Use node2vec library (our Phase 4 notebook approach)

# Original approach (preferred):
from ge import Struc2Vec
model = Struc2Vec(G, 10, 80, workers=4, verbose=40)
model.train(embed_size=128)
embeddings = model.get_embeddings()

# Convert to array and save
embedding_array = np.array(list(embeddings.values()))
np.save('128_corr_struc2vec_adjgat.npy', embedding_array)
```

---

## 4. Factor Data Handling

### Question: What about the 22 Alpha158 factor CSVs?

**Answer**: Factors are loaded **separately** by the dataset loader during training, NOT included in flow.npz.

**Dataset Loader Logic** (from `lib/Multitask_Stockformer_utils.py`):
```python
# Load flow.npz (returns only)
Traffic = np.load(args.traffic_file)['result']  # Shape: (660, 191)

# Load factors separately from CSVs
path = args.factor_dir  # Points to Alpha_158 directory
files = os.listdir(path)
for file in files:
    df = pd.read_csv(file, index_col=0)  # Shape: (660, 191)
    # Concatenate along feature dimension
# Result: bonus_all shape (660, 191, 22)
```

**Requirements**:
1. Factor CSVs must have **same timesteps** as flow.npz
2. All 22 factor files must have shape (660, 191)
3. Currently we have 22 files with (660, 191) ✓ and 9 files with (95, 191) ✗

**Action**: Ensure only the 22 full-period factors (660 rows) are kept in Alpha_158 directory. Remove or exclude the 9 partial-period factors.

---

## 5. Implementation Steps (Execution Order)

### Step 1: Generate Returns from Raw Data
**Script**: `generate_returns_from_raw.py`
- Input: `data/NIFTY200/raw/*.csv` (191 files, 660 rows each)
- Output: `data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/label.csv` (660, 191)
- Validation: Check for NaN values, verify date range

### Step 2: Generate flow.npz
**Script**: `generate_flow.py`
- Input: `label.csv` (660, 191)
- Output: `flow.npz` with `result` key (660, 191)
- Validation: Verify shape, check min/max values

### Step 3: Generate trend_indicator.npz
**Script**: `generate_trend_indicator.py`
- Input: `label.csv` (660, 191)
- Output: `trend_indicator.npz` with `result` key (660, 191)
- Validation: Verify binary values (0 or 1 only)

### Step 4: Generate corr_adj.npy
**Script**: `generate_correlation_matrix.py`
- Input: `label.csv` (660, 191)
- Output: `corr_adj.npy` (191, 191)
- Validation: Check diagonal = 1.0, symmetric matrix

### Step 5: Generate adjgat embeddings
**Script**: `generate_graph_embeddings.py`
- Input: `corr_adj.npy` (191, 191)
- Output: `128_corr_struc2vec_adjgat.npy` (191, 128)
- Validation: Check no NaN values, verify shape

### Step 6: Clean Factor Directory
**Script**: Manual or automated cleanup
- Keep only 22 factor CSVs with (660, 191) shape
- Remove 9 partial-period factors with (95, 191) shape
- Validation: Verify all remaining factors have same date range

---

## 6. Expected Final Output

After Phase 4 completion, we should have:

```
data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/
├── label.csv                              (660, 191) ✓
├── flow.npz
│   └── result                             (660, 191) ✓
├── trend_indicator.npz
│   └── result                             (660, 191) ✓
├── corr_adj.npy                           (191, 191) ✓
├── 128_corr_struc2vec_adjgat.npy          (191, 128) ✓
└── data.edgelist                          Text file with graph edges

data/NIFTY200/Alpha_158_2022-01-01_2024-08-31/
├── BETA20.csv                             (660, 191) ✓
├── BETA30.csv                             (660, 191) ✓
├── ... (20 more files)
└── [REMOVED: 9 partial-period factors]
```

**Total**: 22 factor CSVs with (660, 191) shape

---

## 7. Phase 5 Configuration Update

After Phase 4 is corrected, Phase 5 config will need:

```ini
[file]
traffic = ./data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/flow.npz
indicator = ./data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/trend_indicator.npz
adj = ./data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/corr_adj.npy
adjgat = ./data/NIFTY200/Stock_NIFTY_2022-01-01_2024-08-31/128_corr_struc2vec_adjgat.npy
factor_dir = ./data/NIFTY200/Alpha_158_2022-01-01_2024-08-31
```

**Phase 5 Dataset Loader** will then:
1. Load `Traffic` from flow.npz['result'] → shape (660, 191)
2. Load `indicator` from trend_indicator.npz['result'] → shape (660, 191)
3. Load 22 factor CSVs from `factor_dir` → shape (660, 191, 22)
4. Concatenate into `bonus_all` → shape (660, 191, 22)
5. Apply seq2instance to create training samples

**Feature Dimension**:
- `infea = 22 + 2 = 24` (22 factors + 2 temporal embeddings)

---

## 8. Validation Checklist

After Phase 4 implementation, verify:

- [ ] label.csv has 660 rows (full period) not 95 rows
- [ ] flow.npz['result'] is 2D (660, 191), not 3D
- [ ] trend_indicator.npz['result'] matches flow shape (660, 191)
- [ ] corr_adj.npy diagonal all equals 1.0
- [ ] adjgat.npy has no NaN values
- [ ] All 22 factor CSVs have identical shape (660, 191)
- [ ] Factor CSVs date range matches label.csv date range
- [ ] No partial-period factors remain in Alpha_158 directory

---

## 9. Next Actions

1. **Review this plan** with user
2. **Implement Step 1**: Generate returns from raw data
3. **Execute Steps 2-5**: Generate all 4 required files
4. **Verify outputs**: Run validation checks
5. **Resume Phase 5**: Update config and continue model configuration
