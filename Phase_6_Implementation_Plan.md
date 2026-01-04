# Phase 6: Model Training - Implementation Plan

**Date Created:** 2026-01-03  
**Current Status:** Phase 5 Complete → Ready for Phase 6  
**Objective:** Train Multitask-Stockformer on NIFTY-200 with optimizations

---

## 📋 PHASE 6 OVERVIEW

### Major Goals
1. **Train Model:** Dual-task learning (regression + classification) for 185 stocks, 22 factors
2. **Optimize Training:** Add DataLoader, early stopping, better monitoring
3. **Fix Issues:** Remove hardcoded paths from Chinese implementation
4. **Validate Performance:** Test MAE < 0.015, Accuracy > 0.52
5. **Enable Reproducibility:** Proper checkpointing and logging

### What Changed from Chinese Implementation
| Aspect | Chinese Original | Our Adaptation | Reason |
|--------|-----------------|----------------|--------|
| Data Loading | Manual arrays | PyTorch DataLoader | 2-4x faster, parallel workers |
| Stopping | Fixed 100 epochs | Early stopping (patience=30) | Save time, prevent overfitting |
| Paths | Hardcoded | Config-driven | Portability, maintainability |
| Monitoring | Basic logs | TensorBoard + detailed logs | Better debugging |
| Epochs | 100 | Start with 20 for testing | Validate setup before full run |

---

## 📥 INPUTS (From Phase 5)

**Config:** `config/Multitask_NIFTY200_Alpha158.conf` (validated ✓)  
**Data:** 185 stocks, 22 factors, 660 days (all aligned ✓)  
**Splits:** Train 474 samples, Val 62 samples, Test 61 samples  

---

## 📤 OUTPUTS

1. **Model:** `cpt/NIFTY200/saved_model_Alpha158_MVP`
2. **Logs:** `log/NIFTY200/log_Alpha158_MVP`
3. **TensorBoard:** `runs/Multitask_Stockformer/NIFTY200_Alpha158_MVP/`
4. **Predictions:** `output/Multitask_output_NIFTY200_Alpha158_MVP/{classification,regression}/*.csv`

---

## 🛠️ IMPLEMENTATION TASKS

### Task 6.1: Fix Hardcoded Paths
**File:** `MultiTask_Stockformer_train.py`

**Changes:**
```python
# Line ~89: TensorBoard path
tensorboard_folder = f'./runs/Multitask_Stockformer/{args.Dataset}_Alpha158_MVP'

# Lines ~242-249: Output paths
output_dir = f'./output/Multitask_output_{args.Dataset}_Alpha158_MVP'
os.makedirs(f'{output_dir}/classification', exist_ok=True)
os.makedirs(f'{output_dir}/regression', exist_ok=True)

# Line ~57: Add factor_dir parameter
parser.add_argument('--factor_dir', default=config['file']['factor_dir'])
```

---

### Task 6.2: Add DataLoader with Multi-Workers
**File:** `MultiTask_Stockformer_train.py`

**Changes:**
```python
from torch.utils.data import DataLoader

# Replace manual batching with DataLoader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                        shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                         shuffle=False, num_workers=2, pin_memory=True)
```

**Validation:** Training should be 2x faster, GPU utilization >80%

---

### Task 6.3: Add Early Stopping
**File:** `MultiTask_Stockformer_train.py`

**Changes:**
```python
# In train() function
early_stop_patience = 30
epochs_no_improve = 0
best_mae = float('inf')

for epoch in range(max_epochs):
    # ... training ...
    val_mae = validate()
    
    if val_mae < best_mae:
        best_mae = val_mae
        epochs_no_improve = 0
        save_model()
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= early_stop_patience:
        log_string(log, f"Early stopping at epoch {epoch}")
        break
```

---

### Task 6.4: Create Pre-Training Validation
**File:** `validate_before_training.py` (NEW)

**Purpose:** Quick checks before expensive training

**Tests:**
1. Config loads correctly
2. All data files exist and have correct shapes
3. Dataset loader works (load 1 batch)
4. Model initializes and moves to GPU
5. Forward pass works (no shape errors)
6. Backward pass works (gradients flow)

**Expected:** "ALL CHECKS PASSED - Ready to train"

---

### Task 6.5: Update Config for Testing
**File:** `config/Multitask_NIFTY200_Alpha158.conf`

**Change for initial test:**
```ini
[train]
max_epoch = 20    # Start with 20, not 100
```

**After validation, update to 100 epochs**

---

### Task 6.6: Run Training
**Commands:**
```bash
# Pre-check
python validate_before_training.py --config config/Multitask_NIFTY200_Alpha158.conf

# Training (20 epochs first)
python MultiTask_Stockformer_train.py --config config/Multitask_NIFTY200_Alpha158.conf

# Monitor in separate terminals
tail -f log/NIFTY200/log_Alpha158_MVP
tensorboard --logdir=./runs/Multitask_Stockformer/
```

**Success:** Training loss decreases, val MAE < 0.015, no crashes

---

### Task 6.7: Analyze Results
**File:** `analyze_training.py` (NEW - optional)

Quick script to plot training curves and show final metrics

---

## 📊 CHECKLIST

### Pre-Training:
- [ ] Phase 5 verification passed
- [ ] Output directories created (`mkdir -p cpt/NIFTY200 log/NIFTY200 output runs`)
- [ ] GPU available (`nvidia-smi`)

### Implementation:
- [ ] Task 6.1: Paths fixed
- [ ] Task 6.2: DataLoader added
- [ ] Task 6.3: Early stopping added
- [ ] Task 6.4: Validation script created and passes
- [ ] Task 6.5: Config updated (20 epochs)
- [ ] Task 6.6: Training runs successfully
- [ ] Task 6.7: Results analyzed

### Validation:
- [ ] Training completes without errors
- [ ] Val MAE improves over epochs
- [ ] Model checkpoint saved
- [ ] TensorBoard logs generated
- [ ] Test predictions CSV files created

---

## 🚀 EXECUTION ORDER

**Day 1 (2-3 hours):**
1. Task 6.1: Fix paths (30 min)
2. Task 6.2: Add DataLoader (30 min)
3. Task 6.3: Add early stopping (30 min)
4. Task 6.4: Create validation script (1 hour)
5. Run validation script (5 min)

**Day 2 (1-2 hours):**
6. Task 6.5: Update config (2 min)
7. Task 6.6: Run training (20 min actual runtime)
8. Task 6.7: Analyze results (30 min)

**If 20-epoch results good → Update max_epoch=100 and rerun**

---

## ⚠️ CRITICAL NOTES

**Performance Targets:**
- Test MAE < 0.015 (minimum viable)
- Test Accuracy > 0.52 (better than random)
- Training should complete without OOM errors

**Debugging:**
- If OOM: Reduce batch_size to 8
- If NaN loss: Reduce learning_rate to 0.0001
- If slow: Check DataLoader num_workers (try 0 if issues)

**Success Indicators:**
- ✅ Training loss decreases smoothly
- ✅ Val MAE improves in first 10 epochs
- ✅ GPU memory < 15GB
- ✅ Epoch time ~30-40 seconds

---

## 📝 DELIVERABLES

1. `MultiTask_Stockformer_train.py` (modified)
2. `validate_before_training.py` (new)
3. `analyze_training.py` (new, optional)
4. Trained model checkpoint
5. Training logs
6. TensorBoard logs
7. Test predictions CSVs

---

**End of Phase 6 Implementation Plan**
