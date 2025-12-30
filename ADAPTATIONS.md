# Stockformer Adaptations & Custom Analysis

This fork contains the original Stockformer code plus additional scripts and analysis for: 
1. **Reproducing** the paper's results on Chinese stocks
2. **Analyzing** model architecture and generalization
3. **Testing** on multiple time periods and datasets
4. **Preparing** for adaptation to NIFTY-50 Indian stocks

## Custom Files Added

### Analysis & Documentation
- **`Stockformer_Architecture_Notes. ipynb`** - Comprehensive breakdown of model architecture
  - Parameter distribution (976K total parameters)
  - Data flow diagrams
  - Hyperparameter settings
  - Input/output specifications
  - Ablation study questions

### Testing & Validation Scripts
- **`analyze_model_architecture.py`** - Inspects trained model structure
  - Loads saved models
  - Prints parameter counts per layer
  - Validates architecture matches paper specifications
  
- **`test_generalization.py`** - Tests model generalization across subsets
  - Trains on one dataset subset
  - Tests on all other time periods
  - Measures generalization gap
  - Generates `generalization_results.log`

- **`run_inference.py`** - Inference utility script
  - Loads trained models
  - Generates predictions on new data
  - Supports batch processing

### Project Planning
- **`stockformer. ipynb`** - Project roadmap and setup guide
  - Reproduction strategy (14 independent models)
  - Data structure explanation
  - Setup tasks and training workflow
  - Backtest procedures
  - Transition plan to NIFTY-50

### Logs & Results
- **`training_subset11.log`** - Sample training log from Subset 11
- **`generalization_results.log`** - Generalization test results across all subsets

---

## Reproduction Progress

### ✅ Completed
- [x] Downloaded dataset:  `Stock_CN_2018-03-01_2020-10-29`
- [x] Analyzed model architecture (976K parameters)
- [x] Created analysis documentation
- [x] Developed generalization testing framework

### 🚀 Next Steps
1. Train on Subset 12 (largest, most robust)
2. Run full backtesting suite
3. Validate IC, Sharpe, drawdown metrics
4. Scale to all 14 subsets
5. Compare results to published paper

---

## NIFTY-50 Adaptation Plan

### Phase 1: Reproduction (Current)
- Complete reproduction on Chinese stock data
- Validate all metrics match paper

### Phase 2: Data Preparation
- Collect 5-7 years of NIFTY-50 historical data
- Adapt preprocessing for Indian market structure
- Generate 158 technical factors (or use Qlib's Alpha158)

### Phase 3: Training
- Train separate models for NIFTY-50 universe
- Adjust for 50 stocks (vs 255 in original)
- Tune hyperparameters for Indian market dynamics

### Phase 4: Backtesting
- Implement NIFTY-specific backtesting
- Account for Indian market trading rules
- Compare to NIFTY-50 TRI benchmark

---

## Key Insights

### Architecture Overview# Stockformer Adaptations & Custom Analysis

This fork contains the original Stockformer code plus additional scripts and analysis for: 
1. **Reproducing** the paper's results on Chinese stocks
2. **Analyzing** model architecture and generalization
3. **Testing** on multiple time periods and datasets
4. **Preparing** for adaptation to NIFTY-50 Indian stocks

## Custom Files Added

### Analysis & Documentation
- **`Stockformer_Architecture_Notes. ipynb`** - Comprehensive breakdown of model architecture
  - Parameter distribution (976K total parameters)
  - Data flow diagrams
  - Hyperparameter settings
  - Input/output specifications
  - Ablation study questions

### Testing & Validation Scripts
- **`analyze_model_architecture.py`** - Inspects trained model structure
  - Loads saved models
  - Prints parameter counts per layer
  - Validates architecture matches paper specifications
  
- **`test_generalization.py`** - Tests model generalization across subsets
  - Trains on one dataset subset
  - Tests on all other time periods
  - Measures generalization gap
  - Generates `generalization_results.log`

- **`run_inference.py`** - Inference utility script
  - Loads trained models
  - Generates predictions on new data
  - Supports batch processing

### Project Planning
- **`stockformer. ipynb`** - Project roadmap and setup guide
  - Reproduction strategy (14 independent models)
  - Data structure explanation
  - Setup tasks and training workflow
  - Backtest procedures
  - Transition plan to NIFTY-50

### Logs & Results
- **`training_subset11.log`** - Sample training log from Subset 11
- **`generalization_results.log`** - Generalization test results across all subsets

---

## Reproduction Progress

### ✅ Completed
- [x] Downloaded dataset:  `Stock_CN_2018-03-01_2020-10-29`
- [x] Analyzed model architecture (976K parameters)
- [x] Created analysis documentation
- [x] Developed generalization testing framework

### 🚀 Next Steps
1. Train on Subset 12 (largest, most robust)
2. Run full backtesting suite
3. Validate IC, Sharpe, drawdown metrics
4. Scale to all 14 subsets
5. Compare results to published paper

---

## NIFTY-50 Adaptation Plan

### Phase 1: Reproduction (Current)
- Complete reproduction on Chinese stock data
- Validate all metrics match paper

### Phase 2: Data Preparation
- Collect 5-7 years of NIFTY-50 historical data
- Adapt preprocessing for Indian market structure
- Generate 158 technical factors (or use Qlib's Alpha158)

### Phase 3: Training
- Train separate models for NIFTY-50 universe
- Adjust for 50 stocks (vs 255 in original)
- Tune hyperparameters for Indian market dynamics

### Phase 4: Backtesting
- Implement NIFTY-specific backtesting
- Account for Indian market trading rules
- Compare to NIFTY-50 TRI benchmark

---

## Key Insights

### Architecture Overview
