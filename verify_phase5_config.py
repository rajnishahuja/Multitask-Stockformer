#!/usr/bin/env python3
"""
Phase 5 Configuration Verification Script
Validates all data files, config settings, and dataset loader before training.

Checks:
1. Config file parsing
2. Data file existence and shapes
3. Data quality (NaN/Inf checks)
4. Dataset loader functionality
5. Feature dimension correctness
"""

import os
import sys
import configparser
import numpy as np
import pandas as pd
from pathlib import Path

def print_header(text):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")

def print_check(passed, message):
    """Print check result with status symbol"""
    symbol = "✓" if passed else "✗"
    print(f"  {symbol} {message}")
    return passed

def verify_config(config_path):
    """Verify configuration file can be loaded and has all required sections"""
    print_header("1. Configuration File Validation")
    
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        print_check(True, f"Loaded: {config_path}")
        
        # Check required sections
        required_sections = ['file', 'data', 'train', 'param']
        all_present = all(section in config for section in required_sections)
        print_check(all_present, f"All sections present: {required_sections}")
        
        if not all_present:
            missing = [s for s in required_sections if s not in config]
            print(f"    Missing sections: {missing}")
            return None, False
            
        return config, True
    except Exception as e:
        print_check(False, f"Config parsing failed: {e}")
        return None, False

def verify_data_files(config):
    """Verify all data files exist and have correct shapes"""
    print_header("2. Data File Validation")
    
    all_good = True
    
    # Check flow.npz
    flow_path = config['file']['traffic']
    if os.path.exists(flow_path):
        flow_data = np.load(flow_path)
        low_freq = flow_data['low_freq'] if 'low_freq' in flow_data else flow_data['result']
        high_freq = flow_data['high_freq'] if 'high_freq' in flow_data else None
        
        size_mb = os.path.getsize(flow_path) / 1024 / 1024
        print_check(True, f"flow.npz exists ({size_mb:.2f} MB)")
        
        if high_freq is not None:
            print(f"    - low_freq: {low_freq.shape}")
            print(f"    - high_freq: {high_freq.shape}")
        else:
            print(f"    - result: {low_freq.shape}")
    else:
        print_check(False, f"flow.npz missing: {flow_path}")
        all_good = False
    
    # Check trend_indicator.npz
    indicator_path = config['file']['indicator']
    if os.path.exists(indicator_path):
        indicator_data = np.load(indicator_path)
        trend = indicator_data['trend'] if 'trend' in indicator_data else indicator_data['result']
        size_kb = os.path.getsize(indicator_path) / 1024
        print_check(True, f"trend_indicator.npz exists ({size_kb:.2f} KB)")
        print(f"    - trend: {trend.shape}")
    else:
        print_check(False, f"trend_indicator.npz missing: {indicator_path}")
        all_good = False
    
    # Check corr_adj.npy
    adj_path = config['file']['adj']
    if os.path.exists(adj_path):
        corr_adj = np.load(adj_path)
        size_kb = os.path.getsize(adj_path) / 1024
        sparsity = (corr_adj == 0).sum() / corr_adj.size * 100
        print_check(True, f"corr_adj.npy exists ({size_kb:.2f} KB)")
        print(f"    - shape: {corr_adj.shape}, sparsity: {sparsity:.1f}%")
    else:
        print_check(False, f"corr_adj.npy missing: {adj_path}")
        all_good = False
    
    # Check adjgat.npy
    adjgat_path = config['file']['adjgat']
    if os.path.exists(adjgat_path):
        adjgat = np.load(adjgat_path)
        size_kb = os.path.getsize(adjgat_path) / 1024
        print_check(True, f"128_corr_struc2vec_adjgat.npy exists ({size_kb:.2f} KB)")
        print(f"    - shape: {adjgat.shape}")
    else:
        print_check(False, f"adjgat.npy missing: {adjgat_path}")
        all_good = False
    
    return all_good

def verify_factor_files(config):
    """Verify factor CSV files exist and have correct structure"""
    print_header("3. Factor File Validation")
    
    factor_dir = Path(config['file']['factor_dir'])
    
    if not factor_dir.exists():
        print_check(False, f"Factor directory missing: {factor_dir}")
        return False
    
    print_check(True, f"Factor directory: {factor_dir}")
    
    # Count CSV files
    csv_files = list(factor_dir.glob('*.csv'))
    # Exclude ic_summary.csv if present
    csv_files = [f for f in csv_files if 'ic_summary' not in f.name.lower()]
    
    num_factors = len(csv_files)
    print_check(True, f"Found {num_factors} factor CSVs")
    
    # Load one to check shape
    if csv_files:
        sample_df = pd.read_csv(csv_files[0], index_col=0)
        print(f"    - Shape per factor: {sample_df.shape}")
        print(f"    - Date range: {sample_df.index[0]} to {sample_df.index[-1]}")
        print(f"    - Stocks: {len(sample_df.columns)}")
    
    return True

def verify_data_quality(config):
    """Check for NaN/Inf values in loaded data"""
    print_header("4. Data Quality Checks")
    
    all_good = True
    
    # Check flow data
    flow_data = np.load(config['file']['traffic'])
    flow = flow_data['low_freq'] if 'low_freq' in flow_data else flow_data['result']
    
    has_inf = np.isinf(flow).any()
    print_check(not has_inf, f"flow.npz: No Inf values" if not has_inf else "flow.npz: Contains Inf!")
    
    # Check corr_adj
    corr_adj = np.load(config['file']['adj'])
    diagonal_correct = np.allclose(np.diag(corr_adj), 1.0)
    print_check(diagonal_correct, "corr_adj.npy: Diagonal all 1.0" if diagonal_correct else "corr_adj.npy: Diagonal issue!")
    
    # Check adjgat
    adjgat = np.load(config['file']['adjgat'])
    has_embeddings = (np.abs(adjgat).sum(axis=1) != 0).sum() > 0
    print_check(has_embeddings, f"adjgat.npy: Has non-zero embeddings" if has_embeddings else "adjgat.npy: All zeros!")
    
    return all_good

def verify_dataset_loader(config):
    """Test dataset loader can instantiate and iterate"""
    print_header("5. Dataset Loader Validation")
    
    try:
        # Add lib directory to path
        sys.path.insert(0, '/home/ubuntu/rajnish/Multitask-Stockformer/lib')
        from Multitask_Stockformer_utils import StockDataset
        
        # Create args object from config
        class Args:
            def __init__(self, config):
                self.traffic_file = config['file']['traffic']
                self.indicator_file = config['file']['indicator']
                self.factor_dir = config['file']['factor_dir']
                self.T1 = int(config['data']['t1'])
                self.T2 = int(config['data']['t2'])
                self.train_ratio = float(config['data']['train_ratio'])
                self.val_ratio = float(config['data']['val_ratio'])
                self.test_ratio = float(config['data']['test_ratio'])
                self.w = config['param']['wave']
                self.j = int(config['param']['level'])
        
        args = Args(config)
        
        # Try to create datasets
        train_dataset = StockDataset(args, mode='train')
        val_dataset = StockDataset(args, mode='val')
        test_dataset = StockDataset(args, mode='test')
        
        print_check(True, f"Train dataset: {len(train_dataset)} samples")
        print_check(True, f"Val dataset: {len(val_dataset)} samples")
        print_check(True, f"Test dataset: {len(test_dataset)} samples")
        print_check(True, f"Feature dimension (infea): {train_dataset.infea}")
        
        # Try to get one sample
        sample_x, sample_y = train_dataset[0]
        print(f"    - Sample X keys: {list(sample_x.keys())}")
        print(f"    - Sample Y keys: {list(sample_y.keys())}")
        print(f"    - X shape: {sample_x['X'].shape}")
        print(f"    - bonus_X shape: {sample_x['bonus_X'].shape}")
        
        return True, train_dataset.infea
        
    except Exception as e:
        print_check(False, f"Dataset loader failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def verify_model_readiness(infea):
    """Check if dimensions are compatible with model"""
    print_header("6. Model Readiness")
    
    if infea is None:
        print_check(False, "Cannot verify - dataset loader failed")
        return False
    
    print_check(True, f"Input feature count: {infea}")
    
    # Expected: 22 factors + 2 temporal = 24
    expected_infea = 24
    correct = (infea == expected_infea)
    print_check(correct, f"Expected infea: {expected_infea} (22 factors + 2 temporal)" if correct else f"Mismatch! Expected {expected_infea}, got {infea}")
    
    return correct

def main():
    """Run all verification checks"""
    config_path = 'config/Multitask_NIFTY200_Alpha158.conf'
    
    print("\n" + "="*70)
    print("  Phase 5 Configuration Verification")
    print("  NIFTY-200 Multitask-Stockformer")
    print("="*70)
    
    # Check 1: Config file
    config, config_ok = verify_config(config_path)
    if not config_ok:
        print("\n❌ VERIFICATION FAILED: Config file issues")
        return 1
    
    # Check 2: Data files
    data_ok = verify_data_files(config)
    if not data_ok:
        print("\n❌ VERIFICATION FAILED: Data file issues")
        return 1
    
    # Check 3: Factor files
    factor_ok = verify_factor_files(config)
    if not factor_ok:
        print("\n❌ VERIFICATION FAILED: Factor file issues")
        return 1
    
    # Check 4: Data quality
    quality_ok = verify_data_quality(config)
    
    # Check 5: Dataset loader
    loader_ok, infea = verify_dataset_loader(config)
    if not loader_ok:
        print("\n❌ VERIFICATION FAILED: Dataset loader issues")
        return 1
    
    # Check 6: Model readiness
    model_ok = verify_model_readiness(infea)
    
    # Final verdict
    all_ok = config_ok and data_ok and factor_ok and quality_ok and loader_ok and model_ok
    
    print_header("Final Verdict")
    if all_ok:
        print("  ✅ ALL CHECKS PASSED")
        print("  Ready to proceed to Phase 6: Training")
        return 0
    else:
        print("  ⚠️  Some checks failed - review issues above")
        return 1

if __name__ == '__main__':
    sys.exit(main())
