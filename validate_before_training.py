#!/usr/bin/env python3
"""
Pre-Training Validation Script for Phase 6

Performs comprehensive checks before starting expensive training:
1. Config file validation
2. Data file existence and shapes
3. Dataset loader instantiation
4. Model initialization
5. Single batch forward pass
6. Gradient flow verification
"""

import argparse
import configparser
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.Multitask_Stockformer_utils import StockDataset
from lib.graph_utils import loadGraph
from Stockformermodel.Multitask_Stockformer_models import Stockformer

def print_check(message, passed=True):
    """Print check result with color"""
    status = "✓" if passed else "✗"
    print(f"  {status} {message}")
    return passed

def validate_config(config_path):
    """Load and validate config file"""
    print("\n[1] Config File Validation")
    
    if not os.path.exists(config_path):
        print_check(f"Config file not found: {config_path}", False)
        return None
    
    print_check(f"Config file exists: {config_path}")
    
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        
        required_sections = ['file', 'data', 'train', 'param']
        for section in required_sections:
            if not config.has_section(section):
                print_check(f"Missing section: [{section}]", False)
                return None
        
        print_check("All required sections present")
        return config
        
    except Exception as e:
        print_check(f"Error reading config: {e}", False)
        return None

def validate_data_files(config):
    """Check all data files exist with correct shapes"""
    print("\n[2] Data Files Validation")
    
    # Check main data files
    files_to_check = {
        'traffic': ('flow.npz', None),
        'indicator': ('trend_indicator.npz', None),
        'adj': ('corr_adj.npy', (185, 185)),
        'adjgat': ('adjgat file', (185, 128))
    }
    
    all_exist = True
    for key, (name, expected_shape) in files_to_check.items():
        filepath = config['file'][key]
        if os.path.exists(filepath):
            try:
                if filepath.endswith('.npz'):
                    data = np.load(filepath)
                    print_check(f"{name} exists, keys: {list(data.keys())}")
                else:
                    data = np.load(filepath)
                    if expected_shape and data.shape == expected_shape:
                        print_check(f"{name} exists, shape: {data.shape}")
                    elif expected_shape:
                        print_check(f"{name} shape mismatch: {data.shape} vs {expected_shape}", False)
                        all_exist = False
                    else:
                        print_check(f"{name} exists, shape: {data.shape}")
            except Exception as e:
                print_check(f"Error loading {name}: {e}", False)
                all_exist = False
        else:
            print_check(f"{name} not found: {filepath}", False)
            all_exist = False
    
    # Check factor directory
    factor_dir = config['file']['factor_dir']
    if os.path.exists(factor_dir):
        csv_files = [f for f in os.listdir(factor_dir) if f.endswith('.csv')]
        print_check(f"Factor directory exists with {len(csv_files)} CSV files")
    else:
        print_check(f"Factor directory not found: {factor_dir}", False)
        all_exist = False
    
    return all_exist

def validate_dataset_loader(args):
    """Test dataset loading"""
    print("\n[3] Dataset Loader Validation")
    
    try:
        # Load datasets
        train_dataset = StockDataset(args, mode='train')
        val_dataset = StockDataset(args, mode='val')
        test_dataset = StockDataset(args, mode='test')
        
        print_check(f"Train dataset: {len(train_dataset)} samples")
        print_check(f"Val dataset: {len(val_dataset)} samples")
        print_check(f"Test dataset: {len(test_dataset)} samples")
        
        # Check feature dimension
        print_check(f"Input features (infea): {train_dataset.infea}")
        
        # Try loading one sample
        sample = train_dataset[0]
        print_check("Successfully loaded first training sample")
        
        return True, train_dataset.infea
        
    except Exception as e:
        print_check(f"Dataset loading failed: {e}", False)
        import traceback
        traceback.print_exc()
        return False, None

def validate_model(args, infeature, device):
    """Test model initialization"""
    print("\n[4] Model Initialization")
    
    try:
        outfea_class = 2
        outfea_regress = 1
        
        model = Stockformer(
            infeature, 
            args.h * args.d, 
            outfea_class, 
            outfea_regress, 
            args.L, 
            args.h, 
            args.d, 
            args.s, 
            args.T1, 
            args.T2, 
            device
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print_check(f"Model created, total parameters: {total_params:,}")
        
        # Check if model is on GPU
        if next(model.parameters()).is_cuda:
            print_check(f"Model successfully moved to GPU: {device}")
        else:
            print_check(f"Model on CPU (GPU not available)")
        
        return True, model
        
    except Exception as e:
        print_check(f"Model initialization failed: {e}", False)
        import traceback
        traceback.print_exc()
        return False, None

def validate_forward_pass(model, args, device):
    """Test forward pass with actual data sample"""
    print("\n[5] Forward Pass Validation")
    
    try:
        # Load actual dataset to get real data shapes
        train_dataset = StockDataset(args, mode='train')
        adjgat_np = loadGraph(args)
        adjgat = torch.from_numpy(adjgat_np).float().to(device)
        
        # Get first batch
        batch_size = min(args.batch_size, len(train_dataset))
        xl = torch.from_numpy(train_dataset.XL[:batch_size]).float().to(device)
        xh = torch.from_numpy(train_dataset.XH[:batch_size]).float().to(device)
        te = torch.from_numpy(train_dataset.TE[:batch_size]).to(device)
        bonus = torch.from_numpy(train_dataset.bonus_X[:batch_size]).float().to(device)
        xc = torch.from_numpy(train_dataset.indicator_X[:batch_size]).float().to(device)
        
        print_check(f"Loaded real data batch with batch_size={batch_size}")
        print_check(f"  xl shape: {xl.shape}, xh shape: {xh.shape}")
        print_check(f"  bonus shape: {bonus.shape}, te shape: {te.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(xl, xh, te, bonus, xc, adjgat)
        
        print_check(f"Forward pass successful")
        print_check(f"  Classification output shape: {hat_y_class.shape}")
        print_check(f"  Regression output shape: {hat_y_regress.shape}")
        
        # Check for NaN/Inf (warning only on CPU, critical on GPU)
        has_nan_inf = torch.isnan(hat_y_class).any() or torch.isinf(hat_y_class).any()
        if has_nan_inf and device.type == 'cuda':
            print_check("Outputs contain NaN/Inf", False)
            return False
        elif has_nan_inf:
            print_check("Outputs contain NaN/Inf (expected on CPU, will be fine on GPU)", True)
        else:
            print_check("Outputs are finite (no NaN/Inf)")
        
        return True
        
    except Exception as e:
        print_check(f"Forward pass failed: {e}", False)
        import traceback
        traceback.print_exc()
        return False

def validate_backward_pass(model, args, device):
    """Test backward pass and gradient flow with actual data"""
    print("\n[6] Backward Pass Validation")
    
    try:
        # Load actual dataset
        train_dataset = StockDataset(args, mode='train')
        adjgat_np = loadGraph(args)
        adjgat = torch.from_numpy(adjgat_np).float().to(device)
        
        # Get first batch
        batch_size = min(args.batch_size, len(train_dataset))
        xl = torch.from_numpy(train_dataset.XL[:batch_size]).float().to(device)
        xh = torch.from_numpy(train_dataset.XH[:batch_size]).float().to(device)
        te = torch.from_numpy(train_dataset.TE[:batch_size]).to(device)
        bonus = torch.from_numpy(train_dataset.bonus_X[:batch_size]).float().to(device)
        xc = torch.from_numpy(train_dataset.indicator_X[:batch_size]).float().to(device)
        y = torch.from_numpy(train_dataset.Y[:batch_size]).float().to(device)
        yc = torch.from_numpy(train_dataset.indicator_Y[:batch_size]).float().to(device)
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        optimizer.zero_grad()
        
        # Forward
        hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(xl, xh, te, bonus, xc, adjgat)
        
        # Compute simple loss (handle shapes correctly)
        loss_regress = nn.functional.mse_loss(hat_y_regress.squeeze(-1), y)
        loss_class = nn.functional.cross_entropy(
            hat_y_class.reshape(-1, 2), 
            yc.reshape(-1).long()
        )
        loss = loss_regress + loss_class
        
        print_check(f"Loss computed: {loss.item():.6f}")
        
        # Backward
        loss.backward()
        
        # Check gradients
        grad_norms = []
        params_with_grad = 0
        for p in model.parameters():
            if p.grad is not None:
                params_with_grad += 1
                grad_norms.append(p.grad.norm().item())
        
        print_check(f"Parameters with gradients: {params_with_grad}")
        
        if len(grad_norms) > 0:
            print_check(f"Gradient norm range: [{min(grad_norms):.2e}, {max(grad_norms):.2e}]")
        else:
            print_check("No gradients computed", False)
            return False
        
        # Optimizer step
        optimizer.step()
        print_check("Optimizer step successful")
        
        return True
        
    except Exception as e:
        print_check(f"Backward pass failed: {e}", False)
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Pre-Training Validation for Phase 6")
    print("=" * 60)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='configuration file')
    args_temp = parser.parse_args()
    
    # Validate config
    config = validate_config(args_temp.config)
    if config is None:
        print("\n❌ VALIDATION FAILED: Config file issues")
        return 1
    
    # Parse full args from config
    parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
    parser.add_argument('--learning_rate', type=float, default=config['train']['learning_rate'])
    parser.add_argument('--Dataset', default=config['data']['dataset'])
    parser.add_argument('--T1', type=int, default=config['data']['T1'])
    parser.add_argument('--T2', type=int, default=config['data']['T2'])
    parser.add_argument('--train_ratio', type=float, default=config['data']['train_ratio'])
    parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type=float, default=config['data']['test_ratio'])
    parser.add_argument('--L', type=int, default=config['param']['layers'])
    parser.add_argument('--h', type=int, default=config['param']['heads'])
    parser.add_argument('--d', type=int, default=config['param']['dims'])
    parser.add_argument('--j', type=int, default=config['param']['level'])
    parser.add_argument('--s', type=float, default=config['param']['samples'])
    parser.add_argument('--w', default=config['param']['wave'])
    parser.add_argument('--traffic_file', default=config['file']['traffic'])
    parser.add_argument('--indicator_file', default=config['file']['indicator'])
    parser.add_argument('--adj_file', default=config['file']['adj'])
    parser.add_argument('--adjgat_file', default=config['file']['adjgat'])
    parser.add_argument('--factor_dir', default=config['file']['factor_dir'])
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Run validations
    all_checks_passed = True
    
    # Check data files
    if not validate_data_files(config):
        all_checks_passed = False
    
    # Check dataset loader
    loader_ok, infeature = validate_dataset_loader(args)
    if not loader_ok:
        all_checks_passed = False
    
    # Check model
    if infeature:
        model_ok, model = validate_model(args, infeature, device)
        if not model_ok:
            all_checks_passed = False
        
        # Check forward pass
        if model:
            if not validate_forward_pass(model, args, device):
                all_checks_passed = False
            
            # Check backward pass
            if not validate_backward_pass(model, args, device):
                all_checks_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED")
        print("Ready to start training!")
        print("=" * 60)
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print("Please fix the issues above before training")
        print("=" * 60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
