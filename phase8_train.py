#!/usr/bin/env python3
"""
Phase 8: Multitask-Stockformer Training Script with Resume Capability

Features:
- Resume training from checkpoint
- Save best model and training state after each epoch
- Configurable start epoch

Usage:
    python phase8_train.py --config config/Phase8_NIFTY_Subset10.conf
    python phase8_train.py --config config/Phase8_NIFTY_Subset10.conf --resume  # Resume from last checkpoint
"""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from scipy.stats import spearmanr
import argparse
import configparser
import math
import csv
import random
import json
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from lib.Multitask_Stockformer_utils import log_string, _compute_regression_loss, _compute_class_loss, _compute_regression_loss_with_listmle, metric, save_to_csv, StockDataset
from lib.graph_utils import loadGraph
from Stockformermodel.Multitask_Stockformer_models import Stockformer

import os
from torch.utils.tensorboard import SummaryWriter

# Force unbuffered output
import sys
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

# Initialize argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--resume", action='store_true', help='Resume training from last checkpoint')

# First parse to get config file only
args, unknown = parser.parse_known_args()

# Read configuration file
config = configparser.ConfigParser()
config.read(args.config)

# Add other configuration parameters
parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
parser.add_argument('--seed', type=int, default=config['train']['seed'])
parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
parser.add_argument('--max_epoch', type=int, default=config['train']['max_epoch'])
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
parser.add_argument('--model_file', default=config['file']['model'])
parser.add_argument('--log_file', default=config['file']['log'])

# Final argument parsing
args = parser.parse_args()

# Check and create log file directory
log_directory = os.path.dirname(args.log_file)
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    print(f"Directory created for log file: {log_directory}")

# Open log file in append mode if resuming
log_mode = 'a' if args.resume else 'w'
log = open(args.log_file, log_mode)

# Check and create model file directory
model_directory = os.path.dirname(args.model_file)
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    print(f"Directory created for model file: {model_directory}")

# Checkpoint file for resume capability
checkpoint_file = args.model_file + '_checkpoint.json'

print(f"Model file path: {args.model_file}")
print(f"Checkpoint file: {checkpoint_file}")
if args.resume:
    print("Resume mode: ON")

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# Dynamic TensorBoard path based on dataset name
tensorboard_folder = f'./runs/Multitask_Stockformer/{args.Dataset}_Alpha158_MVP'

if not os.path.exists(tensorboard_folder):
    os.makedirs(tensorboard_folder)
    log_string(log, f"Folder created: {tensorboard_folder}")
else:
    log_string(log, f"Folder already exists: {tensorboard_folder}")

# Determine the name for the new subfolder
subfolders = [f.name for f in os.scandir(tensorboard_folder) if f.is_dir()]
versions = [int(folder.replace('version', '')) for folder in subfolders if folder.startswith('version')]
next_version = 0 if not versions else max(versions) + 1

# If resuming, use the latest version
if args.resume and versions:
    next_version = max(versions)
    
new_folder = os.path.join(tensorboard_folder, f'version{next_version}')

if not os.path.exists(new_folder):
    os.makedirs(new_folder)
    log_string(log, f"Subfolder created: {new_folder}")

tensor_writer = SummaryWriter(new_folder)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(epoch, best_mae, epochs_no_improve, optimizer_state=None):
    """Save training state to checkpoint file"""
    checkpoint = {
        'epoch': int(epoch),
        'best_mae': float(best_mae),  # Convert numpy float32 to Python float
        'epochs_no_improve': int(epochs_no_improve),
    }
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f)
    
    # Also save optimizer state separately for full resume
    if optimizer_state is not None:
        optimizer_file = args.model_file + '_optimizer.pt'
        torch.save(optimizer_state, optimizer_file)


def load_checkpoint():
    """Load training state from checkpoint file"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint
    return None


def res(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat, epoch, log, tensor_writer):
    model.eval()
    num_val = valXL.shape[0]
    num_batch = math.ceil(num_val / args.batch_size)

    pred_class = []
    pred_regress = []
    label_class = []
    label_regress = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(valXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(valXH[start_idx : end_idx]).float().to(device)
                xc = torch.from_numpy(valXC[start_idx : end_idx]).float().to(device)
                te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)
                bonus = torch.from_numpy(bonus_valX[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                yc = valYC[start_idx : end_idx]

                hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(xl, xh, te, bonus, xc, adjgat)

                pred_class.append(hat_y_class.cpu().numpy())
                pred_regress.append(hat_y_regress.cpu().numpy())
                label_class.append(yc)
                label_regress.append(y)
    
    pred_class = np.concatenate(pred_class, axis=0)
    pred_regress = np.concatenate(pred_regress, axis=0)
    label_class = np.concatenate(label_class, axis=0)
    label_regress = np.concatenate(label_regress, axis=0)

    accs = []
    maes = []
    rmses = []
    mapes = []

    for i in range(pred_class.shape[1]):
        acc, mae, rmse, mape = metric(pred_regress[:, i, :], label_regress[:, i, :], pred_class[:, i, :], label_class[:, i, :])
        accs.append(acc)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, f'step {i+1}, acc: {acc:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')

    avg_acc = np.mean(accs)
    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    avg_mape = np.mean(mapes)
    log_string(log, f'average, acc: {avg_acc:.4f}, mae: {avg_mae:.4f}, rmse: {avg_rmse:.4f}, mape: {avg_mape:.4f}')

    tensor_writer.add_scalar('Val/Average_Accuracy', avg_acc, epoch)
    tensor_writer.add_scalar('Val/Average_MAE', avg_mae, epoch)
    tensor_writer.add_scalar('Val/Average_RMSE', avg_rmse, epoch)
    tensor_writer.add_scalar('Val/Average_MAPE', avg_mape, epoch)
    
    # === Phase 8 Continued: TopK Precision ===
    # TopK Precision: % of predicted top 10 that are in actual top 10
    # NOTE: We dropped Rank IC because it showed no correlation with TopK Precision.
    #       A model can have good TopK but poor overall ranking (and vice versa).
    #       Since we only trade TopK stocks, TopK Precision is what matters.
    topk_precisions = []
    K = 10  # Top K stocks for precision calculation
    
    for i in range(pred_regress.shape[0]):  # For each sample
        for t in range(pred_regress.shape[1]):  # For each timestep
            preds = pred_regress[i, t, :]
            actuals = label_regress[i, t, :]
            
            # Skip if too few valid values
            valid_mask = ~(np.isnan(preds) | np.isnan(actuals))
            if valid_mask.sum() < K:
                continue
            
            preds_valid = preds[valid_mask]
            actuals_valid = actuals[valid_mask]
            
            # TopK Precision
            pred_top_k = set(np.argsort(preds_valid)[-K:])
            actual_top_k = set(np.argsort(actuals_valid)[-K:])
            precision = len(pred_top_k & actual_top_k) / K
            topk_precisions.append(precision)
    
    avg_topk_precision = np.mean(topk_precisions) if topk_precisions else 0.0
    
    log_string(log, f'TopK Precision: {avg_topk_precision:.1%}')
    tensor_writer.add_scalar('Val/TopK_Precision', avg_topk_precision, epoch)
    
    return avg_acc, avg_mae, avg_rmse, avg_mape, avg_topk_precision


def test_res(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat):
    model.eval()
    num_val = valXL.shape[0]
    num_batch = math.ceil(num_val / args.batch_size)

    pred_class = []
    pred_regress = []
    label_class = []
    label_regress = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(valXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(valXH[start_idx : end_idx]).float().to(device)
                xc = torch.from_numpy(valXC[start_idx : end_idx]).float().to(device)
                te = torch.from_numpy(valTE[start_idx : end_idx]).to(device)
                bonus = torch.from_numpy(bonus_valX[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]
                yc = valYC[start_idx : end_idx]

                hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(xl, xh, te, bonus, xc, adjgat)

                pred_class.append(hat_y_class.cpu().numpy())
                pred_regress.append(hat_y_regress.cpu().numpy())
                label_class.append(yc)
                label_regress.append(y)
    
    pred_class = np.concatenate(pred_class, axis=0)
    pred_regress = np.concatenate(pred_regress, axis=0)
    label_class = np.concatenate(label_class, axis=0)
    label_regress = np.concatenate(label_regress, axis=0)

    accs = []
    maes = []
    rmses = []
    mapes = []

    for i in range(pred_regress.shape[1]):
        acc, mae, rmse, mape = metric(pred_regress[:, i, :], label_regress[:, i, :], pred_class[:, i, :], label_class[:, i, :])
        accs.append(acc)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log,'step %d, acc: %.4f, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, acc, mae, rmse, mape))
    
    avg_acc = np.mean(accs)
    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    avg_mape = np.mean(mapes)
    log_string(log, 'average, acc: %.4f, mae: %.4f, rmse: %.4f, mape: %.4f' % (avg_acc, avg_mae, avg_rmse, avg_mape))
    
    # Dynamic output paths based on dataset name - all outputs in one folder
    output_dir = f'./output/{args.Dataset}'
    os.makedirs(output_dir, exist_ok=True)
    
    save_to_csv(f'{output_dir}/classification_pred.csv', pred_class[:, -1, :])
    save_to_csv(f'{output_dir}/classification_label.csv', label_class[:, -1])
    save_to_csv(f'{output_dir}/regression_pred.csv', pred_regress[:, -1, :])
    save_to_csv(f'{output_dir}/regression_label.csv', label_regress[:, -1])

    return avg_acc, avg_mae, avg_rmse, avg_mape


def train(model, trainXL, trainXH, trainXC, bonus_trainX, trainTE, trainY, trainYL, trainYC, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat):
    num_train = trainXL.shape[0]
    best_mae = float('inf')
    best_topk_prec = 0.0  # Track best TopK Precision separately
    early_stop_patience = 30
    epochs_no_improve = 0
    start_epoch = 1
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)  # Phase 8: added L2 regularization
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=20,    
        verbose=False, threshold=0.001, threshold_mode='rel', 
        cooldown=0, min_lr=2e-6, eps=1e-08
    )
    
    # Resume from checkpoint if requested
    if args.resume:
        checkpoint = load_checkpoint()
        if checkpoint is not None:
            start_epoch = checkpoint['epoch'] + 1
            best_mae = checkpoint['best_mae']
            epochs_no_improve = checkpoint['epochs_no_improve']
            
            # Load model weights
            if os.path.exists(args.model_file):
                model.load_state_dict(torch.load(args.model_file))
                log_string(log, f"Resumed from epoch {checkpoint['epoch']} with best_mae={best_mae:.4f}")
            
            # Load optimizer state if available
            optimizer_file = args.model_file + '_optimizer.pt'
            if os.path.exists(optimizer_file):
                optimizer.load_state_dict(torch.load(optimizer_file))
                log_string(log, f"Optimizer state restored")
        else:
            log_string(log, "No checkpoint found, starting fresh")
    
    for epoch in tqdm(range(start_epoch, args.max_epoch + 1)):
        model.train()
        train_l_sum, batch_count, start = 0.0, 0, time.time()
        
        # Shuffle training data
        permutation = np.random.permutation(num_train)
        trainXL = trainXL[permutation]
        trainXH = trainXH[permutation]
        trainXC = trainXC[permutation]
        trainTE = trainTE[permutation]
        trainY = trainY[permutation]
        trainYL = trainYL[permutation]
        trainYC = trainYC[permutation]
        bonus_trainX = bonus_trainX[permutation]
        num_batch = math.ceil(num_train / args.batch_size)

        with tqdm(total=num_batch, desc=f"Epoch {epoch}") as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                xl = torch.from_numpy(trainXL[start_idx : end_idx]).float().to(device)
                xh = torch.from_numpy(trainXH[start_idx : end_idx]).float().to(device)
                xc = torch.from_numpy(trainXC[start_idx : end_idx]).float().to(device)
                y = torch.from_numpy(trainY[start_idx : end_idx]).float().to(device)
                yl = torch.from_numpy(trainYL[start_idx : end_idx]).float().to(device)
                yc = torch.from_numpy(trainYC[start_idx : end_idx]).float().to(device)
                te = torch.from_numpy(trainTE[start_idx : end_idx]).to(device)
                bonus = torch.from_numpy(bonus_trainX[start_idx : end_idx]).float().to(device)
                
                optimizer.zero_grad()
                hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(xl, xh, te, bonus, xc, adjgat)

                # Phase 8 Step 5b: Use ListMLE loss (50% MAE + 50% ListMLE ranking loss)
                # ListMLE optimizes probability of correct full ranking
                loss_regress = _compute_regression_loss_with_listmle(y, hat_y_regress, listmle_weight=0.5) + \
                               _compute_regression_loss_with_listmle(yl, hat_y_l_regress, listmle_weight=0.5)
                loss_class = _compute_class_loss(yc, hat_y_class) + _compute_class_loss(yc, hat_y_l_class)
                loss = loss_regress + loss_class

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                train_l_sum += loss.cpu().item()
                batch_count += 1
                pbar.update(1)

        log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
              % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))

        tensor_writer.add_scalar('training loss', train_l_sum / batch_count, epoch)

        # Validation
        acc, mae, rmse, mape, topk_prec = res(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat, epoch, log, tensor_writer)
        lr_scheduler.step(mae)

        # Check for MAE improvement (original model selection)
        if mae < best_mae:
            best_mae = mae
            # Save best MAE model
            torch.save(model.state_dict(), args.model_file)
            log_string(log, f'Epoch {epoch}: New best mae: {best_mae:.4f}, Model saved.')
        
        # Check for TopK Precision improvement (primary model selection for ListMLE)
        # Early stopping is now based on TopK, not MAE
        if topk_prec > best_topk_prec:
            best_topk_prec = topk_prec
            epochs_no_improve = 0  # Reset early stopping counter
            topk_model_file = args.model_file + '_topk'
            torch.save(model.state_dict(), topk_model_file)
            log_string(log, f'Epoch {epoch}: New best TopK Precision: {best_topk_prec:.1%}, TopK model saved.')
        else:
            epochs_no_improve += 1
            log_string(log, f'Epoch {epoch}: No TopK improvement for {epochs_no_improve} epochs (best: {best_topk_prec:.1%})')
        
        # Save checkpoint after every epoch (for resume capability)
        save_checkpoint(epoch, best_mae, epochs_no_improve, optimizer.state_dict())
        
        # Early stopping check - now based on TopK Precision
        if epochs_no_improve >= early_stop_patience:
            log_string(log, f'Early stopping triggered after {epoch} epochs (no TopK improvement for {early_stop_patience} epochs)')
            break


def test(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat):
    try:
        model.load_state_dict(torch.load(args.model_file))
        total_params = sum(p.numel() for p in model.parameters())
        log_string(log, 'Total parameters: {}'.format(total_params))
    except EOFError:
        print(f"Error: Unable to load model from {args.model_file}")
        return

    acc, mae, rmse, mape = test_res(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat)
    return acc, mae, rmse, mape


if __name__ == '__main__':
    log_string(log, "loading data....")
    outfea_class = 2
    outfea_regress = 1
    
    train_dataset = StockDataset(args, mode='train')
    val_dataset = StockDataset(args, mode='val')
    test_dataset = StockDataset(args, mode='test')
    
    # Train data
    trainXL = train_dataset.XL
    trainXH = train_dataset.XH
    trainXC = train_dataset.indicator_X
    trainTE = train_dataset.TE
    trainY = train_dataset.Y
    trainYL = train_dataset.YL
    trainYC = train_dataset.indicator_Y
    bonus_trainX = train_dataset.bonus_X
    
    # Val data
    valXL = val_dataset.XL
    valXH = val_dataset.XH
    valXC = val_dataset.indicator_X
    valTE = val_dataset.TE
    valY = val_dataset.Y
    valYL = val_dataset.YL
    valYC = val_dataset.indicator_Y
    bonus_valX = val_dataset.bonus_X
    
    # Test data
    testXL = test_dataset.XL
    testXH = test_dataset.XH
    testXC = test_dataset.indicator_X
    testTE = test_dataset.TE
    testY = test_dataset.Y
    testYL = test_dataset.YL
    testYC = test_dataset.indicator_Y
    bonus_testX = test_dataset.bonus_X
    
    # Feature count
    infeature = train_dataset.infea
    
    # Load graph
    adjgat = loadGraph(args)
    adjgat = torch.from_numpy(adjgat).float().to(device)
    log_string(log, "loading end....")

    log_string(log, "constructing model begin....")
    model = Stockformer(infeature, args.h*args.d, outfea_class, outfea_regress, args.L, args.h, args.d, args.s, args.T1, args.T2, device).to(device)
    log_string(log, "constructing model end....")

    log_string(log, "training begin....")
    train(model, trainXL, trainXH, trainXC, bonus_trainX, trainTE, trainY, trainYL, trainYC, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat)
    log_string(log, "training end....")

    log_string(log, "testing begin....")
    test(model, testXL, testXH, testXC, bonus_testX, testTE, testY, testYC, adjgat)
    log_string(log, "testing end....")
