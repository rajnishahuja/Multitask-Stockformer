#!/usr/bin/env python3
"""
Phase 8 Step 6: Run inference on test data using saved model
Generates prediction files for backtesting
"""

import numpy as np
import torch
import math
import argparse
import configparser
import os

from lib.Multitask_Stockformer_utils import log_string, metric, save_to_csv, StockDataset
from lib.graph_utils import loadGraph
from Stockformermodel.Multitask_Stockformer_models import Stockformer

# Config
CONFIG_FILE = "config/Phase8_NIFTY_Subset10_Alpha360.conf"
MODEL_FILE = "./output/NIFTY200_Subset10_Alpha360/best_model"  # or best_model_topk

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

# Build args object
class Args:
    cuda = config['train']['cuda']
    seed = int(config['train']['seed'])
    batch_size = int(config['train']['batch_size'])
    T1 = int(config['data']['T1'])
    T2 = int(config['data']['T2'])
    train_ratio = float(config['data']['train_ratio'])
    val_ratio = float(config['data']['val_ratio'])
    test_ratio = float(config['data']['test_ratio'])
    L = int(config['param']['layers'])
    h = int(config['param']['heads'])
    d = int(config['param']['dims'])
    j = int(config['param']['level'])
    s = float(config['param']['samples'])
    w = config['param']['wave']
    traffic_file = config['file']['traffic']
    indicator_file = config['file']['indicator']
    adj_file = config['file']['adj']
    adjgat_file = config['file']['adjgat']
    factor_dir = config['file']['factor_dir']
    model_file = MODEL_FILE

args = Args()
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

print("="*60)
print("PHASE 8 STEP 6: INFERENCE")
print("="*60)
print(f"Model: {MODEL_FILE}")
print(f"Device: {device}")

# Load test data
print("\nLoading test data...")
test_dataset = StockDataset(args, mode='test')
testXL = test_dataset.XL
testXH = test_dataset.XH
testXC = test_dataset.indicator_X
testTE = test_dataset.TE
testY = test_dataset.Y
testYC = test_dataset.indicator_Y
bonus_testX = test_dataset.bonus_X
infeature = test_dataset.infea

print(f"Test samples: {testXL.shape[0]}")
print(f"Input features: {infeature}")

# Load graph
adjgat = loadGraph(args)
adjgat = torch.from_numpy(adjgat).float().to(device)

# Build model
outfea_class = 2
outfea_regress = 1
model = Stockformer(infeature, args.h*args.d, outfea_class, outfea_regress, args.L, args.h, args.d, args.s, args.T1, args.T2, device).to(device)

# Load weights
model.load_state_dict(torch.load(MODEL_FILE))
print(f"Model loaded from {MODEL_FILE}")

# Run inference
model.eval()
num_test = testXL.shape[0]
num_batch = math.ceil(num_test / args.batch_size)

pred_class = []
pred_regress = []
label_class = []
label_regress = []

print("\nRunning inference...")
with torch.no_grad():
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_test, (batch_idx + 1) * args.batch_size)

        xl = torch.from_numpy(testXL[start_idx:end_idx]).float().to(device)
        xh = torch.from_numpy(testXH[start_idx:end_idx]).float().to(device)
        xc = torch.from_numpy(testXC[start_idx:end_idx]).float().to(device)
        te = torch.from_numpy(testTE[start_idx:end_idx]).to(device)
        bonus = torch.from_numpy(bonus_testX[start_idx:end_idx]).float().to(device)
        y = testY[start_idx:end_idx]
        yc = testYC[start_idx:end_idx]

        hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(xl, xh, te, bonus, xc, adjgat)

        pred_class.append(hat_y_class.cpu().numpy())
        pred_regress.append(hat_y_regress.cpu().numpy())
        label_class.append(yc)
        label_regress.append(y)

pred_class = np.concatenate(pred_class, axis=0)
pred_regress = np.concatenate(pred_regress, axis=0)
label_class = np.concatenate(label_class, axis=0)
label_regress = np.concatenate(label_regress, axis=0)

# Metrics
print("\nTest Metrics:")
for i in range(pred_regress.shape[1]):
    acc, mae, rmse, mape = metric(pred_regress[:, i, :], label_regress[:, i, :], pred_class[:, i, :], label_class[:, i, :])
    print(f"  Step {i+1}: acc={acc:.4f}, mae={mae:.4f}, rmse={rmse:.4f}")

# TopK Precision
K = 10
topk_precisions = []
for i in range(pred_regress.shape[0]):
    for t in range(pred_regress.shape[1]):
        preds = pred_regress[i, t, :]
        actuals = label_regress[i, t, :]
        valid = ~(np.isnan(preds) | np.isnan(actuals))
        if valid.sum() >= K:
            pred_top_k = set(np.argsort(preds[valid])[-K:])
            actual_top_k = set(np.argsort(actuals[valid])[-K:])
            topk_precisions.append(len(pred_top_k & actual_top_k) / K)

print(f"  TopK Precision: {np.mean(topk_precisions):.1%}")

# Save outputs
output_dir = './output/NIFTY200_Subset10_Alpha360'
save_to_csv(f'{output_dir}/regression_pred.csv', pred_regress[:, -1, :])
save_to_csv(f'{output_dir}/regression_label.csv', label_regress[:, -1])
save_to_csv(f'{output_dir}/classification_pred.csv', pred_class[:, -1, :])
save_to_csv(f'{output_dir}/classification_label.csv', label_class[:, -1])

print(f"\nPredictions saved to {output_dir}/")
print("="*60)
