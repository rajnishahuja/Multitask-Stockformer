
import pandas as pd
import numpy as np
import os

# Configuration - use consolidated output folder
DATASET = "NIFTY200_Subset10"
OUTPUT_DIR = f"./output/{DATASET}"
PRED_FILE = f"{OUTPUT_DIR}/regression_pred.csv"
LABEL_FILE = f"{OUTPUT_DIR}/regression_label.csv"
UNIVERSE_FILE = f"./data/{DATASET}/fno_universe_2020_04.txt"
ALL_DATES_FILE = f"./data/{DATASET}/dataset/label.csv"
T1 = 20
T2 = 2
TEST_RATIO = 0.125

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Raw Predictions (Days x Stocks)
    print("Loading predictions...")
    preds = pd.read_csv(PRED_FILE, header=None).values
    labels = pd.read_csv(LABEL_FILE, header=None).values
    print(f"Prediction Shape: {preds.shape}") # Expected (62, 143)
    
    # 2. Get Stock Symbols
    print("Loading stock symbols...")
    with open(UNIVERSE_FILE, 'r') as f:
        stocks = [line.strip() for line in f if line.strip()]
    
    # Filter stocks if needed (the training used 143, universe might have 144)
    if len(stocks) != preds.shape[1]:
        print(f"Warning: Stock count mismatch! Universe: {len(stocks)}, Preds: {preds.shape[1]}")
        # EQUITASBNK failed (known).
        stocks = [s for s in stocks if s != 'EQUITAS']
        stocks.sort()
        print(f"Adjusted stock count: {len(stocks)}")
        
    if len(stocks) != preds.shape[1]:
        raise ValueError(f"Still mismatch! {len(stocks)} vs {preds.shape[1]}")
        
    # 3. Get Dates
    print("Loading dates...")
    full_labels = pd.read_csv(ALL_DATES_FILE, usecols=[0], header=None)
    all_dates = full_labels[0].tolist()
    total_len = len(all_dates)
    print(f"Total dates: {total_len}")
    
    # Calculate Test Split
    test_len = round(total_len * TEST_RATIO)
    test_dates = all_dates[-test_len:]
    print(f"Test split length: {test_len}")
    
    # Calculate Prediction Dates (offset by P+Q-1)
    offset = T1 + T2 - 1
    pred_dates = test_dates[offset:]
    
    print(f"Prediction dates length: {len(pred_dates)}") # Should match 62
    
    if len(pred_dates) != preds.shape[0]:
        print(f"Mismatch in dates! Expected {len(pred_dates)}, got {preds.shape[0]}")
        min_len = min(len(pred_dates), preds.shape[0])
        pred_dates = pred_dates[:min_len]
        preds = preds[:min_len]
        labels = labels[:min_len]
    
    # 4. Create DataFrames (Stocks = Index, Dates = Columns) for Backtest Script
    # Transpose: (Days, Stocks) -> (Stocks, Days)
    pred_df = pd.DataFrame(preds.T, index=stocks, columns=pred_dates)
    label_df = pd.DataFrame(labels.T, index=stocks, columns=pred_dates)
    
    # 5. Save (filenames match phase8_backtest.py)
    pred_out = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    label_out = os.path.join(OUTPUT_DIR, "test_labels.csv")
    
    pred_df.to_csv(pred_out)
    label_df.to_csv(label_out)
    
    print(f"Saved to {pred_out}")
    print(f"Saved to {label_out}")

if __name__ == "__main__":
    main()
