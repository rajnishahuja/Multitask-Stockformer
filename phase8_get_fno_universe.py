"""
Phase 8: Step 1 - Get Historical F&O Universe from Bhavcopy
============================================================

This script downloads F&O bhavcopy from NSE for a specified date
and extracts the list of F&O eligible stocks.

Usage:
    python phase8_get_fno_universe.py

Output:
    - data/NIFTY200_Subset10/fno_universe_2020_04.txt (list of F&O stocks)
    - data/NIFTY200_Subset10/fno_bhavcopy_2020_04_01.csv (raw bhavcopy)
"""

import os
import pandas as pd
from datetime import date, timedelta
from jugaad_data.nse import bhavcopy_fo_save, bhavcopy_fo_raw

# Configuration
TARGET_DATE = date(2020, 4, 1)  # Start of our training period
OUTPUT_DIR = "./data/NIFTY200_Subset10"
BHAVCOPY_FILE = os.path.join(OUTPUT_DIR, f"fno_bhavcopy_{TARGET_DATE.strftime('%Y_%m_%d')}.csv")
UNIVERSE_FILE = os.path.join(OUTPUT_DIR, f"fno_universe_{TARGET_DATE.strftime('%Y_%m')}.txt")

def find_trading_day(start_date, max_attempts=10):
    """Find the nearest trading day (bhavcopy might not exist on holidays)"""
    current_date = start_date
    for i in range(max_attempts):
        try:
            print(f"Trying date: {current_date}")
            # Test if we can get data for this date
            df = bhavcopy_fo_raw(current_date)
            if df is not None and len(df) > 0:
                return current_date
        except Exception as e:
            print(f"  No data for {current_date}: {e}")
        current_date = current_date + timedelta(days=1)
    raise ValueError(f"Could not find trading day within {max_attempts} days of {start_date}")

def download_fno_bhavcopy(target_date, output_file):
    """Download F&O bhavcopy for a specific date"""
    print(f"\n{'='*60}")
    print(f"Downloading F&O Bhavcopy for {target_date}")
    print(f"{'='*60}")
    
    # Find the nearest trading day
    trading_date = find_trading_day(target_date)
    print(f"\nUsing trading date: {trading_date}")
    
    # Download the bhavcopy
    try:
        df = bhavcopy_fo_raw(trading_date)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ Bhavcopy saved to: {output_file}")
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        return df, trading_date
    except Exception as e:
        print(f"✗ Error downloading bhavcopy: {e}")
        raise

def extract_fno_universe(bhavcopy_df, output_file):
    """Extract unique stock symbols from F&O bhavcopy"""
    print(f"\n{'='*60}")
    print("Extracting F&O Stock Universe")
    print(f"{'='*60}")
    
    # Show available columns
    print(f"\nColumns in bhavcopy: {list(bhavcopy_df.columns)}")
    
    # Find the symbol column (might be named differently)
    symbol_col = None
    for col in ['SYMBOL', 'Symbol', 'TckrSymb', 'UNDERLYING']:
        if col in bhavcopy_df.columns:
            symbol_col = col
            break
    
    if symbol_col is None:
        print("Available columns:", bhavcopy_df.columns.tolist())
        raise ValueError("Could not find symbol column in bhavcopy")
    
    # Find instrument type column
    inst_col = None
    for col in ['INSTRUMENT', 'Instrument', 'FinInstrmTp']:
        if col in bhavcopy_df.columns:
            inst_col = col
            break
    
    print(f"\nUsing columns: symbol='{symbol_col}', instrument='{inst_col}'")
    
    if inst_col:
        print(f"\nInstrument types in bhavcopy:")
        print(bhavcopy_df[inst_col].value_counts())
        
        # Filter for stock futures (FUTSTK) - these are the F&O eligible stocks
        # Also include OPTSTK for stock options
        stock_instruments = ['FUTSTK', 'OPTSTK', 'STF', 'STO']
        mask = bhavcopy_df[inst_col].isin(stock_instruments)
        stock_df = bhavcopy_df[mask]
        print(f"\nFiltered to stock F&O instruments: {len(stock_df)} rows")
    else:
        stock_df = bhavcopy_df
    
    # Extract unique symbols
    symbols = stock_df[symbol_col].unique()
    symbols = sorted([s for s in symbols if s and isinstance(s, str)])
    
    # Remove index symbols (like NIFTY, BANKNIFTY)
    index_symbols = ['NIFTY', 'BANKNIFTY', 'NIFTYIT', 'FINNIFTY', 'MIDCPNIFTY']
    symbols = [s for s in symbols if s not in index_symbols]
    
    print(f"\n✓ Found {len(symbols)} unique F&O eligible stocks")
    print(f"\nFirst 20 symbols: {symbols[:20]}")
    print(f"Last 20 symbols: {symbols[-20:]}")
    
    # Save to file
    with open(output_file, 'w') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    
    print(f"\n✓ Universe saved to: {output_file}")
    
    return symbols

def main():
    print("="*60)
    print("Phase 8: Get Historical F&O Universe")
    print("="*60)
    print(f"\nTarget Date: {TARGET_DATE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Step 1: Download bhavcopy
    bhavcopy_df, actual_date = download_fno_bhavcopy(TARGET_DATE, BHAVCOPY_FILE)
    
    # Step 2: Extract universe
    symbols = extract_fno_universe(bhavcopy_df, UNIVERSE_FILE)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Trading date used: {actual_date}")
    print(f"✓ Bhavcopy file: {BHAVCOPY_FILE}")
    print(f"✓ Universe file: {UNIVERSE_FILE}")
    print(f"✓ Total F&O eligible stocks: {len(symbols)}")
    print(f"\nNext Step: Use this universe for data download in Phase 8 Step 2")
    
    return symbols

if __name__ == "__main__":
    main()
