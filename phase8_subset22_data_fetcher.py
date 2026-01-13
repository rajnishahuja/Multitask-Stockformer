#!/usr/bin/env python3
"""
Phase 9: Yahoo Finance Data Fetcher (Subset 22 - Live Trading)

Downloads historical OHLCV data for Subset 22 (Jun 2023 - Jan 2026).
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

# Check for yfinance
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    os.system("pip install yfinance")
    import yfinance as yf

# Configuration
FROM_DATE = "2023-06-01"
TO_DATE = "2026-01-11" # Include Jan 10
OUTPUT_DIR = "./data/NIFTY200_Subset22"
RAW_YAHOO_DIR = os.path.join(OUTPUT_DIR, "raw_yahoo")
UNIVERSE_FILE = os.path.join(OUTPUT_DIR, "fno_stocks.txt")
LOG_FILE = os.path.join(OUTPUT_DIR, "subset22_fetcher.log")

# Data quality thresholds
REQUIRED_COVERAGE = 0.80

# Symbol mapping (Inherited from Phase 8, updated if needed)
SYMBOL_MAPPING = {
    'AMARAJABAT': 'ARE&M',           
    'CADILAHC': 'ZYDUSLIFE',         
    'CENTURYTEX': 'ABREL',           
    'EQUITAS': 'EQUITASBNK',         
    'GMRINFRA': 'GMRAIRPORT',        
    'HDFC': 'HDFCBANK',              
    'IBULHSGFIN': 'IBULLSLTD',       
    'INFRATEL': 'INDUSTOWER',        
    'L&TFH': 'LTF',                  
    'MCDOWELL-N': 'UNITDSPR',        
    'MINDTREE': 'LTIM',              
    'MOTHERSUMI': 'MOTHERSON',       
    'NIITTECH': 'COFORGE',           
    'PEL': 'POONAWALLA',             
    'PVR': 'PVRINOX',                
    'SRTRANSFIN': 'SHRIRAMFIN',      
    'TATAMOTORS': 'TATAMOTORS',      # Ensure standard symbol
    'UJJIVAN': 'UJJIVANSFB',         
}

SKIP_SYMBOLS = set()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_yahoo_symbol(symbol: str) -> str:
    mapped_symbol = SYMBOL_MAPPING.get(symbol, symbol)
    return f"{mapped_symbol}.NS"

def fetch_yahoo_data(symbol: str, from_date: str, to_date: str, 
                     logger: logging.Logger) -> Optional[pd.DataFrame]:
    yahoo_symbol = get_yahoo_symbol(symbol)
    
    try:
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(start=from_date, end=to_date, auto_adjust=False)
        
        if df.empty:
            logger.warning(f"No data returned for {symbol} ({yahoo_symbol})")
            return None
        
        df = df.reset_index()
        
        df = df.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj_Close',
            'Volume': 'Volume'
        })
        
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        df = df[[c for c in columns_to_keep if c in df.columns]]
        
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        logger.info(f"✓ Fetched {len(df)} records for {symbol} ({yahoo_symbol})")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {symbol} ({yahoo_symbol}): {e}")
        return None

def validate_data(df: pd.DataFrame, symbol: str, logger: logging.Logger) -> Dict:
    from_dt = datetime.strptime(FROM_DATE, '%Y-%m-%d').date()
    to_dt = datetime.strptime(TO_DATE, '%Y-%m-%d').date()
    # Handle future ToDate
    today = datetime.now().date()
    if to_dt > today:
        to_dt = today

    validation = {
        'symbol': symbol,
        'total_records': len(df),
        'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
        'coverage': 0.0,
        'zero_volume_days': 0,
        'large_gaps': 0,
        'data_quality': 'PASS'
    }
    
    total_days = (to_dt - from_dt).days
    expected_trading_days = int(total_days * 252 / 365)
    actual_days = len(df)
    
    validation['coverage'] = actual_days / expected_trading_days if expected_trading_days > 0 else 0
    
    if validation['coverage'] < REQUIRED_COVERAGE:
        validation['data_quality'] = 'FAIL'
        logger.warning(f"{symbol}: Insufficient coverage {validation['coverage']:.1%}")
        return validation

    zero_vol = (df['Volume'] == 0).sum()
    if zero_vol > len(df) * 0.1:
        validation['data_quality'] = 'WARN'
    
    return validation

def load_universe() -> List[str]:
    if not os.path.exists(UNIVERSE_FILE):
        print(f"Error: Universe file not found: {UNIVERSE_FILE}")
        sys.exit(1)
    
    with open(UNIVERSE_FILE, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    return symbols

def main():
    os.makedirs(RAW_YAHOO_DIR, exist_ok=True)
    logger = setup_logging()
    
    print("="*80)
    print("Phase 9: Yahoo Finance Data Fetcher (Subset 22)")
    print("="*80)
    
    symbols = load_universe()
    print(f"\n✓ Loaded {len(symbols)} stocks from F&O universe")
    
    # Download data
    successful = 0
    failed = []
    
    for idx, symbol in enumerate(symbols, 1):
        mapped = SYMBOL_MAPPING.get(symbol, symbol)
        print(f"\n[{idx}/{len(symbols)}] Processing {symbol} -> {mapped}.NS...")
        
        df = fetch_yahoo_data(symbol, FROM_DATE, TO_DATE, logger)
        
        if df is not None and len(df) > 0:
            validation = validate_data(df, symbol, logger)
            
            if validation['data_quality'] != 'FAIL':
                output_file = os.path.join(RAW_YAHOO_DIR, f"{mapped}.csv")
                df.to_csv(output_file, index=False)
                print(f"✓ Saved {len(df)} records to {output_file}")
                successful += 1
            else:
                print(f"⚠ Skipped {symbol} due to insufficient data coverage")
                failed.append(symbol)
            
            time.sleep(0.3)
        else:
            failed.append(symbol)
            print(f"✗ Failed to download {symbol}")
    
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"Successful: {successful}/{len(symbols)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed symbols: {failed}")

if __name__ == "__main__":
    main()
