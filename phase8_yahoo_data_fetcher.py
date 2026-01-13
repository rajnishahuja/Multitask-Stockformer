#!/usr/bin/env python3
"""
Phase 8: Yahoo Finance Data Fetcher

Downloads historical OHLCV data with ADJUSTED prices from Yahoo Finance.
Yahoo provides split/dividend adjusted prices which are more accurate for ML models.

Usage:
    python phase8_yahoo_data_fetcher.py

Output:
    - data/NIFTY200_Subset10/raw_yahoo/{SYMBOL}.csv (one file per stock)
    - data/NIFTY200_Subset10/yahoo_data_quality_report.txt
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
FROM_DATE = "2020-04-01"
TO_DATE = "2022-11-30"
OUTPUT_DIR = "./data/NIFTY200_Subset10"
RAW_YAHOO_DIR = os.path.join(OUTPUT_DIR, "raw_yahoo")
UNIVERSE_FILE = os.path.join(OUTPUT_DIR, "fno_universe_2020_04.txt")
LOG_FILE = os.path.join(OUTPUT_DIR, "yahoo_fetcher.log")

# Data quality thresholds
REQUIRED_COVERAGE = 0.80  # Require 80% coverage of trading days

# Symbol mapping for renamed stocks (old F&O symbol -> Yahoo NSE symbol)
SYMBOL_MAPPING = {
    'AMARAJABAT': 'ARE&M',           # Amara Raja Batteries -> Amara Raja Energy & Mobility
    'CADILAHC': 'ZYDUSLIFE',         # Cadila Healthcare -> Zydus Lifesciences
    'CENTURYTEX': 'ABREL',           # Century Textiles -> Aditya Birla Real Estate
    'EQUITAS': 'EQUITASBNK',         # Equitas Holdings -> Equitas Small Finance Bank
    'GMRINFRA': 'GMRAIRPORT',        # GMR Infra -> GMR Airports
    'HDFC': 'HDFCBANK',              # HDFC merged with HDFC Bank
    'IBULHSGFIN': 'IBULLSLTD',       # Indiabulls Housing -> Indiabulls Ltd
    'INFRATEL': 'INDUSTOWER',        # Bharti Infratel -> Indus Towers
    'L&TFH': 'LTF',                  # L&T Finance Holdings -> LTF
    'MCDOWELL-N': 'UNITDSPR',        # United Spirits
    'MINDTREE': 'LTIM',              # Mindtree merged with L&T Infotech
    'MOTHERSUMI': 'MOTHERSON',       # Motherson Sumi -> Motherson
    'NIITTECH': 'COFORGE',           # NIIT Tech -> Coforge
    'PEL': 'POONAWALLA',             # Piramal Enterprises split
    'PVR': 'PVRINOX',                # PVR merged with INOX
    'SRTRANSFIN': 'SHRIRAMFIN',      # Shriram Transport -> Shriram Finance
    'TATAMOTORS': 'TMPV',            # Tata Motors -> TMPV post demerger
    'UJJIVAN': 'UJJIVANSFB',         # Ujjivan -> Ujjivan Small Finance Bank
}

# Stocks to skip (no viable successor)
SKIP_SYMBOLS = set()

def setup_logging():
    """Configure logging"""
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
    """Convert NSE symbol to Yahoo Finance format"""
    # Apply mapping for renamed stocks
    mapped_symbol = SYMBOL_MAPPING.get(symbol, symbol)
    # Add .NS suffix for NSE
    return f"{mapped_symbol}.NS"

def fetch_yahoo_data(symbol: str, from_date: str, to_date: str, 
                     logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Fetch historical data from Yahoo Finance with adjusted prices"""
    yahoo_symbol = get_yahoo_symbol(symbol)
    
    try:
        # Download data from Yahoo Finance
        ticker = yf.Ticker(yahoo_symbol)
        df = ticker.history(start=from_date, end=to_date, auto_adjust=False)
        
        if df.empty:
            logger.warning(f"No data returned for {symbol} ({yahoo_symbol})")
            return None
        
        # Reset index to get Date as column
        df = df.reset_index()
        
        # Rename columns to match our format
        df = df.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Adj Close': 'Adj_Close',
            'Volume': 'Volume'
        })
        
        # Select columns we need (include both Close and Adj_Close)
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        df = df[[c for c in columns_to_keep if c in df.columns]]
        
        # Convert Date to date only (remove time)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        
        logger.info(f"✓ Fetched {len(df)} records for {symbol} ({yahoo_symbol})")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching {symbol} ({yahoo_symbol}): {e}")
        return None

def validate_data(df: pd.DataFrame, symbol: str, logger: logging.Logger) -> Dict:
    """Validate data completeness and quality"""
    from_dt = datetime.strptime(FROM_DATE, '%Y-%m-%d').date()
    to_dt = datetime.strptime(TO_DATE, '%Y-%m-%d').date()
    
    validation = {
        'symbol': symbol,
        'total_records': len(df),
        'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
        'coverage': 0.0,
        'zero_volume_days': 0,
        'large_gaps': 0,
        'large_gaps_adj': 0,
        'data_quality': 'PASS'
    }
    
    # Calculate expected trading days (~252 days/year)
    total_days = (to_dt - from_dt).days
    expected_trading_days = int(total_days * 252 / 365)
    actual_days = len(df)
    
    validation['coverage'] = actual_days / expected_trading_days if expected_trading_days > 0 else 0
    
    if validation['coverage'] < REQUIRED_COVERAGE:
        validation['data_quality'] = 'FAIL'
        logger.warning(f"{symbol}: Insufficient coverage {validation['coverage']:.1%}")
        return validation
    
    # Check for zero volume days
    zero_vol = (df['Volume'] == 0).sum()
    validation['zero_volume_days'] = int(zero_vol)
    if zero_vol > len(df) * 0.1:
        validation['data_quality'] = 'WARN'
    
    # Check for large price gaps in raw Close
    df_sorted = df.sort_values('Date')
    pct_change = df_sorted['Close'].pct_change().abs()
    large_gaps = (pct_change > 0.40).sum()
    validation['large_gaps'] = int(large_gaps)
    
    # Check for large price gaps in Adjusted Close
    if 'Adj_Close' in df_sorted.columns:
        adj_pct_change = df_sorted['Adj_Close'].pct_change().abs()
        large_gaps_adj = (adj_pct_change > 0.40).sum()
        validation['large_gaps_adj'] = int(large_gaps_adj)
        
        # If adjusted has fewer gaps, that's good!
        if large_gaps > 0 and large_gaps_adj < large_gaps:
            logger.info(f"{symbol}: Adjusted prices reduced gaps from {large_gaps} to {large_gaps_adj}")
    
    if validation['large_gaps_adj'] > 0:
        validation['data_quality'] = 'WARN'
    
    return validation

def load_universe() -> List[str]:
    """Load F&O universe symbols"""
    if not os.path.exists(UNIVERSE_FILE):
        print(f"Error: Universe file not found: {UNIVERSE_FILE}")
        print("Please run phase8_get_fno_universe.py first!")
        sys.exit(1)
    
    with open(UNIVERSE_FILE, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    return symbols

def main():
    # Setup
    os.makedirs(RAW_YAHOO_DIR, exist_ok=True)
    logger = setup_logging()
    
    print("="*80)
    print("Phase 8: Yahoo Finance Data Fetcher (Adjusted Prices)")
    print("="*80)
    
    # Load universe
    symbols = load_universe()
    print(f"\n✓ Loaded {len(symbols)} stocks from F&O universe")
    
    # Apply symbol mapping
    available_symbols = []
    for symbol in symbols:
        if symbol in SKIP_SYMBOLS:
            continue
        mapped = SYMBOL_MAPPING.get(symbol, symbol)
        available_symbols.append((symbol, mapped))
    
    print(f"✓ {len(available_symbols)} symbols to download")
    print(f"\nDate range: {FROM_DATE} to {TO_DATE}")
    print(f"Output directory: {RAW_YAHOO_DIR}")
    print("="*80 + "\n")
    
    # Download data
    validation_results = []
    successful = 0
    failed = []
    
    for idx, (original_symbol, mapped_symbol) in enumerate(available_symbols, 1):
        print(f"\n[{idx}/{len(available_symbols)}] Processing {original_symbol} -> {mapped_symbol}.NS...")
        
        # Fetch data
        df = fetch_yahoo_data(original_symbol, FROM_DATE, TO_DATE, logger)
        
        if df is not None and len(df) > 0:
            # Validate data
            validation = validate_data(df, original_symbol, logger)
            validation_results.append(validation)
            
            # Save to CSV if passes quality check
            if validation['data_quality'] != 'FAIL':
                # Use original symbol for filename for consistency
                output_file = os.path.join(RAW_YAHOO_DIR, f"{mapped_symbol}.csv")
                df.to_csv(output_file, index=False)
                print(f"✓ Saved {len(df)} records to {output_file}")
                successful += 1
            else:
                print(f"⚠ Skipped {original_symbol} due to insufficient data coverage")
                failed.append(original_symbol)
            
            # Rate limiting (Yahoo is generally more lenient)
            time.sleep(0.3)
        else:
            failed.append(original_symbol)
            print(f"✗ Failed to download {original_symbol}")
    
    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"Successful: {successful}/{len(available_symbols)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"\nFailed symbols: {', '.join(failed[:20])}{'...' if len(failed) > 20 else ''}")
    
    # Generate quality report
    report_file = os.path.join(OUTPUT_DIR, "yahoo_data_quality_report.txt")
    with open(report_file, 'w') as f:
        f.write("Phase 8: Yahoo Finance Data Quality Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date Range: {FROM_DATE} to {TO_DATE}\n")
        f.write(f"Data Source: Yahoo Finance (Adjusted Prices)\n")
        f.write(f"Universe: F&O stocks as of Apr 2020\n")
        f.write(f"Total stocks in universe: {len(symbols)}\n")
        f.write(f"Successfully downloaded: {successful}\n")
        f.write(f"Failed/Skipped: {len(failed)}\n\n")
        
        f.write("Validation Summary:\n")
        f.write("-"*60 + "\n")
        
        pass_count = sum(1 for v in validation_results if v['data_quality'] == 'PASS')
        warn_count = sum(1 for v in validation_results if v['data_quality'] == 'WARN')
        fail_count = sum(1 for v in validation_results if v['data_quality'] == 'FAIL')
        
        f.write(f"PASS: {pass_count}\n")
        f.write(f"WARN: {warn_count}\n")
        f.write(f"FAIL: {fail_count}\n\n")
        
        # Show warnings if any
        warnings = [v for v in validation_results if v['data_quality'] == 'WARN']
        if warnings:
            f.write("Warnings (large price gaps or zero volume):\n")
            for w in warnings:
                f.write(f"  {w['symbol']}: gaps={w['large_gaps']} (raw), gaps_adj={w['large_gaps_adj']} (adjusted)\n")
            f.write("\n")
        
        if failed:
            f.write(f"Failed symbols:\n{', '.join(failed)}\n")
    
    print(f"\n✓ Quality report saved to: {report_file}")
    print(f"\nSummary:")
    print(f"  PASS: {pass_count}")
    print(f"  WARN: {warn_count}")
    print(f"  FAIL: {fail_count}")
    
    # Compare adjusted vs raw gaps
    total_raw_gaps = sum(v['large_gaps'] for v in validation_results)
    total_adj_gaps = sum(v['large_gaps_adj'] for v in validation_results)
    print(f"\nPrice gaps comparison:")
    print(f"  Raw Close gaps (>40%): {total_raw_gaps}")
    print(f"  Adj Close gaps (>40%): {total_adj_gaps}")
    if total_raw_gaps > total_adj_gaps:
        print(f"  ✓ Adjusted prices eliminated {total_raw_gaps - total_adj_gaps} false gaps!")

if __name__ == "__main__":
    main()
