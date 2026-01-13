"""
Phase 8: Zerodha Data Fetcher for Bear Market Test (Subset 10)
================================================================

Downloads historical OHLCV data for F&O eligible stocks (Apr 2020 - Nov 2022).

Usage:
    python phase8_zerodha_data_fetcher.py

Prerequisites:
    - Run phase8_get_fno_universe.py first to create the F&O universe list
    - Zerodha account with API access

Output:
    - data/NIFTY200_Subset10/raw/{SYMBOL}.csv (one file per stock)
    - data/NIFTY200_Subset10/data_quality_report.txt
"""

import os
import sys
import time
import logging
import json
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from kiteconnect import KiteConnect

# ============================================================================
# CONFIGURATION - Update these for your setup
# ============================================================================

# Zerodha API credentials (same as Phase 2)
API_KEY = "a3vlmmcvyt40udoq"
API_SECRET = "xin86nvnojty5996zzexbu7chc040zy0"

# Date range for Subset 10 (Apr 2020 - Nov 2022)
FROM_DATE = "2020-04-01"
TO_DATE = "2022-11-30"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data/NIFTY200_Subset10")
RAW_DATA_DIR = os.path.join(OUTPUT_DIR, "raw")
UNIVERSE_FILE = os.path.join(OUTPUT_DIR, "fno_universe_2020_04.txt")
TOKEN_FILE = os.path.join(OUTPUT_DIR, "zerodha_token.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "zerodha_fetcher.log")

# Data quality thresholds
REQUIRED_COVERAGE = 0.80  # Require 80% coverage of trading days

# ============================================================================
# LOGGING SETUP
# ============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# TOKEN MANAGEMENT
# ============================================================================

def save_token(access_token: str) -> dict:
    """Save access token with timestamp to file"""
    token_data = {
        'access_token': access_token,
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().date().isoformat()
    }
    
    with open(TOKEN_FILE, 'w') as f:
        json.dump(token_data, f, indent=2)
    
    logger.info(f"Token saved to {TOKEN_FILE}")
    return token_data

def load_token() -> Optional[str]:
    """Load access token if valid (same day)"""
    if not os.path.exists(TOKEN_FILE):
        logger.info("No saved token found")
        return None
    
    try:
        with open(TOKEN_FILE, 'r') as f:
            token_data = json.load(f)
        
        # Check if token is from today
        saved_date = date.fromisoformat(token_data['date'])
        today = datetime.now().date()
        
        if saved_date == today:
            logger.info(f"✓ Found valid token from {token_data['timestamp']}")
            return token_data['access_token']
        else:
            logger.info(f"Token expired (from {saved_date}), need fresh login")
            return None
    
    except Exception as e:
        logger.error(f"Error loading token: {e}")
        return None

# ============================================================================
# ZERODHA API CONNECTION
# ============================================================================

def initialize_kite() -> KiteConnect:
    """Initialize KiteConnect with authentication"""
    kite = KiteConnect(api_key=API_KEY)
    
    # Try to load existing token
    access_token = load_token()
    
    if access_token:
        kite.set_access_token(access_token)
        print("✅ Using saved session - no login required!")
    else:
        # Manual login required
        print("\n" + "="*80)
        print("MANUAL LOGIN REQUIRED (One-time per day)")
        print("="*80)
        print(f"\n1. Open this URL in your browser:\n   {kite.login_url()}")
        print("\n2. After login, you'll be redirected to: http://127.0.0.1/?request_token=...")
        request_token = input("\n3. Paste the 'request_token' value here: ").strip()
        
        try:
            data = kite.generate_session(request_token, api_secret=API_SECRET)
            access_token = data["access_token"]
            kite.set_access_token(access_token)
            save_token(access_token)
            print("✅ Login Successful! Token saved for today.")
        except Exception as e:
            print(f"❌ Login Failed: {e}")
            raise
    
    # Test connection
    try:
        profile = kite.profile()
        print(f"\n✅ Connected to Zerodha API")
        print(f"User: {profile['user_name']}")
        print(f"Email: {profile['email']}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        raise
    
    return kite

# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

def fetch_historical_data(kite: KiteConnect, symbol: str, instrument_token: int,
                         from_date: str, to_date: str, 
                         max_retries: int = 5) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV data with exponential backoff retry"""
    from_dt = datetime.strptime(from_date, '%Y-%m-%d')
    to_dt = datetime.strptime(to_date, '%Y-%m-%d')
    
    for attempt in range(max_retries):
        try:
            historical_data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_dt,
                to_date=to_dt,
                interval='day'
            )
            
            if not historical_data:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
            df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            
            logger.info(f"✓ Fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            wait_time = 1 * (2 ** attempt)
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {symbol}: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"✗ Failed after {max_retries} attempts: {symbol}")
                return None
    
    return None

def validate_data(df: pd.DataFrame, symbol: str, from_date: str, to_date: str) -> Dict:
    """Validate data completeness and quality"""
    from_dt = datetime.strptime(from_date, '%Y-%m-%d').date()
    to_dt = datetime.strptime(to_date, '%Y-%m-%d').date()
    
    validation = {
        'symbol': symbol,
        'total_records': len(df),
        'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
        'coverage': 0.0,
        'zero_volume_days': 0,
        'large_gaps': 0,
        'data_quality': 'PASS'
    }
    
    # Calculate expected trading days (rough estimate: 252 days/year)
    total_days = (to_dt - from_dt).days
    expected_trading_days = int(total_days * 252 / 365)
    actual_days = len(df)
    
    validation['coverage'] = actual_days / expected_trading_days if expected_trading_days > 0 else 0
    
    if validation['coverage'] < REQUIRED_COVERAGE:
        validation['data_quality'] = 'FAIL'
        logger.warning(f"{symbol}: Insufficient coverage {validation['coverage']:.1%}")
    
    # Check for zero volume days
    zero_vol = (df['Volume'] == 0).sum()
    validation['zero_volume_days'] = int(zero_vol)
    if zero_vol > len(df) * 0.1:
        validation['data_quality'] = 'WARN'
        logger.warning(f"{symbol}: {zero_vol} zero volume days")
    
    # Check for large price gaps (>40% change - possible unadjusted data)
    df_sorted = df.sort_values('Date')
    pct_change = df_sorted['Close'].pct_change().abs()
    large_gaps = (pct_change > 0.40).sum()
    validation['large_gaps'] = int(large_gaps)
    if large_gaps > 0:
        validation['data_quality'] = 'WARN'
        logger.warning(f"{symbol}: {large_gaps} days with >40% price changes")
    
    return validation

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("Phase 8: Zerodha Data Fetcher for Bear Market Test (Subset 10)")
    print("="*80)
    print(f"\nDate Range: {FROM_DATE} to {TO_DATE}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Check if universe file exists
    if not os.path.exists(UNIVERSE_FILE):
        print(f"\n❌ Error: Universe file not found: {UNIVERSE_FILE}")
        print("Please run phase8_get_fno_universe.py first!")
        sys.exit(1)
    
    # Load F&O universe
    with open(UNIVERSE_FILE, 'r') as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    print(f"\n✓ Loaded {len(symbols)} stocks from F&O universe")
    print(f"First 10: {symbols[:10]}")
    
    # Initialize Zerodha connection
    kite = initialize_kite()
    
    # Fetch instrument tokens
    print("\nFetching NSE instruments list...")
    instruments = kite.instruments("NSE")
    instrument_map = {i['tradingsymbol']: i['instrument_token'] for i in instruments}
    print(f"✓ Fetched {len(instruments)} NSE instruments")
    
    # Check which symbols are available
    available_symbols = []
    missing_symbols = []
    for symbol in symbols:
        if symbol in instrument_map:
            available_symbols.append(symbol)
        else:
            missing_symbols.append(symbol)
            logger.warning(f"No instrument token for: {symbol}")
    
    print(f"\n✓ {len(available_symbols)} symbols available on NSE")
    if missing_symbols:
        print(f"⚠ {len(missing_symbols)} symbols not found: {missing_symbols[:10]}...")
    
    # Download data
    print("\n" + "="*80)
    print(f"Starting data download for {len(available_symbols)} stocks")
    print("="*80 + "\n")
    
    validation_results = []
    successful = 0
    failed = []
    
    for idx, symbol in enumerate(available_symbols, 1):
        print(f"\n[{idx}/{len(available_symbols)}] Processing {symbol}...")
        
        instrument_token = instrument_map[symbol]
        
        # Fetch data
        df = fetch_historical_data(kite, symbol, instrument_token, FROM_DATE, TO_DATE)
        
        if df is not None and len(df) > 0:
            # Validate data
            validation = validate_data(df, symbol, FROM_DATE, TO_DATE)
            validation_results.append(validation)
            
            # Save to CSV only if passes quality check
            if validation['data_quality'] != 'FAIL':
                output_file = os.path.join(RAW_DATA_DIR, f"{symbol}.csv")
                df.to_csv(output_file, index=False)
                print(f"✓ Saved {len(df)} records to {output_file}")
                successful += 1
            else:
                print(f"⚠ Skipped {symbol} due to insufficient data coverage")
                failed.append(symbol)
            
            # Rate limiting (Zerodha API limit)
            time.sleep(0.5)
        else:
            failed.append(symbol)
            print(f"✗ Failed to download {symbol}")
    
    # Generate quality report
    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE")
    print("="*80)
    print(f"Successful: {successful}/{len(available_symbols)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed symbols: {', '.join(failed[:20])}{'...' if len(failed) > 20 else ''}")
    
    # Save quality report
    report_file = os.path.join(OUTPUT_DIR, "data_quality_report.txt")
    with open(report_file, 'w') as f:
        f.write("Phase 8: Data Quality Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Date Range: {FROM_DATE} to {TO_DATE}\n")
        f.write(f"Universe: F&O stocks as of Apr 2020\n")
        f.write(f"Total stocks in universe: {len(symbols)}\n")
        f.write(f"Successfully downloaded: {successful}\n")
        f.write(f"Failed/Skipped: {len(failed)}\n\n")
        
        f.write("Validation Summary:\n")
        f.write("-"*60 + "\n")
        
        # Count by quality status
        pass_count = sum(1 for v in validation_results if v['data_quality'] == 'PASS')
        warn_count = sum(1 for v in validation_results if v['data_quality'] == 'WARN')
        fail_count = sum(1 for v in validation_results if v['data_quality'] == 'FAIL')
        
        f.write(f"PASS: {pass_count}\n")
        f.write(f"WARN: {warn_count}\n")
        f.write(f"FAIL: {fail_count}\n\n")
        
        if failed:
            f.write(f"Failed symbols:\n{', '.join(failed)}\n")
    
    print(f"\n✓ Quality report saved to: {report_file}")
    print(f"\nNext Step: Run phase8_factor_engineering.py to calculate Alpha158 factors")
    
    return successful

if __name__ == "__main__":
    main()
