
import pandas as pd
import os
from datetime import date
from jugaad_data.nse import bhavcopy_fo_save, bhavcopy_fo_raw

# Configuration
TARGET_DATE = date(2025, 1, 10) # Using a Friday in Jan 2025 as proxy for "current" if 2026 data not in library
# Actually, the user said "It is 11th Jan 2026". The library might not have 2026 data yet if it's not live-updated.
# Let's try to get the most recent valid bhavcopy. 
# Safe bet: Use last valid trading day of 2024 or 2025.
# Let's try Dec 2024 first? Or assume the library works for recent dates.
# Let's try specific date: 2024-08-30 (known trading day).
# Or better: let's try a few recent dates until one works.

OUTPUT_DIR = "./data/NIFTY200_Subset22"

def get_fno_stocks(target_date):
    print(f"============================================================")
    print(f"Downloading F&O Bhavcopy for {target_date}")
    print(f"============================================================")
    
    try:
        # Save bhavcopy to a temporary file
        output_path = f"/tmp/fo_bhav_{target_date}.csv"
        bhavcopy_fo_save(target_date, "/tmp")
        
        # Rename to a predictable name if save adds random chars? 
        # jugaad-data saves as "fo{date}bhav.csv".
        expected_filename = f"fo{target_date.strftime('%d%b%Y').upper()}bhav.csv"
        full_path = os.path.join("/tmp", expected_filename)
        
        if not os.path.exists(full_path):
             # Try listing dir to find it
             print(f"Could not find {expected_filename} in /tmp")
             return None

        df = pd.read_csv(full_path)
        
        # Filter for Equity Derivatives
        df = df[df['INSTRUMENT'] == 'FUTSTK']
        
        stocks = df['SYMBOL'].unique().tolist()
        stocks.sort()
        
        print(f"Found {len(stocks)} F&O stocks.")
        return stocks

    except Exception as e:
        print(f"Error downloading for {target_date}: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Try a few dates in reverse order to find latest available
    # Since today is Jan 11 2026, let's look for Jan 9 2026.
    # If the library relies on static URLs/logic, it might fail for 2026.
    # Let's try 2024-08-30 (Fri) as a fallback if 2026 fails.
    
    dates_to_try = [
        date(2025, 1, 10), # Fri
        date(2025, 1, 3),  # Fri
        date(2024, 8, 30), # Fri (known good)
        date(2020, 4, 1)   # Original (fallback)
    ]
    
    stocks = None
    used_date = None
    
    for d in dates_to_try:
        stocks = get_fno_stocks(d)
        if stocks:
            used_date = d
            break
            
    if stocks:
        output_file = os.path.join(OUTPUT_DIR, "fno_stocks.txt")
        with open(output_file, "w") as f:
            for stock in stocks:
                f.write(stock + "\n")
        print(f"Saved {len(stocks)} stocks to {output_file} (Date: {used_date})")
    else:
        print("Failed to fetch F&O list.")
