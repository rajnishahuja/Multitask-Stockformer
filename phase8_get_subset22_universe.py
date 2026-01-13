
import pandas as pd
import os
from datetime import date
from jugaad_data.nse import bhavcopy_fo_save

# Using start of Subset 22 Training Period
TARGET_DATE = date(2023, 6, 1) # Thursday
# If Thu is holiday, we might need a backup
BACKUP_DATE = date(2023, 6, 2) # Friday

OUTPUT_DIR = "./data/NIFTY200_Subset22"

def get_fno_stocks(target_date):
    print(f"============================================================")
    print(f"Downloading F&O Bhavcopy for {target_date}")
    print(f"============================================================")
    
    try:
        # Save bhavcopy to a temporary file
        bhavcopy_fo_save(target_date, "/tmp")
        
        # Construct filename: foDDMMMYYYYbhav.csv
        # Note: Library saves as 'Jun' (Title case), not 'JUN' (Upper)
        expected_filename = f"fo{target_date.strftime('%d%b%Y')}bhav.csv"
        full_path = os.path.join("/tmp", expected_filename)
        
        if not os.path.exists(full_path):
             print(f"Could not find {expected_filename} in /tmp")
             return None

        df = pd.read_csv(full_path)
        
        # Filter for Equity Derivatives (FUTSTK)
        df = df[df['INSTRUMENT'] == 'FUTSTK']
        
        stocks = df['SYMBOL'].unique().tolist()
        stocks = [s for s in stocks if s != 'SYMBOL'] # cleanup
        stocks.sort()
        
        print(f"Found {len(stocks)} F&O stocks on {target_date}.")
        return stocks

    except Exception as e:
        print(f"Error downloading for {target_date}: {e}")
        return None

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    stocks = get_fno_stocks(TARGET_DATE)
    if not stocks:
        print(f"Retrying with backup date {BACKUP_DATE}...")
        stocks = get_fno_stocks(BACKUP_DATE)
            
    if stocks:
        output_file = os.path.join(OUTPUT_DIR, "fno_stocks.txt")
        with open(output_file, "w") as f:
            for stock in stocks:
                f.write(stock + "\n")
        print(f"Saved {len(stocks)} stocks to {output_file}")
    else:
        print("Failed to fetch F&O list.")
