from jugaad_data.nse import bhavcopy_fo_save
from datetime import date
import pandas as pd
import os

OUTPUT_FILE = "data/experiment_fno_universe_2024_06.txt"
BHAV_DIR = "data/"

# Try dates in June 2024
dates_to_try = [
    date(2024, 6, 3), # Monday
    date(2024, 6, 4),
    date(2024, 6, 5)
]

data_downloaded = False
success_date = None
downloaded_file_path = None

for d in dates_to_try:
    print(f"Trying to download bhavcopy for {d}...")
    try:
        bhavcopy_fo_save(d, BHAV_DIR)
        # Construct expected filename
        # Format: foDDMMMYYYYbhav.csv (e.g. fo03Jun2024bhav.csv)
        # Note: Month is title case (Jun)
        fname = f"fo{d.strftime('%d%b%Y')}bhav.csv"
        # jugaad-data handles Title Case months? 
        # Actually standard NSE format is fo03JUN2024bhav.csv usually?
        # Let's check what exists
        
        # Check case variations
        candidates = [
            f"fo{d.strftime('%d%b%Y')}bhav.csv",
            f"fo{d.strftime('%d%b%Y').upper()}bhav.csv",
        ]
        
        found = False
        for c in candidates:
             p = os.path.join(BHAV_DIR, c)
             if os.path.exists(p):
                 downloaded_file_path = p
                 found = True
                 break
        
        if found:
             print(f"Success downloading for {d}. File: {downloaded_file_path}")
             success_date = d
             data_downloaded = True
             break
        else:
             print(f"Download seemed to work but file not found: {candidates}")

    except Exception as e:
        print(f"Failed for {d}: {e}")

if not data_downloaded:
    print("Could not download ANY bhavcopy.")
    exit(1)

# Extract Universe
print(f"Extracting universe from {downloaded_file_path}...")
df = pd.read_csv(downloaded_file_path)

# Filter for FUTSTK
# Column names might be 'INSTRUMENT', 'SYMBOL', etc.
# Check columns
df.columns = [c.strip().upper() for c in df.columns]

if 'INSTRUMENT' in df.columns:
    stocks = df[df['INSTRUMENT'] == 'FUTSTK']['SYMBOL'].unique()
else:
    # Older format might be differnet, but 2024 should be standard
    print(f"Unknown columns: {df.columns}")
    exit(1)

print(f"Found {len(stocks)} F&O stocks.")

with open(OUTPUT_FILE, 'w') as f:
    for s in sorted(stocks):
        f.write(s + "\n")
        
print(f"Saved universe to {OUTPUT_FILE}")
