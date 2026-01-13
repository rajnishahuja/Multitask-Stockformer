# Symbol Mapping for Phase 8: Renamed/Merged Stocks (2020 -> 2026)
# These stocks from Apr 2020 F&O list have been renamed, merged, or restructured

SYMBOL_MAPPING = {
    'AMARAJABAT': 'ARE&M',           # Amara Raja Batteries -> Amara Raja Energy & Mobility
    'CADILAHC': 'ZYDUSLIFE',         # Cadila Healthcare -> Zydus Lifesciences
    'CENTURYTEX': 'ABREL',           # Century Textiles -> Aditya Birla Real Estate
    'EQUITAS': 'EQUITASBNK',         # Equitas Holdings -> Equitas Small Finance Bank
    'GMRINFRA': 'GMRAIRPORT',        # GMR Infra -> GMR Airports (restructured)
    'HDFC': 'HDFCBANK',              # HDFC merged with HDFC Bank (Jul 2023)
    'IBULHSGFIN': 'IBULLSLTD',       # Indiabulls Housing -> Indiabulls Ltd
    'INFRATEL': 'INDUSTOWER',        # Bharti Infratel -> Indus Towers
    'L&TFH': 'LTF',                  # L&T Finance Holdings -> LTF
    'MCDOWELL-N': 'UNITDSPR',        # United Spirits (traded under different symbol)
    'MINDTREE': 'LTIM',              # Mindtree merged with L&T Infotech
    'MOTHERSUMI': 'MOTHERSON',       # Motherson Sumi -> Motherson
    'NIITTECH': 'COFORGE',           # NIIT Tech -> Coforge
    'PEL': 'POONAWALLA',             # Piramal Enterprises split
    'PVR': 'PVRINOX',                # PVR merged with INOX
    'SRTRANSFIN': 'SHRIRAMFIN',      # Shriram Transport -> Shriram Finance
    'TATAMOTORS': 'TMPV',      # Tata Motors - trying DVR variant (original may have API issue)
    'UJJIVAN': 'UJJIVANSFB',         # Ujjivan -> Ujjivan Small Finance Bank
}

# Stocks to skip (no viable successor or delisted)
SKIP_SYMBOLS = set()  # Empty - all have successors now

def apply_symbol_mapping(symbols, instrument_map):
    """Apply symbol mapping and return available symbols with their tokens"""
    available = []
    mapped = []
    skipped = []
    
    for symbol in symbols:
        if symbol in SKIP_SYMBOLS:
            skipped.append(symbol)
            continue
            
        # Check if original symbol exists
        if symbol in instrument_map:
            available.append(symbol)
            continue
        
        # Check if we have a mapping
        if symbol in SYMBOL_MAPPING:
            new_symbol = SYMBOL_MAPPING[symbol]
            if new_symbol and new_symbol in instrument_map:
                available.append(new_symbol)
                mapped.append(f"{symbol} -> {new_symbol}")
            else:
                # Original symbol might still work for historical data
                skipped.append(f"{symbol} (mapped to {new_symbol} but not found)")
        else:
            skipped.append(symbol)
    
    return available, mapped, skipped
