"""
Alpha158 Factor Implementation using Pandas
Directly implements the 158 Alpha factors from Qlib using pandas operations.
Formulas extracted from qlib.contrib.data.loader.Alpha158DL.get_feature_config()
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def compute_alpha158_factors(ohlcv_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute all 158 Alpha factors for given OHLCV data.
    
    Args:
        ohlcv_data: Dict mapping symbol -> DataFrame with columns [Date, Open, High, Low, Close, Volume]
    
    Returns:
        DataFrame with MultiIndex (date, symbol) and 158 feature columns
    """
    all_factors = []
    
    for symbol, df in ohlcv_data.items():
        df = df.copy().sort_values('Date')
        df = df.set_index('Date')
        
        # Extract price and volume series
        open_price = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
        
        factors = {}
        
        # ========== KBAR Features (9 features) ==========
        factors['KMID'] = (close - open_price) / open_price
        factors['KLEN'] = (high - low) / open_price
        factors['KMID2'] = (close - open_price) / (high - low + 1e-12)
        factors['KUP'] = (high - np.maximum(open_price, close)) / open_price
        factors['KUP2'] = (high - np.maximum(open_price, close)) / (high - low + 1e-12)
        factors['KLOW'] = (np.minimum(open_price, close) - low) / open_price
        factors['KLOW2'] = (np.minimum(open_price, close) - low) / (high - low + 1e-12)
        factors['KSFT'] = (2 * close - high - low) / open_price
        factors['KSFT2'] = (2 * close - high - low) / (high - low + 1e-12)
        
        # ========== Price Features (4 features, windows=[0]) ==========
        factors['OPEN0'] = open_price / close
        factors['HIGH0'] = high / close
        factors['LOW0'] = low / close
        # VWAP approximation: (high + low + close) / 3
        vwap = (high + low + close) / 3
        factors['VWAP0'] = vwap / close
        
        # ========== Rolling Features (145 features) ==========
        windows = [5, 10, 20, 30, 60]
        
        # ROC - Rate of Change (5 features)
        for d in windows:
            factors[f'ROC{d}'] = close.shift(d) / close
        
        # MA - Moving Average (5 features)
        for d in windows:
            factors[f'MA{d}'] = close.rolling(d).mean() / close
        
        # STD - Standard Deviation (5 features)
        for d in windows:
            factors[f'STD{d}'] = close.rolling(d).std() / close
        
        # BETA - Slope/Trend (5 features)
        for d in windows:
            # Linear regression slope
            def slope(series):
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                y = series.values
                valid = ~np.isnan(y)
                if valid.sum() < 2:
                    return np.nan
                return np.polyfit(x[valid], y[valid], 1)[0]
            
            factors[f'BETA{d}'] = close.rolling(d).apply(slope, raw=False) / close
        
        # RSQR - R-squared (5 features)
        for d in windows:
            def rsquare(series):
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                y = series.values
                valid = ~np.isnan(y)
                if valid.sum() < 2:
                    return np.nan
                y_valid = y[valid]
                x_valid = x[valid]
                slope, intercept = np.polyfit(x_valid, y_valid, 1)
                y_pred = slope * x_valid + intercept
                ss_res = np.sum((y_valid - y_pred) ** 2)
                ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
                return 1 - (ss_res / (ss_tot + 1e-12))
            
            factors[f'RSQR{d}'] = close.rolling(d).apply(rsquare, raw=False)
        
        # RESI - Residual (5 features)
        for d in windows:
            def residual(series):
                if len(series) < 2:
                    return np.nan
                x = np.arange(len(series))
                y = series.values
                valid = ~np.isnan(y)
                if valid.sum() < 2:
                    return np.nan
                y_valid = y[valid]
                x_valid = x[valid]
                slope, intercept = np.polyfit(x_valid, y_valid, 1)
                y_pred = slope * x_valid[-1] + intercept
                return y_valid[-1] - y_pred
            
            factors[f'RESI{d}'] = close.rolling(d).apply(residual, raw=False) / close
        
        # MAX - Maximum High (5 features)
        for d in windows:
            factors[f'MAX{d}'] = high.rolling(d).max() / close
        
        # MIN - Minimum Low (5 features)
        for d in windows:
            factors[f'MIN{d}'] = low.rolling(d).min() / close
        
        # QTLU - 80% Quantile (5 features)
        for d in windows:
            factors[f'QTLU{d}'] = close.rolling(d).quantile(0.8) / close
        
        # QTLD - 20% Quantile (5 features)
        for d in windows:
            factors[f'QTLD{d}'] = close.rolling(d).quantile(0.2) / close
        
        # RANK - Percentile Rank (5 features)
        for d in windows:
            factors[f'RANK{d}'] = close.rolling(d).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        
        # RSV - Relative Strength Value (5 features)
        for d in windows:
            min_low = low.rolling(d).min()
            max_high = high.rolling(d).max()
            factors[f'RSV{d}'] = (close - min_low) / (max_high - min_low + 1e-12)
        
        # IMAX - Days since maximum (5 features)
        for d in windows:
            factors[f'IMAX{d}'] = high.rolling(d).apply(lambda x: (len(x) - 1 - x.argmax()) / d, raw=False)
        
        # IMIN - Days since minimum (5 features)
        for d in windows:
            factors[f'IMIN{d}'] = low.rolling(d).apply(lambda x: (len(x) - 1 - x.argmin()) / d, raw=False)
        
        # IMXD - IMAX - IMIN difference (5 features)
        for d in windows:
            idx_max = high.rolling(d).apply(lambda x: len(x) - 1 - x.argmax(), raw=False)
            idx_min = low.rolling(d).apply(lambda x: len(x) - 1 - x.argmin(), raw=False)
            factors[f'IMXD{d}'] = (idx_max - idx_min) / d
        
        # CORR - Correlation between close and log(volume) (5 features)
        for d in windows:
            log_vol = np.log(volume + 1)
            factors[f'CORR{d}'] = close.rolling(d).corr(log_vol)
        
        # CORD - Correlation between returns and volume changes (5 features)
        for d in windows:
            returns = close / close.shift(1)
            vol_change = np.log(volume / volume.shift(1) + 1)
            factors[f'CORD{d}'] = returns.rolling(d).corr(vol_change)
        
        # CNTP - Count of positive days (5 features)
        for d in windows:
            pos = (close > close.shift(1)).astype(float)
            factors[f'CNTP{d}'] = pos.rolling(d).mean()
        
        # CNTN - Count of negative days (5 features)
        for d in windows:
            neg = (close < close.shift(1)).astype(float)
            factors[f'CNTN{d}'] = neg.rolling(d).mean()
        
        # CNTD - Difference between positive and negative days (5 features)
        for d in windows:
            pos = (close > close.shift(1)).astype(float)
            neg = (close < close.shift(1)).astype(float)
            factors[f'CNTD{d}'] = pos.rolling(d).mean() - neg.rolling(d).mean()
        
        # SUMP - Sum of gains ratio (5 features)
        for d in windows:
            gain = np.maximum(close - close.shift(1), 0)
            abs_change = np.abs(close - close.shift(1))
            factors[f'SUMP{d}'] = gain.rolling(d).sum() / (abs_change.rolling(d).sum() + 1e-12)
        
        # SUMN - Sum of losses ratio (5 features)
        for d in windows:
            loss = np.maximum(close.shift(1) - close, 0)
            abs_change = np.abs(close - close.shift(1))
            factors[f'SUMN{d}'] = loss.rolling(d).sum() / (abs_change.rolling(d).sum() + 1e-12)
        
        # SUMD - Difference between gains and losses (5 features)
        for d in windows:
            gain = np.maximum(close - close.shift(1), 0)
            loss = np.maximum(close.shift(1) - close, 0)
            abs_change = np.abs(close - close.shift(1))
            factors[f'SUMD{d}'] = (gain.rolling(d).sum() - loss.rolling(d).sum()) / (abs_change.rolling(d).sum() + 1e-12)
        
        # VMA - Volume Moving Average (5 features)
        for d in windows:
            factors[f'VMA{d}'] = volume.rolling(d).mean() / (volume + 1e-12)
        
        # VSTD - Volume Standard Deviation (5 features)
        for d in windows:
            factors[f'VSTD{d}'] = volume.rolling(d).std() / (volume + 1e-12)
        
        # WVMA - Weighted Volume Moving Average (5 features)
        for d in windows:
            price_change = np.abs(close / close.shift(1) - 1) * volume
            factors[f'WVMA{d}'] = price_change.rolling(d).std() / (price_change.rolling(d).mean() + 1e-12)
        
        # VSUMP - Volume sum of gains ratio (5 features)
        for d in windows:
            vol_gain = np.maximum(volume - volume.shift(1), 0)
            abs_vol_change = np.abs(volume - volume.shift(1))
            factors[f'VSUMP{d}'] = vol_gain.rolling(d).sum() / (abs_vol_change.rolling(d).sum() + 1e-12)
        
        # VSUMN - Volume sum of losses ratio (5 features)
        for d in windows:
            vol_loss = np.maximum(volume.shift(1) - volume, 0)
            abs_vol_change = np.abs(volume - volume.shift(1))
            factors[f'VSUMN{d}'] = vol_loss.rolling(d).sum() / (abs_vol_change.rolling(d).sum() + 1e-12)
        
        # VSUMD - Volume difference between gains and losses (5 features)
        for d in windows:
            vol_gain = np.maximum(volume - volume.shift(1), 0)
            vol_loss = np.maximum(volume.shift(1) - volume, 0)
            abs_vol_change = np.abs(volume - volume.shift(1))
            factors[f'VSUMD{d}'] = (vol_gain.rolling(d).sum() - vol_loss.rolling(d).sum()) / (abs_vol_change.rolling(d).sum() + 1e-12)
        
        # Create DataFrame for this symbol
        factor_df = pd.DataFrame(factors)
        factor_df['symbol'] = symbol
        factor_df['date'] = factor_df.index
        all_factors.append(factor_df)
    
    # Combine all symbols
    result = pd.concat(all_factors, ignore_index=True)
    result = result.set_index(['date', 'symbol'])
    result = result.sort_index()
    
    print(f"✓ Computed {result.shape[1]} Alpha158 factors")
    print(f"  Expected: 158 (9 KBAR + 4 PRICE + 145 ROLLING)")
    print(f"  Data shape: {result.shape}")
    
    return result


def get_feature_names() -> List[str]:
    """Return the ordered list of all 158 Alpha158 feature names."""
    names = []
    
    # KBAR (9 features)
    names += ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2']
    
    # PRICE (4 features)
    names += ['OPEN0', 'HIGH0', 'LOW0', 'VWAP0']
    
    # ROLLING (145 features)
    windows = [5, 10, 20, 30, 60]
    for base in ['ROC', 'MA', 'STD', 'BETA', 'RSQR', 'RESI', 'MAX', 'MIN', 'QTLU', 'QTLD', 
                 'RANK', 'RSV', 'IMAX', 'IMIN', 'IMXD', 'CORR', 'CORD', 'CNTP', 'CNTN', 'CNTD',
                 'SUMP', 'SUMN', 'SUMD', 'VMA', 'VSTD', 'WVMA', 'VSUMP', 'VSUMN', 'VSUMD']:
        for d in windows:
            names.append(f'{base}{d}')
    
    return names
