# NIFTY-200 Rolling Windows Definition

**Created:** 2026-01-11  
**Purpose:** Define 14 rolling windows for NIFTY market testing, similar to original CSI-300 project  
**Format:** 2-year training, 4-month validation, 4-month test (75/12.5/12.5 split)

---

## Rolling Windows Table

| Subset | Train Start | Train End | Val Start | Val End | Test Start | Test End | Market Regime | NIFTY Context |
|--------|-------------|-----------|-----------|---------|------------|----------|---------------|---------------|
| **1** | 2018-03-01 | 2020-02-28 | 2020-03-02 | 2020-06-30 | 2020-07-01 | 2020-10-30 | 🔴 CRASH | COVID crash, V-recovery |
| **2** | 2018-06-01 | 2020-05-29 | 2020-06-01 | 2020-09-30 | 2020-10-01 | 2021-01-29 | 🟢 BULL | Post-COVID rally |
| **3** | 2018-09-01 | 2020-08-31 | 2020-09-01 | 2020-12-31 | 2021-01-01 | 2021-04-30 | 🟢 BULL | Continuous rally |
| **4** | 2018-12-01 | 2020-11-30 | 2020-12-01 | 2021-03-31 | 2021-04-01 | 2021-07-30 | 🟢 BULL | Second COVID wave |
| **5** | 2019-03-01 | 2021-02-26 | 2021-03-01 | 2021-06-30 | 2021-07-01 | 2021-10-29 | 🟢 BULL | IPO boom, approaching peak |
| **6** | 2019-06-01 | 2021-05-31 | 2021-06-01 | 2021-09-30 | 2021-10-01 | 2022-01-31 |  CORRECTION | Oct 2021 peak, correction |
| **7** | 2019-09-01 | 2021-08-31 | 2021-09-01 | 2021-12-31 | 2022-01-03 | 2022-04-29 | � VOLATILE | Rate hike fears |
| **8** | 2019-12-01 | 2021-11-30 | 2021-12-01 | 2022-03-31 | 2022-04-01 | 2022-07-29 | 🟠 VOLATILE | Russia-Ukraine war |
| **9** | 2020-03-01 | 2022-02-28 | 2022-03-01 | 2022-06-30 | 2022-07-01 | 2022-10-31 | � BEAR | FII outflows |
| **10** | 2020-06-01 | 2022-05-31 | 2022-06-01 | 2022-09-30 | 2022-10-01 | 2023-01-31 | 🟡 BEAR→RECOVERY | **Phase 8 in progress** |
| **11** | 2020-09-01 | 2022-08-31 | 2022-09-01 | 2022-12-30 | 2023-01-02 | 2023-04-28 | 🔴 CRISIS | Adani crisis (Jan 2023) |
| **12** | 2020-12-01 | 2022-11-30 | 2022-12-01 | 2023-03-31 | 2023-04-03 | 2023-07-31 | 🟡 RECOVERY | Post-Adani stabilization |
| **13** | 2021-03-01 | 2023-02-28 | 2023-03-01 | 2023-06-30 | 2023-07-03 | 2023-10-31 | 🟢 BULL | Strong recovery |
| **14** | 2021-06-01 | 2023-05-31 | 2023-06-01 | 2023-09-29 | 2023-10-02 | 2024-01-31 | 🟢 BULL | Pre-election rally |
| **15** | 2021-09-01 | 2023-08-31 | 2023-09-01 | 2023-12-29 | 2024-01-02 | 2024-04-30 | 🟢 BULL | Election anticipation |
| **16** | 2021-12-01 | 2023-11-30 | 2023-12-01 | 2024-03-29 | 2024-04-01 | 2024-07-31 | 🟢→🟡 | Election period |
| **17** | 2022-03-01 | 2024-02-29 | 2024-03-01 | 2024-06-28 | 2024-07-01 | 2024-10-31 | 🟡 MIXED | Post-election volatility |
| **18** | 2022-06-01 | 2024-05-31 | 2024-06-03 | 2024-09-30 | 2024-10-01 | 2025-01-31 | 🟢 BULL | FII return, new highs |
| **19** | 2022-09-01 | 2024-08-30 | 2024-09-02 | 2024-12-31 | 2025-01-02 | 2025-04-30 | 🟢→🟡 | US election impact |
| **20** | 2022-12-01 | 2024-11-29 | 2024-12-02 | 2025-03-31 | 2025-04-01 | 2025-07-31 | 🟢 BULL | Consolidation |
| **21** | 2023-03-01 | 2025-02-28 | 2025-03-03 | 2025-06-30 | 2025-07-01 | 2025-10-31 | 🟢 BULL | Mid-2025 |
| **22** | 2023-06-01 | 2025-05-30 | 2025-06-02 | 2025-09-30 | 2025-10-01 | 2026-01-10 | 🟢 BULL | **LIVE TRADING** ⭐ |

### Current Implementation Status
| Subset | Period | Regime | Status |
|--------|--------|--------|--------|
| 10 | Train: Jun 2020-May 2022, Test: Oct 2022-Jan 2023 | 🟡 BEAR→RECOVERY | 🔄 Training in progress |
| **22** | Train: Jun 2023-May 2025, Test: Oct 2025-Jan 2026 | 🟢 BULL | ⭐ **LIVE TRADING TARGET** |

---

## Market Regime Legend

| Symbol | Regime | Description |
|--------|--------|-------------|
| 🔴 | CRASH/CRISIS | Major correction >15%, extreme volatility, market panic |
| 🟠 | CORRECTION | 10-15% decline, elevated uncertainty |
| 🟡 | SIDEWAYS/MIXED | Choppy market, no clear trend |
| 🟢 | BULL | Sustained uptrend, positive returns |

---

## Key NIFTY Events (2018-2026)

| Date | Event | NIFTY Impact |
|------|-------|--------------|
| Feb 2018 | LTCG tax announcement | -4% drop |
| Sep 2018 | IL&FS crisis | NBFC sector crash |
| Oct 2018 | Global sell-off | -8% monthly |
| May 2019 | NDA election win | New high 12,041 |
| Mar 2020 | COVID crash | **-38%** (7,610 low) |
| Nov 2020 | Pre-COVID recovery | +84% from Mar lows |
| Oct 2021 | All-time high | ~18,600 |
| Oct-Dec 2021 | Global rate hike fears | -11% correction |
| Feb-Jun 2022 | Russia-Ukraine + rates | -15% from peak |
| Jan 2023 | Adani crisis | -5% shock, quick recovery |
| Jun 2024 | NDA election (reduced majority) | -3% initial drop |
| Sep-Dec 2024 | FII return + US election rally | New highs ~24,000 |
| Jan 2025 | **Current period** | Consolidation |

---

## Stock Universe Selection

### Approach A: Fixed Universe (Current)
- Use current NIFTY-200 constituents
- Remove stocks without sufficient history for the period
- **Pros**: Simple, reproducible
- **Cons**: Survivorship bias

### Approach B: Market-Cap Based (Recommended for Production)
```python
# Pseudo-code
def get_universe(period_start_date, n_stocks=200):
    # Get market cap as of period start
    market_caps = get_market_cap(date=period_start_date)
    # Filter stocks with available data
    available = filter_data_available(market_caps, min_days=500)
    # Select top N by market cap
    return available.sort_values('market_cap', ascending=False).head(n_stocks)
```

### Approach C: Historical NSE Constituents
- Use NSE IndexInclExcl.xls reports
- Most accurate but requires manual tracking

---

## Data Requirements per Subset

| Subset | Data Start | Data End | Total Days | Zerodha Download |
|--------|------------|----------|------------|------------------|
| 1-4 | 2018-01-01 | 2021-05-31 | ~850 | Required |
| 5-7 | 2019-01-01 | 2022-02-28 | ~800 | Required |
| 8-10 | 2019-10-01 | 2022-11-30 | ~800 | Required |
| 11-14 | 2020-07-01 | 2023-11-30 | ~850 | Partially available |

---

## Comparison to Original CSI-300 Project

| Aspect | Original (CSI-300) | NIFTY-200 Adaptation |
|--------|-------------------|----------------------|
| Subsets | 14 rolling windows | 14 rolling windows |
| Period | 2018-03-01 to 2024-03-01 | 2018-01-01 to 2024-08-31 |
| Universe | 255 stocks | ~185-200 stocks |
| Key Crashes | 2015 China crash, 2020 COVID | 2020 COVID, 2022 correction, 2023 Adani |
| Constituent Approach | Unknown (likely fixed) | To be decided |

---

## Next Steps

1. ✅ **Subset 10** (Bear Market): Training in progress (Apr 2020 - Nov 2022)
2. ⏳ **Subset 20** (Live Trading): Download data Oct 2022 - Jan 2026, prepare for weekly predictions
3. 🔮 **Full 20 subsets**: Experiment across all market regimes

### Subset 20 - Live Trading Window Details

| Aspect | Value |
|--------|-------|
| **Data Period** | Oct 2022 - Jan 10, 2026 |
| **Trading Days** | ~580 |
| **Train Period** | Oct 2022 - Sep 2024 (~500 days, 75%) |
| **Val Period** | Oct 2024 - Dec 2024 (~60 days, 12.5%) |
| **Test/Live Period** | Jan 2025 onwards |
| **Rebalance** | Weekly (every Friday) |
| **Universe** | F&O stocks (current list) |
