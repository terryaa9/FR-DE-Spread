"""
STEP 2 — FEATURE ENGINEERING
==============================
Four fundamental drivers of the FR-DE day-ahead spread:
  1. Residual load differential (FR - DE)
  2. TTF gas price
  3. French nuclear availability (lagged, look-ahead free)
  4. Wind/solar forecasts (FR and DE)

All features are look-ahead free:
  - Wind/solar/load: TSO-published day-ahead forecasts
  - Nuclear: lagged 1h (proxy for known availability)
  - TTF: daily price, known before day-ahead auction

Target: spread = fr_price - de_price
"""

import pandas as pd
import numpy as np


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv('processed/clean_merged_v3.csv', index_col=0, parse_dates=True)
print(f"Loaded {len(df)} rows")


# ============================================================
# TARGET
# ============================================================

df['spread'] = df['fr_price'] - df['de_price']

print(f"\nSpread stats:")
print(f"  Mean:   {df['spread'].mean():.1f} EUR/MWh")
print(f"  Std:    {df['spread'].std():.1f} EUR/MWh")
print(f"  Median: {df['spread'].median():.1f} EUR/MWh")


# ============================================================
# RESIDUAL LOAD
# ============================================================
# Load minus wind minus solar — how much thermal generation is needed.
# FR: use forecast wind/solar (look-ahead free).

df['de_residual_load'] = df['de_load'] - df['de_wind'] - df['de_solar']
df['fr_residual_load'] = df['fr_load'] - df['fr_wind_forecast'] - df['fr_solar_forecast']
df['residual_load_diff'] = df['fr_residual_load'] - df['de_residual_load']


# ============================================================
# NUCLEAR — LAGGED TO AVOID LOOK-AHEAD
# ============================================================
# Nuclear is actual generation, not a forecast.
# Shift by 1 hour so model only sees last hour's output.
# Reactors don't change hourly, so this is a close proxy for
# what a trader knows before the day-ahead auction.

df['fr_nuclear_lag1'] = df['fr_nuclear'].shift(1)

# Deviation from 30-day rolling mean captures outage shocks.
df['fr_nuclear_deviation'] = (
    df['fr_nuclear'].shift(1) -
    df['fr_nuclear'].rolling(720).mean().shift(1)  # 720h = 30 days
)


# ============================================================
# TIME FEATURES
# ============================================================

df['hour'] = df.index.hour
df['month'] = df.index.month
df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)

# Peak hours: 8-20 CET = 7-19 UTC
df['is_peak'] = df['hour'].isin(range(7, 20)).astype(int)


# ============================================================
# REGIME LABELS (for analysis.py regime splits)
# ============================================================
# Median splits: balanced comparison, half the data in each regime.

ttf_median = df['ttf_close'].median()
df['regime_gas'] = np.where(df['ttf_close'] >= ttf_median, 'high_gas', 'low_gas')
print(f"\nGas regime split at TTF = EUR {ttf_median:.1f}/MWh")

de_wind_median = df['de_wind'].median()
df['regime_wind'] = np.where(df['de_wind'] >= de_wind_median, 'high_wind', 'low_wind')
print(f"Wind regime split at DE wind = {de_wind_median:.0f} MW")

# Use lagged nuclear for regime label too — no look-ahead
fr_nuc_median = df['fr_nuclear_lag1'].median()
df['regime_nuclear'] = np.where(
    df['fr_nuclear_lag1'] >= fr_nuc_median, 'high_nuclear', 'low_nuclear'
)
print(f"Nuclear regime split at FR nuclear = {fr_nuc_median:.0f} MW")

df['regime_peak'] = np.where(df['is_peak'] == 1, 'peak', 'off_peak')


# ============================================================
# FEATURE LIST
# ============================================================

feature_columns = [
    # Residual load differential — core fundamental driver
    'residual_load_diff',
    'fr_residual_load',
    'de_residual_load',
    'fr_load',
    'de_load',

    # Gas price
    'ttf_close',

    # Nuclear availability (look-ahead free)
    'fr_nuclear_lag1',
    'fr_nuclear_deviation',

    # Wind & solar forecasts
    'fr_wind_forecast',
    'fr_solar_forecast',
    'de_wind',
    'de_solar',

    # Time
    'hour',
    'month',
    'is_weekend',
    'is_peak',
]

print(f"\nFeatures ({len(feature_columns)}):")
for f in feature_columns:
    print(f"  {f}")


# ============================================================
# FINALISE
# ============================================================

rows_before = len(df)
df = df.dropna()
rows_after = len(df)
print(f"\nDropped {rows_before - rows_after} rows from lagging/rolling")
print(f"Final: {rows_after} rows")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

df.to_csv('processed/model_ready_v3.csv')

with open('processed/feature_columns_v3.txt', 'w') as f:
    for col in feature_columns:
        f.write(col + '\n')

print("\nSaved to processed/model_ready_v3.csv")
