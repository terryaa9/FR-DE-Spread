"""
STEP 1 — DATA LOADING & CLEANING
==================================
Loads all datasets and merges into one clean hourly table.
All features are look-ahead free: forecasts for wind/solar/load,
lagged actuals for nuclear, daily TTF forward-filled.
"""

import pandas as pd
import numpy as np


# ============================================================
# BLOCK 1: LOAD PRICES
# ============================================================

de_price = pd.read_csv('raw/de_day_ahead_price.csv', index_col=0)
de_price.index = pd.to_datetime(de_price.index, utc=True)
de_price.columns = ['de_price']
de_price = de_price[~de_price.index.duplicated(keep='first')]

fr_price = pd.read_csv('raw/fr_day_ahead_price.csv', index_col=0)
fr_price.index = pd.to_datetime(fr_price.index, utc=True)
fr_price.columns = ['fr_price']
fr_price = fr_price[~fr_price.index.duplicated(keep='first')]

print(f"DE prices: {len(de_price)} rows")
print(f"FR prices: {len(fr_price)} rows")


# ============================================================
# BLOCK 2: LOAD FRENCH GENERATION (for nuclear only)
# ============================================================
# We only need nuclear from this file. Wind and solar will come
# from the forecast file instead (no look-ahead).

fr_gen = pd.read_csv('raw/fr_generation.csv', header=[0, 1], index_col=0)
fr_gen = fr_gen.xs('Actual Aggregated', axis=1, level=1)
fr_gen.index = pd.to_datetime(fr_gen.index, utc=True)

# Resample 15-min to hourly
fr_gen = fr_gen.resample('h').mean()

# Extract only nuclear
fr_nuclear = fr_gen[['Nuclear']].rename(columns={'Nuclear': 'fr_nuclear'})
print(f"FR nuclear: {len(fr_nuclear)} rows")


# ============================================================
# BLOCK 3: LOAD FRENCH WIND/SOLAR FORECAST (look-ahead free)
# ============================================================

fr_ws_fcst = pd.read_csv('raw/fr_wind_solar_forecast.csv', index_col=0)
fr_ws_fcst.index = pd.to_datetime(fr_ws_fcst.index, utc=True)

fr_ws_fcst = fr_ws_fcst.rename(columns={
    'Solar': 'fr_solar_forecast',
    'Wind Onshore': 'fr_wind_onshore_forecast',
    'Wind Offshore': 'fr_wind_offshore_forecast',
})

# Combine onshore + offshore into total wind forecast
fr_ws_fcst['fr_wind_forecast'] = (
    fr_ws_fcst['fr_wind_onshore_forecast'].fillna(0) +
    fr_ws_fcst['fr_wind_offshore_forecast'].fillna(0)
)

fr_ws_fcst = fr_ws_fcst[['fr_solar_forecast', 'fr_wind_forecast']]
print(f"FR wind/solar forecast: {len(fr_ws_fcst)} rows")


# ============================================================
# BLOCK 4: LOAD FRENCH LOAD FORECAST
# ============================================================

fr_load = pd.read_csv('raw/fr_load_forecast.csv', index_col=0)
fr_load.index = pd.to_datetime(fr_load.index, utc=True)
fr_load = fr_load.rename(columns={'Forecasted Load': 'fr_load'})
print(f"FR load forecast: {len(fr_load)} rows")


# ============================================================
# BLOCK 5: LOAD GERMAN DATA (forecasts)
# ============================================================

de_data = pd.read_csv('raw/de_power_real.csv', index_col=0)
de_data.index = pd.to_datetime(de_data.index, utc=True)
de_data = de_data.rename(columns={
    'load_forecast': 'de_load',
    'wind_forecast': 'de_wind',
    'solar_forecast': 'de_solar',
})
de_data = de_data[['de_load', 'de_wind', 'de_solar']]
de_data = de_data.resample('h').mean()
print(f"DE data (forecasts): {len(de_data)} rows")


# ============================================================
# BLOCK 6: LOAD TTF GAS PRICE
# ============================================================
# Daily data — forward-filled to hourly after the merge.

ttf = pd.read_csv('raw/ttf_gas_price.csv', index_col=0)
ttf = ttf[['Close']].rename(columns={'Close': 'ttf_close'})
ttf.index = pd.to_datetime(ttf.index, utc=True)
ttf['ttf_close'] = pd.to_numeric(ttf['ttf_close'], errors='coerce')

print(f"TTF gas: {len(ttf)} rows (daily)")
print(f"  Range: EUR {ttf['ttf_close'].min():.1f} to EUR {ttf['ttf_close'].max():.1f}")


# ============================================================
# BLOCK 7: MERGE EVERYTHING
# ============================================================

df = pd.concat([
    de_price[['de_price']],
    fr_price[['fr_price']],
    fr_nuclear[['fr_nuclear']],
    fr_ws_fcst[['fr_solar_forecast', 'fr_wind_forecast']],
    fr_load[['fr_load']],
    de_data[['de_load', 'de_wind', 'de_solar']],
], axis=1, join='inner')

# TTF is daily — forward-fill so each day's price covers all 24 hours.
ttf_hourly = ttf.reindex(df.index, method='ffill')
df['ttf_close'] = ttf_hourly['ttf_close']

print(f"\nMerged dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# BLOCK 8: HANDLE MISSING VALUES
# ============================================================

print(f"\nMissing values:")
print(df.isnull().sum())

df = df.interpolate(method='linear', limit=3)
rows_before = len(df)
df = df.dropna()
rows_after = len(df)
print(f"\nDropped {rows_before - rows_after} rows")
print(f"Final: {rows_after} rows")


# ============================================================
# BLOCK 9: SAVE
# ============================================================

df.to_csv('processed/clean_merged_v3.csv')
print(f"\nSaved to processed/clean_merged_v3.csv")
