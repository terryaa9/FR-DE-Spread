"""
GERMAN POWER DATA DOWNLOAD
============================
Downloads German (DE_LU) day-ahead forecasts from ENTSO-E:
  - Load forecast
  - Wind & solar forecast (combined query)

Saves to raw/de_power_real.csv with columns:
  load_forecast, wind_forecast, solar_forecast

15-min resolution, resampled to hourly in data_loader.py.
Requires a free ENTSO-E API key from:
https://transparency.entsoe.eu
"""

from entsoe import EntsoePandasClient
import pandas as pd
import os
from dotenv import load_dotenv
from utils import download_in_chunks

load_dotenv()
API_KEY = os.getenv('ENTSOE_API_KEY')
client = EntsoePandasClient(api_key=API_KEY)

os.makedirs('raw', exist_ok=True)


# 1. DE load forecast
print("Downloading DE load forecast...")
de_load = download_in_chunks(client.query_load_forecast, 'DE_LU')
de_load = de_load.iloc[:, 0].rename('load_forecast')

# 2. DE wind & solar forecast
print("Downloading DE wind & solar forecast...")
de_ws = download_in_chunks(client.query_wind_and_solar_forecast, 'DE_LU')

# de_ws is a DataFrame with columns like 'Wind Onshore', 'Wind Offshore', 'Solar'
# Combine onshore + offshore into a single wind_forecast column
de_ws['wind_forecast'] = de_ws.filter(like='Wind').sum(axis=1)
de_ws = de_ws.rename(columns={'Solar': 'solar_forecast'})
de_ws = de_ws[['wind_forecast', 'solar_forecast']]

# 3. Merge
print("Merging...")
df = pd.concat([de_load, de_ws], axis=1)
df.index.name = 'timestamp'

df = df[['load_forecast', 'wind_forecast', 'solar_forecast']]

df.to_csv('raw/de_power_real.csv')
print(f"\nDone. {len(df)} rows saved to raw/de_power_real.csv")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Columns: {list(df.columns)}")
