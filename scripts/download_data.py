"""
ENTSO-E DATA DOWNLOAD
======================
Downloads all required datasets from the ENTSO-E Transparency
Platform API. Requires a free API key from:
https://transparency.entsoe.eu

Replace 'YOUR_API_KEY_HERE' with your personal token.
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


# 1. Day-ahead prices
print("Downloading DE prices...")
de_price = download_in_chunks(client.query_day_ahead_prices, 'DE_LU')

print("Downloading FR prices...")
fr_price = download_in_chunks(client.query_day_ahead_prices, 'FR')

# 2. French generation (for nuclear)
print("Downloading FR generation...")
fr_generation = download_in_chunks(client.query_generation, 'FR')

# 3. French load forecast
print("Downloading FR load forecast...")
fr_load = download_in_chunks(client.query_load_forecast, 'FR')
fr_load = fr_load.iloc[:, 0]

# 4. French wind/solar forecast
print("Downloading FR wind/solar forecast...")
fr_ws_forecast = download_in_chunks(client.query_wind_and_solar_forecast, 'FR')

# Save all
de_price.to_csv('raw/de_day_ahead_price.csv')
fr_price.to_csv('raw/fr_day_ahead_price.csv')
fr_generation.to_csv('raw/fr_generation.csv')
fr_load.to_csv('raw/fr_load_forecast.csv')
fr_ws_forecast.to_csv('raw/fr_wind_solar_forecast.csv')

print("\nAll ENTSO-E downloads complete.")
