"""
TTF GAS PRICE DOWNLOAD
=======================
Downloads Dutch TTF natural gas front-month futures.
Tries multiple tickers in case one is delisted.
No API key required.
"""

import yfinance as yf
import pandas as pd
import os

os.makedirs('raw', exist_ok=True)

# TTF tickers to try in order (NG=F is Henry Hub / US gas — not a valid TTF proxy)
TICKERS = ['TTF=F', 'TF=F']

print("Downloading TTF gas price...")

ttf = None
for ticker in TICKERS:
    print(f"  Trying {ticker}...")
    try:
        data = yf.download(ticker, start='2022-01-01', end='2025-01-06', progress=False)
        if len(data) > 100:
            ttf = data
            print(f"  Success with {ticker}: {len(data)} rows")
            break
        else:
            print(f"  {ticker} returned {len(data)} rows — skipping")
    except Exception as e:
        print(f"  {ticker} failed: {e}")

if ttf is None or len(ttf) == 0:
    raise RuntimeError(
        "Could not download TTF gas price data. "
        "Check that one of these tickers is available on yfinance: "
        + str(TICKERS)
    )

# Flatten multi-level columns if present (yfinance sometimes returns MultiIndex)
if isinstance(ttf.columns, pd.MultiIndex):
    ttf.columns = [col[0] for col in ttf.columns]

ttf.to_csv('raw/ttf_gas_price.csv')
print(f"Done. {len(ttf)} daily rows saved to raw/ttf_gas_price.csv")
