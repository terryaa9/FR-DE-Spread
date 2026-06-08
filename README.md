# FR-DE Spread Model

## Project Overview
Models the France-Germany day-ahead electricity price spread using physical fundamentals: residual load differential, TTF gas price, French nuclear availability, and wind/solar forecasts. A Random Forest is trained on 2022-2024 hourly data and evaluated across four market regimes (gas price, DE wind, FR nuclear, peak/off-peak).

---

## Folder Structure

```
data/
├── raw/                  # Original downloaded data — do not modify
│   ├── de_day_ahead_price.csv       # German day-ahead electricity prices
│   ├── de_power_real.csv            # German load, wind & solar forecasts
│   ├── fr_day_ahead_price.csv       # French day-ahead electricity prices
│   ├── fr_generation.csv            # French realized generation (nuclear extracted)
│   ├── fr_load_forecast.csv         # French load forecast
│   ├── fr_wind_solar_forecast.csv   # French wind & solar forecast
│   └── ttf_gas_price.csv            # TTF natural gas prices
│
├── processed/            # Cleaned and feature-engineered data
│   ├── clean_merged.csv             # Merged and cleaned hourly dataset
│   ├── model_ready.csv              # Final model-ready feature matrix
│   └── feature_columns.txt          # Feature columns used in model
│
├── scripts/              # Python scripts for data download, processing, and analysis
│   ├── main.py                      # Pipeline entry point — runs all steps in order
│   ├── utils.py                     # Shared download utility
│   ├── download_data.py             # Downloads FR/DE prices and FR forecasts
│   ├── download_de_data.py          # Downloads German load/wind/solar forecasts
│   ├── download_ttf.py              # Downloads TTF gas price
│   ├── data_loader.py               # Cleans and merges all datasets
│   ├── features.py                  # Feature engineering and regime labels
│   └── analysis.py                  # Random Forest model and regime analysis
│
├── outputs/              # Generated charts and results
│   ├── baseline_analysis.png        # Predicted vs actual, feature importance
│   ├── regime_analysis.png          # Feature importance across four regimes
│   ├── regime_summary.csv           # R² and MAE per regime
│   └── feature_importance.csv       # Baseline model feature importance
│
├── .gitignore
└── README.md
```

---

## Data Sources
- **ENTSO-E Transparency Platform** — electricity prices, generation, and load/wind/solar forecasts
- **yfinance** — TTF natural gas front-month futures

## Setup
1. Get a free API key from [transparency.entsoe.eu](https://transparency.entsoe.eu)
2. Create a `.env` file with: `ENTSOE_API_KEY=your_key_here`
3. Install dependencies: `pip install entsoe-py yfinance pandas scikit-learn matplotlib python-dotenv`
4. Run: `python main.py`

## Notes
- The `.env` file contains API credentials — never commit it to version control.
- All features are look-ahead free: wind/solar/load use TSO day-ahead forecasts, nuclear is lagged 1 hour.
- Re-run `python main.py` to refresh data; steps are skipped if output files already exist.
