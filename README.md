# FR-DE Spread Model

## Project Overview
Models the France-Germany day-ahead electricity price spread using physical fundamentals: residual load differential, TTF gas price, French nuclear availability, and wind/solar forecasts. A Random Forest is trained on 2022-2024 hourly data and evaluated across four market regimes (gas price, DE wind, FR nuclear, peak/off-peak).

**See [NOTES.md](NOTES.md) for the methodology, the look-ahead audit, and results.**

---

## Folder Structure

```
raw/                      # Original downloaded data — do not modify
├── de_day_ahead_price.csv       # German day-ahead electricity prices
├── de_power_real.csv            # German load, wind & solar forecasts
├── fr_day_ahead_price.csv       # French day-ahead electricity prices
├── fr_generation.csv            # French realized generation (nuclear extracted)
├── fr_load_forecast.csv         # French load forecast
├── fr_wind_solar_forecast.csv   # French wind & solar forecast
└── ttf_gas_price.csv            # TTF natural gas prices

processed/                # Cleaned and feature-engineered data
├── clean_merged_v3.csv          # Merged and cleaned hourly dataset
├── model_ready_v3.csv           # Final model-ready feature matrix
└── feature_columns_v3.txt       # Feature columns used in model

scripts/                  # Python scripts for data download, processing, and analysis
├── main.py                      # Pipeline entry point — runs all steps in order
├── utils.py                     # Shared download utility
├── download_data.py             # Downloads FR/DE prices and FR forecasts
├── download_de_data.py          # Downloads German load/wind/solar forecasts
├── download_ttf.py              # Downloads TTF gas price
├── data_loader.py               # Cleans and merges all datasets
├── features.py                  # Feature engineering and regime labels
└── analysis.py                  # Random Forest model and regime analysis

outputs/                  # Generated charts and results
├── baseline_analysis.png        # Predicted vs actual, feature importance
├── regime_analysis.png          # Feature importance across four regimes
├── regime_summary.csv           # R² and MAE per regime
└── feature_importance.csv       # Baseline model feature importance

requirements.txt          # Pinned dependencies
.gitignore
README.md
```

> `main.py` can be run from anywhere (it locates the repo root itself). **Individual scripts should be run from the repository root**, since their paths are resolved relative to the project root, not the `scripts/` folder.

---

## Data Sources
- **ENTSO-E Transparency Platform** — electricity prices, generation, and load/wind/solar forecasts
- **yfinance** — TTF natural gas front-month futures

## Setup
1. Get a free API key from [transparency.entsoe.eu](https://transparency.entsoe.eu)
2. Create a `.env` file with: `ENTSOE_API_KEY=your_key_here`
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python scripts/main.py`

## Notes

- **All features respect the day-ahead information set** (known at the ~noon-D-1 auction that fixes the prices being modelled):
  - Wind / solar / load use the TSOs' published **day-ahead forecasts**.
  - **TTF gas** is lagged one day — delivery day D uses the previous day's settled close, since day D's close does not exist at auction time.
  - **Nuclear** uses the **previous day's** generation profile (24h lag) as a causal proxy for the availability published for day D; reactor output is slow-moving.
- Re-run `python scripts/main.py` to refresh data; steps are skipped if output files already exist.
