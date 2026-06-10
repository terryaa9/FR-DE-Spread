"""
MAIN — FR-DE SPREAD PROJECT
=============================
Runs the full pipeline in order:
  1. Download ENTSO-E data (FR & DE prices, flows, FR forecasts)
  2. Download DE wind/solar/load forecasts
  3. Download TTF gas price
  4. Clean & merge → clean_merged_v3.csv
  5. Feature engineering → model_ready_v3.csv
  6. Random Forest model + four regime splits → PNGs + CSVs

Downloads are skipped if the output files already exist.
Any step failure stops the pipeline immediately.

Usage:
  python main.py
"""

import subprocess
import sys
import os
import time

# All paths in the pipeline are relative to the repo root, so make that the
# working directory no matter where this script is invoked from.
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# PIPELINE DEFINITION
# ============================================================
# Each step: (script, output_file_to_check_for_skip, description)
# If output_file is None, the step always runs.

STEPS = [
    (
        'scripts/download_data.py',
        'raw/fr_day_ahead_price.csv',
        'Download ENTSO-E data (FR/DE prices, flows, FR forecasts)',
    ),
    (
        'scripts/download_de_data.py',
        'raw/de_power_real.csv',
        'Download DE load/wind/solar forecasts',
    ),
    (
        'scripts/download_ttf.py',
        'raw/ttf_gas_price.csv',
        'Download TTF gas price',
    ),
    (
        'scripts/data_loader.py',
        'processed/clean_merged_v3.csv',
        'Clean & merge all datasets',
    ),
    (
        'scripts/features.py',
        'processed/model_ready_v3.csv',
        'Feature engineering',
    ),
    (
        'scripts/analysis.py',
        None,
        'Random Forest model + four regime splits',
    ),
]


# ============================================================
# RUNNER
# ============================================================

def run_step(script, skip_if_exists, description, step_num, total):
    header = f"[{step_num}/{total}] {description}"
    print(f"\n{'='*60}")
    print(header)
    print('='*60)

    if skip_if_exists and os.path.exists(skip_if_exists):
        print(f"  SKIPPED — {skip_if_exists} already exists.")
        return

    start = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,   # stream output live
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        print(f"  Fix {script} and re-run.")
        sys.exit(1)

    print(f"\n  Done in {elapsed:.1f}s")


def main():
    print("FR-DE SPREAD PROJECT — FULL PIPELINE")
    print(f"Working directory: {os.getcwd()}")

    total = len(STEPS)
    pipeline_start = time.time()

    for i, (script, skip_file, desc) in enumerate(STEPS, start=1):
        run_step(script, skip_file, desc, i, total)

    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE in {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print("\nOutputs:")
    print("  outputs/regime_analysis.png   — feature importance across 4 regime splits")
    print("  outputs/baseline_analysis.png — predicted vs actual, feature importance")
    print("  outputs/regime_summary.csv    — R² and MAE per regime")
    print("  outputs/feature_importance.csv")


if __name__ == '__main__':
    main()
