"""
STEP 3 — SPREAD MODEL & REGIME ANALYSIS
=========================================
1. Baseline Random Forest: predicts FR-DE spread from fundamentals
2. Four regime splits: shows how spread drivers change across
   market conditions (gas, wind, nuclear, peak/off-peak)
3. Feature importance comparison across regimes
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# BLOCK 1: LOAD DATA
# ============================================================

df = pd.read_csv('processed/model_ready_v3.csv', index_col=0, parse_dates=True)

with open('processed/feature_columns_v3.txt', 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]

target = 'spread'

print(f"Loaded {len(df)} rows, {len(feature_columns)} features")


# ============================================================
# BLOCK 2: TRAIN / TEST SPLIT
# ============================================================
# Temporal split: train on historical data, test on unseen future.
# Train includes the 2022 energy crisis (high gas) and 2023
# normalisation — the model learns both regimes.
# Test is Jul 2024 – Jan 2025 (calmer market).

split_date = '2024-07-01'
train = df[df.index < split_date]
test = df[df.index >= split_date]

X_train = train[feature_columns]
y_train = train[target]
X_test = test[feature_columns]
y_test = test[target]

print(f"Train: {len(train)} rows ({train.index[0].date()} to {train.index[-1].date()})")
print(f"Test:  {len(test)} rows ({test.index[0].date()} to {test.index[-1].date()})")


# ============================================================
# BLOCK 3: BASELINE MODEL
# ============================================================

print("\n" + "="*60)
print("BASELINE MODEL — ALL DATA")
print("="*60)

model_baseline = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

model_baseline.fit(X_train, y_train)

pred_train = model_baseline.predict(X_train)
pred_test = model_baseline.predict(X_test)

print(f"\nTRAIN — MAE: {mean_absolute_error(y_train, pred_train):.1f} EUR/MWh | R²: {r2_score(y_train, pred_train):.4f}")
print(f"TEST  — MAE: {mean_absolute_error(y_test, pred_test):.1f} EUR/MWh | R²: {r2_score(y_test, pred_test):.4f}")

# Feature importance
imp_baseline = pd.DataFrame({
    'feature': feature_columns,
    'importance': model_baseline.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
for _, row in imp_baseline.iterrows():
    bar = '#' * int(row['importance'] * 50)
    print(f"  {row['feature']:<25} {row['importance']:.4f}  {bar}")


# ============================================================
# BLOCK 4: REGIME ANALYSIS
# ============================================================
# For each regime, we train a separate model on the train data
# within that regime, test on the test data within that regime,
# and extract feature importances.
#
# Shows how drivers change across market conditions.

regimes = {
    'Gas Price': ('regime_gas', ['high_gas', 'low_gas']),
    'DE Wind': ('regime_wind', ['high_wind', 'low_wind']),
    'FR Nuclear': ('regime_nuclear', ['high_nuclear', 'low_nuclear']),
    'Peak/Off-Peak': ('regime_peak', ['peak', 'off_peak']),
}

# Store results for comparison
regime_results = {}

for regime_name, (regime_col, regime_values) in regimes.items():
    print(f"\n{'='*60}")
    print(f"REGIME: {regime_name}")
    print(f"{'='*60}")

    for value in regime_values:
        # Filter train and test data for this regime
        train_regime = train[train[regime_col] == value]
        test_regime = test[test[regime_col] == value]

        if len(train_regime) < 500 or len(test_regime) < 100:
            print(f"\n  {value}: insufficient data (train={len(train_regime)}, test={len(test_regime)})")
            continue

        X_tr = train_regime[feature_columns]
        y_tr = train_regime[target]
        X_te = test_regime[feature_columns]
        y_te = test_regime[target]

        # Train regime-specific model
        model_regime = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        model_regime.fit(X_tr, y_tr)

        pred_regime = model_regime.predict(X_te)
        mae = mean_absolute_error(y_te, pred_regime)
        r2 = r2_score(y_te, pred_regime)

        # Feature importance for this regime
        imp = pd.DataFrame({
            'feature': feature_columns,
            'importance': model_regime.feature_importances_
        }).sort_values('importance', ascending=False)

        # Spread stats in this regime
        avg_spread = test_regime['spread'].mean()
        std_spread = test_regime['spread'].std()

        print(f"\n  {value}:")
        print(f"    Train: {len(train_regime)} hours | Test: {len(test_regime)} hours")
        print(f"    Avg spread: {avg_spread:.1f} EUR/MWh | Std: {std_spread:.1f} EUR/MWh")
        print(f"    MAE: {mae:.1f} EUR/MWh | R²: {r2:.4f}")
        print(f"    Top 5 drivers:")
        for _, r in imp.head(5).iterrows():
            print(f"      {r['feature']:<25} {r['importance']:.4f}")

        # Store for visualisation
        regime_results[f"{regime_name}: {value}"] = {
            'importance': imp,
            'mae': mae,
            'r2': r2,
            'avg_spread': avg_spread,
            'std_spread': std_spread,
            'n_hours': len(test_regime),
        }


# ============================================================
# BLOCK 5: VISUALISATION — REGIME COMPARISON
# ============================================================
# Four panels, one per regime. Each shows side-by-side feature
# importances for the two sub-regimes.

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('FR-DE Spread Drivers Across Market Regimes',
             fontsize=16, fontweight='bold')

regime_pairs = [
    ('Gas Price: high_gas', 'Gas Price: low_gas', 'Gas Price Regime', axes[0, 0]),
    ('DE Wind: high_wind', 'DE Wind: low_wind', 'DE Wind Regime', axes[0, 1]),
    ('FR Nuclear: high_nuclear', 'FR Nuclear: low_nuclear', 'FR Nuclear Regime', axes[1, 0]),
    ('Peak/Off-Peak: peak', 'Peak/Off-Peak: off_peak', 'Peak vs Off-Peak', axes[1, 1]),
]

for key_a, key_b, title, ax in regime_pairs:
    if key_a not in regime_results or key_b not in regime_results:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        ax.set_title(title)
        continue

    imp_a = regime_results[key_a]['importance'].head(8)
    imp_b = regime_results[key_b]['importance'].set_index('feature')

    features = imp_a['feature'].values
    vals_a = imp_a['importance'].values
    vals_b = [imp_b.loc[f, 'importance'] if f in imp_b.index else 0 for f in features]

    y_pos = np.arange(len(features))
    bar_height = 0.35

    bars1 = ax.barh(y_pos - bar_height/2, vals_a, bar_height,
                     label=key_a.split(': ')[1], color='steelblue', alpha=0.8)
    bars2 = ax.barh(y_pos + bar_height/2, vals_b, bar_height,
                     label=key_b.split(': ')[1], color='coral', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')

    # Add spread stats to title
    stats_a = regime_results[key_a]
    stats_b = regime_results[key_b]
    ax.set_title(f"{title}\n"
                 f"{key_a.split(': ')[1]}: spread={stats_a['avg_spread']:.1f}±{stats_a['std_spread']:.1f} | "
                 f"{key_b.split(': ')[1]}: spread={stats_b['avg_spread']:.1f}±{stats_b['std_spread']:.1f}",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('outputs/regime_analysis.png', dpi=150, bbox_inches='tight')
print("\nSaved regime analysis to outputs/regime_analysis.png")


# ============================================================
# BLOCK 6: BASELINE MODEL VISUALISATION
# ============================================================

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('FR-DE Spread Model — Baseline Analysis',
              fontsize=14, fontweight='bold')

# Plot 1: Predicted vs Actual spread
ax = axes2[0, 0]
ax.scatter(y_test, pred_test, alpha=0.1, s=5, color='steelblue')
spread_min = min(y_test.min(), pred_test.min())
spread_max = max(y_test.max(), pred_test.max())
ax.plot([spread_min, spread_max], [spread_min, spread_max], 'r--', linewidth=1)
ax.set_xlabel('Actual Spread (EUR/MWh)')
ax.set_ylabel('Predicted Spread (EUR/MWh)')
ax.set_title(f'Predicted vs Actual (R²={r2_score(y_test, pred_test):.3f})')
ax.grid(True, alpha=0.3)

# Plot 2: Feature importance (top 12)
ax = axes2[0, 1]
top_12 = imp_baseline.head(12)
ax.barh(range(len(top_12)), top_12['importance'].values, color='steelblue')
ax.set_yticks(range(len(top_12)))
ax.set_yticklabels(top_12['feature'].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Importance')
ax.set_title('Feature Importance — Baseline Model')
ax.grid(True, alpha=0.3, axis='x')

# Plot 3: Spread by hour of day
ax = axes2[1, 0]
hourly_spread = test.groupby('hour')['spread'].agg(['mean', 'std'])
ax.plot(hourly_spread.index, hourly_spread['mean'], 'o-', color='steelblue', linewidth=2)
ax.fill_between(hourly_spread.index,
                hourly_spread['mean'] - hourly_spread['std'],
                hourly_spread['mean'] + hourly_spread['std'],
                alpha=0.2, color='steelblue')
ax.set_xlabel('Hour (UTC)')
ax.set_ylabel('Spread (EUR/MWh)')
ax.set_title('Average FR-DE Spread by Hour')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Plot 4: Spread vs DE residual load (coloured by gas regime)
ax = axes2[1, 1]
high_gas = test[test['regime_gas'] == 'high_gas']
low_gas = test[test['regime_gas'] == 'low_gas']
ax.scatter(high_gas['de_residual_load'], high_gas['spread'],
           alpha=0.05, s=5, color='red', label='High gas')
ax.scatter(low_gas['de_residual_load'], low_gas['spread'],
           alpha=0.05, s=5, color='blue', label='Low gas')
ax.set_xlabel('DE Residual Load (MW)')
ax.set_ylabel('Spread (EUR/MWh)')
ax.set_title('Spread vs DE Residual Load — by Gas Regime')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/baseline_analysis.png', dpi=150, bbox_inches='tight')
print("Saved baseline analysis to outputs/baseline_analysis.png")


# ============================================================
# BLOCK 7: SAVE RESULTS
# ============================================================

imp_baseline.to_csv('outputs/feature_importance.csv', index=False)

# Save regime summary
regime_summary = []
for key, val in regime_results.items():
    regime_summary.append({
        'regime': key,
        'avg_spread': val['avg_spread'],
        'std_spread': val['std_spread'],
        'mae': val['mae'],
        'r2': val['r2'],
        'n_hours': val['n_hours'],
        'top_driver': val['importance'].iloc[0]['feature'],
        'top_driver_importance': val['importance'].iloc[0]['importance'],
    })
pd.DataFrame(regime_summary).to_csv('outputs/regime_summary.csv', index=False)
print("Saved all results.")
