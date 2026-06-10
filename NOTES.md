# Methodology & Results — FR-DE Day-Ahead Spread

## Approach
I model the hourly France–Germany day-ahead price spread (`fr_price − de_price`)
with a Random Forest on physical fundamentals: residual-load differential, TTF
gas, French nuclear availability, and wind/solar forecasts. These are fairly
basic fundamental drivers, but the ones I understand best — I avoided
adding more intricate data points that I didn't fully grasp. Data spans
Feb 2022 – Jan 2025 (ENTSO-E + TTF futures). The split is temporal: I train
through 2024-06-30 (including the 2022 energy crisis) and test on Jul 2024 –
Jan 2025 (a calmer market), so the test set is genuinely out-of-sample in time.


## Information-set discipline (look-ahead audit)
The prices being modelled are fixed at the day-ahead auction (~12:00 CET on D-1),
so every feature must be known by then. During cleanup I found and corrected
three leaks where this was violated:

| Issue | Problem | Fix |
|---|---|---|
| **TTF gas** (top feature) | The daily `Close` is an end-of-day settlement, forward-filled onto the *same* delivery day → leakage of future price | Lag one day: delivery D uses D-1's settled close |
| **FR nuclear** | A 1h lag of actual generation is a *real-time* information set, not day-ahead | Use the previous-day profile (24h lag) as a causal proxy for published availability |
| **Regime thresholds** | Median split points computed over the full sample → test distribution leaks into the splits | Fit thresholds on the train period only |

A note on how two of these crept in: the nuclear 1h-lag is a remnant from an
earlier version of the project, where I was trying to model the intraday spread
rather than the day-ahead one. And the regime threshold leak I genuinely just
forgot, I had the regime idea on a whim and split the whole dataset out of
habit, instead of splitting only the train portion.

I also replaced bidirectional linear interpolation with causal forward-fill,
which only uses past rows to fill gaps.

## Headline result
Removing the leakage left out-of-sample performance **essentially unchanged**:

| | Train R² | Test R² | Test MAE (€/MWh) |
|---|---|---|---|
| Before (leaky) | 0.72 | 0.11 | 19.8 |
| After (corrected) | 0.72 | 0.13 | 19.4 |

It looks like my mistakes earlier and the leakage did not affect the baseline
model's performance too much — the gas and nuclear features are slow-moving, so
the previous-day values carry almost the same information as the leaked same-day
ones. The top two drivers are unchanged: TTF gas (0.38) and residual-load
differential (0.33). Nuclear's importance does drop (0.09 → 0.07), which is the
footprint of the lag fix.

## Regime analysis (corrected, train-fit thresholds)
Driver structure and skill vary sharply by market condition:

| Regime | Test hours | R² | Top driver |
|---|---|---|---|
| Low gas | 4,270 | 0.15 | residual-load diff |
| High gas | 263 | −0.99 | residual-load diff |
| High DE wind | 2,078 | 0.29 | TTF gas |
| Low DE wind | 2,455 | −0.01 | TTF gas |
| High FR nuclear | 4,064 | 0.12 | residual-load diff |
| Low FR nuclear | 469 | 0.37 | TTF gas |
| Peak | 2,457 | 0.05 | TTF gas |
| Off-peak | 2,076 | 0.28 | TTF gas |

The model does best in low-volatility regimes — low-nuclear, high-wind and
off-peak. In these calm hours the two zones' prices converge — e.g. strong
German wind pulls DE prices down toward France's nuclear-cheapened level — so
the spread is small. It breaks down in high-variance regimes (low wind, peak), where
scarcity pricing and interconnector congestion drive swings the fundamentals
can't explain, especially since I don't have data on the interconnector in the
feature set. (One caveat: low-nuclear is only 469 test hours, so treat its
R² as indicative.)

The high-gas regime is a different kind of failure. Its 263 test hours are
actually calm (spread −0.4 ± 23.5), so by the logic above it should be
predictable — yet it scores −0.99. It comes down to my threshold leak. Because
I first set the medians on the full dataset, the test period's own
prices helped define the split. Once it's fixed to the train period only, the
2022 energy crisis pushes that median up to ~€48/MWh — so almost none of the calm
2024 test set clears it: only **263 of 4,533 test hours** are "high-gas." The
leaky full-sample median sat lower (~€43) and labelled far more hours high-gas,
masking how low-gas 2024 really was. With only 263 points — points that also
look nothing like the crisis-era hours the high-gas model trained on — the
regime can't be scored meaningfully (hence the −0.99 R²).

In conclusion, both of these regimes — high-gas and low-nuclear — are
essentially untestable once the threshold mistake is fixed: the train-fit
medians, set during the 2022 crisis, leave too few test hours on the extreme
side (263 and 469 of 4,533) to draw reliable conclusions from their R².

## Regime feature importance — before vs after
Unlike the baseline R² (which barely moved), the per-regime driver picture
shifts noticeably.

| Regime | Before (leaky) top drivers | After (fixed) top drivers | What moved |
|---|---|---|---|
| Gas: high | residual .39, ttf .30, month .08, nuc .08 | residual .39, ttf .27, **month .12**, nuc .04 | nuclear ↓, month ↑; test n **1221→263** |
| Gas: low | residual .41, **nuc .17**, ttf .09 | residual .42, ttf .14, **nuc .10** | nuclear demoted from #2 to #3 |
| Wind: high | ttf .48, residual .31 | ttf .48, residual .31 | ~unchanged |
| Wind: low | ttf .32, de_solar .27, residual .11, nuc .07 | ttf .34, de_solar .26, residual .11 | nuclear drops out of top 5 |
| Nuclear: high | residual .47, ttf .35 | residual .43, **ttf .40** | ttf ↑ |
| Nuclear: low | **residual .51**, ttf .30 | **ttf .44**, residual .35 | **top driver flips** |
| Peak | ttf .42, residual .36, nuc .05 | ttf .42, residual .35, nuc .04 | nuclear ↓ |
| Off-peak | ttf .35, residual .23, de_wind .15 | ttf .36, residual .22, de_wind .16 | ~unchanged |

**1. Nuclear lag fix (feature relabeling).** Same effect as in the baseline but
sharper in some regimes. In **low-gas** hours the leaked 1h nuclear was the #2
driver (0.17); with the previous-day proxy it falls to 0.10 and drops below TTF.
The nuclear leak was overstating nuclear's role specifically in low-gas, peak
and low-wind conditions.

**2. Train-only thresholds (regime relabeling).** The bigger structural change —
it alters *which hours fall in which regime*, so each regime model trains and
tests on a different slice of data. Beyond the gas-regime collapse described
above, this is what flips the low-nuclear regime's top driver from residual-load
to TTF: its membership changes, so a different model is being fit.

Takeaway: removing the nuclear leak demotes nuclear most where it was overstated,
and the train-fit thresholds reveal a structurally low-gas test period — a
stronger, more honest story than the single-number baseline R².

## Honest limitations
- Single temporal split; a walk-forward / expanding-window evaluation would
  better characterise stability across regimes.
- The TTF fix isn't perfectly clean either: D-1's close settles ~18:00 CET, a
  few hours *after* the noon auction. I tested the strict two-day lag (D uses
  D-2's close): test R² 0.127 vs 0.129 — negligible, so I kept D-1's close,
  which is the closer proxy for what traders actually knew at the auction. The
  same few-hour residual applies to the 24h nuclear proxy.
- Nuclear uses a 24h-lagged actual-generation proxy rather than REMIT published
  availability.
