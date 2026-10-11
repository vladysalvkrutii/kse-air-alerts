# Forecasting air raid alerts in Ukraine

KSE CS selection, Stage 2. Hourly, per-oblast probability that a new air raid alert starts within the
next 3 hours, built and evaluated so that no forecast ever uses information from after its forecast time.

## Problem framing

- **Forecast time t:** every whole hour (UTC).
- **Target:** for each of 24 regions, `y(t) = 1` if a new alert starts in the region in `[t, t + 3h)`.
  "Starts", not "is active": otherwise "an alert running now keeps running" would win for free.
- **Output:** a probability; scored with the **Brier score** (mean squared difference between the
  probability and the 0/1 outcome; lower is better), **within-region AUC** (does the model rank risky
  hours above quiet ones inside the same region; 0.5 = coin flip) and a **calibration** table.
  A pooled AUC across regions was rejected: it mostly rewards knowing which regions are busier.

## Data

[Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset),
`datasets/volunteer_data_en.csv` (eTryvoga volunteer channel, MIT license), pinned to commit
`bf0c323` (2026-10-11 04:50 UTC). Columns: `region, started_at, finished_at, naive`, UTC, oblast level.

- 115,986 alerts used, 2022-02-25 → 2026-10-10. 24 regions (Luhansk excluded: permanent alert not in the
  data; Crimea not in the data).
- The last 24 h of the snapshot are dropped: alerts still running at download time are not in the file.
- **Why not the official data** in the same repo: it ends on 2026-09-07, as does the public feed of its
  source Telegram channel (@air_alert_ua). That fails "evaluate on the latest period".
  alerts.in.ua needs a token and only exposes the last month.
- Known data issues: ~10% of alerts are `naive` (no end message; the dataset invents start + 30 min);
  volunteer and official data agree only moderately (e.g. Lviv/Volyn far busier in volunteer data).
  The target is therefore "alert as reported by eTryvoga".

## Method

**Evaluation (walk-forward).** The model is refit every week using only the past, then forecasts the
following week, like a deployed system would be retrained.

| Window | Forecast times (UTC) | Used for |
|---|---|---|
| Validation | 2026-08-15 04:00 → 2026-09-12 04:00 (4 weekly folds) | all choices (features, training length) |
| **Test** | 2026-09-12 04:00 → 2026-10-10 01:00 (last forecast time; 4 weekly folds) | final numbers only (`--final-test`) |

At each refit, a training example at time t' is used only if its label window `[t', t'+3h)` has closed
before the fold starts, so no label from the future enters training.

**Baselines.** base rate (per region, last 8 weeks); climatology (region × Kyiv hour-of-day; a
day-of-week × hour version overfits); persistence (raw 0/1: "did an alert start in the last 3 h");
calibrated persistence (the same signal turned into per-region probabilities); seasonal naive (same
window 24 h / 168 h ago).

**Model.** Logistic regression (scikit-learn, standardized features, region one-hot), trained on the last
16 weeks (chosen on validation among 8/16/26/40). Features, all from information available at t:
alerts started in the last 1/3/6/24/168 h (log), hours since the last start, whether an alert is running
at t and for how long, number of regions with a start in the last 3 h / running now, Kyiv hour of day
(sin/cos). Chosen because every weight can be read and explained, it outputs probabilities directly, and
it trains in seconds.

**Leakage protection** (`tests/test_no_leakage.py`). Features at t are computed from (a) the full data,
(b) only what an observer had at t, (c) that plus fake future alerts and fake end times; all three must
be identical and NaN-free. "Known at t" means: alerts started before t, and an end only if its end
message arrived by t. `naive` alerts never got one, so they count as running until the next alert starts
in the region. The test checks itself with two deliberately leaky features, which must make it fail.

## Results

Test window (12 Sep – 10 Oct 2026, 16,080 region-hours, 49.2% positive):

| Forecaster | Brier ↓ | Within-region AUC | Brier skill vs climatology |
|---|---|---|---|
| **Logistic regression (16 w)** | **0.1778** | **0.588** | **+7.7%** |
| Calibrated persistence | 0.1913 | 0.503 | +0.7% |
| Climatology (hour-of-day) | 0.1927 | 0.551 | 0 |
| Base rate | 0.1934 | 0.500 | −0.4% |
| Climatology (day-of-week × hour) | 0.1988 | 0.546 | −3.2% |
| Seasonal naive 24 h | 0.3325 | 0.532 | −73% |
| Seasonal naive 168 h | 0.3458 | 0.510 | −79% |
| Persistence (raw 0/1) | 0.3521 | 0.512 | −83% |

The model has the best Brier score in each of the 4 test weeks (0.183 / 0.169 / 0.174 / 0.186 vs best
baseline 0.198 / 0.182 / 0.186 / 0.198, i.e. 6.2–7.6% better each week; 7.0% overall vs calibrated
persistence). Validation: 0.1593 vs 0.1683 for the best baseline.

These numbers are from a re-run of the identical pipeline on the newest snapshot. The first run, on
snapshot `7ccd436` (test 11 Sep – 9 Oct), gave 0.1762 vs 0.1892 (−6.9%); see `NOTES.md`.

Calibration on test (`results/test_calibration.csv`): accurate above 40% (predicted 0.65 → observed 0.66,
0.94 → 0.94); under-predicts at 10–30% (0.26 → 0.34).

## Limitations

- **The gain is modest.** About 7% better Brier than the best baseline. Within a region the ranking signal
  is weak (AUC 0.59 on test, 0.62 on validation). Most of the skill comes from "how active has this
  region been in the last week" plus whether an alert is running right now.
- **Under-prediction:** alert frequency has been rising, and a model trained on the last 16 weeks lags it.
- **One label source:** eTryvoga volunteer data. It disagrees with official data in some regions; we
  cannot tell which one is right. Oblast level only (official alerts are mostly raion level since Dec 2025).
- **Backtest vs live:** the snapshot is assumed to be what was known in real time. Checked on three snapshots
  (3 Oct vs 10 Oct: no past row changed, one row added late; 10 Oct vs 11 Oct: no change at all).
- **Not used:** text/Telegram/news features and neighbouring-region features (see Future work: adding
  them after the test was run would mean tuning on the test window).
- The test window is only 4 weeks; small differences between baselines are within noise.

## Future work

The current test window has been looked at, so any change below must be judged on validation and then
on a **new** test window made only of data after this snapshot (after 2026-10-10). Re-using the current
window would be tuning on the test.

- **Text features from Telegram:** Ukrainian Air Force (@kpszsu) statements and monitoring channels
  (e.g. reported launches of drones or missiles), using only posts published before t. Telegram posts can
  be edited or deleted later, so they must be archived as first published (or edited posts dropped);
  otherwise a later edit leaks into an earlier forecast. The leakage test would cover these features too.
- **Neighbouring regions:** activity in adjacent oblasts in the last hours (attacks move across regions).
- **Fixing under-prediction:** recalibrate on the most recent weeks, or add faster-reacting trend features.
- **Non-linear model** (e.g. gradient boosting), kept only if it beats logistic regression on validation.
- **Second label source:** cross-check against alerts.in.ua (needs a token) or a revived official feed,
  to measure how noisy the volunteer labels are; raion-level targets if fresh raion data becomes available.
- **Other targets:** 1 h / 6 h / 12 h horizons, and a nationwide "mass attack" target
  (≥ K regions under alert at once).
- **Uncertainty:** confidence intervals for the Brier difference, e.g. by resampling whole weeks.
- **Live check:** archive a dataset snapshot every day to confirm the backtest data equals what was known
  in real time.

## How to run

Windows / Python 3.12:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run.py                # download pinned data, validation scores
.venv\Scripts\python -m pytest -q          # leakage tests (needs the data from the line above)
.venv\Scripts\python run.py --final-test   # held-out test scores
```

Outputs go to `results/`. To use a newer snapshot, change `DATA_SHA` and `DATA_SNAPSHOT_UTC` in
`airalerts/config.py`; all windows move with it.

## Repository layout

```
run.py                     one command: download -> data -> baselines + model -> scores
airalerts/config.py        snapshot, horizon, windows, chosen training length
airalerts/download.py      fetch the pinned CSV
airalerts/data.py          load alerts, hourly start counts, target
airalerts/features.py      features computable at t
airalerts/baselines.py     baselines
airalerts/model.py         logistic regression
airalerts/evaluate.py      walk-forward folds, Brier / AUC / calibration
tests/test_no_leakage.py   leakage and target-boundary tests
NOTES.md                   decisions, failures, what changed and why
```

## How AI was used

Two AI tools were used throughout the project, in two roles:

- **Claude Code: engineering.** It proposed options, wrote and ran the code, and showed real output
  at every step.
- **A separate Claude chat: reviewer, for the whole project, not just the start.** It helped me
  write the kickoff prompt. After that I usually pasted Claude Code's answers into it to get a
  second opinion, then went back to Claude Code with questions. Several of my key questions came
  out of that review, including:
  - whether the "alert running at t" feature leaked through the end times the dataset invents for
    `naive` alerts. It did, and fixing it changed both the feature and the leakage test;
  - making sure baseline scores were reported on validation, not on the test window;
  - why the effect of a recent alert reversed in frontline regions. The answer, an alert already
    running at t blocks a new start, led to the "active alert" features.

Final decisions (target, horizon, training window, when to run the test) were mine, usually after Claude
Code laid out the options and gave a recommendation. Other bugs (missing
features at the last hour, a baseline with missing forecasts) came up while running the pipeline and
were fixed and recorded. See `NOTES.md` for the full trail.
