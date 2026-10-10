# NOTES — decisions, failures, surprises

## 2026-10-10 (Sat) — data check, framing, baselines

### Data sources checked (actually downloaded/probed, not from memory)
| Source | Result |
|---|---|
| Vadimkin dataset, `volunteer_data_en.csv` (eTryvoga) | Fresh: last alert 2026-10-10 04:47 UTC. 25 regions, oblast level, since 2022-02-25. **Chosen.** |
| Vadimkin dataset, `official_data_en.csv` | Repo commits daily but data **ends 2026-09-07**; fails "latest period". |
| Telegram @air_alert_ua (public preview) | Feed also ends 2026-09-07 06:09 UTC mid-stream, which explains the official file. Reason unknown. |
| alerts.in.ua API | Exists; returns 401 without token; history endpoint is `month_ago` only. Not used. |
| Kaggle missile attacks | Not checked (text/extra features cut for time). |

Volunteer-data checks:
- No silent history rewriting: 3 Oct vs 10 Oct snapshots identical for 114,542 past rows (1 late-added row).
- ~10% of alerts are `naive`: no end message, `finished_at` is invented as start+30 min.
- Only moderate agreement with official data (Aug 2026 oblast-hour "active": 77% agree, corr 0.43).
  Lviv/Volyn have 3–5 alerts/day in volunteer data vs ~0 in official. Unexplained; our target is
  "alert as reported by eTryvoga".

### Decisions
- **Target (option A):** at each hour t (UTC), per region, y = 1 if a new alert starts in [t, t+3h).
  "Starts" rather than "active" so that persistence of an ongoing alert isn't a free win.
- Regions: 24 (Luhansk excluded: permanent alert, not in data; Crimea absent).
- Snapshot pinned to dataset commit `7ccd436` (2026-10-10 05:01 UTC). Last 24 h dropped (alerts still
  running at snapshot time are missing). DATA_END = 2026-10-09 05:00 UTC.
- **Test window fixed before seeing any results:** forecast times 2026-09-11 05:00 → 2026-10-09 02:00 UTC
  (28 days). Only scored by `run.py --final-test`, once, at the end.
- Validation: 4 weekly walk-forward folds (2026-08-14 → 2026-09-11). Each fold refits using only labels
  whose 3 h window closed before the fold starts (no label from the future).
- Features will be built mainly from alert **start** times; end times only when the end was before t
  (naive ends are invented after the fact).
- Kyiv DST verified with tzdata 2026e (UTC+3 summer, UTC+2 from 25 Oct 2026). UTC internally;
  Kyiv time only for calendar features.
- Model plan: logistic regression (explainable), compared to the baselines.
- Text/Telegram features: dropped (8 h total budget).

### Surprises / mistakes
- I first said the official dataset's collector was broken; the Telegram check showed the source
  channel itself stopped on 2026-09-07. Corrected.
- Windows console (cp1251) crashed printing emoji → scripts force UTF-8 output.
- Climatology by day-of-week × hour scored **worse** than a flat per-region rate (Brier 0.1756 vs 0.1685).
  Checked it was not a bug (hand-verified target; smoother variants converge to the flat rate):
  8 weeks × 168 cells = 8 samples per cell = noise. Kept hour-of-day climatology as the main reference.
- Pooled AUC (0.83) was misleading: it measured "which region is busier". Switched to mean
  within-region AUC, where all baselines are ~0.47–0.54.
- "Alert in last 3 h" raises the chance of another start in quiet regions (Vinnytsia 0.40 vs 0.17) but
  lowers it in busy frontline ones (Sumy 0.71 vs 0.84) — effects cancel when pooled. The model needs
  region-specific effects.
- Fitted baselines under-predict (mean p 0.40 vs observed 0.45): alert frequency is rising.

### Validation results (baselines only)
| forecaster | Brier ↓ | within-region AUC | BSS vs climatology |
|---|---|---|---|
| persistence_calibrated | 0.1663 | 0.508 | +0.017 |
| base_rate | 0.1685 | 0.469 | +0.004 |
| climatology (hour-of-day) | 0.1691 | 0.533 | 0 |
| climatology_how (dow × hour) | 0.1756 | 0.534 | −0.038 |
| seasonal_naive_24h | 0.2940 | 0.539 | −0.739 |
| persistence (raw 0/1) | 0.3043 | 0.530 | −0.800 |
| seasonal_naive_168h | 0.3154 | 0.508 | −0.865 |

Bar for the model: Brier < 0.1663 on validation.

## 2026-10-11 (Sun) — features, leakage test, logistic regression

### Checks on the "flipping persistence" effect (user's hypothesis: ongoing alert blocks new starts)
- Confirmed in part: in the data a new alert **never** starts while one is running in the same region
  (0.000 overlaps), and in every region P(new start | alert running at t) < P(new start | recent alert
  already ended). E.g. Sumy 0.62 vs 0.74, Kyiv oblast 0.63 vs 0.75.
- Not the whole story: in Sumy/Chernihiv the "nothing started in 3 h" state still has the highest rate.
- `naive` (invented) end times barely matter: ignoring them changes the "active" flag in 0.66% of rows.

### Features (all from information before t; `airalerts/features.py`)
starts in last 1/3/6/24/168 h (log), hours since last start, alert running at t, minutes it has run,
number of regions with a start in last 3 h / running now, Kyiv hour-of-day (sin/cos), region (one-hot).

### Leakage test (`tests/test_no_leakage.py`)
Features at t are computed (a) from the full data, (b) from only what was known at t (alerts started
before t; end times after t blanked), (c) from known data + 500 fake future alerts and fake end times.
All must be identical. The test also checks itself: two deliberately leaky features (peeking at future
starts / at a running alert's true end) must make it fail. Plus an exact-boundary test of the target window.

### Model: logistic regression, refit each fold
- Training window compared on validation: 8w 0.1600, **16w 0.1580**, 26w 0.1584, 40w 0.1591 (Brier).
  Short windows react to the trend but are noisy; the "since Dec 2025" window (40w) is slightly worse.
- Beats the best baseline (persistence_calibrated 0.1663) in all 4 folds; within-region AUC 0.61 vs ≤0.54.
- Treated as suspicious, checked (on the version that still had interactions): coefficients sensible
  (strongest = last-week activity; running alert in a busy region lowers p), calibration good (0.04→0.04, 0.45→0.47, 0.93→0.95; slight under-prediction
  mid-range due to rising trend), no single feature group dominates (each ablation costs ~0.0005–0.001).

### Failed idea
- Interactions "recent activity × weekly busyness", added to model the sign flip: validation Brier
  0.1583 with vs 0.1580 without. Removed — region intercepts + last-week activity already cover it.

### Leakage bug found by the user: invented end times of `naive` alerts
- The "alert running at t" feature used `finished_at` for `naive` alerts, but that value is invented
  later by the dataset (start + 30 min) because no end message ever arrived. A live observer at t would
  see such an alert as still running. My earlier check (0.66% of flags change) measured the size, not
  the legitimacy — it was leakage of a post-hoc value.
- The leakage test had the same blind spot: it trusted any `finished_at <= t`. Tightened `known_at()`
  to hide naive ends and the naive flag itself → test **failed** on the old code (every region looked
  "active", because naive alerts since 2022 never closed).
- Fix (real-time rule): only the region's latest start can be running; it ends when its end message
  arrives (non-naive `finished_at <= t`) or when a new alert starts in the region (alerts never overlap).
  Test passes again.
- Validation after fix: logreg_16w 0.1571 (was 0.1580); 16w still best → **MODEL_TRAIN_WEEKS = 16**.
  Score went slightly up, not down: the leak injected an invented state, not true future information.

### Plan change
Final test run today (Sun) instead of Monday; on Monday re-pin to the newest snapshot and re-run the
identical pipeline (no code changes), record both. The rule is "no decisions after seeing test", not
"run it once".
