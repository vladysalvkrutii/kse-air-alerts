"""Central configuration. All timestamps are UTC unless the name says otherwise."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"

# --- Data source: Vadimkin/ukrainian-air-raid-sirens-dataset, volunteer (eTryvoga) file.
# Pinned to a commit so every run sees the same snapshot. To refresh: update both lines.
DATA_REPO = "Vadimkin/ukrainian-air-raid-sirens-dataset"
DATA_SHA = "7ccd4367b7057a4541ce109ca09cf1f0220f6a75"
DATA_SNAPSHOT_UTC = pd.Timestamp("2026-10-10T05:01:45Z")  # commit time of DATA_SHA
DATA_FILE = "volunteer_data_en.csv"

# Luhanska oblast is under a permanent alert (not in the data); Crimea is absent entirely.
EXCLUDED_REGIONS = {"Luhanska oblast"}

KYIV_TZ = "Europe/Kyiv"

# --- Forecast task: at hour t, P(a new alert starts in the region within [t, t + H)).
HORIZON_H = 3

# Alerts still running at snapshot time are not in the file yet, so the last day is incomplete.
CENSOR_H = 24
DATA_END = DATA_SNAPSHOT_UTC.floor("h") - pd.Timedelta(hours=CENSOR_H)

# --- Evaluation windows (forecast times t). Test is touched once, at the very end.
TEST_DAYS = 28
TEST_START = DATA_END - pd.Timedelta(days=TEST_DAYS)
VAL_FOLDS = 4  # weekly walk-forward folds immediately before TEST_START
FOLD_DAYS = 7

# --- Model: training-window lengths compared on validation; the chosen one is used on test.
TRAIN_WEEKS_GRID = [8, 16, 26, 40]   # 40 weeks ~ since the Dec 2025 switch to raion-level alerts
MODEL_TRAIN_WEEKS = 16               # chosen on validation (lowest Brier), 2026-10-11
