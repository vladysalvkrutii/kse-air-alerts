"""One command: download -> build hourly data -> walk-forward evaluation.

    python run.py                # validation folds only (safe to run any time)
    python run.py --final-test   # ALSO scores the held-out test window; run once, at the end
"""
import argparse
import sys

import pandas as pd

from airalerts import config, data, download, evaluate, features, model
from airalerts.baselines import BASELINES

sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-test", action="store_true")
    args = ap.parse_args()

    download.download()
    alerts = data.load_alerts()
    starts = data.hourly_starts(alerts)
    y = data.make_target(starts)
    print(f"[data] alerts={len(alerts):,} regions={starts.shape[1]} "
          f"hours={starts.index.min()} .. {starts.index.max()} (DATA_END={config.DATA_END})")
    print(f"[data] target y: shape={y.shape}, known={int(y.notna().sum().sum()):,}, "
          f"overall rate={y.stack().mean():.3f}")

    split = "test" if args.final_test else "validation"
    for i, (f0, f1) in enumerate(evaluate.folds(args.final_test)):
        print(f"[{split}] fold {i}: t in [{f0}, {f1})")
    print(f"[split] TEST_START={config.TEST_START}")

    preds = {name: evaluate.predict_walk_forward(fn, starts, y, args.final_test)
             for name, fn in BASELINES.items()}

    if args.final_test:
        assert config.MODEL_TRAIN_WEEKS, "choose MODEL_TRAIN_WEEKS on validation before testing"
        grid = [config.MODEL_TRAIN_WEEKS]
    else:
        grid = config.TRAIN_WEEKS_GRID
    first_fold = evaluate.folds(args.final_test)[0][0]
    feat_times = y.loc[first_fold - pd.Timedelta(weeks=max(grid)):].dropna(how="all").index
    feats = features.build_features(alerts, feat_times, list(y.columns))
    print(f"[features] {feats.shape[0]:,} rows x {feats.shape[1]} cols, "
          f"t in [{feat_times.min()}, {feat_times.max()}]")
    for w in grid:
        fn = model.make_logreg(feats, w)
        preds[fn.model_name] = evaluate.predict_walk_forward(fn, starts, y, args.final_test)
    tab = evaluate.score_table(preds)
    print(f"\n=== {split} scores (pooled over folds) ===")
    print(tab.round(4).to_string())

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / f"{split}_scores.csv"
    tab.round(5).to_csv(out)
    print(f"\n[saved] {out}")

    # calibration of the chosen model: when it says p, how often does it happen?
    name = f"logreg_{config.MODEL_TRAIN_WEEKS}w"
    if name in preds:
        p = preds[name]
        bins = pd.cut(p.p, [i / 10 for i in range(11)], include_lowest=True)
        cal = p.groupby(bins, observed=True).agg(n=("y", "size"), mean_p=("p", "mean"), observed=("y", "mean"))
        print(f"\n=== {split} calibration, {name} ===")
        print(cal.round(3).to_string())
        cal.round(4).to_csv(config.RESULTS_DIR / f"{split}_calibration.csv")


if __name__ == "__main__":
    main()
