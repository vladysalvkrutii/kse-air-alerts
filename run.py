"""One command: download -> build hourly data -> walk-forward evaluation.

    python run.py                # validation folds only (safe to run any time)
    python run.py --final-test   # ALSO scores the held-out test window; run once, at the end
"""
import argparse
import sys

import pandas as pd

from airalerts import config, data, download, evaluate
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
    tab = evaluate.score_table(preds)
    print(f"\n=== {split} scores (pooled over folds) ===")
    print(tab.round(4).to_string())

    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / f"{split}_scores.csv"
    tab.round(5).to_csv(out)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
