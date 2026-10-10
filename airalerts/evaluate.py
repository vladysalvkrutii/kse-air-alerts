"""Walk-forward evaluation.

Validation: VAL_FOLDS weekly folds right before TEST_START. For each fold the forecaster
is refit with train_end = fold start, then predicts every hour in the fold.
Test: one fold [TEST_START, last valid t], only when final_test=True.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config


def folds(final_test: bool = False) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of forecast-time windows, end exclusive."""
    if final_test:
        last_t = config.DATA_END - pd.Timedelta(hours=config.HORIZON_H)
        return [(config.TEST_START, last_t + pd.Timedelta(hours=1))]
    step = pd.Timedelta(days=config.FOLD_DAYS)
    starts = [config.TEST_START - step * (config.VAL_FOLDS - k) for k in range(config.VAL_FOLDS)]
    return [(s, s + step) for s in starts]


def predict_walk_forward(forecaster, starts, y, final_test: bool = False) -> pd.DataFrame:
    """Long frame: t, region, fold, p, y."""
    out = []
    for i, (f0, f1) in enumerate(folds(final_test)):
        if not final_test:
            assert f1 <= config.TEST_START, "validation fold overlaps the test window"
        times = y.loc[f0:f1 - pd.Timedelta(hours=1)].index
        p = forecaster(starts, y, f0, times)
        long = (p.stack().rename("p").to_frame()
                 .join(y.loc[times].stack().rename("y"))
                 .rename_axis(["t", "region"]).reset_index())
        long["fold"] = i
        out.append(long)
    return pd.concat(out, ignore_index=True)


def within_region_auc(pred: pd.DataFrame) -> float:
    """Mean over regions of the AUC computed inside each region. A pooled AUC would mostly
    reward knowing which regions are busier, not which hours are risky."""
    aucs = [roc_auc_score(g.y, g.p) for _, g in pred.groupby("region")
            if g.y.nunique() > 1 and g.p.nunique() > 1]
    return float(np.mean(aucs)) if aucs else float("nan")


def score(pred: pd.DataFrame) -> dict:
    pred = pred.dropna(subset=["p", "y"])
    p, yv = pred.p.to_numpy(), pred.y.to_numpy()
    return {
        "n": len(pred),
        "brier": float(np.mean((p - yv) ** 2)),
        "auc_within_region": within_region_auc(pred),
        "mean_p": float(p.mean()),
        "obs_rate": float(yv.mean()),
    }


def score_table(preds: dict[str, pd.DataFrame], reference: str = "climatology") -> pd.DataFrame:
    """Pooled scores plus Brier skill score vs the reference: 1 - brier/brier_ref (>0 = better)."""
    rows = {name: score(p) for name, p in preds.items()}
    tab = pd.DataFrame(rows).T
    tab["bss_vs_" + reference] = 1 - tab.brier / tab.loc[reference, "brier"]
    per_fold = pd.DataFrame({name: p.groupby("fold").apply(lambda d: score(d)["brier"])
                             for name, p in preds.items()}).T
    per_fold.columns = [f"brier_fold{c}" for c in per_fold.columns]
    return tab.join(per_fold).sort_values("brier")
