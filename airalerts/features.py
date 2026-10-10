"""Features for forecast time t, per region. Everything here must be computable from what
was known at t:
- an alert is known only if started_at < t;
- its end is known only if finished_at <= t. `finished_at` may be NaT (= "not ended yet as of now"),
  which is exactly what the leakage test feeds in.
"""
import numpy as np
import pandas as pd

from . import config

LOOKBACKS_H = [1, 3, 6, 24, 168]
SINCE_CAP_H = 168.0


def _hourly_counts(alerts: pd.DataFrame, hours: pd.DatetimeIndex, regions) -> pd.DataFrame:
    """counts[h, r] = alerts started in [h, h+1h)."""
    return (alerts.assign(h=alerts.started_at.dt.floor("h"))
                  .groupby(["h", "region"]).size()
                  .unstack(fill_value=0)
                  .reindex(index=hours, columns=regions, fill_value=0))


def _per_region_state(alerts: pd.DataFrame, times: pd.DatetimeIndex, regions):
    """For each t, region: is an alert running at t, minutes it has run, hours since last start."""
    t = times.values
    far = np.datetime64("2262-01-01")
    active, run_min, since = {}, {}, {}
    for r in regions:
        g = alerts[alerts.region == r]
        s = np.sort(g.started_at.values)
        f = np.sort(g.finished_at.fillna(pd.Timestamp(far, tz="UTC")).values)
        n_started = np.searchsorted(s, t, side="left")      # starts strictly before t
        n_ended = np.searchsorted(f, t, side="right")       # ends at or before t
        is_active = (n_started - n_ended) > 0
        if len(s):
            hrs = (t - s[np.maximum(n_started - 1, 0)]) / np.timedelta64(1, "h")
            hrs = np.where(n_started > 0, hrs, np.nan)   # no alert yet before t
        else:
            hrs = np.full(len(t), np.nan)
        active[r] = is_active.astype(float)
        run_min[r] = np.where(is_active, hrs * 60, 0.0)
        since[r] = np.nan_to_num(np.minimum(hrs, SINCE_CAP_H), nan=SINCE_CAP_H)
    mk = lambda d: pd.DataFrame(d, index=times)
    return mk(active), mk(run_min), mk(since)


def build_features(alerts: pd.DataFrame, times: pd.DatetimeIndex, regions) -> pd.DataFrame:
    """Long frame indexed by (t, region). `times` must be whole hours (UTC)."""
    alerts = alerts[alerts.started_at < times.max()]  # nothing at/after the last t is ever needed
    hours = pd.date_range(times.min() - pd.Timedelta(hours=max(LOOKBACKS_H)), times.max(),
                          freq="h", inclusive="left")
    counts = _hourly_counts(alerts, hours, regions)

    wide = {}
    for k in LOOKBACKS_H:
        # starts in [t-k, t): rolling sum over hours t-k .. t-1, i.e. shifted by one hour
        wide[f"starts_{k}h"] = counts.rolling(k, min_periods=1).sum().shift(1).reindex(times)
    active, run_min, since = _per_region_state(alerts, times, regions)
    wide["active"], wide["active_minutes"], wide["hours_since_start"] = active, run_min, since
    nat_starts_3h = wide["starts_3h"].gt(0).sum(axis=1)
    nat_active = active.sum(axis=1)

    long = pd.concat({k: v.stack() for k, v in wide.items()}, axis=1)
    long.index.names = ["t", "region"]
    tt = long.index.get_level_values("t")
    long["nat_regions_started_3h"] = nat_starts_3h.reindex(tt).values
    long["nat_regions_active"] = nat_active.reindex(tt).values

    # transforms for a linear model
    for k in LOOKBACKS_H:
        long[f"log_starts_{k}h"] = np.log1p(long.pop(f"starts_{k}h"))
    long["log_hours_since_start"] = np.log1p(long.pop("hours_since_start"))
    long["log_active_minutes"] = np.log1p(long.pop("active_minutes"))
    hour = tt.tz_convert(config.KYIV_TZ).hour
    for n in (1, 2):
        long[f"hour_sin{n}"] = np.sin(2 * np.pi * n * hour / 24)
        long[f"hour_cos{n}"] = np.cos(2 * np.pi * n * hour / 24)
    # Tried: interactions of recent activity with weekly busyness (to capture the sign flip
    # between quiet and frontline regions). Validation Brier 0.1583 with vs 0.1580 without,
    # so they were removed (see NOTES.md).
    return long
