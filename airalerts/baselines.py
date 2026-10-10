"""Baseline forecasters.

Every baseline has the same signature:
    fn(starts, y, train_end, times) -> wide DataFrame of P(y=1), index=times
- inputs at forecast time t may use only starts in hours < t;
- anything fitted may use only labels y[t'] whose window is closed: t' + H <= train_end.
"""
import pandas as pd

from . import config

H = config.HORIZON_H
FIT_WEEKS = 8      # how much recent history the fitted baselines look at
SHRINK_K = 4.0     # pseudo-count pulling sparse hour-of-week cells toward the hour-of-day rate


def _train_labels(y: pd.DataFrame, train_end: pd.Timestamp) -> pd.DataFrame:
    last_t = train_end - pd.Timedelta(hours=H)  # label window [t, t+H) must close by train_end
    first_t = train_end - pd.Timedelta(weeks=FIT_WEEKS)
    return y.loc[first_t:last_t]  # .loc is inclusive: last window is [train_end-H, train_end)


def _kyiv_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert(config.KYIV_TZ)
    return pd.DataFrame({"hour": local.hour, "dow": local.dayofweek}, index=index)


def recent_started(starts: pd.DataFrame, lookback_h: int) -> pd.DataFrame:
    """1 if any alert started in [t - lookback_h, t). Uses only hours strictly before t."""
    return (starts.rolling(lookback_h).sum().shift(1) > 0).astype(float)


def base_rate(starts, y, train_end, times):
    rate = _train_labels(y, train_end).mean()
    return pd.DataFrame([rate.values] * len(times), index=times, columns=y.columns)


def climatology(starts, y, train_end, times):
    """Region x Kyiv hour-of-day rate over the last FIT_WEEKS."""
    tr = _train_labels(y, train_end)
    by_hour = tr.groupby(_kyiv_calendar(tr.index).hour.values).mean()
    return by_hour.reindex(_kyiv_calendar(times).hour.values).set_axis(times)


def climatology_how(starts, y, train_end, times):
    """Region x Kyiv (day-of-week, hour) rate over the last FIT_WEEKS, shrunk toward
    the region x hour rate so 8-sample cells are not too noisy."""
    tr = _train_labels(y, train_end)
    cal = _kyiv_calendar(tr.index)
    by_hour = tr.groupby(cal.hour.values).mean()
    g = tr.groupby([cal.dow.values, cal.hour.values])
    cell_mean, cell_n = g.mean(), g.count()
    prior = by_hour.reindex(cell_mean.index.get_level_values(1)).set_axis(cell_mean.index)
    shrunk = (cell_mean * cell_n + prior * SHRINK_K) / (cell_n + SHRINK_K)
    cal_p = _kyiv_calendar(times)
    idx = pd.MultiIndex.from_arrays([cal_p.dow.values, cal_p.hour.values])
    return shrunk.reindex(idx).set_axis(times)


def persistence(starts, y, train_end, times):
    """Raw persistence: 'whatever happened in the last H hours happens in the next H'."""
    return recent_started(starts, H).reindex(times)


def persistence_calibrated(starts, y, train_end, times):
    """P(y=1 | alert started in last H hours or not), per region, learned on training labels.
    A stronger, probabilistic version of persistence."""
    tr = _train_labels(y, train_end)
    state_tr = recent_started(starts, H).reindex(tr.index)
    p1 = tr.where(state_tr == 1).mean()
    p0 = tr.where(state_tr == 0).mean()
    state = recent_started(starts, H).reindex(times)
    return state * p1 + (1 - state) * p0


def seasonal_naive_24h(starts, y, train_end, times):
    """Same window yesterday: y[t - 24h] (its window closes at t - 21h, known at t)."""
    return y.shift(24).reindex(times)


def seasonal_naive_168h(starts, y, train_end, times):
    """Same window last week: y[t - 168h]."""
    return y.shift(168).reindex(times)


BASELINES = {
    "base_rate": base_rate,
    "climatology": climatology,
    "climatology_how": climatology_how,
    "persistence": persistence,
    "persistence_calibrated": persistence_calibrated,
    "seasonal_naive_24h": seasonal_naive_24h,
    "seasonal_naive_168h": seasonal_naive_168h,
}
