"""Load raw alerts and turn them into hourly per-region matrices.

Wide matrices everywhere: index = hour start (UTC), columns = region.
"""
import pandas as pd

from . import config


def load_alerts() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_DIR / config.DATA_FILE)
    for c in ["started_at", "finished_at"]:
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    df = df[~df.region.isin(config.EXCLUDED_REGIONS)]
    return df.sort_values("started_at").reset_index(drop=True)


def hourly_starts(alerts: pd.DataFrame) -> pd.DataFrame:
    """counts[h, r] = number of alerts in region r that started in [h, h+1h).

    Only starts before DATA_END are kept: later hours are treated as unknown.
    """
    a = alerts[alerts.started_at < config.DATA_END]
    hours = pd.date_range(a.started_at.min().floor("h"), config.DATA_END,
                          freq="h", inclusive="left")
    counts = (a.assign(h=a.started_at.dt.floor("h"))
               .groupby(["h", "region"]).size()
               .unstack(fill_value=0)
               .reindex(hours, fill_value=0))
    counts.index.name = "t"
    return counts


def make_target(starts: pd.DataFrame, horizon_h: int = config.HORIZON_H) -> pd.DataFrame:
    """y[t, r] = 1 if any alert starts in [t, t + horizon_h). NaN where the window
    runs past the end of known data."""
    window_sum = starts.rolling(horizon_h).sum().shift(-(horizon_h - 1))
    return (window_sum > 0).astype(float).where(window_sum.notna())
