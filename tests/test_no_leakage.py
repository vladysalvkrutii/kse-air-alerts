"""Features at time t must not change when we remove or scramble everything that was not
yet known at t. Runs on the real dataset (python run.py downloads it first)."""
import numpy as np
import pandas as pd
import pytest

from airalerts import config, data
from airalerts.features import build_features


@pytest.fixture(scope="module")
def alerts():
    if not (config.RAW_DIR / config.DATA_FILE).exists():
        pytest.skip("data not downloaded; run `python run.py` first")
    return data.load_alerts()


def known_at(alerts: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """What an observer at time t would have: alerts started before t, ends only if <= t."""
    k = alerts[alerts.started_at < t].copy()
    k.loc[k.finished_at > t, "finished_at"] = pd.NaT
    return k


def scrambled_future(alerts: pd.DataFrame, t: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Everything known at t, plus a block of fake future alerts and fake future end times."""
    rng = np.random.default_rng(seed)
    k = known_at(alerts, t)
    k.loc[k.finished_at.isna(), "finished_at"] = t + pd.to_timedelta(rng.integers(1, 600), "min")
    n = 500
    fake_start = t + pd.to_timedelta(rng.integers(0, 72 * 60, n), "min")
    fake = pd.DataFrame({
        "region": rng.choice(alerts.region.unique(), n),
        "started_at": fake_start,
        "finished_at": fake_start + pd.Timedelta(minutes=30),
        "naive": False,
    })
    return pd.concat([k, fake], ignore_index=True).sort_values("started_at")


# forecast times inside validation and test windows, incl. one exactly on an alert start hour
TIMES = [pd.Timestamp("2026-08-20 03:00", tz="UTC"),
         config.TEST_START,
         pd.Timestamp("2026-10-01 18:00", tz="UTC")]


@pytest.mark.parametrize("t", TIMES)
def test_features_use_only_past(alerts, t):
    regions = sorted(alerts.region.unique())
    times = pd.DatetimeIndex([t])
    full = build_features(alerts, times, regions)
    past_only = build_features(known_at(alerts, t), times, regions)
    pd.testing.assert_frame_equal(full, past_only)
    for seed in range(3):
        fake = build_features(scrambled_future(alerts, t, seed), times, regions)
        pd.testing.assert_frame_equal(full, fake)


@pytest.mark.parametrize("peek", ["future_start", "future_end"])
def test_detector_catches_a_leak(alerts, peek):
    """Sanity check of the test itself: deliberately leaky features must make it fail."""
    t = TIMES[2]
    regions = sorted(alerts.region.unique())
    times = pd.DatetimeIndex([t])

    def leaky_build(a):
        f = build_features(a, times, regions)
        if peek == "future_start":   # alerts starting in the next 3 h
            x = a[(a.started_at >= t) & (a.started_at < t + pd.Timedelta(hours=3))].groupby("region").size()
        else:                        # true end time of an alert still running at t
            x = a[(a.started_at < t)].groupby("region").finished_at.max().sub(t).dt.total_seconds()
        f["leak"] = x.reindex(f.index.get_level_values("region")).fillna(0).values
        return f

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(leaky_build(alerts), leaky_build(known_at(alerts, t)))


def test_target_window_after_t(alerts):
    """y[t] must count only starts in [t, t+H): a start at exactly t counts, one at t-1s does not."""
    r = "Kyiv City"
    t = pd.Timestamp("2026-09-01 12:00", tz="UTC")
    base = alerts[alerts.region != r]
    for start, expected in [(t - pd.Timedelta(seconds=1), 0.0), (t, 1.0),
                            (t + pd.Timedelta(hours=config.HORIZON_H) - pd.Timedelta(seconds=1), 1.0),
                            (t + pd.Timedelta(hours=config.HORIZON_H), 0.0)]:
        one = pd.DataFrame({"region": [r], "started_at": [start],
                            "finished_at": [start + pd.Timedelta(minutes=30)], "naive": [False]})
        y = data.make_target(data.hourly_starts(pd.concat([base, one])))
        assert y.loc[t, r] == expected, (start, y.loc[t, r])
