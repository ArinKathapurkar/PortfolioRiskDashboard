import datetime as dt
import os

import pandas as pd

from backend.data import cache


def _write_prices(path, last_day, written_on):
    df = pd.DataFrame({"AAA": [1.0, 2.0]}, index=pd.to_datetime([last_day - dt.timedelta(days=1), last_day]))
    df.to_parquet(path)
    ts = dt.datetime.combine(written_on, dt.time(12)).timestamp()
    os.utime(path, (ts, ts))


def test_last_completed_weekday():
    thu, fri, sat, mon = dt.date(2026, 10, 29), dt.date(2026, 10, 30), dt.date(2026, 10, 31), dt.date(2026, 11, 2)
    assert cache._last_completed_weekday(thu) == dt.date(2026, 10, 28)
    assert cache._last_completed_weekday(sat) == fri
    assert cache._last_completed_weekday(mon) == fri


def test_missing_file_is_stale(tmp_path):
    assert not cache._is_fresh(tmp_path / "nope.parquet")


def test_fresh_when_data_reaches_yesterday(tmp_path):
    # fetch excludes today, so data through yesterday must count as fresh
    path = tmp_path / "p.parquet"
    today = dt.date(2026, 10, 29)
    _write_prices(path, last_day=dt.date(2026, 10, 28), written_on=dt.date(2026, 10, 28))
    assert cache._is_fresh(path, today=today)


def test_stale_when_data_is_old(tmp_path):
    path = tmp_path / "p.parquet"
    _write_prices(path, last_day=dt.date(2026, 10, 27), written_on=dt.date(2026, 10, 28))
    assert not cache._is_fresh(path, today=dt.date(2026, 10, 29))


def test_fresh_when_written_today_even_if_data_lags(tmp_path):
    # e.g. yesterday was a market holiday, so the data can't reach it
    path = tmp_path / "p.parquet"
    today = dt.date(2026, 10, 29)
    _write_prices(path, last_day=dt.date(2026, 10, 27), written_on=today)
    assert cache._is_fresh(path, today=today)


def test_load_or_fetch_uses_cache_then_refetches_when_forced(tmp_path):
    path = tmp_path / "p.parquet"
    calls = []

    def fetch():
        calls.append(1)
        return pd.DataFrame({"AAA": [1.0]}, index=pd.to_datetime(["2026-10-28"]))

    cache.load_or_fetch(fetch, path=path)
    cache.load_or_fetch(fetch, path=path)   # written today -> served from disk
    assert len(calls) == 1
    cache.load_or_fetch(fetch, path=path, force=True)
    assert len(calls) == 2
