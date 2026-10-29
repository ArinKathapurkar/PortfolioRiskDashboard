# cache.py — persistence layer. Turns a live fetch into a cached read.
# Job: hold a fresh copy of the price matrix on disk so the network gets
# hit at most once per trading day, and every other module reads the file.

from pathlib import Path
import datetime as dt
import pandas as pd

# Cache lives in the same data/ folder. parquet = typed, compressed, keeps the index.
CACHE_DIR = Path(__file__).parent
PRICES_PATH = CACHE_DIR / "prices.parquet"


def _last_completed_weekday(today: dt.date) -> dt.date:
    """Most recent weekday strictly before today. fetch_prices excludes today
    (yfinance's end date is exclusive, and today's bar is partial anyway)."""
    day = today - dt.timedelta(days=1)
    while day.weekday() >= 5:      # 5=Sat, 6=Sun
        day -= dt.timedelta(days=1)
    return day


def _is_fresh(path: Path, today: dt.date | None = None) -> bool:
    """Fresh = file was written today, or already includes the last completed weekday."""
    if not path.exists():
        return False
    today = today or dt.date.today()

    # Already fetched today: covers market holidays, where the data can never
    # reach the expected date and we'd otherwise refetch on every call.
    if dt.date.fromtimestamp(path.stat().st_mtime) == today:
        return True

    cached = pd.read_parquet(path)
    if cached.empty:
        return False
    return cached.index.max().date() >= _last_completed_weekday(today)


def load_or_fetch(fetch_fn, path: Path = PRICES_PATH, force: bool = False):
    """Return cached prices if fresh, else fetch → save → return.

    fetch_fn: a zero-arg callable that returns the price DataFrame
              (you'll pass get_prices).
    force:    set True to bypass the cache and re-download.
    """
    if not force and _is_fresh(path):
        return pd.read_parquet(path)

    df = fetch_fn()
    df.to_parquet(path)
    return df

