# cache.py — persistence layer. Turns a live fetch into a cached read.
# Job: hold a fresh copy of the price matrix on disk so the network gets
# hit at most once per trading day, and every other module reads the file.

from pathlib import Path
import datetime as dt
import pandas as pd

# Cache lives in the same data/ folder. parquet = typed, compressed, keeps the index.
CACHE_DIR = Path(__file__).parent
PRICES_PATH = CACHE_DIR / "prices.parquet"


def _is_fresh(path: Path) -> bool:
    """Fresh = file exists and already includes the last completed weekday."""
    if not path.exists():
        return False
    cached = pd.read_parquet(path)
    if cached.empty:
        return False
    last_cached = cached.index.max().date()

    # Most recent weekday (Mon–Fri). Doesn't account for market holidays —
    # worst case that triggers one extra harmless refetch.
    today = dt.date.today()
    last_weekday = today
    while last_weekday.weekday() >= 5:      # 5=Sat, 6=Sun
        last_weekday -= dt.timedelta(days=1)

    return last_cached >= last_weekday


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

