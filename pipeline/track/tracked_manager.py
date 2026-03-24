"""
Tracked manager: load/save tracked URLs, add from curated, prune by days_losing/days_stale.
"""
import json
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd

from config.config import TRACKED_FILE, HISTORY_DIR, DAYS_LOSING_REMOVE, DAYS_STALE_REMOVE

logger = logging.getLogger(__name__)

HISTORY_FILE = HISTORY_DIR / "post_scores.csv"


def load_tracked() -> dict[str, dict]:
    """Load tracked URLs. Returns {url: {category, source, added_date}}."""
    if not TRACKED_FILE.exists():
        return {}
    try:
        with open(TRACKED_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load tracked: %s", e)
        return {}


def save_tracked(tracked: dict[str, dict]) -> None:
    """Save tracked URLs."""
    TRACKED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACKED_FILE, "w", encoding="utf-8") as f:
        json.dump(tracked, f, indent=2)


def _load_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=["url", "date", "score"])
    try:
        return pd.read_csv(HISTORY_FILE, encoding="utf-8")
    except Exception:
        return pd.DataFrame(columns=["url", "date", "score"])


def _compute_days_losing_stale(hist: pd.DataFrame, url: str) -> tuple[int, int]:
    """Return (days_losing, days_stale) for url. Count consecutive days from most recent."""
    rows = hist[hist["url"] == url].sort_values("date", ascending=False)
    if len(rows) < 2:
        return 0, 0
    rows = rows.reset_index(drop=True)
    days_losing = 0
    days_stale = 0
    for i in range(len(rows) - 1):
        curr = int(rows.iloc[i]["score"])
        prev = int(rows.iloc[i + 1]["score"])
        if curr < prev:
            days_losing += 1
            days_stale = 0
        elif curr == prev or abs(curr - prev) / max(prev, 1) < 0.01:
            days_stale += 1
            days_losing = 0
        else:
            break
    return days_losing, days_stale


def prune_tracked(tracked: dict[str, dict]) -> dict[str, dict]:
    """Remove URLs that have been losing >= DAYS_LOSING or stale >= DAYS_STALE."""
    hist = _load_history()
    if hist.empty:
        return tracked

    kept = {}
    removed = 0
    for url, meta in tracked.items():
        days_losing, days_stale = _compute_days_losing_stale(hist, url)
        if days_losing >= DAYS_LOSING_REMOVE or days_stale >= DAYS_STALE_REMOVE:
            removed += 1
            continue
        kept[url] = meta
    if removed:
        logger.info("Pruned %d tracked URLs (losing/stale)", removed)
    return kept


def add_from_curated(tracked: dict[str, dict], curated: list[dict]) -> dict[str, dict]:
    """Add curated URLs to tracked. curated = [{url, category, source}]."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for c in curated:
        url = c.get("url")
        if not url:
            continue
        if url not in tracked:
            tracked[url] = {
                "category": c.get("category", ""),
                "source": c.get("source", ""),
                "added_date": today,
            }
    return tracked


def get_tracked_reddit_urls(tracked: dict[str, dict]) -> list[str]:
    """Return Reddit URLs from tracked."""
    return [u for u, m in tracked.items() if m.get("source") == "reddit"]
