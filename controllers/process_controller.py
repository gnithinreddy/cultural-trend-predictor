"""
Process controller: orchestrates clean + classify, saves to processed.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

from config.config import REDDIT_RAW, NEWS_RAW, PROCESSED_DIR
from pipeline.process.cleaner import clean
from pipeline.classify.post_classifier import classify
from pipeline.trend.trend_scorer import score_trends

logger = logging.getLogger(__name__)


def _read_csv(path) -> pd.DataFrame | None:
    """Read CSV, return None on failure."""
    try:
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    except Exception as e:
        logger.warning("Failed to read %s: %s", path.name, e)
        return None


def run_process(skip_trend: bool = False) -> int:
    """
    Load raw Reddit + News (parallel), merge, clean, classify, save to processed.
    Returns 0 on success, 1 on failure.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    reddit_path = REDDIT_RAW / f"{today}.csv"
    news_path = NEWS_RAW / f"{today}.csv"
    processed_path = PROCESSED_DIR / f"{today}.csv"

    if not reddit_path.exists() or not news_path.exists():
        logger.error("Raw data not found for today. Run collection first.")
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dfs = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_reddit = ex.submit(_read_csv, reddit_path)
        f_news = ex.submit(_read_csv, news_path)
        for fut in as_completed([f_reddit, f_news]):
            df = fut.result()
            if df is not None and not df.empty:
                dfs.append(df)

    if not dfs:
        logger.error("No raw data to process.")
        return 1

    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d raw rows", len(df))

    df = clean(df)
    logger.info("Cleaned: %d rows", len(df))

    df = classify(df)
    logger.info("Classified: %d rows", len(df))

    df = score_trends(df, date_str=today, skip_pytrends=skip_trend)
    logger.info("Trend scores computed")

    df.to_csv(processed_path, index=False, encoding="utf-8")
    logger.info("Saved to %s", processed_path)
    return 0
