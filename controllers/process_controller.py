"""
Process controller: orchestrates clean + classify, saves to processed.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

from config.config import REDDIT_RAW, NEWS_RAW, PROCESSED_DIR, POSTS_PER_BUCKET
from pipeline.process.cleaner import clean
from pipeline.classify.post_classifier import classify
from pipeline.trend.trend_scorer import score_trends, append_refetched_to_history
from pipeline.sentiment.sentiment_analyzer import add_sentiment
from pipeline.emerging.emerging_topics import save_keyword_counts_for_date
from pipeline.track.refetcher import refetch_reddit_scores
from pipeline.track.tracked_manager import (
    load_tracked,
    save_tracked,
    prune_tracked,
    add_from_curated,
    get_tracked_reddit_urls,
)

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

    # Re-fetch tracked Reddit URLs and append to history
    tracked = load_tracked()
    reddit_urls = get_tracked_reddit_urls(tracked)
    if reddit_urls:
        refetched = refetch_reddit_scores(reddit_urls)
        if refetched:
            append_refetched_to_history(refetched, today)

    df = clean(df)
    logger.info("Cleaned: %d rows", len(df))

    df = classify(df)
    logger.info("Classified: %d rows", len(df))

    df = score_trends(df, date_str=today, skip_pytrends=skip_trend)
    logger.info("Trend scores computed")

    df = add_sentiment(df)
    logger.info("Sentiment added")

    save_keyword_counts_for_date(df, today)
    logger.info("Keyword counts saved for burst detection")

    # Add curated (top N per bucket per category) to tracked, then prune
    curated = []
    for cat in df["category"].dropna().unique():
        sub = df[df["category"] == cat]
        rising = sub[sub["trend_score"] >= 70].nlargest(POSTS_PER_BUCKET, "trend_score")
        falling = sub[sub["trend_score"] <= 30].nsmallest(POSTS_PER_BUCKET, "trend_score")
        stable = sub[(sub["trend_score"] > 30) & (sub["trend_score"] < 70)]
        new = stable.sort_values("created_at", ascending=False).head(POSTS_PER_BUCKET) if "created_at" in sub.columns else stable.head(POSTS_PER_BUCKET)
        for _, row in pd.concat([rising, falling, new]).iterrows():
            # Only track Reddit URLs (we re-fetch Reddit scores only)
            if row.get("source") == "reddit":
                curated.append({"url": row["url"], "category": cat, "source": row["source"]})
    tracked = add_from_curated(tracked, curated)
    tracked = prune_tracked(tracked)
    save_tracked(tracked)

    df.to_csv(processed_path, index=False, encoding="utf-8")
    logger.info("Saved to %s", processed_path)
    return 0
