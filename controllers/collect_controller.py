"""
Collect controller: orchestrates data collection from Reddit and News.
Called by the check pipeline when raw data for today is missing.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from config.config import REDDIT_RAW, NEWS_RAW
from pipeline.collect.reddit_collector import collect as collect_reddit
from pipeline.collect.news_collector import collect as collect_news

logger = logging.getLogger(__name__)

COLUMNS = ["id", "source", "title", "url", "created_at", "score", "subreddit", "author", "body"]


def run_collect() -> int:
    """
    Collect raw data from all sources (Reddit, News) in parallel.
    Saves to data/raw/reddit/YYYY-MM-DD.csv and data/raw/news/YYYY-MM-DD.csv
    Returns 0 on success, 1 on failure.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    REDDIT_RAW.mkdir(parents=True, exist_ok=True)
    NEWS_RAW.mkdir(parents=True, exist_ok=True)

    reddit_path = REDDIT_RAW / f"{today}.csv"
    news_path = NEWS_RAW / f"{today}.csv"

    reddit_posts, news_posts = [], []
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_reddit = ex.submit(collect_reddit)
        f_news = ex.submit(collect_news)
        reddit_posts = f_reddit.result()
        news_posts = f_news.result()

    if reddit_posts:
        df_reddit = pd.DataFrame(reddit_posts)[COLUMNS]
        df_reddit.to_csv(reddit_path, index=False, encoding="utf-8")
        logger.info("Saved %d Reddit posts to %s", len(reddit_posts), reddit_path.name)
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(reddit_path, index=False, encoding="utf-8")
        logger.warning("No Reddit posts collected; created empty file")

    if news_posts:
        df_news = pd.DataFrame(news_posts)[COLUMNS]
        df_news.to_csv(news_path, index=False, encoding="utf-8")
        logger.info("Saved %d News articles to %s", len(news_posts), news_path.name)
    else:
        pd.DataFrame(columns=COLUMNS).to_csv(news_path, index=False, encoding="utf-8")
        logger.warning("No News articles collected; created empty file")

    if not reddit_posts and not news_posts:
        logger.error("Collection failed: no data from any source.")
        return 1

    logger.info("Collection complete: %s, %s", reddit_path.name, news_path.name)
    return 0
