"""
Reddit collector: fetches hot posts from configured subreddits.
"""
import hashlib
import logging
import time
from datetime import datetime
from typing import Any

import praw

from config.config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    REDDIT_SUBREDDITS,
    REDDIT_POST_LIMIT,
)

logger = logging.getLogger(__name__)


def _make_id(url: str) -> str:
    """Generate stable id from URL."""
    return "reddit_" + hashlib.md5(url.encode()).hexdigest()[:12]


def collect() -> list[dict[str, Any]]:
    """
    Fetch hot posts from configured subreddits.
    Returns list of dicts with: id, source, title, url, created_at, score, subreddit, author, body
    """
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        logger.warning("Reddit credentials missing. Skipping Reddit collection.")
        return []

    reddit = None
    for attempt in range(3):
        try:
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT,
            )
            break
        except Exception as e:
            logger.warning("Reddit connect attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.error("Failed to connect to Reddit after 3 attempts")
                return []

    subs = "+".join(REDDIT_SUBREDDITS)
    posts = []
    seen_urls = set()

    try:
        for submission in reddit.subreddit(subs).hot(limit=REDDIT_POST_LIMIT):
            url = f"https://www.reddit.com{submission.permalink}"
            if url in seen_urls:
                continue
            seen_urls.add(url)

            created = datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d %H:%M:%S")
            body = (submission.selftext or "")[:5000] if submission.selftext else ""

            posts.append({
                "id": _make_id(url),
                "source": "reddit",
                "title": (submission.title or "")[:500],
                "url": url,
                "created_at": created,
                "score": submission.score,
                "subreddit": submission.subreddit.display_name,
                "author": str(submission.author) if submission.author else "",
                "body": body,
            })
    except Exception as e:
        logger.error("Error fetching Reddit posts: %s", e)
        return posts

    logger.info("Collected %d Reddit posts", len(posts))
    return posts
