"""
Refetcher: re-fetch Reddit post scores by URL for tracked posts.
"""
import logging
import re
import time
import praw

from config.config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

logger = logging.getLogger(__name__)

_REDDIT_ID_RE = re.compile(r"/comments/([a-z0-9]+)/", re.I)


def _extract_reddit_id(url: str) -> str | None:
    """Extract submission ID from Reddit URL."""
    m = _REDDIT_ID_RE.search(url)
    return m.group(1) if m else None


def refetch_reddit_scores(urls: list[str]) -> dict[str, int]:
    """
    Re-fetch current scores for Reddit URLs.
    Returns {url: score}. Skips non-Reddit URLs and failures.
    """
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        return {}

    reddit_urls = [u for u in urls if "reddit.com" in u]
    if not reddit_urls:
        return {}

    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
    except Exception as e:
        logger.warning("Refetcher: Reddit connect failed: %s", e)
        return {}

    result = {}
    for url in reddit_urls:
        try:
            sid = _extract_reddit_id(url)
            if not sid:
                continue
            sub = reddit.submission(id=sid)
            result[url] = sub.score
            time.sleep(0.5)
        except Exception as e:
            logger.debug("Refetch failed for %s: %s", url[:50], e)
    logger.info("Refetched %d Reddit scores", len(result))
    return result
