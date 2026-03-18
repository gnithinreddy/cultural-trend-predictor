"""
News collector: fetches top headlines from NewsAPI or RSS feeds (fallback).
RSS feeds fetched in parallel. Retries on transient failures.
"""
import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import feedparser
import requests

from config.config import NEWS_API_KEY, NEWS_CATEGORIES, NEWS_PAGE_SIZE

logger = logging.getLogger(__name__)

NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
VALID_NEWS_CATEGORIES = {"business", "entertainment", "general", "health", "science", "sports", "technology"}

RSS_FEEDS = [
    "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
]
_HTML_RE = re.compile(r"<[^>]+>")


def _make_id(url: str) -> str:
    """Generate stable id from URL."""
    return "news_" + hashlib.md5(url.encode()).hexdigest()[:12]


def _collect_from_newsapi() -> list[dict[str, Any]]:
    """Fetch from NewsAPI. Returns empty list if key missing or on error."""
    if not NEWS_API_KEY:
        return []

    posts = []
    seen_urls = set()

    for category in NEWS_CATEGORIES:
        if category not in VALID_NEWS_CATEGORIES:
            continue
        data = None
        for attempt in range(3):
            try:
                resp = requests.get(
                    NEWS_API_URL,
                    params={
                        "apiKey": NEWS_API_KEY,
                        "country": "us",
                        "category": category,
                        "pageSize": min(NEWS_PAGE_SIZE, 100),
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException as e:
                logger.warning("NewsAPI %s attempt %d failed: %s", category, attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt)
            except ValueError as e:
                logger.warning("NewsAPI response invalid for %s: %s", category, e)
                break
        if data is None:
            continue

        articles = data.get("articles", [])
        for a in articles:
            url = a.get("url") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            published = a.get("publishedAt") or ""
            if published:
                try:
                    dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    created = dt.strftime("%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    created = published[:19].replace("T", " ")
            else:
                created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            posts.append({
                "id": _make_id(url),
                "source": "news",
                "title": (a.get("title") or "")[:500],
                "url": url,
                "created_at": created,
                "score": 0,
                "subreddit": "",
                "author": a.get("author") or "",
                "body": (a.get("description") or "")[:5000],
            })

    return posts


def _parse_rss_feed(feed_url: str) -> list[dict[str, Any]]:
    """Parse a single RSS feed. Returns list of post dicts. Retries up to 2 times."""
    posts = []
    for attempt in range(3):
        try:
            parsed = feedparser.parse(feed_url, request_headers={"User-Agent": "cultural_trend_predictor/1.0"})
            break
        except Exception as e:
            logger.warning("RSS feed %s attempt %d failed: %s", feed_url[:40], attempt + 1, e)
            if attempt < 2:
                time.sleep(1)
            else:
                return posts

    for entry in parsed.get("entries", [])[:50]:
        url = entry.get("link") or ""
        if not url:
            continue

        published = entry.get("published_parsed") or entry.get("updated_parsed")
        if published and len(published) >= 6:
            try:
                created = datetime(*published[:6]).strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        else:
            created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        summary = entry.get("summary", "") or entry.get("description", "")
        if hasattr(summary, "replace"):
            summary = _HTML_RE.sub("", summary)[:5000]
        summary = summary if isinstance(summary, str) else ""

        posts.append({
            "id": _make_id(url),
            "source": "news",
            "title": (entry.get("title") or "")[:500],
            "url": url,
            "created_at": created,
            "score": 0,
            "subreddit": "",
            "author": entry.get("author", "") or entry.get("source", {}).get("title", "") or "",
            "body": summary,
        })
    return posts


def _collect_from_rss() -> list[dict[str, Any]]:
    """Fetch from RSS feeds in parallel. No API key required."""
    posts_by_url = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_parse_rss_feed, url): url for url in RSS_FEEDS}
        for fut in as_completed(futures):
            for p in fut.result():
                url = p["url"]
                if url not in posts_by_url:
                    posts_by_url[url] = p
    return list(posts_by_url.values())


def collect() -> list[dict[str, Any]]:
    """
    Fetch top headlines from NewsAPI (if key set) or RSS feeds (fallback).
    Returns list of dicts with: id, source, title, url, created_at, score, subreddit, author, body
    """
    posts = _collect_from_newsapi()
    if not posts:
        logger.info("NewsAPI unavailable or empty; using RSS feeds.")
        posts = _collect_from_rss()

    logger.info("Collected %d News articles", len(posts))
    return posts
