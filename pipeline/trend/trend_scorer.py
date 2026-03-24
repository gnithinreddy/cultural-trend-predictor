"""
Trend scorer: computes trend_score (0-100) for Reddit and News.
Reddit: velocity (score/age), optionally growth from history.
News: keywords → Google Trends (pytrends).
"""
import logging
import time
from datetime import datetime, timezone
import pandas as pd

from config.config import HISTORY_DIR, PYTRENDS_DELAY_SEC, HISTORY_RETENTION_DAYS
from pipeline.trend.keyword_extractor import extract_keywords

logger = logging.getLogger(__name__)

HISTORY_FILE = HISTORY_DIR / "post_scores.csv"


def _parse_created_at(s) -> datetime | None:
    """Parse created_at to datetime."""
    if pd.isna(s) or not s:
        return None
    try:
        s = str(s).replace("T", " ").split(".")[0].split("+")[0][:19]
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def _reddit_velocity_score(score: float, created_at: str, now: datetime | None = None) -> float:
    """Velocity = score / hours_since_created. Higher = more momentum."""
    created = _parse_created_at(created_at)
    if created is None:
        return 0.0
    now = now or datetime.now(timezone.utc)
    hours = max((now - created).total_seconds() / 3600, 0.5)
    return float(score) / hours


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Normalize to 0-100."""
    if series.empty or series.max() == series.min():
        return pd.Series(50.0, index=series.index)
    return (series - series.min()) / (series.max() - series.min()) * 100


def _get_pytrends_interest(keywords: str) -> float | None:
    """Query Google Trends, return growth-based score (0-100). Retries up to 2 times."""
    if not keywords or not str(keywords).strip():
        return None
    kw = str(keywords).strip()[:50]
    for attempt in range(3):
        try:
            from pytrends.request import TrendReq
            pytrends = TrendReq()
            pytrends.build_payload([kw], timeframe="now 7-d")
            result = pytrends.interest_over_time()
            if result is None or result.empty:
                return None
            col = [c for c in result.columns if c != "isPartial"]
            if not col:
                return None
            vals = result[col[0]].dropna()
            if len(vals) < 2:
                return float(vals.iloc[-1]) if len(vals) == 1 else None
            latest = vals.iloc[-1]
            prev = vals.iloc[-4] if len(vals) >= 4 else vals.iloc[0]
            growth = (latest - prev) / max(prev, 1) if prev else 0
            score = 50 + 25 * max(-1, min(1, growth))
            return max(0.0, min(100.0, score))
        except Exception as e:
            logger.debug("pytrends '%s' attempt %d failed: %s", kw[:30], attempt + 1, e)
            if attempt < 2:
                time.sleep(PYTRENDS_DELAY_SEC * (attempt + 1))
            else:
                return None


def _load_history() -> pd.DataFrame:
    """Load history CSV if exists."""
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=["url", "date", "score"])
    try:
        return pd.read_csv(HISTORY_FILE, encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to load history: %s", e)
        return pd.DataFrame(columns=["url", "date", "score"])


def _prune_history(hist: pd.DataFrame, retention_days: int) -> pd.DataFrame:
    """Keep only last retention_days of history."""
    if hist.empty or "date" not in hist.columns:
        return hist
    try:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).strftime("%Y-%m-%d")
        pruned = hist[hist["date"] >= cutoff]
        if len(pruned) < len(hist):
            logger.info("Pruned history: removed %d rows older than %d days", len(hist) - len(pruned), retention_days)
        return pruned
    except Exception as e:
        logger.warning("History prune failed: %s", e)
        return hist


def append_refetched_to_history(refetched: dict[str, int], date_str: str) -> None:
    """Append re-fetched Reddit scores to history. refetched = {url: score}."""
    if not refetched:
        return
    rows = [{"url": url, "date": date_str, "score": score} for url, score in refetched.items()]
    new_df = pd.DataFrame(rows)
    hist = _load_history()
    combined = pd.concat([hist, new_df], ignore_index=True)
    combined = _prune_history(combined, HISTORY_RETENTION_DAYS)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(HISTORY_FILE, index=False, encoding="utf-8")
    logger.info("Appended %d refetched scores to history", len(rows))


def _append_history(df: pd.DataFrame, date_str: str) -> None:
    """Append today's (url, score) to history, then prune old rows."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in df.iterrows():
        url = row.get("url")
        score = row.get("score", 0)
        if url and pd.notna(url):
            rows.append({"url": str(url), "date": date_str, "score": int(score) if pd.notna(score) else 0})
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    hist = _load_history()
    combined = pd.concat([hist, new_df], ignore_index=True)
    combined = _prune_history(combined, HISTORY_RETENTION_DAYS)
    combined.to_csv(HISTORY_FILE, index=False, encoding="utf-8")
    logger.info("Updated history: %d new rows", len(rows))


def score_trends(df: pd.DataFrame, date_str: str, skip_pytrends: bool = False) -> pd.DataFrame:
    """
    Add trend_score (0-100) and keywords (for news) to DataFrame.
    Reddit: velocity-based, min-max normalized.
    News: keywords from title → pytrends → trend_score.
    """
    if df.empty:
        return df

    df = df.copy()
    now = datetime.now(timezone.utc)

    # Reddit: use history growth when 2+ days, else velocity
    reddit_mask = df["source"] == "reddit"
    if reddit_mask.any():
        hist = _load_history()
        rdf = df.loc[reddit_mask].copy()
        scores = []
        velocity_indices = []
        for idx, row in rdf.iterrows():
            url = row.get("url")
            score = row.get("score", 0)
            if pd.isna(url):
                scores.append(50.0)
                continue
            url_hist = hist[hist["url"] == url].sort_values("date", ascending=False)
            if len(url_hist) >= 2:
                latest = int(url_hist.iloc[0]["score"])
                prev = int(url_hist.iloc[1]["score"])
                growth = (latest - prev) / max(prev, 1) if prev else 0
                score_val = 50 + 25 * max(-1, min(1, growth))
                scores.append(max(0, min(100, score_val)))
            else:
                created = pd.to_datetime(row.get("created_at"), errors="coerce")
                if pd.notna(created):
                    if created.tzinfo is None:
                        created = created.tz_localize("UTC")
                    hours = max((now - created).total_seconds() / 3600, 0.5)
                    scores.append(float(score) / hours if pd.notna(score) else 0)
                else:
                    scores.append(50.0)
                velocity_indices.append(len(scores) - 1)
        if velocity_indices:
            vel_vals = [scores[i] for i in velocity_indices]
            norm_vel = _min_max_normalize(pd.Series(vel_vals))
            for j, i in enumerate(velocity_indices):
                scores[i] = norm_vel.iloc[j]
        df.loc[reddit_mask, "trend_score"] = scores
        df.loc[reddit_mask, "keywords"] = df.loc[reddit_mask].apply(
            lambda r: extract_keywords(str(r.get("title", "")), str(r.get("body", ""))[:200]),
            axis=1,
        )

    # News: keywords + pytrends
    news_mask = df["source"] == "news"
    if news_mask.any():
        df.loc[news_mask, "keywords"] = df.loc[news_mask].apply(
            lambda r: extract_keywords(str(r.get("title", "")), str(r.get("body", ""))[:200]),
            axis=1,
        )
        if skip_pytrends:
            df.loc[news_mask, "trend_score"] = 50.0
        else:
            scores = []
            seen_kw = {}
            for idx in df.loc[news_mask].index:
                kw = df.at[idx, "keywords"]
                if kw in seen_kw:
                    scores.append(seen_kw[kw])
                else:
                    s = _get_pytrends_interest(kw)
                    s = s if s is not None else 50.0
                    seen_kw[kw] = s
                    scores.append(s)
                    time.sleep(PYTRENDS_DELAY_SEC)
            df.loc[news_mask, "trend_score"] = scores

    # Ensure trend_score for any missing
    if "trend_score" not in df.columns:
        df["trend_score"] = 50.0
    df["trend_score"] = df["trend_score"].fillna(50.0)

    # Update history (for future multi-day growth)
    _append_history(df, date_str)

    logger.info("Trend scores computed: Reddit %d, News %d", reddit_mask.sum(), news_mask.sum())
    return df
