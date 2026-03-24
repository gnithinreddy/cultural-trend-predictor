"""
Sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).
Lightweight, works well for social media and news headlines.
"""
import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

_SentimentLabel = Literal["positive", "negative", "neutral"]


def _get_analyzer():
    """Lazy load VADER analyzer."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError as e:
        logger.warning("VADER not available: %s", e)
        return None


def _score_to_label(compound: float) -> _SentimentLabel:
    """Convert compound score (-1 to 1) to label. Relaxed thresholds for headlines."""
    if compound >= 0.02:
        return "positive"
    if compound <= -0.02:
        return "negative"
    return "neutral"


def add_sentiment(df: pd.DataFrame, text_col: str = "title", body_col: str = "body") -> pd.DataFrame:
    """
    Add sentiment columns to DataFrame.
    Adds: sentiment_score (compound -1 to 1), sentiment_label (positive/negative/neutral).
    """
    if df.empty:
        return df

    analyzer = _get_analyzer()
    if analyzer is None:
        df = df.copy()
        df["sentiment_score"] = 0.0
        df["sentiment_label"] = "neutral"
        return df

    scores = []
    labels = []
    for _, row in df.iterrows():
        text = str(row.get(text_col, ""))
        body = str(row.get(body_col, ""))[:200] if body_col in df.columns else ""
        combined = f"{text} {body}".strip() or "no text"
        try:
            result = analyzer.polarity_scores(combined)
            compound = result["compound"]
            scores.append(compound)
            labels.append(_score_to_label(compound))
        except Exception as e:
            logger.debug("Sentiment failed for row: %s", e)
            scores.append(0.0)
            labels.append("neutral")

    df = df.copy()
    df["sentiment_score"] = scores
    df["sentiment_label"] = labels
    return df
