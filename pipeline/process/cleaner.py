"""
Data cleaner: deduplication, text cleaning, type coercion.
Uses vectorized pandas operations for speed.
"""
import logging
import re

import pandas as pd

from models.schema import REQUIRED_COLUMNS, OPTIONAL_COLUMNS, VALID_SOURCES

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_MAP = {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "\u200b": "", "\ufffd": ""}


def _strip_html_series(s: pd.Series) -> pd.Series:
    """Vectorized HTML strip for a Series."""
    s = s.fillna("").astype(str)
    s = s.str.replace(_HTML_TAG_RE, "", regex=True)
    for old, new in _ENTITY_MAP.items():
        s = s.str.replace(old, new, regex=False)
    return s.str.split().str.join(" ")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data: dedup by URL, strip HTML, fix encoding, coerce types.
    Returns cleaned DataFrame.
    """
    if df.empty:
        return df

    out = df.copy()

    # Deduplicate by URL
    before = len(out)
    out = out.drop_duplicates(subset=["url"], keep="first")
    if len(out) < before:
        logger.info("Removed %d duplicate rows (by URL)", before - len(out))

    # Ensure required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    # Text cleaning: title, body (vectorized)
    for col in ["title", "body"]:
        if col in out.columns:
            out[col] = _strip_html_series(out[col])

    # Type coercion (vectorized)
    if "score" in out.columns:
        out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0).astype("Int64")
    if "created_at" in out.columns:
        s = out["created_at"].fillna("").astype(str).str.strip()
        s = s.str.replace("T", " ", regex=False).str.split(".", regex=False).str[0].str.split("+", regex=False).str[0]
        out["created_at"] = s.str[:19]

    # Filter invalid sources
    if "source" in out.columns:
        invalid = ~out["source"].isin(VALID_SOURCES)
        if invalid.any():
            out = out[~invalid]
            logger.info("Dropped %d rows with invalid source", invalid.sum())

    # Fill missing optional columns
    for col in OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col != "score" else 0

    return out.reset_index(drop=True)
