"""
Check pipeline: validates data readiness before each step.
When launching the app, main calls this first.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

from config.config import REDDIT_RAW, NEWS_RAW, PROCESSED_DIR

logger = logging.getLogger(__name__)


def _raw_paths_for_today() -> tuple[Path, Path]:
    """Return paths for today's raw Reddit and News CSVs."""
    today = _today()
    return (
        REDDIT_RAW / f"{today}.csv",
        NEWS_RAW / f"{today}.csv",
    )


def _processed_path_for_today() -> Path:
    """Return path for today's processed CSV."""
    today = _today()
    return PROCESSED_DIR / f"{today}.csv"


def _file_has_data(path: Path) -> bool:
    """Check if CSV has at least one data row (beyond header)."""
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        return len(lines) > 1
    except (OSError, UnicodeDecodeError):
        return False


def raw_data_exists_for_today() -> bool:
    """Check if raw data was collected for today (both files exist AND have data)."""
    reddit_path, news_path = _raw_paths_for_today()
    return _file_has_data(reddit_path) and _file_has_data(news_path)


def _raw_newer_than_processed() -> bool:
    """True if raw files are newer than processed (need to re-process)."""
    reddit_path, news_path = _raw_paths_for_today()
    processed_path = _processed_path_for_today()
    if not processed_path.exists():
        return True
    proc_mtime = processed_path.stat().st_mtime
    if reddit_path.exists() and reddit_path.stat().st_mtime > proc_mtime:
        return True
    if news_path.exists() and news_path.stat().st_mtime > proc_mtime:
        return True
    return False


def processed_data_ready_for_today() -> bool:
    """Processed is ready only if it exists AND raw hasn't been updated since."""
    path = _processed_path_for_today()
    if not path.exists():
        return False
    if _raw_newer_than_processed():
        return False
    return True


def ensure_raw_data_ready() -> None:
    """
    Check: did we collect raw data for today (files with actual rows)?
    - If yes: skip collection.
    - If no: run collection.
    """
    if raw_data_exists_for_today():
        logger.info("Raw data for today already exists. Skipping collection.")
        return

    logger.info("Raw data for today not found or empty. Running collection...")
    from controllers.collect_controller import run_collect
    run_collect()


def ensure_processed_data_ready(skip_trend: bool = False) -> bool:
    """
    Check: if raw exists with data, ensure processed is ready.
    - Processed ready (exists and not stale): skip.
    - Processed missing or stale: run clean + classify + trend.
    - Raw missing: return False (cannot process).
    skip_trend: if True, skip pytrends (faster).
    Returns True if processed is ready, False otherwise.
    """
    if not raw_data_exists_for_today():
        logger.warning("Raw data missing or empty. Cannot process. Run collection first.")
        return False

    if processed_data_ready_for_today():
        logger.info("Processed data for today already exists and is up to date. Skipping process.")
        return True

    logger.info("Processed data missing or stale (raw was updated). Running clean + classify + trend...")
    from controllers.process_controller import run_process
    ok = run_process(skip_trend=skip_trend) == 0
    if not ok:
        logger.error("Process failed.")
    return ok
