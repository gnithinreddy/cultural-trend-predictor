"""
Configuration for Cultural Trend Predictor.
Paths and settings; credentials loaded from .env.
"""
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env
def _load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        import dotenv
        dotenv.load_dotenv(env_path)
    except ImportError:
        pass


_load_env()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REDDIT_RAW = RAW_DIR / "reddit"
NEWS_RAW = RAW_DIR / "news"
PROCESSED_DIR = DATA_DIR / "processed"
HISTORY_DIR = DATA_DIR / "history"

# Trend settings
HISTORY_RETENTION_DAYS = 30
PYTRENDS_DELAY_SEC = 2

# Credentials (from .env)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "cultural_trend_predictor/1.0")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Collection settings
REDDIT_SUBREDDITS = [
    "technology", "environment", "politics", "sports", "business",
    "science", "entertainment", "media", "news", "worldnews",
]
REDDIT_POST_LIMIT = 100
NEWS_CATEGORIES = [
    "technology", "general", "sports", "business",
    "entertainment", "science", "health",
]
NEWS_PAGE_SIZE = 100
