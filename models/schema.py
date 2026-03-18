"""
Data schema for Cultural Trend Predictor.
"""
REQUIRED_COLUMNS = ["id", "source", "title", "url", "created_at"]
OPTIONAL_COLUMNS = ["score", "subreddit", "author", "body", "category", "trend_score", "keywords"]
VALID_SOURCES = ["reddit", "news"]
VALID_CATEGORIES = [
    "tech", "politics", "sports", "business", "entertainment",
    "science", "environment", "health", "general",
]
