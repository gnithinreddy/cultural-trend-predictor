# Cultural Trend Predictor

MVC app that collects Reddit + News, classifies content, scores trends, and displays curated posts per category.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with Reddit credentials (required) and optional NewsAPI key
```

For spaCy (better keyword extraction for News):
```bash
python -m spacy download en_core_web_sm
```

## Launch

```bash
python main.py              # Default: check pipeline + dashboard
python main.py run          # Same
python main.py run --skip-trend   # Skip pytrends (faster)
python main.py collect      # Collect raw data only
python main.py process       # Process (clean + classify + trend)
python main.py process --skip-trend   # Skip trend scoring (faster)
```

## Pipeline

1. **Collect** – Reddit (praw) + News (NewsAPI or RSS fallback), parallel
2. **Clean** – Dedup, strip HTML, coerce types
3. **Classify** – Zero-shot into 9 categories (tech, politics, sports, etc.)
4. **Trend** – Reddit: velocity (score/age). News: keywords → Google Trends
5. **Dashboard** – Filters, sort, curated view (5 rising/falling/new per category)

## Structure

```
config/         Configuration, paths, credentials
models/         Schema, valid categories
pipeline/
  collect/      Reddit, News collectors
  process/      Cleaner
  classify/     Zero-shot classifier
  trend/        Keyword extractor, trend scorer
controllers/    Collect, process orchestration
views/          Streamlit dashboard
data/
  raw/          Collected CSVs
  processed/    Cleaned + classified + trend scores
  history/      Post scores over time (for future growth-based trend)
```

## Requirements

- Python 3.10+
- Reddit API credentials (create app at reddit.com/prefs/apps)
- Optional: NewsAPI key, GPU for faster classification
