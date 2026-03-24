"""
Emerging topics: keywords/phrases gaining traction in rising posts.
Includes: category-specific, source-specific, trend-weighted, temporal burst,
co-occurrence network, and topic diversity.
"""
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import pandas as pd

from config.config import HISTORY_DIR
try:
    from config.config import KEYWORD_HISTORY_FILE
except ImportError:
    KEYWORD_HISTORY_FILE = HISTORY_DIR / "keyword_counts.json"
from pipeline.trend.keyword_extractor import (
    extract_keywords,
    extract_entities,
    extract_phrases,
)

logger = logging.getLogger(__name__)

_EMERGING_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "news", "says", "say", "said", "year", "years", "day", "days", "new",
    "first", "one", "two", "last", "time", "people", "world", "u", "s",
    "nan", "none", "apparently", "reveals", "freaks", "prices", "underground",
    "get", "got", "gets", "make", "made", "makes", "take", "took", "takes",
    "see", "saw", "sees", "know", "knew", "knows", "think", "thought", "thinks",
    "like", "just", "also", "even", "back", "into", "than", "when", "what",
    "how", "why", "who", "which", "this", "that", "these", "those", "it",
    "week", "series", "private", "lectures",
})


def _is_valid_phrase(p: str) -> bool:
    if not p or len(p) < 4:
        return False
    if "nan" in p or "none" in p:
        return False
    words = p.split()
    if any(w in _EMERGING_STOPWORDS for w in words):
        return False
    return True


def _extract_rake_phrases(text: str, max_phrases: int = 10) -> list[str]:
    if not text or not str(text).strip():
        return []
    try:
        from rake_nltk import Rake
        r = Rake()
        r.extract_keywords_from_text(str(text)[:1000])
        phrases = r.get_ranked_phrases()[:max_phrases]
        return [p.lower().strip() for p in phrases if 2 <= len(p.split()) <= 5 and len(p) > 4]
    except Exception as e:
        logger.debug("RAKE extraction failed: %s", e)
        return []


def _extract_ngrams(text: str, min_n: int = 2, max_n: int = 3, max_phrases: int = 15) -> list[str]:
    if not text or not str(text).strip():
        return []
    words = re.sub(r"[^\w\s]", " ", str(text)).split()
    words = [w.lower() for w in words if len(w) >= 2 and w.lower() not in _EMERGING_STOPWORDS]
    result = []
    for n in range(min_n, max_n + 1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            if _is_valid_phrase(phrase):
                result.append(phrase)
            if len(result) >= max_phrases:
                return result[:max_phrases]
    return result[:max_phrases]


def _get_phrases_for_post(row: pd.Series, use_rake: bool = True) -> tuple[list[str], list[str]]:
    title = str(row.get("title", ""))
    body = str(row.get("body", ""))[:200]
    kw = row.get("keywords")
    text = f"{title} {body}".strip()

    entities = extract_entities(text)
    phrases = []

    for e in entities:
        if 2 <= len(e.split()) <= 4:
            phrases.append(e)

    for p in extract_phrases(text, 2, 4):
        if p not in phrases:
            phrases.append(p)

    if use_rake:
        for p in _extract_rake_phrases(text, 5):
            if p not in phrases:
                phrases.append(p)

    if kw and str(kw).strip() and str(kw).lower() != "nan":
        ngrams = _extract_ngrams(str(kw), 2, 3, max_phrases=8)
    else:
        kw_extracted = extract_keywords(title, body)
        ngrams = _extract_ngrams(kw_extracted or title, 2, 3, max_phrases=8)

    for ng in ngrams:
        if ng not in phrases and _is_valid_phrase(ng):
            phrases.append(ng)

    if len(phrases) < 3:
        words = re.sub(r"[^\w\s]", " ", title).split()
        for w in words:
            wl = w.lower()
            if len(wl) >= 4 and wl not in _EMERGING_STOPWORDS:
                phrases.append(wl)

    return phrases, entities


def _core_extract(
    df: pd.DataFrame,
    rising_threshold: float = 70,
    category: str | None = None,
    source: str | None = None,
    trend_weight: bool = True,
    use_rake: bool = True,
    tfidf_weight: bool = True,
    ner_boost: float = 2.0,
) -> tuple[Counter[str], Counter[str], set[str], pd.DataFrame]:
    """Core extraction: returns (rising_counts, all_counts, entity_set, subset_used)."""
    subset = df.copy()
    if category and "category" in df.columns:
        subset = subset[subset["category"] == category]
    if source and "source" in df.columns:
        subset = subset[subset["source"] == source]
    if subset.empty:
        return Counter(), Counter(), set(), subset

    rising = subset[subset["trend_score"] >= rising_threshold] if "trend_score" in subset.columns else pd.DataFrame()
    if rising.empty:
        return Counter(), Counter(), set(), subset

    rising_counts: Counter[str] = Counter()
    all_counts: Counter[str] = Counter()
    entity_set: set[str] = set()

    for _, row in rising.iterrows():
        phrases, entities = _get_phrases_for_post(row, use_rake=use_rake)
        entity_set.update(entities)
        weight = 1.0
        if trend_weight and "trend_score" in row:
            weight = max(0.5, (row["trend_score"] - rising_threshold) / 30)
        for p in phrases:
            p_clean = p.lower().strip()
            if p_clean and _is_valid_phrase(p_clean) and len(p_clean) >= 3:
                rising_counts[p_clean] += weight

    for _, row in subset.iterrows():
        phrases, _ = _get_phrases_for_post(row, use_rake=use_rake)
        for p in phrases:
            p_clean = p.lower().strip()
            if p_clean and _is_valid_phrase(p_clean) and len(p_clean) >= 3:
                all_counts[p_clean] += 1

    return rising_counts, all_counts, entity_set, subset


def _score_and_dedupe(
    rising_counts: Counter[str],
    all_counts: Counter[str],
    entity_set: set[str],
    n_rising: int,
    top_n: int,
    min_count: float,
    tfidf_weight: bool,
    ner_boost: float,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for term, rise_val in rising_counts.items():
        if rise_val < min_count:
            continue
        all_count = all_counts.get(term, 0)
        if tfidf_weight and all_count > 0:
            score = rise_val * (1.0 / (1.0 + all_count / max(n_rising, 1)))
        else:
            score = float(rise_val)
        if term in entity_set:
            score *= ner_boost
        scored.append((term, score))

    def _sort_key(item):
        term, score = item
        wc = len(term.split())
        lp = 0 if 2 <= wc <= 3 else (4 - wc)
        return (-score, lp)

    scored.sort(key=_sort_key)

    result: list[tuple[str, float]] = []
    for term, score in scored[: top_n * 3]:
        term_words = set(term.split())
        if any(
            len(e.split()) < len(term.split()) and set(e.split()).issubset(term_words)
            for e, _ in result
        ):
            continue
        result.append((term, score))
        if len(result) >= top_n:
            break
    return result[:top_n]


def get_emerging_topics(
    df: pd.DataFrame,
    rising_threshold: float = 70,
    top_n: int = 20,
    min_count: float = 1,
    category: str | None = None,
    source: str | None = None,
    trend_weight: bool = True,
    use_rake: bool = True,
    tfidf_weight: bool = True,
    ner_boost: float = 2.0,
) -> list[tuple[str, float]]:
    """Extract emerging topics. Supports category and source filters, trend weighting."""
    if df.empty or "trend_score" not in df.columns:
        return []

    rising_counts, all_counts, entity_set, subset = _core_extract(
        df, rising_threshold, category, source, trend_weight, use_rake, tfidf_weight, ner_boost
    )
    rising = subset[subset["trend_score"] >= rising_threshold] if "trend_score" in subset.columns else pd.DataFrame()
    if rising.empty:
        return []

    return _score_and_dedupe(
        rising_counts, all_counts, entity_set, len(rising), top_n, min_count, tfidf_weight, ner_boost
    )


def get_emerging_topics_by_category(
    df: pd.DataFrame,
    rising_threshold: float = 70,
    top_n_per_cat: int = 15,
    min_count: float = 0.5,
    **kwargs,
) -> dict[str, list[tuple[str, float]]]:
    """Emerging topics per category."""
    if "category" not in df.columns:
        return {}
    result = {}
    for cat in df["category"].dropna().unique():
        topics = get_emerging_topics(
            df, category=cat, top_n=top_n_per_cat, rising_threshold=rising_threshold,
            min_count=min_count, **kwargs
        )
        if topics:
            result[str(cat)] = topics
    return result


def get_emerging_topics_by_source(
    df: pd.DataFrame,
    rising_threshold: float = 70,
    top_n_per_source: int = 10,
    **kwargs,
) -> dict[str, list[tuple[str, float]]]:
    """Emerging topics per source (reddit, news)."""
    if "source" not in df.columns:
        return {}
    result = {}
    for src in ["reddit", "news"]:
        if src in df["source"].values:
            topics = get_emerging_topics(df, source=src, top_n=top_n_per_source, rising_threshold=rising_threshold, **kwargs)
            if topics:
                result[src] = topics
    return result


# --- Keyword history for temporal burst ---
def _load_keyword_history() -> dict[str, dict[str, int]]:
    """Load {date: {term: count}}."""
    if not KEYWORD_HISTORY_FILE.exists():
        return {}
    try:
        with open(KEYWORD_HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load keyword history: %s", e)
        return {}


def _save_keyword_history(hist: dict) -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    with open(KEYWORD_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)


def save_keyword_counts_for_date(df: pd.DataFrame, date_str: str, rising_threshold: float = 70) -> None:
    """Save keyword counts from rising posts for temporal burst detection."""
    rising = df[(df["trend_score"] >= rising_threshold)] if "trend_score" in df.columns else pd.DataFrame()
    if rising.empty:
        return
    counts: Counter[str] = Counter()
    for _, row in rising.iterrows():
        phrases, _ = _get_phrases_for_post(row, use_rake=True)
        for p in phrases:
            p_clean = p.lower().strip()
            if p_clean and _is_valid_phrase(p_clean):
                counts[p_clean] += 1

    hist = _load_keyword_history()
    hist[date_str] = dict(counts)
    # Prune old dates (keep 14 days)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).strftime("%Y-%m-%d")
    hist = {d: v for d, v in hist.items() if d >= cutoff}
    _save_keyword_history(hist)


def get_burst_topics(
    df: pd.DataFrame,
    date_str: str | None = None,
    baseline_days: int = 7,
    top_n: int = 15,
    min_baseline: float = 0.5,
) -> list[tuple[str, float, float]]:
    """
    Topics spiking today vs baseline. Returns [(term, today_count, pct_change), ...].
    """
    hist = _load_keyword_history()
    if not hist:
        return []

    date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_counts = hist.get(date_str, {})
    if not today_counts:
        return []

    dates = sorted(hist.keys(), reverse=True)
    baseline_dates = [d for d in dates if d != date_str][:baseline_days]
    baseline: Counter[str] = Counter()
    for d in baseline_dates:
        baseline.update(hist.get(d, {}))

    scored = []
    for term, today_val in today_counts.items():
        if today_val < 1:
            continue
        base_val = baseline.get(term, 0) + min_baseline
        pct = (today_val - base_val) / base_val * 100 if base_val > 0 else 100
        scored.append((term, today_val, pct))

    scored.sort(key=lambda x: -x[2])
    return scored[:top_n]


# --- Co-occurrence network ---
def _get_word_pairs(phrases: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for p in phrases:
        words = [w for w in p.split() if w not in _EMERGING_STOPWORDS and len(w) >= 2]
        for i in range(len(words)):
            for j in range(i + 1, min(i + 3, len(words))):
                pairs.append((words[i], words[j]))
    return pairs


def get_cooccurrence_topics(
    df: pd.DataFrame,
    rising_threshold: float = 70,
    top_n_clusters: int = 5,
    min_cluster_size: int = 2,
) -> list[list[str]]:
    """
    Topic clusters from co-occurrence network + community detection.
    Returns list of clusters, each cluster = list of related terms.
    """
    rising = df[df["trend_score"] >= rising_threshold] if "trend_score" in df.columns else pd.DataFrame()
    if rising.empty:
        return []

    try:
        import networkx as nx
        try:
            from networkx.algorithms.community import louvain_communities
        except ImportError:
            from networkx.algorithms.community.modularity_max import greedy_modularity_communities as louvain_communities
    except ImportError as e:
        logger.debug("networkx not available: %s", e)
        return []

    G = nx.Graph()
    for _, row in rising.iterrows():
        phrases, _ = _get_phrases_for_post(row, use_rake=True)
        pairs = _get_word_pairs(phrases)
        for a, b in pairs:
            if G.has_edge(a, b):
                G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
            else:
                G.add_edge(a, b, weight=1)

    if G.number_of_nodes() < 3:
        return []

    try:
        communities = list(louvain_communities(G))
    except Exception as e:
        logger.debug("Louvain failed: %s", e)
        return []

    clusters = []
    for comm in sorted(communities, key=len, reverse=True)[:top_n_clusters]:
        if len(comm) >= min_cluster_size:
            clusters.append(sorted(comm, key=lambda w: -G.degree(w) if w in G else 0)[:10])
    return clusters


# --- Topic diversity (MMR-style) ---
def _jaccard_sim(a: str, b: str) -> float:
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def apply_diversity_filter(
    topics: list[tuple[str, float]],
    lambda_param: float = 0.7,
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """
    MMR-style diversity: prefer topics that are relevant but not redundant.
    """
    if len(topics) <= top_n:
        return topics

    selected: list[tuple[str, float]] = []
    remaining = list(topics)

    while len(selected) < top_n and remaining:
        best_idx = 0
        best_mmr = -1e9
        for i, (term, score) in enumerate(remaining):
            sim_max = max((_jaccard_sim(term, s[0]) for s in selected), default=0)
            mmr = lambda_param * score - (1 - lambda_param) * sim_max
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected
