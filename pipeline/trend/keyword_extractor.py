"""
Keyword extractor: extracts meaningful keywords from text for Google Trends.
Uses spaCy NER + noun chunks. Falls back to simple extraction if spaCy unavailable.
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "need",
    "dare", "ought", "used", "up", "out", "over", "after", "before",
    "news", "live", "updates", "update", "breaking", "latest", "report",
})

_nlp = None


def _load_spacy():
    """Lazy load spaCy model. Returns None if unavailable (use fallback)."""
    global _nlp
    if _nlp is not None and _nlp is not False:
        return _nlp
    if _nlp is False:
        return None
    try:
        import spacy  # noqa: F401
        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except OSError:
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            import spacy  # noqa: F401
            _nlp = spacy.load("en_core_web_sm")
            return _nlp
        except Exception as e:
            logger.warning("spaCy not available: %s. Using fallback.", e)
            _nlp = False
            return None
    except Exception as e:
        logger.warning("spaCy not available: %s. Using fallback.", e)
        _nlp = False
        return None


def _extract_simple(title: str, max_words: int = 3) -> str:
    """Fallback: take longest non-stopword words."""
    words = re.sub(r"[^\w\s]", " ", title).split()
    words = [w for w in words if len(w) > 2 and w.lower() not in _STOPWORDS]
    words = sorted(set(words), key=len, reverse=True)[:max_words]
    return " ".join(words) if words else title[:50]


def extract_keywords(title: str, body: str = "", max_terms: int = 3) -> str:
    """
    Extract meaningful keywords for Google Trends query.
    Returns 2-3 word phrase. Uses spaCy NER + noun chunks if available.
    """
    text = (title or "") + " " + (body or "")[:200]
    text = text.strip()
    if not text:
        return ""

    nlp = _load_spacy()
    if nlp is None:
        return _extract_simple(title or text, max_terms)

    try:
        doc = nlp(text[:500])
    except Exception as e:
        logger.debug("spaCy parse failed: %s", e)
        return _extract_simple(title or text, max_terms)

    terms = []

    # 1. Named entities (PERSON, ORG, GPE, EVENT)
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "EVENT", "PRODUCT"):
            t = ent.text.strip()
            if len(t) > 1 and t.lower() not in _STOPWORDS:
                terms.append(t)

    # 2. Noun chunks (phrases)
    for chunk in doc.noun_chunks:
        t = chunk.text.strip()
        if 2 <= len(t.split()) <= 4 and t.lower() not in _STOPWORDS:
            if t not in terms:
                terms.append(t)

    # 3. Fallback: nouns
    if len(terms) < max_terms:
        for token in doc:
            if token.pos_ == "NOUN" and token.text not in _STOPWORDS and len(token.text) > 2:
                if token.text not in terms:
                    terms.append(token.text)

    # 4. Fallback: simple
    if not terms:
        return _extract_simple(title or text, max_terms)

    # Take top max_terms, join
    result = " ".join(terms[:max_terms])[:80]
    return result.strip()
