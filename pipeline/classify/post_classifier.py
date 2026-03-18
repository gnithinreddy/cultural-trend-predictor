"""
Post classifier: zero-shot classification into categories.
Uses GPU if available. Optimized with larger batches and torch.no_grad.
"""
import logging

import pandas as pd
import torch

from models.schema import VALID_CATEGORIES

logger = logging.getLogger(__name__)

# Larger batch size on GPU for throughput; smaller on CPU to avoid OOM
_GPU_BATCH_SIZE = 64
_CPU_BATCH_SIZE = 16


def _get_device() -> str:
    """Return 'cuda' if GPU available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_pipeline():
    """Load zero-shot pipeline. Lazy load to avoid import at module level."""
    from transformers import pipeline
    device = _get_device()
    logger.info("Using device: %s", device)
    return pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        device=0 if device == "cuda" else -1,
    )


def classify(df: pd.DataFrame, text_col: str = "title", batch_size: int | None = None) -> pd.DataFrame:
    """
    Classify each row into one of VALID_CATEGORIES using zero-shot.
    Adds/overwrites 'category' column.
    """
    if df.empty:
        return df

    try:
        pipe = _load_pipeline()
    except Exception as e:
        logger.error("Failed to load classifier: %s", e)
        df = df.copy()
        df["category"] = "general"
        return df

    device = _get_device()
    batch_size = batch_size or (_GPU_BATCH_SIZE if device == "cuda" else _CPU_BATCH_SIZE)
    labels = VALID_CATEGORIES.copy()

    texts = df[text_col].fillna("").astype(str)
    texts = texts.str[:512].replace("", "no title")
    texts = texts.tolist()

    results = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            if not batch:
                continue
            try:
                out = pipe(batch, labels, multi_label=False)
                if isinstance(out, dict):
                    out = [out]
                for o in out:
                    top = (o.get("labels") or ["general"])[0] if o else "general"
                    results.append(top)
            except Exception as e:
                logger.warning("Batch classification failed: %s", e)
                results.extend(["general"] * len(batch))

    df = df.copy()
    df["category"] = results
    logger.info("Classified %d posts into categories", len(df))
    return df
