"""
Streamlit dashboard for Cultural Trend Predictor.
Curated view: 5 rising, 5 falling, 5 new (stable) per category.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from config.config import PROCESSED_DIR

st.set_page_config(page_title="Cultural Trend Predictor", page_icon="📈", layout="wide")
st.title("📈 Cultural Trend Predictor")

RISING_THRESHOLD = 70
FALLING_THRESHOLD = 30
POSTS_PER_BUCKET = 5


@st.cache_data(ttl=60)
def _load_processed(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def _get_latest_processed() -> Path | None:
    if not PROCESSED_DIR.exists():
        return None
    files = list(PROCESSED_DIR.glob("*.csv"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


processed_path = _get_latest_processed()
if processed_path is None or not processed_path.exists():
    st.warning("No processed data. Run: python main.py run")
    st.stop()

df = _load_processed(processed_path)
st.metric("Total Posts", len(df))
st.caption(f"Data: {processed_path.name}")

# Ensure trend_score exists
if "trend_score" not in df.columns:
    df["trend_score"] = 50.0

# Curated view: 5 rising, 5 falling, 5 new per category
categories = sorted(df["category"].dropna().unique().tolist())
cat_tabs = st.tabs(["All"] + categories)

for tab_idx, cat in enumerate(["All"] + categories):
    with cat_tabs[tab_idx]:
        subset = df[df["category"] == cat] if cat != "All" else df

        rising = subset[subset["trend_score"] >= RISING_THRESHOLD].nlargest(POSTS_PER_BUCKET, "trend_score")
        falling = subset[subset["trend_score"] <= FALLING_THRESHOLD].nsmallest(POSTS_PER_BUCKET, "trend_score")
        stable = subset[(subset["trend_score"] > FALLING_THRESHOLD) & (subset["trend_score"] < RISING_THRESHOLD)]
        if "created_at" in stable.columns:
            new = stable.nlargest(POSTS_PER_BUCKET, "created_at")
        else:
            new = stable.head(POSTS_PER_BUCKET)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📈 Rising")
            for _, row in rising.iterrows():
                title = str(row.get("title", ""))[:80]
                score = row.get("trend_score", 0)
                st.markdown(f"**{title}**" + ("..." if len(str(row.get("title", ""))) > 80 else ""))
                st.caption(f"Trend: {score:.0f} · {row.get('source', '')}")
                st.divider()
            if rising.empty:
                st.caption("—")

        with col2:
            st.subheader("📉 Falling")
            for _, row in falling.iterrows():
                title = str(row.get("title", ""))[:80]
                score = row.get("trend_score", 0)
                st.markdown(f"**{title}**" + ("..." if len(str(row.get("title", ""))) > 80 else ""))
                st.caption(f"Trend: {score:.0f} · {row.get('source', '')}")
                st.divider()
            if falling.empty:
                st.caption("—")

        with col3:
            st.subheader("🆕 New")
            for _, row in new.iterrows():
                title = str(row.get("title", ""))[:80]
                score = row.get("trend_score", 0)
                st.markdown(f"**{title}**" + ("..." if len(str(row.get("title", ""))) > 80 else ""))
                st.caption(f"Trend: {score:.0f} · {row.get('source', '')}")
                st.divider()
            if new.empty:
                st.caption("—")

st.divider()
st.subheader("All Posts")
# Filters
col1, col2, col3 = st.columns(3)
with col1:
    cat_filter = st.selectbox("Category", ["All"] + categories)
with col2:
    src_filter = st.selectbox("Source", ["All", "reddit", "news"])
with col3:
    sort_opts = [c for c in ["trend_score", "score", "created_at"] if c in df.columns]
    sort_by = st.selectbox("Sort by", sort_opts or ["score"], index=0)

mask = pd.Series(True, index=df.index)
if cat_filter != "All":
    mask &= df["category"] == cat_filter
if src_filter != "All":
    mask &= df["source"] == src_filter
filtered = df[mask].copy()
asc = sort_by == "created_at"
filtered = filtered.sort_values(by=sort_by, ascending=asc, na_position="last")

cols = [c for c in ["title", "source", "category", "trend_score", "score", "keywords"] if c in filtered.columns]
st.dataframe(filtered[cols].head(100), use_container_width=True, hide_index=True)
