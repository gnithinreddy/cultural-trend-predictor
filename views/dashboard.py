"""
Streamlit dashboard for Cultural Trend Predictor.
Dark theme, KPI cards, date filter, charts, sentiment, share of voice, CSV export.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import config.config as cfg

# Dark theme config
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_COLORS = {"primary": "#6366f1", "reddit": "#ff4500", "news": "#3b82f6", "positive": "#22c55e", "negative": "#ef4444", "neutral": "#64748b"}

# Import emerging topic functions
from pipeline.emerging.emerging_topics import get_emerging_topics
try:
    from pipeline.emerging.emerging_topics import (
        get_emerging_topics_by_category,
        get_emerging_topics_by_source,
        get_burst_topics,
        get_cooccurrence_topics,
        apply_diversity_filter,
    )
except ImportError:
    def get_emerging_topics_by_category(df, **kw): return {}
    def get_emerging_topics_by_source(df, **kw): return {}
    def get_burst_topics(df, **kw): return []
    def get_cooccurrence_topics(df, **kw): return []
    def apply_diversity_filter(x, **kw): return x

PROCESSED_DIR = cfg.PROCESSED_DIR
POSTS_PER_BUCKET = getattr(cfg, "POSTS_PER_BUCKET", 50)
EMERGING_TOPICS_COUNT = getattr(cfg, "EMERGING_TOPICS_COUNT", 20)
EMERGING_TOPICS_PER_CATEGORY = getattr(cfg, "EMERGING_TOPICS_PER_CATEGORY", 15)

st.set_page_config(page_title="Cultural Trend Predictor", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Inject dark theme CSS
_css_path = Path(__file__).resolve().parent.parent / ".streamlit" / "styles.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _render_topic_chips(topics: list) -> None:
    """Render topics as pill-style chips."""
    chips_html = "".join(
        f'<span class="topic-chip">{t.title()} <small>({int(round(s))})</small></span>'
        for t, s in topics
    )
    st.markdown(f'<div style="line-height: 2.2;">{chips_html}</div>', unsafe_allow_html=True)

RISING_THRESHOLD = 70
FALLING_THRESHOLD = 30


@st.cache_data(ttl=60)
def _load_processed(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")


def _get_processed_files() -> list[Path]:
    """List available processed files, newest first."""
    if not PROCESSED_DIR.exists():
        return []
    files = list(PROCESSED_DIR.glob("*.csv"))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


# --- Sidebar ---
files = _get_processed_files()
if not files:
    st.warning("No processed data. Run: python main.py run")
    st.stop()

file_options = {p.name: p for p in files}
with st.sidebar:
    st.header("🎛️ Controls")
    st.markdown("---")
    selected_name = st.selectbox(
        "📆 Date",
        options=list(file_options.keys()),
        index=0,
        help="Select which day's data to view",
    )
    st.markdown("---")

processed_path = file_options[selected_name]
df = _load_processed(processed_path)

with st.sidebar:
    categories = sorted(df["category"].dropna().unique().tolist())
    selected_category = st.radio(
        "🏷️ Category",
        options=["All"] + categories,
        index=0,
        help="Filter view by category",
    )
    st.markdown("---")

# Apply category filter to main view
if selected_category != "All":
    df = df[df["category"] == selected_category].copy()

# Ensure columns exist
if "trend_score" not in df.columns:
    df["trend_score"] = 50.0
if "sentiment_score" not in df.columns or "sentiment_label" not in df.columns:
    from pipeline.sentiment.sentiment_analyzer import add_sentiment
    df = add_sentiment(df)
    _sentiment_computed = True
else:
    _sentiment_computed = False

# --- Header ---
st.title("📊 Cultural Trend Predictor")
st.caption("Track cultural trends from Reddit and News")

# --- KPI Cards ---
rising_count = (df["trend_score"] >= RISING_THRESHOLD).sum()
falling_count = (df["trend_score"] <= FALLING_THRESHOLD).sum()
avg_trend = df["trend_score"].mean()
reddit_count = (df["source"] == "reddit").sum()
news_count = (df["source"] == "news").sum()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Posts", len(df))
with k2:
    st.metric("🚀 Rising", rising_count)
with k3:
    st.metric("↘️ Falling", falling_count)
with k4:
    st.metric("Avg Trend", f"{avg_trend:.1f}")
with k5:
    st.metric("Reddit / News", f"{reddit_count} / {news_count}")

# --- Main tabs ---
tab_overview, tab_emerging, tab_trend, tab_data = st.tabs(["📊 Overview", "🌱 Emerging", "📈 Trend", "📋 Data"])

# --- Tab: Overview (Charts) ---
with tab_overview:
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Trend Score Distribution")
        fig_trend = px.histogram(
            df, x="trend_score", nbins=20,
            color_discrete_sequence=[PLOTLY_COLORS["primary"]],
            labels={"trend_score": "Trend Score", "count": "Posts"},
        )
        fig_trend.update_layout(template=PLOTLY_TEMPLATE, showlegend=False, margin=dict(t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig_trend, use_container_width=True)

    with chart_col2:
        st.subheader("Sentiment Distribution")
        if _sentiment_computed:
            st.caption("Computed on-the-fly · Run `python main.py process` to save")
        if "sentiment_label" in df.columns:
            sent_counts = df["sentiment_label"].value_counts().reindex(["positive", "negative", "neutral"], fill_value=0)
            fig_pie = px.pie(
                values=sent_counts.values,
                names=sent_counts.index,
                color=sent_counts.index,
                color_discrete_map={"positive": PLOTLY_COLORS["positive"], "negative": PLOTLY_COLORS["negative"], "neutral": PLOTLY_COLORS["neutral"]},
                hole=0.4,
            )
            fig_pie.update_layout(
                template=PLOTLY_TEMPLATE,
                margin=dict(t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                showlegend=True,
                height=320,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            if "category" in df.columns:
                with st.expander("Sentiment by Category"):
                    sent_cat = df.groupby(["category", "sentiment_label"]).size().reset_index(name="count")
                    fig_bar = px.bar(
                        sent_cat, x="category", y="count", color="sentiment_label",
                        color_discrete_map={"positive": PLOTLY_COLORS["positive"], "negative": PLOTLY_COLORS["negative"], "neutral": PLOTLY_COLORS["neutral"]},
                        barmode="stack",
                        labels={"category": "Category", "count": "Posts", "sentiment_label": "Sentiment"},
                    )
                    fig_bar.update_layout(template=PLOTLY_TEMPLATE, margin=dict(t=10, b=20), xaxis_tickangle=-45, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.caption("No sentiment data")

    st.subheader("Share of Voice: Reddit vs News")
    if "category" in df.columns and "source" in df.columns:
        sov = df.groupby(["category", "source"]).size().reset_index(name="posts")
        total = sov.groupby("category")["posts"].transform("sum")
        sov["share_pct"] = (sov["posts"] / total * 100).round(1)
        fig_sov = px.bar(
            sov, x="category", y="share_pct", color="source",
            color_discrete_map={"reddit": PLOTLY_COLORS["reddit"], "news": PLOTLY_COLORS["news"]},
            barmode="group",
            labels={"category": "Category", "share_pct": "Share of Voice (%)", "source": "Source"},
        )
        fig_sov.update_layout(template=PLOTLY_TEMPLATE, margin=dict(t=20, b=20), xaxis_tickangle=-45, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320)
        st.plotly_chart(fig_sov, use_container_width=True)
    else:
        st.caption("No category or source data")

    if len(files) >= 2:
        with st.expander("Trend Over Time (multi-day)"):
            daily = []
            for f in files[:14]:
                try:
                    d = _load_processed(f)
                    daily.append({
                        "date": f.stem,
                        "posts": len(d),
                        "rising": (d["trend_score"] >= RISING_THRESHOLD).sum() if "trend_score" in d.columns else 0,
                        "falling": (d["trend_score"] <= FALLING_THRESHOLD).sum() if "trend_score" in d.columns else 0,
                    })
                except Exception:
                    pass
            if daily:
                daily_df = pd.DataFrame(daily)
                fig_time = px.line(daily_df, x="date", y=["posts", "rising", "falling"], markers=True)
                fig_time.update_layout(template=PLOTLY_TEMPLATE, legend_title="Metric", margin=dict(t=20, b=20), xaxis_tickangle=-45, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_time, use_container_width=True)

# --- Tab: Emerging Topics ---
with tab_emerging:
    st.caption("Keywords and phrases gaining traction in rising posts")
    emerging = get_emerging_topics(df, rising_threshold=RISING_THRESHOLD, top_n=EMERGING_TOPICS_COUNT)
    emerging = apply_diversity_filter(emerging, lambda_param=0.7, top_n=EMERGING_TOPICS_COUNT)
    if emerging:
        with st.expander("🏷️ All topics", expanded=True):
            _render_topic_chips(emerging)
    by_cat = get_emerging_topics_by_category(df, rising_threshold=RISING_THRESHOLD, top_n_per_cat=EMERGING_TOPICS_PER_CATEGORY)
    if by_cat:
        with st.expander("🗂️ By Category", expanded=True):
            for cat, topics in sorted(by_cat.items()):
                if topics:
                    st.markdown(f"**{cat.title()}**")
                    _render_topic_chips(topics)
                    st.markdown("")
    by_src = get_emerging_topics_by_source(df, rising_threshold=RISING_THRESHOLD, top_n_per_source=8)
    if by_src:
        with st.expander("📢 By Source", expanded=True):
            for src, topics in by_src.items():
                if topics:
                    st.markdown(f"**{src.title()}**")
                    _render_topic_chips(topics)
                    st.markdown("")
    date_str = processed_path.stem
    burst = get_burst_topics(df, date_str=date_str, top_n=10)
    if burst:
        with st.expander("💥 Burst (spiking vs baseline)"):
            burst_chips = "".join(
                f'<span class="topic-chip">{t.title()} <small>({pct:+.0f}%)</small></span>'
                for t, c, pct in burst
            )
            st.markdown(f'<div style="line-height: 2.2;">{burst_chips}</div>', unsafe_allow_html=True)
    else:
        with st.expander("💥 Burst (spiking vs baseline)"):
            st.caption("Run pipeline for 2+ days to see burst detection.")
    clusters = get_cooccurrence_topics(df, rising_threshold=RISING_THRESHOLD, top_n_clusters=5)
    if clusters:
        with st.expander("🕸️ Topic Clusters"):
            for i, cluster in enumerate(clusters, 1):
                cluster_chips = "".join(f'<span class="topic-chip">{c.title()}</span>' for c in cluster)
                st.markdown(f"**Cluster {i}**")
                st.markdown(f'<div style="line-height: 2.2;">{cluster_chips}</div>', unsafe_allow_html=True)

# --- Tab: Trend ---
with tab_trend:
    st.caption("Rising, falling, and new posts by category")
    categories = sorted(df["category"].dropna().unique().tolist())
    cat_tabs = st.tabs(["All"] + categories)
    for tab_idx, cat in enumerate(["All"] + categories):
        with cat_tabs[tab_idx]:
            subset = df[df["category"] == cat] if cat != "All" else df
            rising = subset[subset["trend_score"] >= RISING_THRESHOLD].nlargest(POSTS_PER_BUCKET, "trend_score")
            falling = subset[subset["trend_score"] <= FALLING_THRESHOLD].nsmallest(POSTS_PER_BUCKET, "trend_score")
            stable = subset[(subset["trend_score"] > FALLING_THRESHOLD) & (subset["trend_score"] < RISING_THRESHOLD)]
            if "created_at" in stable.columns and not stable.empty:
                new = stable.sort_values("created_at", ascending=False).head(POSTS_PER_BUCKET)
            else:
                new = stable.head(POSTS_PER_BUCKET)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("🚀 Rising")
                for _, row in rising.iterrows():
                    title = str(row.get("title", ""))[:80]
                    score = row.get("trend_score", 0)
                    sent = row.get("sentiment_label", "")
                    st.markdown(f"**{title}**" + ("..." if len(str(row.get("title", ""))) > 80 else ""))
                    st.caption(f"Trend: {score:.0f} · {row.get('source', '')} · {sent}")
                    st.divider()
                if rising.empty:
                    st.caption("—")
            with col2:
                st.subheader("↘️ Falling")
                for _, row in falling.iterrows():
                    title = str(row.get("title", ""))[:80]
                    score = row.get("trend_score", 0)
                    sent = row.get("sentiment_label", "")
                    st.markdown(f"**{title}**" + ("..." if len(str(row.get("title", ""))) > 80 else ""))
                    st.caption(f"Trend: {score:.0f} · {row.get('source', '')} · {sent}")
                    st.divider()
                if falling.empty:
                    st.caption("—")
            with col3:
                st.subheader("✨ New")
                for _, row in new.iterrows():
                    title = str(row.get("title", ""))[:80]
                    score = row.get("trend_score", 0)
                    sent = row.get("sentiment_label", "")
                    st.markdown(f"**{title}**" + ("..." if len(str(row.get("title", ""))) > 80 else ""))
                    st.caption(f"Trend: {score:.0f} · {row.get('source', '')} · {sent}")
                    st.divider()
                if new.empty:
                    st.caption("—")

# --- Tab: Data ---
with tab_data:
    st.caption("Filter and sort the full dataset")
    cat_list = sorted(df["category"].dropna().unique().tolist())
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cat_filter = st.selectbox("Category", ["All"] + cat_list, key="data-cat")
    with col2:
        src_filter = st.selectbox("Source", ["All", "reddit", "news"], key="data-src")
    with col3:
        sent_filter = st.selectbox("Sentiment", ["All", "positive", "negative", "neutral"], key="data-sent")
    with col4:
        sort_opts = [c for c in ["trend_score", "score", "created_at", "sentiment_score"] if c in df.columns]
        sort_by = st.selectbox("Sort by", sort_opts or ["score"], index=0, key="data-sort")
    mask = pd.Series(True, index=df.index)
    if cat_filter != "All":
        mask &= df["category"] == cat_filter
    if src_filter != "All":
        mask &= df["source"] == src_filter
    if sent_filter != "All" and "sentiment_label" in df.columns:
        mask &= df["sentiment_label"] == sent_filter
    filtered = df[mask].copy()
    asc = sort_by == "created_at"
    filtered = filtered.sort_values(by=sort_by, ascending=asc, na_position="last")
    cols = [c for c in ["title", "source", "category", "trend_score", "sentiment_label", "score", "keywords"] if c in filtered.columns]
    st.dataframe(filtered[cols].head(100), use_container_width=True, hide_index=True)
    col_dl, _ = st.columns([1, 3])
    with col_dl:
        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            csv_data,
            file_name=f"cultural_trends_{processed_path.stem}.csv",
            mime="text/csv",
            key="download-csv",
        )
