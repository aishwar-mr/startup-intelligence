"""Startup Intelligence & Market Analytics Platform - Streamlit app.

Four screens: Market Overview, Competitor Intelligence, AI Insights, Data Quality.
Artifacts (data/*.parquet|joblib|npz|npy) are built by the Colab notebook or
build_artifacts.py. The app degrades gracefully: it falls back to TF-IDF if the
embedding model is unavailable, and to a rule-based summary if no LLM key is set.
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.resources import (missing_artifacts, load_data, load_tfidf, load_embeddings,
                           load_validation_report)
from src.retrieval import find_competitors, SearchError
from src.saturation import market_saturation, EXPLANATION as SAT_EXPLANATION
from src.analytics import market_overview, category_table, INDICATOR_EXPLANATION
from src.whitespace import white_space
from src.insights import generate_insight
from src.sentiment import gather_mentions, analyze_sentiment, opportunity_score

st.set_page_config(page_title="Startup Intelligence & Market Analytics", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Loading data and models…")
def load_all():
    data = load_data()
    vectorizer, tfidf_matrix = load_tfidf()
    company_emb = load_embeddings()
    embed_model = None
    if company_emb is not None:
        try:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            embed_model, company_emb = None, None  # fall back to TF-IDF
    return data, vectorizer, tfidf_matrix, company_emb, embed_model


@st.cache_data(show_spinner=False)
def cached_overview(_data, n):
    return market_overview(_data), category_table(_data)


# ---------------------------------------------------------------- guard ----
missing = missing_artifacts()
if missing:
    st.title("📊 Startup Intelligence & Market Analytics Platform")
    st.error("Missing data artifacts: **" + ", ".join(missing) + "**\n\n"
             "Run the Colab notebook (or `python build_artifacts.py`) and put the files in `data/`.")
    st.stop()

data, vectorizer, tfidf_matrix, company_emb, embed_model = load_all()
report = load_validation_report()
INDUSTRIES = sorted(data["industry"].unique())
STATUSES = sorted(data["status"].unique())
REGIONS = sorted(data["macro_region"].unique())
STAGES = sorted(data["stage"].unique())
YEARS = data["batch_year"].dropna().astype(int)
YEAR_MIN, YEAR_MAX = (int(YEARS.min()), int(YEARS.max())) if len(YEARS) else (2005, 2026)

st.title("📊 Startup Intelligence & Market Analytics Platform")
st.caption(f"Searching {len(data):,} validated Y Combinator companies · "
           f"retrieval: {'semantic embeddings + TF-IDF' if company_emb is not None else 'TF-IDF'}")

with st.sidebar:
    st.header("Navigate")
    screen = st.radio("Screen", ["Market Overview", "Competitor Intelligence", "AI Insights", "Data Quality"])
    st.markdown("---")
    st.caption("Data: yc-oss public YC directory, Hacker News, GDELT. No private data. "
               "Similarity and scores are model-generated estimates, not market facts.")


def competitor_table(df):
    return st.dataframe(df, width="stretch", hide_index=True, column_config={
        "similarity": st.column_config.NumberColumn("similarity", help="Relevance estimate (0-1), not proof of competition"),
        "website": st.column_config.LinkColumn("website"),
        "yc_url": st.column_config.LinkColumn("YC page"),
    })


# ======================================================= MARKET OVERVIEW ====
if screen == "Market Overview":
    ov, cats = cached_overview(data, len(data))
    c = st.columns(4)
    c[0].metric("Startups analysed", f"{ov['total']:,}")
    c[1].metric("Active / Public", f"{ov['active_pct']}%")
    c[2].metric("Industries", ov["industries"])
    c[3].metric("Latest batch year", ov["latest_batch_year"] or "—")

    a, b = st.columns(2)
    with a:
        st.subheader("By industry")
        d = ov["by_industry"].sort_values()
        st.plotly_chart(px.bar(d, orientation="h", labels={"value": "companies", "index": ""}),
                        width="stretch")
    with b:
        st.subheader("By region")
        d = ov["by_region"].head(10).sort_values()
        st.plotly_chart(px.bar(d, orientation="h", labels={"value": "companies", "index": ""}),
                        width="stretch")

    a, b = st.columns(2)
    with a:
        st.subheader("Companies by batch year")
        st.plotly_chart(px.line(x=ov["by_year"].index, y=ov["by_year"].values,
                                markers=True, labels={"x": "batch year", "y": "companies"}), width="stretch")
    with b:
        st.subheader("By status")
        st.plotly_chart(px.bar(ov["by_status"].sort_values(), orientation="h",
                               labels={"value": "companies", "index": ""}), width="stretch")

    st.subheader("Category saturation & momentum")
    st.caption("Density = size vs the largest category · Momentum = share from the last 3 batch years · "
               "Hint is a heuristic, not advice.")
    st.dataframe(cats, width="stretch", hide_index=True)
    with st.expander("How these indicators are calculated"):
        for k, v in INDICATOR_EXPLANATION.items():
            if k == "limitations":
                st.markdown("**Limitations:** " + " ".join(v))
            else:
                st.markdown(f"**{k}** — {v}")

# ================================================ COMPETITOR INTELLIGENCE ====
elif screen == "Competitor Intelligence":
    col1, col2 = st.columns([3, 2])
    idea = col1.text_input("Describe your startup idea", "AI chatbot for customer support")
    topic = col2.text_input("Topic for live sentiment (short)", "AI customer support",
                            help="A concise phrase retrieves better news than a full sentence.")

    with st.expander("Filters"):
        f1, f2, f3 = st.columns(3)
        sel_ind = f1.multiselect("Industry", INDUSTRIES)
        sel_stat = f2.multiselect("Status", STATUSES)
        sel_reg = f3.multiselect("Region", REGIONS)
        f4, f5 = st.columns(2)
        sel_stage = f4.multiselect("Stage", STAGES)
        yr = f5.slider("Batch year", YEAR_MIN, YEAR_MAX, (YEAR_MIN, YEAR_MAX))
    k = st.slider("Competitors to show", 5, 25, 10)
    run_sentiment = st.checkbox("Also fetch live sentiment + Opportunity Score", value=False,
                                help="Calls Hacker News + GDELT; adds a few seconds.")
    go_btn = st.button("Analyze", type="primary", width="stretch")

    if go_btn:
        filters = {"industries": sel_ind, "statuses": sel_stat, "macro_regions": sel_reg,
                   "stages": sel_stage, "year_range": yr}
        try:
            results, info = find_competitors(idea, data, vectorizer, tfidf_matrix,
                                             embed_model, company_emb, k=k, filters=filters)
        except SearchError as e:
            st.warning(str(e))
            st.stop()

        if results.empty:
            st.warning("No companies match those filters. Try widening them.")
            st.stop()
        if info["low_confidence"]:
            st.info("The closest matches are weak (low similarity) — this idea may be outside the "
                    "YC dataset's coverage. Results are shown but treat them with caution.")

        sat = market_saturation(idea, data, vectorizer, tfidf_matrix)
        m = st.columns(3)
        m[0].metric("Saturation", f"{sat['saturation_score']}/100", sat["label"])
        m[1].metric("Similar companies", sat["competitors"])
        m[2].metric("Retrieval", info["method_used"])

        tabs = st.tabs(["🏢 Competitors", "📊 Saturation", "🧭 White space", "💬 Sentiment"])

        with tabs[0]:
            st.caption(f"{info['total_candidates']:,} companies matched your filters; showing the top {len(results)}. "
                       "Similarity is a relevance estimate, not proof of direct competition.")
            competitor_table(results)

        with tabs[1]:
            g = go.Figure(go.Indicator(mode="gauge+number", value=sat["saturation_score"],
                          title={"text": sat["label"]},
                          gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
                                 "steps": [{"range": [0, 30], "color": "#d9f2d9"},
                                           {"range": [30, 55], "color": "#fff2cc"},
                                           {"range": [55, 75], "color": "#ffd9b3"},
                                           {"range": [75, 100], "color": "#ffb3b3"}]}))
            g.update_layout(height=300, margin=dict(t=40, b=0))
            st.plotly_chart(g, width="stretch")
            cc = st.columns(3)
            cc[0].metric("Breadth", sat["breadth"])
            cc[1].metric("Closeness", sat["closeness"])
            cc[2].metric("Vitality", sat["vitality"])
            with st.expander("What the Saturation Score means"):
                st.markdown(f"**What it is:** {SAT_EXPLANATION['what']}")
                for key, val in SAT_EXPLANATION["inputs"].items():
                    st.markdown(f"- **{key}** — {val}")
                st.markdown(f"**Formula:** `{SAT_EXPLANATION['formula']}`  \n**Bands:** {SAT_EXPLANATION['bands']}")
                st.markdown("**Limitations:** " + " ".join(SAT_EXPLANATION["limitations"]))

        with tabs[2]:
            dom, gaps = white_space(idea, data, vectorizer, tfidf_matrix, embed_model, company_emb)
            if not dom:
                st.info("Needs the tag_list column; rebuild artifacts with the current notebook.")
            else:
                g1, g2 = st.columns(2)
                with g1:
                    st.markdown("**Crowded dimensions** — what competitors all do")
                    if dom:
                        fig = px.bar(pd.DataFrame(dom), x="competitors", y="tag", orientation="h")
                        fig.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(t=10))
                        st.plotly_chart(fig, width="stretch")
                with g2:
                    st.markdown("**White space** — common nearby, rare among competitors")
                    if gaps:
                        fig = px.bar(pd.DataFrame(gaps), x="gap", y="tag", orientation="h",
                                     color="gap", color_continuous_scale="Greens")
                        fig.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(t=10))
                        st.plotly_chart(fig, width="stretch")
                    else:
                        st.caption("No clear adjacent gaps for this idea.")

        with tabs[3]:
            if not run_sentiment:
                st.info("Tick 'Also fetch live sentiment + Opportunity Score' above and re-run.")
            else:
                with st.spinner("Fetching Hacker News + GDELT…"):
                    opp = opportunity_score(idea, data, vectorizer, tfidf_matrix, topic=topic.strip() or idea)
                st.metric("Opportunity", f"{opp['opportunity_score']}/100", opp["verdict"])
                scored = opp["mentions"]
                if scored is None or scored.empty:
                    st.warning("No scorable mentions found — try a broader topic phrase.")
                else:
                    dist = scored["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
                    st.plotly_chart(px.bar(dist, labels={"value": "mentions", "index": ""},
                                    color=dist.index, color_discrete_map={"Positive": "#2e7d32",
                                    "Neutral": "#9e9e9e", "Negative": "#c62828"},
                                    title=f"Sentiment of {opp['mentions_scored']} English mentions"), width="stretch")
                    st.dataframe(scored[["source", "sentiment", "compound", "text"]].head(15),
                                 width="stretch", hide_index=True)
                st.caption("Opportunity rises when the space is open OR the market is unhappy. Heuristic, not advice.")

# ========================================================= AI INSIGHTS ====
elif screen == "AI Insights":
    st.subheader("Evidence-grounded competitive summary")
    st.caption("The summary is built only from the retrieved companies below. If no AI key is "
               "configured, a rule-based summary is shown instead. Inferences are hypotheses, not facts.")
    idea = st.text_input("Startup idea", "AI chatbot for customer support")
    k = st.slider("Evidence records", 5, 12, 8)
    key_present = bool(os.getenv("OPENROUTER_API_KEY"))
    st.caption(f"AI service: {'OpenRouter key detected' if key_present else 'no key set — rule-based mode'}")
    if st.button("Generate summary", type="primary"):
        try:
            results, info = find_competitors(idea, data, vectorizer, tfidf_matrix,
                                             embed_model, company_emb, k=k)
        except SearchError as e:
            st.warning(str(e)); st.stop()
        sat = market_saturation(idea, data, vectorizer, tfidf_matrix)
        ws = white_space(idea, data, vectorizer, tfidf_matrix, embed_model, company_emb)
        with st.spinner("Summarising the landscape…"):
            ins = generate_insight(idea, results, sat, ws)
        tag = "AI model" if ins["source"] == "llm" else "Rule-based"
        st.success(f"Summary generated — {tag}" + (f" ({ins['model_used']})" if ins.get("model_used") else ""))
        for w in ins["warnings"]:
            st.warning(w)
        st.markdown(ins["text"])
        st.markdown("**Evidence used (every claim should trace to these):**")
        ev = pd.DataFrame(ins["evidence"])
        if not ev.empty:
            st.dataframe(ev, width="stretch", hide_index=True,
                         column_config={"website": st.column_config.LinkColumn("website")})

# ========================================================= DATA QUALITY ====
elif screen == "Data Quality":
    st.subheader("Dataset quality report")
    if not report:
        st.info("No validation report found. Rebuild artifacts to generate `data/validation_report.json`.")
        st.stop()
    c = st.columns(4)
    c[0].metric("Records received", f"{report['total_received']:,}")
    c[1].metric("Passed validation", f"{report['passed']:,}")
    c[2].metric("Excluded", report["excluded"])
    c[3].metric("Rules version", report["rules_version"])
    st.caption(f"Deterministic report · input fingerprint `{report['input_fingerprint'][:16]}…` "
               "· raw source data is preserved; cleaning never overwrites it.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Excluded by reason**")
        ex = pd.Series(report["excluded_by_reason"])
        ex = ex[ex > 0]
        st.dataframe(ex.rename("records").reset_index().rename(columns={"index": "reason"}),
                     width="stretch", hide_index=True)
        st.markdown("**Missing values by field**")
        mv = pd.Series(report["missing_by_field"]).rename("missing").reset_index().rename(columns={"index": "field"})
        st.dataframe(mv, width="stretch", hide_index=True)
    with b:
        st.markdown("**Quality flags (records kept but noted)**")
        fl = pd.Series(report["flags_by_type"])
        fl = fl[fl > 0].sort_values(ascending=False)
        st.dataframe(fl.rename("records").reset_index().rename(columns={"index": "flag"}),
                     width="stretch", hide_index=True)
        st.markdown("**Normalisation actions applied (to copies)**")
        na = pd.Series(report["normalisation_actions"])
        na = na[na > 0]
        st.dataframe(na.rename("count").reset_index().rename(columns={"index": "action"}),
                     width="stretch", hide_index=True)

    st.markdown("**Examples of flagged / excluded records**")
    ex_rows = []
    for key, items in report.get("examples", {}).items():
        for it in items[:3]:
            ex_rows.append({"category": key, "id": it["id"], "name": it["name"], "detail": it["detail"]})
    if ex_rows:
        st.dataframe(pd.DataFrame(ex_rows), width="stretch", hide_index=True)
