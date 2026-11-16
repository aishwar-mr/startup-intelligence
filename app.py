"""Startup Intelligence & Market Research Platform - Streamlit UI.

Type a startup idea -> competitors, market-saturation score, community/news
sentiment, and a fused Opportunity Score with a verdict.

Run locally:  streamlit run app.py
Artifacts (data/*.parquet|joblib|npz|npy) are built in the Colab notebook.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.resources import missing_artifacts, load_data, load_tfidf, load_embeddings
from src.retrieval import find_competitors_tfidf, find_competitors_emb
from src.saturation import market_saturation
from src.sentiment import gather_mentions, analyze_sentiment, opportunity_score
from src.whitespace import white_space

st.set_page_config(page_title="Startup Intelligence", page_icon="🚀", layout="wide")


@st.cache_resource(show_spinner="Loading models and data...")
def load_all():
    data = load_data()
    vectorizer, tfidf_matrix = load_tfidf()
    company_emb = load_embeddings()
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return data, vectorizer, tfidf_matrix, company_emb, embed_model


# ---- guard: artifacts present? ----
missing = missing_artifacts()
if missing:
    st.title("🚀 Startup Intelligence & Market Research Platform")
    st.error(
        "Missing data artifacts: **" + ", ".join(missing) + "**\n\n"
        "Run the Colab notebook's save cell and drop these four files into the `data/` folder:\n"
        "`yc_companies.parquet`, `tfidf_vectorizer.joblib`, `tfidf_matrix.npz`, `company_embeddings.npy`."
    )
    st.stop()

data, vectorizer, tfidf_matrix, company_emb, embed_model = load_all()

# ---- header ----
st.title("🚀 Startup Intelligence & Market Research Platform")
st.caption(f"Searching {len(data):,} Y Combinator companies - *has my idea been done, and is the space worth entering?*")

# ---- sidebar ----
with st.sidebar:
    st.header("Settings")
    method = st.radio("Retrieval method", ["Embeddings (semantic)", "TF-IDF (literal)"])
    k = st.slider("Competitors to show", 5, 25, 10)
    run_sentiment = st.checkbox("Run sentiment + Opportunity Score", value=True,
                                help="Fetches live Hacker News + GDELT news. Takes a few seconds.")
    st.markdown("---")
    st.caption("Data: yc-oss public YC directory, Hacker News, GDELT. No private data.")

# ---- input ----
col1, col2 = st.columns([3, 2])
with col1:
    idea = st.text_input("Describe your startup idea", "AI chatbot for customer support")
with col2:
    topic = st.text_input("Topic for news/sentiment (short)", "AI customer support",
                          help="A concise phrase retrieves far better news than a full sentence.")

go_btn = st.button("Analyze", type="primary", width="stretch")

if go_btn and idea.strip():
    # ---------- competitors ----------
    if method.startswith("Embeddings"):
        comp = find_competitors_emb(idea, data, embed_model, company_emb, k=k)
    else:
        comp = find_competitors_tfidf(idea, data, vectorizer, tfidf_matrix, k=k)

    sat = market_saturation(idea, data, vectorizer, tfidf_matrix)

    # ---------- top metrics ----------
    m1, m2, m3 = st.columns(3)
    m1.metric("Saturation", f"{sat['saturation_score']}/100", sat["label"])
    m2.metric("Competitors found", sat["competitors"])

    opp = None
    if run_sentiment:
        with st.spinner("Gathering community + global news sentiment..."):
            opp = opportunity_score(idea, data, vectorizer, tfidf_matrix, topic=topic.strip() or idea)
        m3.metric("Opportunity", f"{opp['opportunity_score']}/100")
    else:
        m3.metric("Opportunity", "- ", "enable in sidebar")

    # ---------- tabs ----------
    tabs = st.tabs(["🏢 Competitors", "📊 Saturation", "💬 Sentiment", "🎯 Opportunity", "🧭 White space"])

    with tabs[0]:
        st.subheader("Most similar existing startups")
        st.dataframe(
            comp,
            width="stretch", hide_index=True,
            column_config={"website": st.column_config.LinkColumn("website")},
        )

    with tabs[1]:
        st.subheader("How crowded is this space?")
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=sat["saturation_score"],
            title={"text": sat["label"]},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333333"},
                   "steps": [{"range": [0, 30], "color": "#d9f2d9"},
                             {"range": [30, 55], "color": "#fff2cc"},
                             {"range": [55, 75], "color": "#ffd9b3"},
                             {"range": [75, 100], "color": "#ffb3b3"}]}))
        gauge.update_layout(height=320, margin=dict(t=40, b=0))
        st.plotly_chart(gauge, width="stretch")
        c = st.columns(3)
        c[0].metric("Breadth", sat["breadth"])
        c[1].metric("Closeness", sat["closeness"])
        c[2].metric("Vitality", sat["vitality"])
        st.caption("score = 100 x (0.5·breadth + 0.3·closeness + 0.2·vitality)")

    with tabs[2]:
        if not run_sentiment:
            st.info("Enable 'Run sentiment' in the sidebar.")
        elif opp is None or opp["mentions"] is None or opp["mentions"].empty:
            st.warning("No sentiment-scorable mentions found for this topic. Try a broader topic phrase.")
        else:
            scored = opp["mentions"]
            dist = scored["sentiment"].value_counts().reindex(
                ["Positive", "Neutral", "Negative"]).fillna(0)
            fig = px.bar(dist, labels={"value": "mentions", "index": ""},
                         color=dist.index,
                         color_discrete_map={"Positive": "#2e7d32", "Neutral": "#9e9e9e", "Negative": "#c62828"},
                         title=f"Sentiment of {opp['mentions_scored']} English mentions")
            st.plotly_chart(fig, width="stretch")

            mentions_all = gather_mentions(topic.strip() or idea)
            gd = mentions_all[mentions_all["source"] == "GDELT News"]
            if not gd.empty and gd["country"].str.strip().ne("").any():
                cov = gd["country"].replace("", "Unknown").value_counts().head(12)
                figc = px.bar(cov, orientation="h", labels={"value": "articles", "index": ""},
                              title="GDELT global coverage - articles by country")
                figc.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(figc, width="stretch")

            st.dataframe(scored[["source", "sentiment", "compound", "text"]].head(20),
                         width="stretch", hide_index=True)

    with tabs[3]:
        if not run_sentiment or opp is None:
            st.info("Enable 'Run sentiment' in the sidebar to compute the Opportunity Score.")
        else:
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=opp["opportunity_score"],
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1565c0"}}))
            g.update_layout(height=320, margin=dict(t=20, b=0))
            st.plotly_chart(g, width="stretch")
            st.subheader(opp["verdict"])
            st.caption(
                f"Saturation {opp['saturation']}/100 · avg sentiment {opp['avg_sentiment']} "
                f"over {opp['mentions_scored']} mentions. "
                "Opportunity rises when the space is open OR the market is unhappy."
            )

    with tabs[4]:
        st.subheader("Where are the gaps?")
        dominant, gaps = white_space(idea, data, embed_model, company_emb)
        if dominant is None:
            st.info("This needs the `tag_list` column. Rebuild artifacts with the final notebook or build_artifacts.py.")
        else:
            cga, cgb = st.columns(2)
            with cga:
                st.markdown("**Crowded dimensions** — what the direct competitors all do")
                if dominant:
                    dd = pd.DataFrame(dominant)
                    fig = px.bar(dd, x="competitors", y="tag", orientation="h")
                    fig.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(t=10))
                    st.plotly_chart(fig, width="stretch")
            with cgb:
                st.markdown("**White space** — common nearby, rare among direct competitors")
                if gaps:
                    gg = pd.DataFrame(gaps)
                    fig = px.bar(gg, x="gap", y="tag", orientation="h", color="gap",
                                 color_continuous_scale="Greens")
                    fig.update_layout(yaxis=dict(autorange="reversed"), height=300, margin=dict(t=10))
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.caption("No clear adjacent gaps found for this idea.")

elif go_btn:
    st.warning("Enter an idea first.")
