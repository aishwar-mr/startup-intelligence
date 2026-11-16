# 🚀 Startup Intelligence & Market Research Platform

Type a startup idea in plain English and get: the most similar existing startups,
a **market-saturation score**, multi-source **sentiment**, and a fused **Opportunity Score**
with an actionable verdict.

Built for a Web Data Mining lab project. Techniques used: information retrieval
(inverted index, TF-IDF, cosine similarity), semantic search (sentence embeddings),
a custom saturation metric, and lexicon-based sentiment analysis.

---

## What it does

| Module | Technique | Data source |
|--------|-----------|-------------|
| Competitor finder | TF-IDF + cosine **and** MiniLM embeddings | YC directory (yc-oss) |
| Saturation score | custom metric (breadth · closeness · vitality) | YC directory |
| Sentiment | VADER on Hacker News + GDELT (+ optional Product Hunt) | public APIs |
| Opportunity Score | saturation × sentiment fusion + 2×2 verdict | derived |

All data is public company metadata and public discussion/news — no PII.

---

## Project layout

```
startup-intelligence/
├── app.py                 # Streamlit UI
├── build_artifacts.py     # regenerate data/ locally (alternative to Colab)
├── requirements.txt
├── src/
│   ├── preprocess.py      # text cleaning
│   ├── resources.py       # load cached artifacts
│   ├── retrieval.py       # TF-IDF + embedding competitor finders
│   ├── saturation.py      # market saturation score
│   └── sentiment.py       # HN/GDELT/PH fetchers + VADER + Opportunity Score
└── data/                  # the 4 artifacts go here (see data/README.md)
```

---

## 1. Get the data artifacts

**Option A — from Colab (recommended):** run the notebook, then download
`yc_companies.parquet`, `tfidf_vectorizer.joblib`, `tfidf_matrix.npz`,
`company_embeddings.npy` from the Files panel and place them in `data/`.

**Option B — locally:** `python build_artifacts.py` (fetches YC + rebuilds everything).

## 2. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, type an idea, click **Analyze**.

## 3. Push to GitHub

```bash
git init
git add .
git commit -m "Startup Intelligence platform"
git branch -M main
git remote add origin https://github.com/<your-username>/startup-intelligence.git
git push -u origin main
```

> The `.npy` artifact is ~10 MB — well under GitHub's 100 MB limit, so plain git is fine (no LFS needed).

## 4. Deploy on Streamlit Community Cloud (free)

1. Go to **share.streamlit.io** → sign in with GitHub.
2. **New app** → pick your repo, branch `main`, main file `app.py`.
3. Deploy. First build takes a few minutes (it installs torch + downloads MiniLM).
4. You get a public `https://<app>.streamlit.app` URL for the viva demo.

**Optional — Product Hunt:** in the app's **Settings → Secrets**, add
`PRODUCTHUNT_TOKEN = "your_token"` to enable that source.

---

## Notes & limitations (good viva talking points)

- **English-only sentiment:** VADER scores English text; GDELT's non-English
  articles are used for global *coverage* but not sentiment. Multilingual
  sentiment is a clear future extension.
- **GDELT throttling:** rapid repeat calls return an empty body, so `fetch_gdelt`
  retries with backoff and results are cached per topic.
- **Saturation constants** (`TAU`, `BREADTH_REF`, `CLOSE_REF`) and the Opportunity
  weights are tunable design choices in `src/saturation.py` and `src/sentiment.py`.
- **Corpus bias:** the company set is Y Combinator only — a strong sample, but not
  the whole startup world.
