# 📊 Startup Intelligence & Market Analytics Platform

*An NLP-powered decision-support application for competitor discovery, market analysis, and
evidence-based opportunity assessment.*

Type a startup idea and get its closest Y Combinator competitors, how crowded the space is, live
community/news sentiment, a differentiation (white-space) view, and an optional evidence-grounded AI
summary — with data-quality validation and tests behind it. All outputs are explainable estimates
over a public dataset, not objective market facts.

> Live demo: _add your Streamlit Cloud URL here_ · Screenshots: `docs/` (add real captures).

## 1. Project overview
A four-screen Streamlit app (Market Overview, Competitor Intelligence, AI Insights, Data Quality)
built on ~6,300 validated YC startups, using TF-IDF + semantic embeddings for retrieval, a custom
saturation metric, multi-source sentiment, and an optional grounded LLM summary.

## 2. Business problem
Manually searching a large startup dataset to understand a competitive landscape is slow and
inconsistent, and gives no sense of how crowded or active a space is. See `docs/Business_Requirements.md`.

## 3. Proposed solution
Searchable NLP matching + structured analytics + explainable scoring + a validated data pipeline,
delivered as a reliable, documented, tested web app.

## 4. Key features
- Competitor search with similarity scores and filters (industry, status, region, stage, batch year).
- Market Overview: distributions + category density/momentum indicators.
- Market Saturation and Opportunity scores, each with an in-app explanation and limitations.
- White-space / differentiation finder.
- Data-quality validation with a deterministic report (raw data preserved).
- Optional evidence-grounded AI summary (OpenRouter) with a rule-based fallback.

## 5. Architecture & technology stack
Python · pandas · scikit-learn (TF-IDF, cosine) · sentence-transformers (MiniLM) · vaderSentiment ·
Plotly · Streamlit · pytest. Logic lives in `src/`; `app.py` is UI only; artifacts are precomputed.
See `docs/Process_Flow.md`.

## 6. Dataset description & limitations
Public `yc-oss` Y Combinator directory (company metadata only; no PII). YC-only, so absolute counts
understate real markets; text similarity can miss differently-worded competitors.

## 7. Data validation methodology
Documented checks, cleaning rules, exclusion/flag logs, and a deterministic fingerprinted report.
Raw data is never overwritten. See `docs/Data_Validation.md`.

## 8. Search & scoring methodology
Cosine similarity over TF-IDF (and MiniLM embeddings when available); saturation = weighted blend of
breadth, closeness and vitality; opportunity = openness × dissatisfaction. All explained in-app and
in `docs/`.

## 9. Testing & validation approach
`python -m pytest` (29 tests): data validation, search/ranking, filters, analytics, AI grounding and
fallback. All four screens pass a headless `AppTest` smoke test. See `docs/Test_Cases.md` and
`docs/Results_and_Validation.md`.

## 10. Screenshots / demo
Add real screenshots of the running app here (not mock-ups).

## 11. Setup & execution
```bash
pip install -r requirements.txt
python build_artifacts.py          # or copy the 4 artifacts from the Colab notebook into data/
streamlit run app.py
python -m pytest                   # run the test suite
```
Optional AI: set `OPENROUTER_API_KEY` (and optionally `OPENROUTER_MODEL`) in the environment or
Streamlit secrets. Without it, summaries are rule-based.

## 12. Known limitations & future improvements
Multilingual sentiment; broader corpus (Crunchbase/Product Hunt); learned saturation weights; trend
detection over time; approximate-nearest-neighbour search for scale. See `docs/Results_and_Validation.md`.

## Repository layout
```
Startup-Intelligence-Platform/
├── app.py                 # Streamlit UI (4 screens)
├── build_artifacts.py     # validate + build TF-IDF/embeddings/report
├── requirements.txt
├── CHANGELOG.md
├── src/                   # preprocess, validation, resources, retrieval,
│                          # saturation, analytics, whitespace, sentiment, insights
├── tests/                 # pytest suite
├── docs/                  # BA docs: requirements, user stories, process flow,
│                          # test cases, results, data validation, user guide
└── data/                  # artifacts (built) + validation logs
```
