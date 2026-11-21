# User Guide

## What this app does
Type a startup idea and get: its closest existing Y Combinator competitors, how crowded the space
is, optional live sentiment, an opportunity verdict, differentiation gaps, and an optional
evidence-grounded AI summary. Everything is an explainable estimate over a public dataset.

## Running it
1. `pip install -r requirements.txt`
2. Build the data once: `python build_artifacts.py` (or run the Colab notebook and copy the four
   `data/` artifacts across). Add `--no-embeddings` for a lighter TF-IDF-only build.
3. `streamlit run app.py` → open http://localhost:8501.

## The four screens
- **Market Overview** — dataset size, industry/region/status/year distributions, and a category
  table with density, activity and momentum. Expand "How these indicators are calculated" for the
  definitions.
- **Competitor Intelligence** — enter an idea, optionally set filters (industry, status, region,
  stage, batch year), and Analyze. Tabs show ranked competitors, the Saturation gauge + its
  explanation, White space (differentiation gaps), and optional live Sentiment + Opportunity.
- **AI Insights** — generates a structured competitive summary grounded only in the retrieved
  companies, which are listed as evidence. Works in rule-based mode with no API key.
- **Data Quality** — the deterministic validation report: records received/passed/excluded, missing
  values, quality flags, normalisation actions, and example flagged records.

## Enabling AI summaries (optional)
1. Create a free key at openrouter.ai.
2. In Streamlit Cloud: **Manage app → Settings → Secrets**, add `OPENROUTER_API_KEY = "sk-or-..."`.
   Locally: `export OPENROUTER_API_KEY=sk-or-...` before `streamlit run`.
3. Optionally set `OPENROUTER_MODEL` to a specific `:free` model id.
Without a key the app shows a rule-based summary; it never fails because the AI is unavailable.

## Reading the numbers
- **Similarity** is a text-based relevance estimate (0–1), not proof of direct competition.
- **Saturation / Opportunity / category hints** are heuristics with hand-set weights — decision
  support, not market facts.
- The **AI summary** may include hypotheses; every company it names should trace to the evidence
  table, and unverified citations are flagged.
