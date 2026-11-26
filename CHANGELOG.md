# Changelog

## v2.1
Added
- README screenshots (`docs/screenshots/`) and expanded setup and scoring notes.

Fixed
- AI Insights: replaced retired free models, added live free-model discovery, and stopped showing
  cut-off summaries.
- Market Overview: "Active / Public" now includes Public companies.
- Docs: refreshed figures to match the bundled data.

## v2.0 — Market Analytics upgrade
Upgraded the Startup Intelligence project per the upgrade specification.

Added
- **Data validation** (`src/validation.py`): documented checks (duplicates, missing values, types,
  placeholders), exclusion/flag logs, geography parsing, and a deterministic fingerprinted report.
  Raw data is preserved.
- **Market Overview analytics** (`src/analytics.py`): industry/region/status/year distributions and
  a category density/activity/momentum table with explained heuristic hints.
- **Search filters & robustness** (`src/retrieval.py`): industry/status/region/stage/year filters,
  empty/short-query handling, low-confidence detection, deterministic ranking.
- **Score explanations**: inputs, formula, range and limitations surfaced for Saturation and the
  category indicators.
- **AI Insights** (`src/insights.py`): evidence-grounded summaries via OpenRouter with a grounding
  check, plus a rule-based fallback when no key is set or the API is down. Keys read from env only.
- **Testing** (`tests/`): 29 pytest tests across data, search, analytics and AI grounding; headless
  Streamlit `AppTest` smoke test of all screens.
- **Business-analysis docs** (`docs/`): requirements, user stories & acceptance criteria, process
  flow, test cases (actual results), results & validation, data-validation methodology, user guide.
- **App restructured** into four screens: Market Overview, Competitor Intelligence, AI Insights,
  Data Quality.

Changed
- App now loads a *validated/cleaned* table; retrieval falls back to TF-IDF when embeddings are
  unavailable; deprecated `use_container_width` replaced with `width="stretch"`.

## v1.0 — Initial platform
Competitor finder (TF-IDF + embeddings), Market Saturation score, multi-source sentiment, Opportunity
Score, white-space finder, Streamlit app, evaluation notebook.
