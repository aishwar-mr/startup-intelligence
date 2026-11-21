# Process Flow

End-to-end workflow from raw data to a decision-support result.

```text
Dataset ingestion (yc-oss API / local raw JSON)
        ↓
Data validation and cleaning (src/validation.py)
   - raw preserved; checks, flags, exclusions, deterministic report
        ↓
Feature preparation / NLP (build_artifacts.py)
   - TF-IDF vectoriser + matrix
   - MiniLM sentence embeddings (optional)
        ↓
User submits a business idea  ──►  optional filters (industry, status, region, stage, year)
        ↓
Competitor search and ranking (src/retrieval.py)
   - cosine similarity (embeddings if available, else TF-IDF)
        ↓
Analytical scoring
   - Market Saturation (src/saturation.py)
   - Market overview & category momentum (src/analytics.py)
   - White-space / differentiation (src/whitespace.py)
   - Live sentiment & Opportunity (src/sentiment.py)  [optional]
        ↓
Optional AI summarisation (src/insights.py)
   - evidence-only prompt → grounding check → or rule-based fallback
        ↓
Results + supporting evidence shown in the Streamlit app (app.py)
        ↓
User inspects evidence / validates / exports
```

## Component responsibilities
| Module | Responsibility |
|--------|----------------|
| `src/validation.py` | Data-quality checks, cleaning rules, exclusion log, deterministic report |
| `src/preprocess.py` | Text normalisation (display and matching) |
| `src/retrieval.py` | Similarity search, filters, empty/low-confidence handling |
| `src/saturation.py` | Market Saturation heuristic + explanation |
| `src/analytics.py` | Market-overview aggregations and category indicators |
| `src/whitespace.py` | Tag-frequency gap analysis |
| `src/sentiment.py` | Hacker News + GDELT sentiment, Opportunity Score |
| `src/insights.py` | Evidence-grounded summary (OpenRouter) + rule-based fallback |
| `src/resources.py` | Load precomputed artifacts |
| `app.py` | Four-screen Streamlit UI |
