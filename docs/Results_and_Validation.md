# Results & Validation

Figures below are the actual outputs of the pipeline for the data bundled in `data/`
(6,406 raw records). The validation run is deterministic; its input fingerprint is
`d7d0ed831c42b2a9…`. The bundled `yc_companies_raw.json` is an earlier snapshot (6,399 records), so
it does not reproduce these exact figures. `python build_artifacts.py` fetches the current
directory, so rebuilding will also change them.

## 1. Data-quality results

| Check | Count |
|---|---|
| Records received | 6,406 |
| Records passing validation | 6,349 |
| Records excluded | 57 |
| Excluded — empty/placeholder description | 57 |
| Excluded — duplicate id | 0 |
| Excluded — probable duplicate | 0 |

**Missing values by field (raw):** long_description 426, tags 834, website 302, one_liner 161,
all_locations 150, team_size 124. (name, industry, batch, status, stage, regions: 0.)

**Quality flags on kept records (examples):** missing_tags 793, missing_website 291,
missing_location 131, team_size_zero 119, short_description 46, unspecified_industry 18,
name_shared_with_other_company 200, website_shared_with_other_company 4, website_normalised 5.

**Normalisation actions (applied to copies, not raw):** whitespace cleanups 4,679, HTML stripped 4,
website scheme added 5.

**Status distribution after cleaning:** Active 4,405 · Inactive 1,075 · Acquired 847 · Public 22.

**Duplicate analysis:** no exact id duplicates and no probable duplicates in this data; 95 names are
shared by different companies (200 records flagged), and 4 records share a website domain with a
differently-named company (flagged for review, not removed).

## 2. Feature build
- TF-IDF matrix: **6,349 × 39,975** (unigrams + bigrams, min_df=2, max_df=0.9, sublinear tf).
- Embeddings: 6,349 × 384 (`all-MiniLM-L6-v2`), when built.

## 3. Functional test results
`python -m pytest` → **29 passed** (see `Test_Cases.md`). Covers data validation, search/ranking,
filters, saturation range, analytics consistency, and AI-insight grounding/fallback.

A separate `AppTest` run (not part of the pytest suite) rendered all four Streamlit screens with
zero exceptions, including empty-query handling, the no-match filter case, and the rule-based AI
summary.

## 4. Example analytical output
Query *"AI chatbot for customer support"* (TF-IDF): 179 companies above the relevance floor;
Saturation **86.2/100 (Crowded)** from 53 similar companies (breadth 1.0, closeness 0.65,
vitality 0.84). Category view: B2B is "Hot & crowded" (density 100, momentum 28%), Industrials
"Emerging" (density 16, momentum 46%). These are heuristic estimates over the YC sample, not
market facts.

## 5. Competitor-matching relevance (to be completed)
Precision@K on a hand-labelled sample has not been measured. Label ~15–20 queries, record the sample
size and labelling method, then report the number here. Do not state a precision figure before the
labels exist.

## 6. Known limitations
- YC-only corpus; absolute counts understate real markets.
- Text similarity misses differently-worded competitors; mitigated (not solved) by embeddings.
- Saturation/opportunity/category hints are heuristics with hand-set weights.
- Live sentiment depends on external APIs (Hacker News, GDELT) that can rate-limit.
