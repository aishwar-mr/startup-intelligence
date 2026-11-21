# Business Requirements

## Problem statement
Market researchers and founders need to identify startups with similar business models and
understand competitive patterns. Manually searching a large startup dataset is slow, inconsistent,
and gives no sense of how crowded or active a space is.

## Proposed solution
A searchable market-intelligence application that uses NLP-based matching (TF-IDF and semantic
embeddings) and structured analytics to identify relevant startups, quantify competitive density,
and present evidence that supports further research. All outputs are explainable estimates over a
public Y Combinator dataset — not objective market facts.

## Scope
In scope: competitor discovery, market-overview analytics, saturation/opportunity indicators,
data-quality validation, and an optional evidence-grounded AI summary.
Out of scope: financial reconciliation, private/PII data, real-time funding or valuation data.

## Business requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| BR-01 | The user can submit a startup idea or business description. | Must |
| BR-02 | The system returns ranked startup matches from the available dataset. | Must |
| BR-03 | The system displays similarity scores and relevant record details. | Must |
| BR-04 | The user can filter results by supported attributes (industry, status, region, stage, batch year). | Must |
| BR-05 | The system displays market saturation and opportunity indicators with explanations. | Must |
| BR-06 | The system validates input and flags data-quality issues, preserving the raw source. | Must |
| BR-07 | The system reports how many records were processed and excluded, with reasons. | Must |
| BR-08 | If AI summaries are enabled, they are grounded in retrieved records and degrade gracefully. | Should |
| BR-09 | Empty/invalid inputs and empty result sets produce helpful messages, not errors. | Must |
| BR-10 | Scores and similarity are clearly labelled as estimates, not market facts. | Must |

## Assumptions and constraints
- Data source is the public `yc-oss` Y Combinator directory (company metadata only).
- The app must run within Streamlit Community Cloud free-tier memory, so a small embedding model
  (`all-MiniLM-L6-v2`) is used and heavy artifacts are precomputed.
- No secret or personal data is sent to any external model; API keys live in environment variables.
