# User Stories & Acceptance Criteria

### US-1 — Competitor discovery
**As** a market research analyst, **I want** to search a startup idea and identify similar startups
**so that** I can evaluate the competitive landscape.

Acceptance criteria:
- The user can enter a non-empty idea or description (BR-01).
- The system returns ranked matches from the available dataset with a similarity score each (BR-02, BR-03).
- The user can inspect the underlying startup information (name, industry, batch, status, region, links).
- Empty or too-short inputs return a helpful message, not an error (BR-09).
- Results are deterministic for a fixed dataset, query and filter set.

### US-2 — Market analysis
**As** a founder, **I want** to view startup distribution and saturation indicators **so that** I can
identify areas for further research.

Acceptance criteria:
- The dashboard shows industry, region, status, and batch-year distributions (BR-05).
- Category density and momentum indicators are shown with an explanation of how they are calculated.
- The saturation scoring methodology, inputs, range and limitations are accessible in the UI.
- Missing data and limitations are communicated (BR-10).

### US-3 — Result filtering
**As** an analyst, **I want** to filter competitor results by industry, status, region, stage and
batch year **so that** I can focus on a relevant subset.

Acceptance criteria:
- Each filter restricts results to matching records only (BR-04).
- The count of candidates after filtering is shown.
- An over-restrictive filter returns a clear "no matches" message rather than an error (BR-09).

### US-4 — Data quality
**As** an application owner, **I want** to identify incomplete and duplicate records **so that**
analysis is not silently affected by poor-quality data.

Acceptance criteria:
- The application identifies configured data-quality issues and reports flagged/excluded records (BR-06, BR-07).
- Cleaning rules are documented and applied to copies; raw source data is preserved.
- The same input dataset produces the same validation summary (deterministic, with a fingerprint).

### US-5 — AI-assisted insights (optional)
**As** an analyst, **I want** a summary of retrieved competitor information **so that** I can quickly
understand common positioning themes.

Acceptance criteria:
- The summary uses only retrieved dataset records, shown alongside it (BR-08).
- Company names cited are checked against the evidence; unsupported citations are flagged.
- Insufficient evidence is explicitly communicated.
- When no API key is set or the service is down, a rule-based summary is shown instead.
