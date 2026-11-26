# Data Validation Methodology

Implemented in `src/validation.py`. The single source of truth for the rules is the
`EXCLUSION_RULES`, `FLAG_RULES` and `NORMALISATION_RULES` dictionaries in that module.

## Principles
1. **Raw data is never modified.** Validation reads the raw records and returns a *cleaned copy*,
   a flag log, an exclusion log and a summary report. A raw snapshot is kept in
   `data/yc_companies_raw.json`; `build_artifacts.py` does not currently update it, so it can be
   older than the built artifacts.
2. **Documented rules, no silent deletion.** Every exclusion and flag has a named rule and a reason;
   excluded records are listed in `validation_exclusions.csv`, not dropped silently.
3. **Deterministic.** Records are processed in id order (ties broken by content), and the report
   carries a SHA-256 fingerprint of the raw input, so the same input always yields the same summary.

## Checks

### Exclusions (record removed from analysis)
| Rule | Meaning |
|---|---|
| `missing_id` | No usable numeric id. |
| `duplicate_id` | Same id as an earlier record; first kept. |
| `missing_name` | Name blank after normalisation. |
| `empty_description` | One-liner and long description both blank or placeholder ("TBD", "-", "."). |
| `probable_duplicate` | Same normalised name AND (same website domain OR identical description); lowest id kept. |

### Flags (record kept, issue noted)
Missing fields (tags, website, location, team size), `team_size_zero`, `short_description`,
`description_equals_name`, `placeholder_text`, `unspecified_industry`, `unspecified_batch`,
`invalid_status`, `invalid_team_size`, `type_coerced`, `website_normalised`,
`name_shared_with_other_company`, `website_shared_with_other_company`.

### Normalisation (applied to the cleaned copy)
Unicode NFKC; strip HTML tags/entities; trim and collapse whitespace; title-case status;
de-duplicate list items (tags/industries/regions); add `http://` scheme to bare websites;
parse `regions` into a specific country, a macro region and a remote policy; derive `batch_year`
from the batch label; order rows by id.

## Outputs
- `yc_companies.parquet` — cleaned table (one row per startup).
- `validation_report.json` — deterministic summary (counts, examples, fingerprint).
- `validation_flags.csv`, `validation_exclusions.csv` — per-record logs.

The Data Quality screen in the app renders the report so reviewers can see exactly what was kept,
flagged and excluded, and why.
