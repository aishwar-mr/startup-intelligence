# Test Cases

Automated with `pytest` (`tests/`). Run `python -m pytest`. Last run: **29 passed**.
Results below are the actual recorded outcomes, not projections. Data/analytics/search tests that
need the built artifacts are skipped automatically if `data/` has not been built yet.

| Test ID | Area | Scenario | Expected result | Actual | Status | Automated test |
|---|---|---|---|---|---|---|
| TC-01 | Data | Duplicate id in input | Second record excluded as `duplicate_id` | Excluded, 1 flagged | Pass | `test_duplicate_id_excluded` |
| TC-02 | Data | Blank / placeholder description | Record excluded as `empty_description` | 2 excluded (blank + "TBD") | Pass | `test_empty_and_placeholder_description_excluded` |
| TC-03 | Data | Missing name | Record excluded as `missing_name` | Excluded | Pass | `test_missing_name_excluded` |
| TC-04 | Data | Probable duplicate (same name + domain) | Lower id kept, other dropped | id 1 kept, id 7 dropped | Pass | `test_probable_duplicate_detected` |
| TC-05 | Data | Tags given as a string | Coerced into a list | `"AI, Support"` → `["AI","Support"]` | Pass | `test_list_given_as_string_is_coerced` |
| TC-06 | Data | Raw input integrity | Raw records unchanged after validation | Unchanged | Pass | `test_raw_input_not_modified` |
| TC-07 | Data | Determinism | Same input (any order) → same summary | Equal report + table | Pass | `test_determinism`, `test_fingerprint_stable` |
| TC-08 | Search | Valid query | Ranked results, descending similarity | 10 ranked results | Pass | `test_valid_query_returns_ranked` |
| TC-09 | Search | Empty query | `SearchError` with helpful message | Raised | Pass | `test_empty_query_raises` |
| TC-10 | Search | Very short query | `SearchError` | Raised | Pass | `test_short_query_raises` |
| TC-11 | Filters | Industry + status filter | Only matching records returned | All Healthcare + Active | Pass | `test_filter_restricts_results` |
| TC-12 | Filters | Over-restrictive filter | Empty result, no error | Empty, `total_candidates=0` | Pass | `test_filter_no_match_returns_empty` |
| TC-13 | Search | Repeatability | Same query → same ids | Identical | Pass | `test_search_is_deterministic` |
| TC-14 | Scores | Saturation output | Score in 0–100 with a valid band | In range, valid label | Pass | `test_saturation_range_and_label` |
| TC-15 | Dashboard | Overview counts | Industry counts sum to total | Consistent | Pass | `test_overview_counts_consistent` |
| TC-16 | Dashboard | Category hints | Only valid hint labels, density max 100 | Valid | Pass | `test_category_hints_valid` |
| TC-17 | AI | Rule-based grounding | Only evidence names appear | Grounded | Pass | `test_rule_based_is_grounded` |
| TC-18 | AI | No evidence | Says insufficient evidence | Communicated | Pass | `test_rule_based_handles_no_evidence` |
| TC-19 | AI | No API key | Falls back to rule-based with warning | Fell back | Pass | `test_no_api_key_falls_back` |
| TC-20 | AI | Invented id in output | Flagged as unverified | Flagged | Pass | `test_grounding_checker_flags_unknown_ids` |
| TC-21 | AI | Valid citations | No grounding warning | No warning | Pass | `test_grounding_checker_passes_valid_ids` |
| TC-22 | AI | API outage | Falls back to rule-based | Fell back | Pass | `test_api_failure_falls_back` |
| TC-23 | AI | API success | Grounded LLM summary returned | Grounded | Pass | `test_api_success_is_grounded` |

## UI smoke test (manual / AppTest)
All four screens (Market Overview, Competitor Intelligence, AI Insights, Data Quality) render with
zero exceptions via `streamlit.testing.v1.AppTest`, including empty-query handling and the
rule-based AI path. See `Results_and_Validation.md`.

## Competitor-matching evaluation (to complete by hand)
The spec asks for a Precision@K evaluation on a hand-labelled sample. The notebook includes a
labelling scaffold: for ~15–20 representative queries, mark each returned company relevant/not,
then compute Precision@10. Record the sample size and labelling method. Do not report accuracy
figures until the labels are actually filled in — the scaffold ships unlabelled on purpose.
