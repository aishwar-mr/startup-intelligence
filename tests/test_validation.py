"""Data validation tests (TC-01, TC-02 and related)."""
import copy

from src.validation import validate, fingerprint, summary_rows


def test_raw_input_not_modified(raw_sample):
    before = copy.deepcopy(raw_sample)
    validate(raw_sample)
    assert raw_sample == before  # cleaning never mutates the raw records


def test_duplicate_id_excluded(raw_sample):
    res = validate(raw_sample)
    assert res.report["excluded_by_reason"]["duplicate_id"] == 1
    assert (res.exclusions["reason"] == "duplicate_id").sum() == 1


def test_missing_name_excluded(raw_sample):
    res = validate(raw_sample)
    assert res.report["excluded_by_reason"]["missing_name"] == 1


def test_empty_and_placeholder_description_excluded(raw_sample):
    res = validate(raw_sample)
    # GhostCo (blank) and TBDStartup (placeholder 'TBD') both go
    assert res.report["excluded_by_reason"]["empty_description"] == 2
    kept_ids = set(res.clean["id"])
    assert 4 not in kept_ids and 5 not in kept_ids


def test_probable_duplicate_detected(raw_sample):
    res = validate(raw_sample)
    assert res.report["excluded_by_reason"]["probable_duplicate"] == 1
    # id 7 (same name + same domain as id 1) is dropped, id 1 kept
    kept = set(res.clean["id"])
    assert 1 in kept and 7 not in kept


def test_list_given_as_string_is_coerced(raw_sample):
    res = validate(raw_sample)
    row = res.clean[res.clean["id"] == 2].iloc[0]
    assert row["tag_list"] == ["AI", "Support"]  # "AI, Support" split into a list


def test_status_case_normalised(raw_sample):
    res = validate(raw_sample)
    assert set(res.clean["status"]) <= {"Active", "Inactive", "Acquired", "Public"}
    assert (res.clean[res.clean["id"] == 2]["status"] == "Active").all()


def test_website_scheme_added(raw_sample):
    res = validate(raw_sample)
    assert res.clean[res.clean["id"] == 2].iloc[0]["website"].startswith("http://")


def test_geography_parsed(raw_sample):
    res = validate(raw_sample)
    row = res.clean[res.clean["id"] == 3].iloc[0]
    assert row["country"] == "Australia"
    assert row["macro_region"] == "Oceania"


def test_missing_counts_reported(raw_sample):
    res = validate(raw_sample)
    mv = res.report["missing_by_field"]
    assert mv["one_liner"] >= 1 and mv["tags"] >= 1 and mv["website"] >= 1


def test_determinism(raw_sample):
    r1 = validate(copy.deepcopy(raw_sample))
    r2 = validate(list(reversed(copy.deepcopy(raw_sample))))  # order must not matter
    assert r1.report["input_fingerprint"] == r2.report["input_fingerprint"]
    assert r1.report["passed"] == r2.report["passed"]
    assert r1.clean.equals(r2.clean)


def test_fingerprint_stable(raw_sample):
    assert fingerprint(raw_sample) == fingerprint(list(reversed(raw_sample)))


def test_summary_rows_shape(raw_sample):
    res = validate(raw_sample)
    sr = summary_rows(res.report)
    assert list(sr.columns) == ["Check", "Count"]
    assert sr.loc[sr["Check"] == "Records received", "Count"].iloc[0] == len(raw_sample)
