"""Search, ranking, filter and analytics tests (TC-03..TC-06)."""
import pytest

from src.retrieval import find_competitors, SearchError
from src.saturation import market_saturation
from src.analytics import market_overview, category_table


def test_valid_query_returns_ranked(artifacts):
    data, vec, mat = artifacts
    res, info = find_competitors("AI chatbot for customer support", data, vec, mat, k=10)
    assert len(res) == 10
    sims = list(res["similarity"])
    assert sims == sorted(sims, reverse=True)  # descending
    assert info["method_used"] == "tfidf"


def test_empty_query_raises(artifacts):
    data, vec, mat = artifacts
    with pytest.raises(SearchError):
        find_competitors("", data, vec, mat)


def test_short_query_raises(artifacts):
    data, vec, mat = artifacts
    with pytest.raises(SearchError):
        find_competitors("ai", data, vec, mat)


def test_filter_restricts_results(artifacts):
    data, vec, mat = artifacts
    res, info = find_competitors("AI chatbot for customer support", data, vec, mat, k=10,
                                 filters={"industries": ["Healthcare"], "statuses": ["Active"]})
    assert (res["industry"] == "Healthcare").all()
    assert (res["status"] == "Active").all()
    assert info["total_candidates"] < len(data)


def test_filter_no_match_returns_empty(artifacts):
    data, vec, mat = artifacts
    res, info = find_competitors("AI chatbot", data, vec, mat,
                                 filters={"industries": ["Healthcare"], "year_range": (1900, 1901)})
    assert res.empty and info["total_candidates"] == 0


def test_search_is_deterministic(artifacts):
    data, vec, mat = artifacts
    a, _ = find_competitors("fintech payments api", data, vec, mat, k=8)
    b, _ = find_competitors("fintech payments api", data, vec, mat, k=8)
    assert list(a["id"]) == list(b["id"])


def test_saturation_range_and_label(artifacts):
    data, vec, mat = artifacts
    s = market_saturation("AI chatbot for customer support", data, vec, mat)
    assert 0 <= s["saturation_score"] <= 100
    assert s["label"] in {"Wide open", "Emerging", "Competitive", "Crowded"}


def test_overview_counts_consistent(artifacts):
    data, vec, mat = artifacts
    ov = market_overview(data)
    assert ov["total"] == len(data)
    assert ov["by_industry"].sum() == len(data)
    live = ov["by_status"].get("Active", 0) + ov["by_status"].get("Public", 0)
    assert ov["active_pct"] == round(100 * live / ov["total"], 1)


def test_category_hints_valid(artifacts):
    data, vec, mat = artifacts
    ct = category_table(data)
    assert set(ct["hint"]) <= {"Hot & crowded", "Emerging", "Mature / cooling", "Quiet", "Unknown"}
    assert ct["density_pct"].max() == 100.0
