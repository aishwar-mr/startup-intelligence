"""AI-insight grounding tests (TC-07). No network: the LLM path is exercised with a stub."""
import pandas as pd

from src import insights
from src.insights import generate_insight, rule_based_summary, _check_grounding, build_evidence

EVIDENCE_DF = pd.DataFrame([
    {"id": 101, "name": "SupportAI", "one_liner": "AI chatbot for support", "industry": "B2B",
     "batch": "Winter 2024", "status": "Active", "country": "USA", "website": "https://a.com", "similarity": 0.3},
    {"id": 102, "name": "HelpBot", "one_liner": "Automated support", "industry": "B2B",
     "batch": "Summer 2023", "status": "Active", "country": "USA", "website": "https://b.com", "similarity": 0.28},
])


def test_rule_based_is_grounded():
    ev = build_evidence(EVIDENCE_DF)
    out = rule_based_summary("AI support tool", ev)
    assert out["source"] == "rule-based"
    # every company named appears in the evidence
    for name in ["SupportAI", "HelpBot"]:
        assert name in out["text"]


def test_rule_based_handles_no_evidence():
    out = rule_based_summary("something obscure", [])
    assert "not enough evidence" in out["text"].lower()


def test_no_api_key_falls_back(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    out = generate_insight("AI support tool", EVIDENCE_DF, api_key=None)
    assert out["source"] == "rule-based"
    assert any("no openrouter_api_key" in w.lower() for w in out["warnings"])


def test_grounding_checker_flags_unknown_ids():
    ev = build_evidence(EVIDENCE_DF)
    text = "Closest competitors: SupportAI [101] and MadeUpCo [999]."
    _, warnings = _check_grounding(text, ev)
    assert any("999" in w for w in warnings)


def test_grounding_checker_passes_valid_ids():
    ev = build_evidence(EVIDENCE_DF)
    text = "SupportAI [101] and HelpBot [102] lead the space."
    _, warnings = _check_grounding(text, ev)
    assert not any("not in the evidence" in w for w in warnings)


def test_api_failure_falls_back(monkeypatch):
    # force the LLM call to fail; generate_insight must return a rule-based summary
    monkeypatch.setattr(insights, "_call_openrouter", lambda *a, **k: (None, None, "simulated outage"))
    out = generate_insight("AI support tool", EVIDENCE_DF, api_key="dummy-key")
    assert out["source"] == "rule-based"
    assert any("unavailable" in w.lower() for w in out["warnings"])


def test_api_success_is_grounded(monkeypatch):
    fake = "Idea overview: a support tool.\nClosest competitors: SupportAI [101], HelpBot [102]."
    monkeypatch.setattr(insights, "_call_openrouter", lambda *a, **k: (fake, "stub-model:free", None))
    out = generate_insight("AI support tool", EVIDENCE_DF, api_key="dummy-key")
    assert out["source"] == "llm" and out["model_used"] == "stub-model:free"
    assert "SupportAI" in out["text"]
