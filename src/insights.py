"""Evidence-grounded AI market summary (optional), with a rule-based fallback.

Design guardrails (from the spec):
- Only the retrieved competitor records are sent to the model - nothing else.
- The prompt forbids inventing names, funding, customers or statistics.
- The output is checked: any company name it cites must appear in the evidence;
  unsupported names are flagged, not shown as fact.
- If no API key is set or the call fails, a deterministic rule-based summary is
  returned instead, so the feature always degrades gracefully.
- The API key is read from the environment, never hard-coded.
"""
import os
import re
import time

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Free models rotate; the primary is tried first, then the fallbacks, then rule-based.
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
FALLBACK_MODELS = ["google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free",
                   "inclusionai/ling-3.2-flash:free"]
MODELS_URL = "https://openrouter.ai/api/v1/models"
# Reasoning models spend part of this budget thinking, so leave generous headroom.
MAX_TOKENS = 3000


def _discover_free_models(tried, limit=4, timeout=15):
    """Ask OpenRouter which free models exist right now (the static list can go stale)."""
    try:
        data = requests.get(MODELS_URL, timeout=timeout).json().get("data", [])
    except Exception:
        return []
    ids = [m["id"] for m in data if str(m.get("id", "")).endswith(":free") and m["id"] not in tried]
    return ids[:limit]

EVIDENCE_COLS = ["id", "name", "one_liner", "industry", "batch", "status", "country", "website"]

SYSTEM_PROMPT = (
    "You are a market-research analyst. You will be given a candidate startup idea and a list of "
    "EVIDENCE records (real Y Combinator companies retrieved for that idea). "
    "Write a short, structured competitive-landscape summary using ONLY the evidence provided. "
    "Rules you must follow:\n"
    "- Refer to companies only by names that appear in the evidence. Never invent a company, "
    "and never state funding amounts, customer names, valuations or market-size statistics.\n"
    "- When you mention a company, cite it as [id] using its evidence id.\n"
    "- Clearly label any interpretation as a hypothesis, not an established fact.\n"
    "- If the evidence is too thin to support a section, say so plainly.\n"
    "Use these sections exactly:\n"
    "Idea overview:\nClosest competitors:\nCommon themes:\nPotential gaps (hypotheses):\nCaveats:"
)


def build_evidence(results_df, limit=8):
    """Turn retrieval results into a compact evidence list (dicts) for the model."""
    ev = []
    for _, r in results_df.head(limit).iterrows():
        ev.append({c: (None if r.get(c) is None else r.get(c)) for c in EVIDENCE_COLS})
    return ev


def _evidence_block(evidence):
    lines = []
    for e in evidence:
        loc = e.get("country") or "?"
        lines.append(f"[{e['id']}] {e['name']} — {e.get('one_liner') or ''} "
                     f"(industry: {e.get('industry')}, batch: {e.get('batch')}, status: {e.get('status')}, region: {loc})")
    return "\n".join(lines)


# --------------------------------------------------------------- rule-based ----

def rule_based_summary(idea, evidence, saturation=None, whitespace=None):
    """Deterministic, fully-grounded summary built from the evidence - no LLM."""
    if not evidence:
        return {"source": "rule-based", "model_used": None, "warnings": [],
                "text": f"No sufficiently similar companies were found for '{idea}', so there is not "
                        "enough evidence to summarise a competitive landscape."}
    names = ", ".join(f"{e['name']} [{e['id']}]" for e in evidence[:5])
    inds = {}
    for e in evidence:
        inds[e.get("industry")] = inds.get(e.get("industry"), 0) + 1
    top_inds = sorted(inds.items(), key=lambda kv: -kv[1])
    ind_txt = ", ".join(f"{k} ({v})" for k, v in top_inds if k)
    parts = [f"**Idea overview:** The idea '{idea}' maps most closely to {len(evidence)} existing "
             f"Y Combinator companies, concentrated in {ind_txt or 'several industries'}."]
    parts.append(f"**Closest competitors:** {names}.")
    if saturation:
        parts.append(f"**Market saturation:** {saturation['saturation_score']}/100 "
                     f"({saturation['label']}), based on {saturation['competitors']} similar companies.")
    if whitespace and whitespace[1]:
        gaps = ", ".join(g["tag"] for g in whitespace[1][:5])
        parts.append(f"**Potential gaps (hypotheses):** tags common nearby but rare among the closest "
                     f"competitors include {gaps}. Treat these as leads to investigate, not conclusions.")
    parts.append("**Caveats:** Based only on the retrieved Y Combinator records; similarity is a "
                 "text-based estimate, not proof of direct competition.")
    return {"source": "rule-based", "model_used": None, "warnings": [], "text": "\n\n".join(parts)}


# ------------------------------------------------------------------- LLM ----

def _call_openrouter(messages, api_key, timeout=45):
    models = [DEFAULT_MODEL] + [m for m in FALLBACK_MODELS if m != DEFAULT_MODEL]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://streamlit.io", "X-Title": "Startup Intelligence Platform"}
    last_err = None
    discovered = False
    i = 0
    while i < len(models):
        model = models[i]
        i += 1
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, timeout=timeout, json={
                "model": model, "messages": messages, "temperature": 0.2,
                "max_tokens": MAX_TOKENS})
            if resp.status_code == 200:
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                text = choice.get("message", {}).get("content", "")
                if choice.get("finish_reason") == "length":  # cut off mid-answer
                    last_err = "truncated response"
                elif text and text.strip():
                    return text.strip(), data.get("model", model), None
                else:
                    last_err = "empty response"
            else:
                last_err = f"HTTP {resp.status_code}"
                if resp.status_code == 429:
                    time.sleep(1.0)
        except Exception as e:  # network / parse
            last_err = str(e)
        if i == len(models) and not discovered:  # static list exhausted -> live discovery
            discovered = True
            models += _discover_free_models(set(models))
    return None, None, last_err


def _check_grounding(text, evidence):
    """Flag any [id] citation not in the evidence; return (clean_text, warnings)."""
    warnings = []
    valid_ids = {str(e["id"]) for e in evidence}
    cited = set(re.findall(r"\[(\d+)\]", text))
    bad = cited - valid_ids
    if bad:
        warnings.append(f"Model cited id(s) not in the evidence: {', '.join(sorted(bad))} "
                        "(these claims are unverified and should be ignored).")
    if not cited:
        warnings.append("Summary did not cite specific evidence records; read it with extra caution.")
    return text, warnings


def generate_insight(idea, results_df, saturation=None, whitespace=None, api_key=None):
    """Return an evidence-grounded summary dict. Falls back to rule-based when needed."""
    evidence = build_evidence(results_df)
    key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        out = rule_based_summary(idea, evidence, saturation, whitespace)
        out["warnings"] = out["warnings"] + ["No OPENROUTER_API_KEY set — showing a rule-based summary."]
        out["evidence"] = evidence
        return out
    if not evidence:
        out = rule_based_summary(idea, evidence, saturation, whitespace)
        out["evidence"] = evidence
        return out

    user = (f"Candidate idea: {idea}\n\nEVIDENCE ({len(evidence)} records):\n"
            + _evidence_block(evidence))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    text, model_used, err = _call_openrouter(messages, key)
    if text is None:
        out = rule_based_summary(idea, evidence, saturation, whitespace)
        out["warnings"] = out["warnings"] + [f"AI service unavailable ({err}); showing a rule-based summary."]
        out["evidence"] = evidence
        return out
    text, warnings = _check_grounding(text, evidence)
    return {"source": "llm", "model_used": model_used, "warnings": warnings,
            "text": text, "evidence": evidence}
