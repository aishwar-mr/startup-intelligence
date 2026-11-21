"""Market Saturation Score - a custom, explainable heuristic (0-100).

score = 100 * (0.5*breadth + 0.3*closeness + 0.2*vitality)
  breadth   - how many companies sit above TAU similarity (number of players)
  closeness - mean similarity of the top-5 matches (how directly on-top competitors are)
  vitality  - share of matched companies still alive (Active/Public weigh highest)

This is a heuristic indicator for prioritising research, NOT an objective market fact.
"""
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from .preprocess import clean

TAU = 0.08          # min cosine similarity to count as "in the same space"
BREADTH_REF = 50    # this many similar companies => breadth fully saturated
CLOSE_REF = 0.40    # mean top-5 similarity at/above this => maximally "close"
WEIGHTS = {"breadth": 0.5, "closeness": 0.3, "vitality": 0.2}
STATUS_WEIGHT = {"Active": 1.0, "Public": 1.0, "Acquired": 0.7, "Inactive": 0.3}

SCORE_RANGE = (0, 100)
BANDS = [(30, "Wide open"), (55, "Emerging"), (75, "Competitive"), (101, "Crowded")]

EXPLANATION = {
    "what": "An estimate of how crowded the space around an idea already is, from 0 (wide open) to 100 (crowded).",
    "inputs": {
        "breadth": f"Number of companies with TF-IDF cosine similarity >= {TAU}, capped at {BREADTH_REF}.",
        "closeness": f"Mean similarity of the 5 nearest companies, scaled so {CLOSE_REF} = 1.0.",
        "vitality": "Share of matched companies still operating, weighting "
                    + ", ".join(f"{k}={v}" for k, v in STATUS_WEIGHT.items()) + ".",
    },
    "formula": "score = 100 * (0.5*breadth + 0.3*closeness + 0.2*vitality)",
    "bands": "0-30 Wide open | 30-55 Emerging | 55-75 Competitive | 75-100 Crowded",
    "limitations": [
        "Heuristic weights and thresholds are design choices, not learned from outcomes.",
        "Only Y Combinator companies are counted, so spaces strong outside YC look emptier than they are.",
        "Similarity is text-based, so differently-worded competitors may be missed.",
    ],
}


def _band(score):
    return next(label for cut, label in BANDS if score < cut)


def market_saturation(idea, data, vectorizer, tfidf_matrix, tau=TAU, candidate_mask=None):
    sims = linear_kernel(vectorizer.transform([clean(idea)]), tfidf_matrix).flatten()
    if candidate_mask is not None:
        sims = np.where(candidate_mask, sims, -1.0)
    order = sims.argsort()[::-1]

    matched = np.where(sims >= tau)[0]
    n = len(matched)

    breadth = min(1.0, n / BREADTH_REF)
    closeness = min(1.0, float(sims[order[:5]].clip(min=0).mean()) / CLOSE_REF) if n else \
        min(1.0, float(sims[order[:5]].clip(min=0).mean()) / CLOSE_REF)
    if n:
        vitality = float(data.iloc[matched]["status"].map(
            lambda s: STATUS_WEIGHT.get(s, 0.5)).mean())
    else:
        vitality = 0.0

    score = 100 * (WEIGHTS["breadth"] * breadth + WEIGHTS["closeness"] * closeness
                   + WEIGHTS["vitality"] * vitality)
    return {
        "saturation_score": round(score, 1),
        "label": _band(score),
        "competitors": int(n),
        "breadth": round(breadth, 3),
        "closeness": round(closeness, 3),
        "vitality": round(vitality, 3),
        "tau": tau,
    }
