"""Market Saturation Score - a custom, explainable metric.

score = 100 * (0.5*breadth + 0.3*closeness + 0.2*vitality)
  breadth   - how many companies sit above TAU similarity (number of players)
  closeness - mean similarity of the top-5 matches (how directly on-top competitors are)
  vitality  - share of matched companies still alive (Active/Public weigh highest)
"""
import numpy as np

from .retrieval import similarities_tfidf

TAU = 0.08          # min cosine similarity to count as "in the same space"
BREADTH_REF = 50    # this many similar companies => breadth fully saturated
CLOSE_REF = 0.40    # mean top-5 similarity at/above this => maximally "close"

STATUS_WEIGHT = {"Active": 1.0, "Public": 1.0, "Acquired": 0.7, "Inactive": 0.3}


def market_saturation(idea, data, vectorizer, tfidf_matrix, tau=TAU):
    sims = similarities_tfidf(idea, vectorizer, tfidf_matrix)
    order = sims.argsort()[::-1]

    matched = np.where(sims >= tau)[0]
    n = len(matched)

    breadth = min(1.0, n / BREADTH_REF)
    closeness = min(1.0, float(sims[order[:5]].mean()) / CLOSE_REF)
    if n:
        vitality = float(data.loc[matched, "status"].map(
            lambda s: STATUS_WEIGHT.get(s, 0.5)).mean())
    else:
        vitality = 0.0

    score = 100 * (0.5 * breadth + 0.3 * closeness + 0.2 * vitality)
    label = ("Wide open" if score < 30 else
             "Emerging" if score < 55 else
             "Competitive" if score < 75 else "Crowded")

    return {
        "saturation_score": round(score, 1),
        "label": label,
        "competitors": int(n),
        "breadth": round(breadth, 3),
        "closeness": round(closeness, 3),
        "vitality": round(vitality, 3),
    }
