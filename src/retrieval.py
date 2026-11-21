"""Competitor retrieval: TF-IDF (literal) and embeddings (semantic), with filters.

Similarity is a model-generated *relevance estimate*, not proof of direct competition.
Results are deterministic for a fixed dataset + query + filter set.
"""
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from .preprocess import clean

DISPLAY_COLS = ["id", "name", "one_liner", "industry", "batch", "status",
                "country", "team_size", "website", "yc_url"]

# a query is treated as "unrelated / too weak" if its best match is below this
MIN_RELEVANCE = 0.03
MIN_QUERY_CHARS = 3


class SearchError(ValueError):
    """Raised for empty or too-short queries so the UI can show a helpful message."""


def _validate_query(idea: str):
    if idea is None or not str(idea).strip():
        raise SearchError("Please enter a startup idea or description to search.")
    if len(clean(idea)) < MIN_QUERY_CHARS:
        raise SearchError("That query is too short — add a few more words describing the idea.")


def _apply_filters(data, industries=None, statuses=None, macro_regions=None,
                   stages=None, year_range=None):
    mask = np.ones(len(data), dtype=bool)
    if industries:
        mask &= data["industry"].isin(industries).to_numpy()
    if statuses:
        mask &= data["status"].isin(statuses).to_numpy()
    if macro_regions:
        mask &= data["macro_region"].isin(macro_regions).to_numpy()
    if stages:
        mask &= data["stage"].isin(stages).to_numpy()
    if year_range:
        lo, hi = year_range
        yr = data["batch_year"]
        mask &= yr.notna().to_numpy() & (yr.fillna(-1).to_numpy() >= lo) & (yr.fillna(-1).to_numpy() <= hi)
    return mask


def similarities(idea, vectorizer, tfidf_matrix, embed_model=None, company_emb=None, method="auto"):
    """Return (sims, method_used). Uses embeddings when available unless method='tfidf'."""
    use_emb = company_emb is not None and embed_model is not None and method in ("auto", "embeddings")
    if use_emb:
        q = embed_model.encode([idea], normalize_embeddings=True)
        return (company_emb @ q[0]), "embeddings"
    q = vectorizer.transform([clean(idea)])
    return linear_kernel(q, tfidf_matrix).flatten(), "tfidf"


def find_competitors(idea, data, vectorizer, tfidf_matrix, embed_model=None, company_emb=None,
                     k=10, method="auto", filters=None):
    """Ranked competitor DataFrame. Adds a 'similarity' column; respects filters.

    Returns (results_df, info) where info has method_used, matches_before_filter,
    low_confidence (bool) and total_candidates.
    """
    _validate_query(idea)
    sims, method_used = similarities(idea, vectorizer, tfidf_matrix, embed_model, company_emb, method)

    mask = np.ones(len(data), dtype=bool)
    if filters:
        mask = _apply_filters(data, **filters)
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        info = {"method_used": method_used, "matches_before_filter": int((sims >= MIN_RELEVANCE).sum()),
                "low_confidence": True, "total_candidates": 0, "best_similarity": 0.0}
        return data.iloc[0:0][DISPLAY_COLS].assign(similarity=[]), info

    order = candidates[np.argsort(sims[candidates])[::-1]]
    top = order[:k]
    best = float(sims[order[0]]) if order.size else 0.0

    out = data.iloc[top][DISPLAY_COLS].copy()
    out.insert(0, "similarity", np.round(sims[top], 3))
    out = out.reset_index(drop=True)
    info = {
        "method_used": method_used,
        "matches_before_filter": int((sims >= MIN_RELEVANCE).sum()),
        "total_candidates": int(candidates.size),
        "best_similarity": round(best, 3),
        "low_confidence": best < MIN_RELEVANCE,
    }
    return out, info
