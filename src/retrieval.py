"""Competitor retrieval: TF-IDF (literal) and embeddings (semantic)."""
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

from .preprocess import clean

DISPLAY_COLS = ["name", "one_liner", "industry", "batch", "status", "website"]


def similarities_tfidf(idea, vectorizer, tfidf_matrix):
    """Cosine similarity of an idea against every company (TF-IDF space)."""
    q = vectorizer.transform([clean(idea)])
    return linear_kernel(q, tfidf_matrix).flatten()


def find_competitors_tfidf(idea, data, vectorizer, tfidf_matrix, k=10):
    sims = similarities_tfidf(idea, vectorizer, tfidf_matrix)
    top = sims.argsort()[::-1][:k]
    out = data.loc[top, DISPLAY_COLS].copy()
    out.insert(0, "similarity", sims[top].round(3))
    return out.reset_index(drop=True)


def find_competitors_emb(idea, data, embed_model, company_emb, k=10):
    q = embed_model.encode([idea], normalize_embeddings=True)
    sims = company_emb @ q[0]
    top = sims.argsort()[::-1][:k]
    out = data.loc[top, DISPLAY_COLS].copy()
    out.insert(0, "similarity", sims[top].round(3))
    return out.reset_index(drop=True)
