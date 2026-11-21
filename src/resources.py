"""Load the precomputed artifacts built by the Colab notebook / build_artifacts.py.

Expected files in ./data :
    yc_companies.parquet      - CLEANED, validated company table (from src.validation)
    tfidf_vectorizer.joblib   - fitted TfidfVectorizer
    tfidf_matrix.npz          - sparse TF-IDF matrix (rows aligned to the table)
    company_embeddings.npy    - (n, 384) MiniLM embeddings (rows aligned, OPTIONAL)
    validation_report.json    - deterministic data-quality report (from src.validation)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REQUIRED = ["yc_companies.parquet", "tfidf_vectorizer.joblib", "tfidf_matrix.npz"]
OPTIONAL = ["company_embeddings.npy", "validation_report.json"]


def missing_artifacts():
    return [f for f in REQUIRED if not (DATA_DIR / f).exists()]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "yc_companies.parquet")
    return df.reset_index(drop=True)


def load_tfidf():
    vectorizer = joblib.load(DATA_DIR / "tfidf_vectorizer.joblib")
    matrix = sparse.load_npz(DATA_DIR / "tfidf_matrix.npz")
    return vectorizer, matrix


def load_embeddings():
    """Return the embedding matrix, or None if it was not built (app falls back to TF-IDF)."""
    path = DATA_DIR / "company_embeddings.npy"
    return np.load(path) if path.exists() else None


def load_validation_report():
    path = DATA_DIR / "validation_report.json"
    if not path.exists():
        return None
    import json
    return json.loads(path.read_text())
