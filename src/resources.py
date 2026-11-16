"""Load the precomputed artifacts built in the Colab notebook (or build_artifacts.py).

Expected files in ./data :
    yc_companies.parquet      - cleaned company table
    tfidf_vectorizer.joblib   - fitted TfidfVectorizer
    tfidf_matrix.npz          - sparse TF-IDF matrix (rows aligned to the table)
    company_embeddings.npy    - (n, 384) MiniLM embeddings (rows aligned to the table)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

REQUIRED = [
    "yc_companies.parquet",
    "tfidf_vectorizer.joblib",
    "tfidf_matrix.npz",
    "company_embeddings.npy",
]


def missing_artifacts():
    return [f for f in REQUIRED if not (DATA_DIR / f).exists()]


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "yc_companies.parquet")


def load_tfidf():
    vectorizer = joblib.load(DATA_DIR / "tfidf_vectorizer.joblib")
    matrix = sparse.load_npz(DATA_DIR / "tfidf_matrix.npz")
    return vectorizer, matrix


def load_embeddings() -> np.ndarray:
    return np.load(DATA_DIR / "company_embeddings.npy")
