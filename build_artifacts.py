"""Regenerate the data artifacts locally (instead of downloading from Colab).

Fetches the YC directory, builds the TF-IDF matrix + vectorizer, encodes
companies with MiniLM, and writes the four files into ./data .

Usage:  python build_artifacts.py
"""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
YC_URL = "https://yc-oss.github.io/api/companies/all.json"

KEEP = ["name", "one_liner", "long_description", "industry", "industries",
        "tags", "batch", "status", "website", "all_locations"]


def to_text(x):
    if isinstance(x, list):
        return " ".join(str(i) for i in x)
    return "" if x is None else str(x)


def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    print("Fetching YC directory...")
    raw = requests.get(YC_URL, timeout=60).json()
    df = pd.DataFrame(raw)
    for col in KEEP:
        if col not in df.columns:
            df[col] = ""
    data = df[KEEP].copy()
    for col in KEEP:
        data[col] = data[col].apply(to_text)
    data = data[(data["one_liner"].str.strip() != "") |
                (data["long_description"].str.strip() != "")].reset_index(drop=True)
    print(f"  kept {len(data):,} companies")

    data["clean_text"] = data.apply(lambda r: clean(
        " ".join([r["name"], r["name"], r["one_liner"], r["one_liner"],
                  r["tags"], r["tags"], r["industry"], r["industries"],
                  r["long_description"]])), axis=1)

    print("Building TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                 min_df=2, max_df=0.9, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(data["clean_text"])

    print("Encoding embeddings (MiniLM)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_texts = data.apply(
        lambda r: f"{r['name']}. {r['one_liner']}. {r['long_description']}. "
                  f"Tags: {r['tags']}. Industry: {r['industry']}", axis=1).tolist()
    company_emb = model.encode(embed_texts, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True)

    print("Saving artifacts...")
    data.to_parquet(DATA_DIR / "yc_companies.parquet", index=False)
    joblib.dump(vectorizer, DATA_DIR / "tfidf_vectorizer.joblib")
    sparse.save_npz(DATA_DIR / "tfidf_matrix.npz", tfidf_matrix)
    np.save(DATA_DIR / "company_embeddings.npy", company_emb)
    print("Done ->", DATA_DIR)


if __name__ == "__main__":
    main()
