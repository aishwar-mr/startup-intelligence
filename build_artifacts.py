"""Build the data artifacts the app loads: validated table, TF-IDF, embeddings, report.

Usage:
    python build_artifacts.py                 # fetch YC data, validate, build everything
    python build_artifacts.py --no-embeddings # skip the embedding step (TF-IDF only)
    python build_artifacts.py --raw data/yc_companies_raw.json  # use a local raw file

Outputs in ./data :
    yc_companies.parquet, tfidf_vectorizer.joblib, tfidf_matrix.npz,
    company_embeddings.npy (unless skipped), validation_report.json,
    validation_flags.csv, validation_exclusions.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import requests
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import clean
from src.validation import validate

DATA_DIR = Path(__file__).resolve().parent / "data"
YC_URL = "https://yc-oss.github.io/api/companies/all.json"


def corpus_text(r):
    return clean(" ".join([r["name"], r["name"], r["one_liner"], r["one_liner"],
                           r["tags"], r["tags"], r["industry"], r["industries"], r["long_description"]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="local raw JSON file instead of fetching")
    ap.add_argument("--no-embeddings", action="store_true")
    args = ap.parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    if args.raw:
        print(f"Loading raw records from {args.raw}")
        records = json.loads(Path(args.raw).read_text())
    else:
        print("Fetching YC directory…")
        records = requests.get(YC_URL, timeout=60).json()
    print(f"  {len(records):,} raw records")

    print("Validating and cleaning…")
    res = validate(records)
    df = res.clean.copy()
    print(f"  passed {res.report['passed']:,} | excluded {res.report['excluded']}")

    json.dump(res.report, open(DATA_DIR / "validation_report.json", "w"), indent=2)
    res.flags.to_csv(DATA_DIR / "validation_flags.csv", index=False)
    res.exclusions.to_csv(DATA_DIR / "validation_exclusions.csv", index=False)

    print("Building TF-IDF…")
    df["_text"] = df.apply(corpus_text, axis=1)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                 min_df=2, max_df=0.9, sublinear_tf=True)
    matrix = vectorizer.fit_transform(df["_text"])
    texts = df["_text"].tolist()
    df = df.drop(columns=["_text"])
    df.to_parquet(DATA_DIR / "yc_companies.parquet", index=False)
    joblib.dump(vectorizer, DATA_DIR / "tfidf_vectorizer.joblib")
    sparse.save_npz(DATA_DIR / "tfidf_matrix.npz", matrix)
    print(f"  TF-IDF matrix {matrix.shape}")

    if not args.no_embeddings:
        print("Encoding embeddings (all-MiniLM-L6-v2)…")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_texts = df.apply(
            lambda r: f"{r['name']}. {r['one_liner']}. {r['long_description']}. "
                      f"Tags: {r['tags']}. Industry: {r['industry']}", axis=1).tolist()
        emb = model.encode(emb_texts, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
        np.save(DATA_DIR / "company_embeddings.npy", emb)
        print(f"  embeddings {emb.shape}")
    else:
        print("Skipping embeddings (TF-IDF only).")

    print("Done ->", DATA_DIR)


if __name__ == "__main__":
    main()
