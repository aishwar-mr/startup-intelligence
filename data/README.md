# data/

Drop the four artifacts built by the Colab notebook (section 6 + 8) here:

| File | Built in | Size |
|------|----------|------|
| `yc_companies.parquet` | Phase 0/1 | ~few MB |
| `tfidf_vectorizer.joblib` | Phase 1 | ~1-2 MB |
| `tfidf_matrix.npz` | Phase 1 | ~few MB |
| `company_embeddings.npy` | Phase 3 | ~10 MB |

Or regenerate them locally: `python build_artifacts.py`
