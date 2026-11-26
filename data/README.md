# data/

Built by `python build_artifacts.py` or the Colab notebook. Commit these so Streamlit Cloud can load them.

| File | What |
|------|------|
| `yc_companies_raw.json` | Raw YC snapshot (may be older than the built files; usable as input with `--raw`) |
| `yc_companies.parquet` | Cleaned, validated company table |
| `tfidf_vectorizer.joblib` | Fitted TF-IDF vectoriser |
| `tfidf_matrix.npz` | Sparse TF-IDF matrix |
| `company_embeddings.npy` | MiniLM embeddings (optional; app falls back to TF-IDF without it) |
| `validation_report.json` | Deterministic data-quality report (shown on the Data Quality screen) |
| `validation_flags.csv` / `validation_exclusions.csv` | Per-record logs |

Rebuild everything: `python build_artifacts.py` (add `--no-embeddings` for a TF-IDF-only build).
