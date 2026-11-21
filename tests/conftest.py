"""Shared fixtures. Integration tests use the real artifacts in data/ when present,
and are skipped otherwise so the suite still runs on a fresh checkout."""
import copy

import joblib
import pytest
from scipy import sparse

from src.resources import DATA_DIR

RAW_SAMPLE = [
    {"id": 1, "name": "SupportAI", "one_liner": "AI chatbot for customer support",
     "long_description": "Automates helpdesk tickets with an AI chatbot.", "industry": "B2B",
     "industries": ["B2B"], "tags": ["AI", "Support"], "batch": "Winter 2024", "status": "Active",
     "website": "https://supportai.com", "all_locations": "San Francisco, CA, USA",
     "regions": ["United States of America", "America / Canada"], "team_size": 10, "stage": "Early",
     "launched_at": 1700000000, "url": "", "slug": "supportai", "subindustry": "B2B -> Ops"},
    {"id": 2, "name": "HelpBot", "one_liner": "Automated customer service chatbot",
     "long_description": "Resolves support tickets automatically for SaaS teams.", "industry": "B2B",
     "industries": ["B2B"], "tags": "AI, Support", "batch": "Summer 2023", "status": "active",
     "website": "helpbot.io", "all_locations": "New York, NY, USA", "regions": ["United States of America"],
     "team_size": 5, "stage": "Early", "launched_at": 1600000000, "url": "", "slug": "helpbot", "subindustry": ""},
    {"id": 3, "name": "ReefDrone", "one_liner": "Underwater drone for coral reef monitoring",
     "long_description": "Autonomous submarine that films reefs for marine biologists.", "industry": "Industrials",
     "industries": ["Industrials"], "tags": ["Robotics", "Ocean"], "batch": "Winter 2023", "status": "Acquired",
     "website": "https://reefdrone.com", "all_locations": "Sydney, Australia", "regions": ["Australia", "Oceania"],
     "team_size": 8, "stage": "Growth", "launched_at": 1650000000, "url": "", "slug": "reefdrone", "subindustry": ""},
    # empty description -> must be excluded
    {"id": 4, "name": "GhostCo", "one_liner": "", "long_description": "", "industry": "B2B",
     "industries": [], "tags": [], "batch": "Winter 2022", "status": "Inactive", "website": "",
     "all_locations": "", "regions": [], "team_size": None, "stage": "Early", "launched_at": 1500000000,
     "url": "", "slug": "ghostco", "subindustry": ""},
    # placeholder one-liner, no long desc -> excluded as empty
    {"id": 5, "name": "TBDStartup", "one_liner": "TBD", "long_description": "", "industry": "B2B",
     "industries": [], "tags": [], "batch": "Spring 2025", "status": "Active", "website": "",
     "all_locations": "", "regions": [], "team_size": 3, "stage": "Early", "launched_at": 1710000000,
     "url": "", "slug": "tbd", "subindustry": ""},
    # missing name -> excluded
    {"id": 6, "name": "   ", "one_liner": "Some product", "long_description": "Does things.",
     "industry": "B2B", "industries": [], "tags": [], "batch": "Winter 2021", "status": "Active",
     "website": "https://x.com", "all_locations": "", "regions": [], "team_size": 2, "stage": "Early",
     "launched_at": 1480000000, "url": "", "slug": "noname", "subindustry": ""},
    # duplicate id of 1 (same name/site) -> excluded as duplicate_id
    {"id": 1, "name": "SupportAI", "one_liner": "AI chatbot for customer support",
     "long_description": "Automates helpdesk tickets with an AI chatbot.", "industry": "B2B",
     "industries": ["B2B"], "tags": ["AI", "Support"], "batch": "Winter 2024", "status": "Active",
     "website": "https://supportai.com", "all_locations": "San Francisco, CA, USA",
     "regions": ["United States of America"], "team_size": 10, "stage": "Early",
     "launched_at": 1700000001, "url": "", "slug": "supportai-dup", "subindustry": ""},
    # probable duplicate of 1 (same name + same domain)
    {"id": 7, "name": "SupportAI", "one_liner": "AI support, resubmitted",
     "long_description": "Same company, new batch.", "industry": "B2B", "industries": ["B2B"],
     "tags": ["AI"], "batch": "Summer 2024", "status": "Active", "website": "https://supportai.com/home",
     "all_locations": "San Francisco, CA, USA", "regions": ["United States of America"], "team_size": 12,
     "stage": "Early", "launched_at": 1705000000, "url": "", "slug": "supportai-2", "subindustry": ""},
]


@pytest.fixture
def raw_sample():
    return copy.deepcopy(RAW_SAMPLE)


def _artifacts_present():
    return all((DATA_DIR / f).exists() for f in
               ["yc_companies.parquet", "tfidf_vectorizer.joblib", "tfidf_matrix.npz"])


@pytest.fixture
def artifacts():
    if not _artifacts_present():
        pytest.skip("real artifacts not built; run build_artifacts.py")
    from src.resources import load_data
    data = load_data()
    vec = joblib.load(DATA_DIR / "tfidf_vectorizer.joblib")
    mat = sparse.load_npz(DATA_DIR / "tfidf_matrix.npz")
    return data, vec, mat
