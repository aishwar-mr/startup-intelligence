"""Market-overview analytics computed from the cleaned, validated table.

All figures are descriptive counts/shares over the available Y Combinator sample.
The category 'momentum' and 'opportunity hint' are heuristic indicators for
prioritising research, explained in INDICATOR_EXPLANATION - not market facts.
"""
import numpy as np
import pandas as pd

RECENT_YEARS = 3          # a batch year within this many years of the latest counts as "recent"
MIN_CATEGORY_SIZE = 15    # categories smaller than this are not scored (too few to be meaningful)

INDICATOR_EXPLANATION = {
    "companies": "Number of validated companies in the category (its competitor density).",
    "active_pct": "Share of the category's companies whose status is Active or Public.",
    "momentum_pct": f"Share of the category's companies from the last {RECENT_YEARS} batch years "
                    "(a proxy for whether new founders are still entering the space).",
    "density_pct": "The category's company count as a percentage of the largest category "
                   "(100 = the most crowded category in the sample).",
    "hint": "A heuristic label, not advice: 'Hot & crowded' = high density and high momentum; "
            "'Emerging' = high momentum but lower density; 'Mature/cooling' = high density, low momentum; "
            "'Quiet' = low density and low momentum. Based on medians across categories.",
    "limitations": [
        "Covers only Y Combinator companies; absolute counts understate each market.",
        "Momentum uses batch year, which is accelerator-admission timing, not company founding.",
        "Categories are YC's own industry labels and can be broad.",
    ],
}


def market_overview(data: pd.DataFrame) -> dict:
    """Return the headline distributions for the Market Overview screen."""
    latest_year = int(data["batch_year"].dropna().max()) if data["batch_year"].notna().any() else None
    by_industry = data["industry"].value_counts()
    by_status = data["status"].value_counts()
    by_stage = data["stage"].value_counts()
    by_region = data["macro_region"].replace("Unspecified", "Unspecified").value_counts()
    by_country = data.loc[data["country"] != "Unspecified", "country"].value_counts()
    by_year = data["batch_year"].dropna().astype(int).value_counts().sort_index()
    remote = data.loc[data["remote_policy"] != "", "remote_policy"].value_counts()
    return {
        "total": int(len(data)),
        "active_pct": round(100 * by_status.get("Active", 0) / max(len(data), 1), 1),
        "industries": int(data["industry"].nunique()),
        "countries": int((data["country"] != "Unspecified").sum() and data["country"].nunique() - 1),
        "latest_batch_year": latest_year,
        "by_industry": by_industry,
        "by_status": by_status,
        "by_stage": by_stage,
        "by_region": by_region,
        "by_country": by_country,
        "by_year": by_year,
        "by_remote": remote,
    }


def category_table(data: pd.DataFrame, recent_years: int = RECENT_YEARS,
                   min_size: int = MIN_CATEGORY_SIZE) -> pd.DataFrame:
    """Per-industry density / activity / momentum with a heuristic opportunity hint."""
    years = data["batch_year"].dropna()
    latest = int(years.max()) if not years.empty else None
    rows = []
    for industry, grp in data.groupby("industry"):
        n = len(grp)
        if n < min_size or industry == "Unspecified":
            continue
        active = grp["status"].isin(["Active", "Public"]).mean()
        gy = grp["batch_year"].dropna()
        momentum = (gy >= (latest - recent_years + 1)).mean() if latest is not None and len(gy) else np.nan
        rows.append({"industry": industry, "companies": n,
                     "active_pct": round(100 * active, 1),
                     "momentum_pct": round(100 * momentum, 1) if momentum == momentum else None})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["density_pct"] = (100 * df["companies"] / df["companies"].max()).round(1)
    dens_med = df["density_pct"].median()
    mom_med = df["momentum_pct"].dropna().median() if df["momentum_pct"].notna().any() else 0

    def hint(r):
        if r["momentum_pct"] is None:
            return "Unknown"
        hot = r["momentum_pct"] >= mom_med
        crowded = r["density_pct"] >= dens_med
        return ("Hot & crowded" if hot and crowded else
                "Emerging" if hot and not crowded else
                "Mature / cooling" if crowded and not hot else "Quiet")

    df["hint"] = df.apply(hint, axis=1)
    return df.sort_values("companies", ascending=False).reset_index(drop=True)
