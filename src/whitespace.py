"""White-space / differentiation finder.

Crowded dimensions = tags shared by the direct competitors.
White space = tags common in the broader neighbourhood (same industries) but rare
among the direct competitors - underserved adjacent angles. Heuristic, descriptive.
"""
from collections import Counter

from .retrieval import similarities

MIN_NEIGHBOURHOOD_FREQ = 0.04


def white_space(idea, data, vectorizer, tfidf_matrix, embed_model=None, company_emb=None,
                n_competitors=40, top=8):
    if "tag_list" not in data.columns:
        return None, None

    sims, _ = similarities(idea, vectorizer, tfidf_matrix, embed_model, company_emb)
    comp_idx = sims.argsort()[::-1][:n_competitors]

    comp_tags = Counter()
    for i in comp_idx:
        comp_tags.update(list(data.iloc[i]["tag_list"]))

    main_inds = set(data.iloc[comp_idx]["industry"].value_counts().index[:3])
    pool = data[data["industry"].isin(main_inds)]
    pool_tags = Counter()
    for tl in pool["tag_list"]:
        pool_tags.update(list(tl))

    n_comp = len(comp_idx)
    n_pool = max(len(pool), 1)
    comp_freq = {t: c / n_comp for t, c in comp_tags.items()}
    pool_freq = {t: c / n_pool for t, c in pool_tags.items()}

    dominant = [{"tag": t, "competitors": c} for t, c in comp_tags.most_common(top)]
    gaps = []
    for t, pf in pool_freq.items():
        cf = comp_freq.get(t, 0.0)
        if pf >= MIN_NEIGHBOURHOOD_FREQ and pf - cf > 0.0:
            gaps.append({"tag": t, "gap": round(pf - cf, 3),
                         "competitor_freq": round(cf, 3), "neighbourhood_freq": round(pf, 3)})
    gaps.sort(key=lambda x: -x["gap"])
    return dominant, gaps[:top]
