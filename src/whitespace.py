"""White-space / differentiation finder.

Looks at the tags of the direct competitors (crowded dimensions = what everyone does),
then compares to the broader neighbourhood (same industries) to surface tags that are
common nearby but rare among the direct competitors = underserved angles.
Requires a 'tag_list' column (raw tag lists) in the company table.
"""
from collections import Counter


def white_space(idea, data, embed_model, company_emb, n_competitors=40, top=8):
    if "tag_list" not in data.columns:
        return None, None

    q = embed_model.encode([idea], normalize_embeddings=True)
    sims = company_emb @ q[0]
    comp_idx = sims.argsort()[::-1][:n_competitors]

    comp_tags = Counter()
    for i in comp_idx:
        comp_tags.update(list(data.loc[i, "tag_list"]))

    main_inds = set(data.loc[comp_idx, "industry"].value_counts().index[:3])
    pool_mask = data["industry"].isin(main_inds)
    pool_tags = Counter()
    for tl in data.loc[pool_mask, "tag_list"]:
        pool_tags.update(list(tl))

    n_comp = len(comp_idx)
    n_pool = int(pool_mask.sum()) or 1
    comp_freq = {t: c / n_comp for t, c in comp_tags.items()}
    pool_freq = {t: c / n_pool for t, c in pool_tags.items()}

    dominant = [{"tag": t, "competitors": c} for t, c in comp_tags.most_common(top)]
    gaps = []
    for t, pf in pool_freq.items():
        cf = comp_freq.get(t, 0.0)
        if pf >= 0.04 and pf - cf > 0.0:
            gaps.append({"tag": t, "gap": round(pf - cf, 3),
                         "competitor_freq": round(cf, 3), "neighbourhood_freq": round(pf, 3)})
    gaps.sort(key=lambda x: -x["gap"])
    return dominant, gaps[:top]
