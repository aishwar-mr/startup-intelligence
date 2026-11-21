"""Data validation and cleaning for the YC startup dataset.

Principles
- Raw records are never modified. Validation works on copies and returns a
  cleaned table, a flag log, an exclusion log and a summary report.
- Every exclusion, flag and normalisation follows a documented rule (the
  *_RULES dictionaries below are the single source of truth for the docs/UI).
- Output is deterministic: records are processed in id order and the report
  carries a SHA-256 fingerprint of the raw input, so the same input always
  produces the same summary.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from .preprocess import clean, normalise_text

RULES_VERSION = "1.0"

# ---------------------------------------------------------------- rules ----

EXCLUSION_RULES = {
    "missing_id": "Record has no usable numeric id, so it cannot be tracked or de-duplicated.",
    "duplicate_id": "Same id as an earlier record; only the first occurrence is kept.",
    "missing_name": "Startup name is blank after normalisation.",
    "empty_description": "One-liner and long description are both blank (or only placeholder "
                         "text such as 'TBD', '-', '.') after normalisation, so NLP has nothing to match.",
    "probable_duplicate": "Same normalised name AND (same website domain OR identical normalised "
                          "description) as a record with a lower id; the lower id is kept.",
}

FLAG_RULES = {
    "short_description": "Combined description is under 25 characters; kept, but matching signal is weak.",
    "description_equals_name": "Description only repeats the startup name; kept, low signal.",
    "placeholder_text": "A text field held placeholder text (e.g. 'TBD', '-', '.') and was treated as blank.",
    "missing_tags": "No tags; white-space analysis cannot use this record.",
    "missing_website": "No website.",
    "missing_location": "No location (blank all_locations and no specific region).",
    "missing_team_size": "Team size not reported.",
    "team_size_zero": "Team size reported as 0 (often inactive companies).",
    "invalid_team_size": "Team size was negative or non-numeric and was set to blank.",
    "unspecified_batch": "Batch is 'Unspecified', so no cohort year.",
    "invalid_batch": "Batch does not match '<Season> <YYYY>' and no cohort year could be read.",
    "unspecified_industry": "Industry is 'Unspecified' or blank.",
    "invalid_status": "Status is not one of Active, Inactive, Acquired, Public.",
    "type_coerced": "A field had an unexpected type and was converted (e.g. text tags split into a list).",
    "name_shared_with_other_company": "Another kept record has the same name but a different website/description; "
                                      "treated as a different company, not a duplicate.",
    "website_shared_with_other_company": "Another record with a different name uses the same website domain; "
                                         "the website may be wrong.",
    "website_normalised": "Website lacked 'http(s)://' and the scheme was added in the cleaned copy.",
}

NORMALISATION_RULES = {
    "whitespace": "Trim and collapse repeated spaces/newlines in all text fields.",
    "html": "Strip HTML tags and decode HTML entities in text fields.",
    "unicode": "Unicode NFKC normalisation.",
    "status_case": "Map status to title case (e.g. 'active' -> 'Active').",
    "lists": "Tags/industries/regions: normalise each item, drop blanks and repeats.",
    "website_scheme": "Add 'http://' to websites missing a scheme.",
    "sort": "Order records by id so output does not depend on API ordering.",
}

IMPORTANT_FIELDS = ["name", "one_liner", "long_description", "industry", "tags", "batch",
                    "status", "website", "all_locations", "regions", "team_size", "stage"]

# expected python types for present (non-None) values
EXPECTED_TYPES = {
    "id": int, "name": str, "slug": str, "one_liner": str, "long_description": str,
    "industry": str, "subindustry": str, "industries": list, "tags": list, "batch": str,
    "status": str, "website": str, "all_locations": str, "regions": list,
    "team_size": int, "stage": str, "launched_at": int, "url": str,
}

VALID_STATUS = ["Active", "Inactive", "Acquired", "Public"]
BATCH_RE = re.compile(r"^(Winter|Spring|Summer|Fall) (\d{4})$")
PLACEHOLDER_RE = re.compile(
    r"^(n/?a|na|tbd|tba|none|null|nil|-+|\.+|\?+|coming soon|todo|test|placeholder)$", re.I)
SHORT_DESCRIPTION_CHARS = 25

MACRO_REGIONS = {"America / Canada", "Europe", "South Asia", "Latin America", "Southeast Asia",
                 "Middle East and North Africa", "Africa", "East Asia", "Oceania", "Asia"}
REMOTE_REGIONS = ["Fully Remote", "Partly Remote", "Remote"]  # most specific first
UNSPECIFIED = "Unspecified"

CLEAN_COLUMNS = ["id", "slug", "name", "one_liner", "long_description", "industry", "subindustry",
                 "industries", "tags", "tag_list", "batch", "batch_year", "status", "stage",
                 "website", "domain", "yc_url", "all_locations", "country", "macro_region",
                 "remote_policy", "team_size", "flags"]


@dataclass
class ValidationResult:
    clean: pd.DataFrame        # records that passed, one row per startup, sorted by id
    flags: pd.DataFrame        # long table: id, name, flag, detail
    exclusions: pd.DataFrame   # id, name, reason, detail
    report: dict               # summary counts (deterministic)


# -------------------------------------------------------------- helpers ----

def _is_blank(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip()) or \
        (isinstance(v, (list, tuple, dict)) and len(v) == 0)


def _type_ok(v, expected) -> bool:
    if v is None:
        return True
    if expected is int:
        return isinstance(v, int) and not isinstance(v, bool)
    return isinstance(v, expected)


def _domain(url: str) -> str:
    u = (url or "").strip().lower()
    u = re.sub(r"^[a-z]+://", "", u)
    u = re.sub(r"^www\.", "", u)
    return u.split("/")[0].split("?")[0].split("#")[0]


def fingerprint(records) -> str:
    """SHA-256 of the raw records, independent of their order (and robust to
    duplicate ids): each record is canonicalised to a JSON string, the strings
    are sorted, then hashed together."""
    canon = [json.dumps(r, sort_keys=True, ensure_ascii=False, default=str) for r in records]
    canon.sort()
    h = hashlib.sha256()
    for c in canon:
        h.update(c.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _norm_list(v, actions, flags):
    if v is None:
        return []
    if isinstance(v, str):  # e.g. "AI, SaaS" instead of ["AI", "SaaS"]
        flags.append(("type_coerced", "list field given as text"))
        v = [p for p in v.split(",")]
    elif not isinstance(v, (list, tuple)):
        flags.append(("type_coerced", f"list field given as {type(v).__name__}"))
        v = [v]
    out, seen = [], set()
    for item in v:
        t = normalise_text(item)
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    if len(out) != len(v):
        actions["lists"] += 1
    return out


def _norm_text(v, actions):
    if v is None:
        return ""
    s = str(v)
    t = normalise_text(s)
    if t != s:
        if re.search(r"<[^>]+>|&[a-z#0-9]+;", s, re.I):
            actions["html"] += 1
        if t != s.strip() or s != s.strip():
            actions["whitespace"] += 1
    return t


def _parse_regions(regions):
    countries = [r for r in regions if r not in MACRO_REGIONS and r not in REMOTE_REGIONS and r != UNSPECIFIED]
    macro = next((r for r in regions if r in MACRO_REGIONS), UNSPECIFIED)
    remote = next((r for r in REMOTE_REGIONS if r in regions), "")
    return (countries[0] if countries else UNSPECIFIED), macro, remote


# ----------------------------------------------------------- validation ----

def validate(records) -> ValidationResult:
    """Validate and clean a list of raw YC company dicts. The input is not modified."""
    actions = Counter()
    missing = Counter({f: 0 for f in IMPORTANT_FIELDS})
    placeholders = Counter()
    type_issues = Counter()
    exclusions, flag_rows = [], []

    # ---- 1. profile raw values (missing + type checks) on the untouched input
    for r in records:
        if not isinstance(r, dict):
            type_issues["<record>"] += 1
            continue
        for f in IMPORTANT_FIELDS:
            if _is_blank(r.get(f)):
                missing[f] += 1
        for f, t in EXPECTED_TYPES.items():
            if not _type_ok(r.get(f), t):
                type_issues[f] += 1

    # ---- 2. order by id (deterministic), drop records without a usable id
    staged = []
    for r in records:
        if not isinstance(r, dict):
            exclusions.append({"id": None, "name": "", "reason": "missing_id", "detail": "record is not an object"})
            continue
        rid = r.get("id")
        if isinstance(rid, str) and rid.strip().isdigit():
            rid = int(rid.strip())
        if not (isinstance(rid, int) and not isinstance(rid, bool)):
            exclusions.append({"id": None, "name": normalise_text(r.get("name")), "reason": "missing_id",
                               "detail": f"id={r.get('id')!r}"})
            continue
        staged.append((rid, r))
    # sort by id, then by canonical content so duplicate-id resolution does not
    # depend on the input ordering (keeps the run fully deterministic)
    staged.sort(key=lambda x: (x[0], json.dumps(x[1], sort_keys=True, ensure_ascii=False, default=str)))
    actions["sort"] = 1

    # ---- 3. normalise each record into a cleaned copy
    seen_ids, rows = set(), []
    for rid, r in staged:
        rec_flags = []
        if rid in seen_ids:
            exclusions.append({"id": rid, "name": normalise_text(r.get("name")), "reason": "duplicate_id",
                               "detail": "id already seen"})
            continue
        seen_ids.add(rid)

        name = _norm_text(r.get("name"), actions)
        one_liner = _norm_text(r.get("one_liner"), actions)
        long_desc = _norm_text(r.get("long_description"), actions)
        for fld, val in (("one_liner", one_liner), ("long_description", long_desc)):
            if val and PLACEHOLDER_RE.match(val):
                placeholders[fld] += 1
                rec_flags.append(("placeholder_text", f"{fld}={val!r}"))
        if one_liner and PLACEHOLDER_RE.match(one_liner):
            one_liner = ""
        if long_desc and PLACEHOLDER_RE.match(long_desc):
            long_desc = ""

        if not name:
            exclusions.append({"id": rid, "name": "", "reason": "missing_name", "detail": ""})
            continue
        combined = f"{one_liner} {long_desc}".strip()
        if not combined:
            had_ph = any(f == "placeholder_text" for f, _ in rec_flags)
            exclusions.append({"id": rid, "name": name, "reason": "empty_description",
                               "detail": "placeholder text only" if had_ph else "both descriptions blank"})
            continue

        status_raw = _norm_text(r.get("status"), actions)
        status = status_raw.title() if status_raw.lower() in {s.lower() for s in VALID_STATUS} else status_raw
        if status != status_raw:
            actions["status_case"] += 1
        if status not in VALID_STATUS:
            rec_flags.append(("invalid_status", f"status={status_raw!r}"))

        batch = _norm_text(r.get("batch"), actions)
        m = BATCH_RE.match(batch)
        batch_year = int(m.group(2)) if m else None
        if batch == UNSPECIFIED:
            rec_flags.append(("unspecified_batch", ""))
        elif not m:
            rec_flags.append(("invalid_batch", f"batch={batch!r}"))

        industry = _norm_text(r.get("industry"), actions) or UNSPECIFIED
        if industry == UNSPECIFIED:
            rec_flags.append(("unspecified_industry", ""))

        tags = _norm_list(r.get("tags"), actions, rec_flags)
        industries = _norm_list(r.get("industries"), actions, rec_flags)
        regions = _norm_list(r.get("regions"), actions, rec_flags)
        if not tags:
            rec_flags.append(("missing_tags", ""))

        website = _norm_text(r.get("website"), actions)
        if website and not re.match(r"^[a-z]+://", website, re.I):
            website = "http://" + website
            actions["website_scheme"] += 1
            rec_flags.append(("website_normalised", ""))
        if not website:
            rec_flags.append(("missing_website", ""))

        all_locations = _norm_text(r.get("all_locations"), actions)
        country, macro, remote = _parse_regions(regions)
        if not all_locations and country == UNSPECIFIED and macro == UNSPECIFIED:
            rec_flags.append(("missing_location", ""))

        ts = r.get("team_size")
        if isinstance(ts, str) and ts.strip().isdigit():
            ts = int(ts.strip())
            rec_flags.append(("type_coerced", "team_size given as text"))
        if ts is None:
            rec_flags.append(("missing_team_size", ""))
        elif isinstance(ts, bool) or not isinstance(ts, int) or ts < 0:
            rec_flags.append(("invalid_team_size", f"team_size={r.get('team_size')!r}"))
            ts = None
        elif ts == 0:
            rec_flags.append(("team_size_zero", ""))

        if len(combined) < SHORT_DESCRIPTION_CHARS:
            rec_flags.append(("short_description", f"{len(combined)} chars"))
        if clean(combined) == clean(name):
            rec_flags.append(("description_equals_name", ""))

        slug = _norm_text(r.get("slug"), actions)
        yc_url = _norm_text(r.get("url"), actions) or (f"https://www.ycombinator.com/companies/{slug}" if slug else "")

        rows.append({
            "id": rid, "slug": slug, "name": name, "one_liner": one_liner, "long_description": long_desc,
            "industry": industry, "subindustry": _norm_text(r.get("subindustry"), actions),
            "industries": " ".join(industries), "tags": " ".join(tags), "tag_list": tags,
            "batch": batch, "batch_year": batch_year, "status": status,
            "stage": _norm_text(r.get("stage"), actions), "website": website, "domain": _domain(website),
            "yc_url": yc_url, "all_locations": all_locations, "country": country, "macro_region": macro,
            "remote_policy": remote, "team_size": ts, "_flags": rec_flags,
            "_norm_name": clean(name), "_norm_desc": clean(combined),
        })

    # ---- 4. probable duplicates (same name AND same domain or same description)
    keep = []
    first_by_key = {}
    for row in rows:  # rows already in id order, so the lowest id wins
        keys = []
        if row["domain"]:
            keys.append(("site", row["_norm_name"], row["domain"]))
        keys.append(("desc", row["_norm_name"], row["_norm_desc"]))
        hit = next((first_by_key[k] for k in keys if k in first_by_key), None)
        if hit is not None:
            exclusions.append({"id": row["id"], "name": row["name"], "reason": "probable_duplicate",
                               "detail": f"matches id {hit}"})
            continue
        for k in keys:
            first_by_key[k] = row["id"]
        keep.append(row)

    # ---- 5. cross-record flags on kept rows (shared names / shared websites)
    by_name, by_domain = defaultdict(list), defaultdict(list)
    for row in keep:
        by_name[row["_norm_name"]].append(row["id"])
        if row["domain"]:
            by_domain[row["domain"]].append(row)
    name_groups = {k: v for k, v in by_name.items() if len(v) > 1}
    for row in keep:
        ids = name_groups.get(row["_norm_name"])
        if ids:
            others = [i for i in ids if i != row["id"]]
            row["_flags"].append(("name_shared_with_other_company", "other ids " + ",".join(map(str, others))))
        same_site = [o for o in by_domain.get(row["domain"], []) if o["_norm_name"] != row["_norm_name"]]
        if same_site:
            row["_flags"].append(("website_shared_with_other_company",
                                  f"{row['domain']} also used by id " + ",".join(str(o["id"]) for o in same_site)))

    # ---- 6. assemble outputs
    for row in keep:
        for flag, detail in row["_flags"]:
            flag_rows.append({"id": row["id"], "name": row["name"], "flag": flag, "detail": detail})
        row["flags"] = ";".join(sorted({f for f, _ in row["_flags"]}))
    clean_df = pd.DataFrame([{c: row[c] for c in CLEAN_COLUMNS} for row in keep], columns=CLEAN_COLUMNS)
    if not clean_df.empty:
        clean_df["batch_year"] = clean_df["batch_year"].astype("Int64")
        clean_df["team_size"] = clean_df["team_size"].astype("Int64")
    flags_df = pd.DataFrame(flag_rows, columns=["id", "name", "flag", "detail"])
    excl_df = pd.DataFrame(exclusions, columns=["id", "name", "reason", "detail"])
    if not excl_df.empty:
        excl_df = excl_df.sort_values(["reason", "id"], na_position="first", kind="stable").reset_index(drop=True)

    flag_counts = flags_df.groupby("flag")["id"].nunique() if not flags_df.empty else pd.Series(dtype=int)
    excl_counts = excl_df["reason"].value_counts() if not excl_df.empty else pd.Series(dtype=int)

    def examples(df, col, val, n=5):
        sub = df[df[col] == val].sort_values("id", na_position="first", kind="stable").head(n)
        return [{"id": (None if pd.isna(i) else int(i)), "name": nm, "detail": dt}
                for i, nm, dt in zip(sub["id"], sub["name"], sub["detail"])]

    report = {
        "rules_version": RULES_VERSION,
        "input_fingerprint": fingerprint([r for r in records if isinstance(r, dict)]),
        "total_received": len(records),
        "passed": int(len(clean_df)),
        "excluded": int(len(excl_df)),
        "excluded_by_reason": {k: int(excl_counts.get(k, 0)) for k in EXCLUSION_RULES},
        "duplicates": {
            "duplicate_id": int(excl_counts.get("duplicate_id", 0)),
            "probable_duplicate": int(excl_counts.get("probable_duplicate", 0)),
            "name_shared_groups": len(name_groups),
            "name_shared_records": int(flag_counts.get("name_shared_with_other_company", 0)),
            "website_shared_records": int(flag_counts.get("website_shared_with_other_company", 0)),
        },
        "missing_by_field": {f: int(missing[f]) for f in IMPORTANT_FIELDS},
        "placeholder_by_field": dict(sorted(placeholders.items())),
        "type_issues_by_field": dict(sorted(type_issues.items())),
        "flags_by_type": {k: int(flag_counts.get(k, 0)) for k in FLAG_RULES},
        "normalisation_actions": {k: int(actions.get(k, 0)) for k in NORMALISATION_RULES},
        "status_counts": {k: int(v) for k, v in sorted(Counter(clean_df["status"]).items())} if len(clean_df) else {},
        "examples": {
            **{f"excluded:{k}": examples(excl_df, "reason", k) for k in EXCLUSION_RULES if excl_counts.get(k, 0)},
            **{f"flag:{k}": examples(flags_df, "flag", k) for k in FLAG_RULES if flag_counts.get(k, 0)},
        },
    }
    return ValidationResult(clean=clean_df, flags=flags_df, exclusions=excl_df, report=report)


def summary_rows(report: dict) -> pd.DataFrame:
    """Flat before/after table for docs and the UI."""
    d = report["duplicates"]
    rows = [
        ("Records received", report["total_received"]),
        ("Records passing validation", report["passed"]),
        ("Records excluded", report["excluded"]),
        ("Duplicate ids", d["duplicate_id"]),
        ("Probable duplicates (name + website/description)", d["probable_duplicate"]),
        ("Names shared by different companies (records)", d["name_shared_records"]),
        ("Websites shared by different companies (records)", d["website_shared_records"]),
    ]
    return pd.DataFrame(rows, columns=["Check", "Count"])
