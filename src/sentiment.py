"""Multi-source sentiment + the Opportunity Score.

Sources (all public, no PII):
  - Hacker News (Algolia API)  - tech-community voice, no auth
  - GDELT DOC 2.0 API          - worldwide news, 65+ languages, no auth
  - Product Hunt (optional)    - early-adopter voice, needs a free token

VADER scores English text only; GDELT non-English articles are kept for
global *coverage* but excluded from sentiment (multilingual sentiment = future work).
"""
import os
import re
import time
from functools import lru_cache

import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .preprocess import clean
from .saturation import market_saturation

_vader = SentimentIntensityAnalyzer()

OPP_W_GAP, OPP_W_DISSAT = 0.5, 0.5
PRODUCTHUNT_TOKEN = os.getenv("PRODUCTHUNT_TOKEN", "")

MENTION_COLS = ["source", "text", "lang", "country", "url"]


def score_sentiment(text: str) -> float:
    return _vader.polarity_scores(text)["compound"]


# ------------------------- fetchers -------------------------

def fetch_hackernews(topic, limit=50):
    url = "https://hn.algolia.com/api/v1/search"
    params = {"query": topic, "tags": "(story,comment)", "hitsPerPage": limit}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        hits = r.json().get("hits", [])
    except Exception as e:
        print("HN fetch failed:", e)
        return pd.DataFrame(columns=MENTION_COLS)
    rows = []
    for h in hits:
        text = h.get("title") or h.get("comment_text") or h.get("story_title") or ""
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            rows.append({"source": "Hacker News", "text": text, "lang": "en",
                         "country": "", "url": f"https://news.ycombinator.com/item?id={h.get('objectID')}"})
    return pd.DataFrame(rows, columns=MENTION_COLS)


def fetch_gdelt(topic, limit=75, timespan="3m", retries=3):
    """Robust GDELT fetch: GDELT throttles rapid repeats with an EMPTY body,
    which breaks json(). We retry with backoff and only parse real JSON."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": topic, "mode": "ArtList", "format": "json",
              "maxrecords": limit, "timespan": timespan, "sort": "datedesc"}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30,
                             headers={"User-Agent": "startup-intel/1.0"})
            body = r.text.strip()
            if r.status_code == 200 and body.startswith("{"):
                arts = r.json().get("articles", [])
                rows = []
                for a in arts:
                    t = (a.get("title") or "").strip()
                    if t:
                        rows.append({"source": "GDELT News", "text": t,
                                     "lang": a.get("language", "Unknown"),
                                     "country": a.get("sourcecountry", ""),
                                     "url": a.get("url", "")})
                return pd.DataFrame(rows, columns=MENTION_COLS)
        except Exception as e:
            print(f"GDELT attempt {attempt + 1} failed:", e)
        time.sleep(1.5 * (attempt + 1))  # back off; lets throttling clear
    print("GDELT returned no data after retries.")
    return pd.DataFrame(columns=MENTION_COLS)


def fetch_producthunt(topic, first=50):
    if not PRODUCTHUNT_TOKEN:
        return pd.DataFrame(columns=MENTION_COLS)
    url = "https://api.producthunt.com/v2/api/graphql"
    headers = {"Authorization": f"Bearer {PRODUCTHUNT_TOKEN}", "Content-Type": "application/json"}
    gql = {"query": "query{ posts(first: 50, order: VOTES){ edges{ node{ name tagline } } } }"}
    try:
        r = requests.post(url, json=gql, headers=headers, timeout=30)
        edges = r.json()["data"]["posts"]["edges"]
    except Exception as e:
        print("Product Hunt fetch failed:", e)
        return pd.DataFrame(columns=MENTION_COLS)
    kws = [w for w in clean(topic).split() if len(w) > 2]
    rows = []
    for e in edges:
        n = e["node"]
        text = f"{n['name']}. {n['tagline']}"
        if any(k in text.lower() for k in kws):
            rows.append({"source": "Product Hunt", "text": text, "lang": "en",
                         "country": "", "url": "https://www.producthunt.com"})
    return pd.DataFrame(rows, columns=MENTION_COLS)


# ------------------------- aggregate -------------------------

@lru_cache(maxsize=128)
def _gather_cached(topic):
    """Cached per topic so GDELT/HN are hit once per topic per session."""
    frames = [fetch_hackernews(topic), fetch_gdelt(topic), fetch_producthunt(topic)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=MENTION_COLS)
    return pd.concat(frames, ignore_index=True)


def gather_mentions(topic):
    return _gather_cached(topic).copy()


def analyze_sentiment(mentions):
    eng = mentions[mentions["lang"].str.lower().isin(["en", "english"]) |
                   mentions["source"].isin(["Hacker News", "Product Hunt"])].copy()
    if eng.empty:
        return eng, {}
    eng["compound"] = eng["text"].apply(score_sentiment)
    eng["sentiment"] = eng["compound"].apply(
        lambda c: "Positive" if c >= 0.05 else ("Negative" if c <= -0.05 else "Neutral"))
    summary = {
        "scored": len(eng),
        "avg_compound": round(float(eng["compound"].mean()), 3),
        "pos_frac": round(float((eng["sentiment"] == "Positive").mean()), 3),
        "neg_frac": round(float((eng["sentiment"] == "Negative").mean()), 3),
        "neu_frac": round(float((eng["sentiment"] == "Neutral").mean()), 3),
    }
    return eng, summary


def opportunity_score(idea, data, vectorizer, tfidf_matrix, topic=None):
    topic = topic or idea
    sat = market_saturation(idea, data, vectorizer, tfidf_matrix)
    mentions = gather_mentions(topic)
    scored, summ = analyze_sentiment(mentions)

    sat_val = sat["saturation_score"]
    gap = 1 - sat_val / 100
    dissat = (1 - summ["avg_compound"]) / 2 if summ else 0.5
    opp = round(100 * (OPP_W_GAP * gap + OPP_W_DISSAT * dissat), 1)

    crowded = sat_val >= 55
    negative = bool(summ) and summ["avg_compound"] < 0.05
    verdict = ("Disruption opportunity - crowded but frustrated users" if crowded and negative else
               "Tough market - crowded with satisfied users" if crowded else
               "Open & underwhelming - space exists, users unhappy" if negative else
               "Greenfield - open space, unproven demand")

    return {
        "idea": idea, "topic": topic,
        "saturation": sat_val, "saturation_label": sat["label"],
        "avg_sentiment": summ.get("avg_compound") if summ else None,
        "mentions_scored": summ.get("scored", 0) if summ else 0,
        "opportunity_score": opp, "verdict": verdict,
        "sat_detail": sat, "mentions": scored, "summary": summ,
    }
