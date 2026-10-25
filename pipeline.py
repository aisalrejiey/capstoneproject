import hashlib
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

import db
import llm
import roles
from llm import log, setting


def run_analysis(progress=None):
    """Extract (sequential, guarded) -> cluster -> rank. Never loops, never retries blindly.
    Returns a dict summarising what happened."""
    out = {"done": 0, "failed": 0, "remaining": 0, "stopped": None,
           "first_error": None, "themes": None, "calls": 0, "empty": 0}
    llm.begin_run()
    max_attempts = setting("MAX_ATTEMPTS", 3)
    max_consec = setting("MAX_CONSECUTIVE_FAILURES", 3)
    rows = db.pending_feedback(max_attempts, setting("MAX_ITEMS_PER_RUN", 30))
    log.info("=== Analysis started: %d item(s) this run ===", len(rows))

    consecutive = 0
    for i, f in enumerate(rows):
        try:
            items = llm.extract_one(f["text"])
            db.save_requests(f["id"], items)
            out["done"] += 1
            out["empty"] += (not items)
            consecutive = 0
        except Exception as e:
            msg = llm.explain(e)
            out["first_error"] = out["first_error"] or msg
            log.error("Extraction failed for feedback #%s: %s", f["id"], msg, exc_info=True)
            if llm.is_fatal(e):                       # config/auth/quota/budget: stop immediately
                out["stopped"] = msg
                break
            db.mark_failed(f["id"], msg)
            out["failed"] += 1
            consecutive += 1
            if consecutive >= max_consec:             # circuit breaker
                out["stopped"] = f"Stopped after {consecutive} failures in a row. Last error: {msg}"
                break
        if progress:
            progress((i + 1) / len(rows))

    if out["stopped"] is None:
        try:
            out["themes"] = cluster_and_rank()
        except Exception as e:
            out["stopped"] = f"Clustering step failed: {type(e).__name__}: {e}"
            log.error("Clustering failed", exc_info=True)

    out["remaining"] = db.counts(max_attempts)["pending"]
    out["calls"] = llm.run_calls()
    log.info("=== Finished: %s ===", out)
    return out


# ---------- clustering ----------
def _vectors(reqs, thr):
    """Returns (reqs, matrix, effective_threshold)."""
    if llm.has_embeddings():
        model = llm.embed_model_name()
        stale = [r for r in reqs if r["embedding"] is None or r["emb_model"] != model]
        if stale:
            vecs = llm.embed([r["summary"] for r in stale])
            db.set_embeddings([(r["id"], v) for r, v in zip(stale, vecs)], model)
            reqs = db.all_requests()
        X = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in reqs])
        return reqs, X, thr
    # No embedding deployment: local TF-IDF (zero API calls). Looser threshold since it's lexical.
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(
        [r["summary"] for r in reqs]).toarray()
    return reqs, X, min(0.95, thr + 0.3)


def wsum(reqs):
    """Volume counted by who raised it (client counts more than developer)."""
    return sum(roles.weight(r["raised_by"]) for r in reqs)


def _age_days(r):
    """Days since the entry's date (never negative)."""
    return max((datetime.now() - datetime.fromisoformat(r["created_at"])).total_seconds() / 86400, 0.0)


def _facts(reqs):
    n = len(reqs)
    fin = [roles.final_urgency(r["urgency"], r["raised_by"], r["created_at"], n) for r in reqs]
    return {"urg": float(np.mean(fin)),                     # 0-5, includes role, age and repeat boosts
            "neg": int(sum(r["sentiment"] == "negative" for r in reqs)),
            "oldest": int(max(_age_days(r) for r in reqs)),
            "repeat": roles.repeat_boost(n)}


def priority(reqs, max_wsum):
    """0-100: weighted volume 40%, final urgency 40%, negativity 20%.
    Final urgency = model urgency x role weight x (1 + age boost + repeat boost), capped at 100%."""
    f = _facts(reqs)
    vol = wsum(reqs) / max_wsum
    return round(100 * (0.4 * vol + 0.4 * f["urg"] / 5 + 0.2 * f["neg"] / len(reqs)), 1)


def explain(reqs, score):
    """One plain-English sentence on why a theme ranks where it does (no API call)."""
    n, f = len(reqs), _facts(reqs)
    level = "High" if score >= 65 else "Medium" if score >= 40 else "Lower"
    cnt = Counter(roles.role_of(r["raised_by"]) or "unspecified" for r in reqs)
    top, top_n = cnt.most_common(1)[0]
    if n == 1:
        parts = [f"a single request from a {top}" if top != "unspecified" else "a single request"]
    elif top != "unspecified" and top_n / n >= 0.6:
        parts = [f"{n} requests, mostly from {top}s"]
    else:
        parts = [f"{n} requests from mixed raisers"]
    parts.append(f"avg urgency {roles.pct(f['urg']):.0f}%")
    if f["neg"]:
        parts.append(f"{f['neg']} of {n} negative" if n > 1 else "negative tone")
    if f["oldest"] >= 10:
        parts.append(f"oldest open {f['oldest']} days (+{roles.age_boost((datetime.now() - timedelta(days=f['oldest'])).isoformat())*100:.0f}% age boost)")
    if f["repeat"] > 0:
        parts.append(f"+{f['repeat']*100:.0f}% repeat boost")
    return f"{level} priority: " + ", ".join(parts) + "."


def cluster_and_rank(distance_threshold=0.45):
    reqs = db.all_requests()
    if not reqs:
        db.replace_themes([], {})          # nothing left -> clear stale themes
        return 0
    reqs, X, thr = _vectors(reqs, distance_threshold)
    if len(reqs) == 1:
        labels = np.array([0])
    else:
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=thr,
                                         metric="cosine", linkage="average").fit_predict(X)
    groups = {}
    for r, l in zip(reqs, labels):
        groups.setdefault(int(l), []).append(r)

    max_w = max(wsum(g) for g in groups.values())
    naming_left = setting("MAX_THEME_CALLS", 15)
    naming_ok = True                                  # flips to False after the first naming error
    themes, assign = [], {}
    for key, g in groups.items():
        summaries = [r["summary"] for r in g]
        sig = hashlib.sha1("|".join(sorted(summaries)).encode()).hexdigest()
        cached = db.cache_get(sig)
        if cached:
            name, desc = cached
        elif len(g) == 1:                              # single request: its summary IS the name (no call)
            name, desc = summaries[0][:60], ""
        elif naming_ok and naming_left > 0:
            naming_left -= 1
            try:
                name, desc = llm.name_theme(summaries)
                db.cache_put(sig, name, desc)
            except Exception as e:
                log.error("Theme naming failed, using fallback names for the rest: %s", e)
                naming_ok = False
                name, desc = summaries[0][:60], ""
        else:
            name, desc = summaries[0][:60], ""
        score = priority(g, max_w)
        themes.append({"key": key, "name": name, "description": desc, "n_requests": len(g),
                       "score": score, "reason": explain(g, score)})
        for r in g:
            assign[r["id"]] = key
    db.replace_themes(themes, assign)
    for t in sorted(themes, key=lambda t: -t["score"]):
        log.info("THEME score=%s n=%d  %s", t["score"], t["n_requests"], t["name"])
    return len(themes)


def refresh_theme_stats():
    """After deleting entries: update counts/scores of existing themes from what is left
    (no API calls). Empty themes disappear. A full re-cluster happens on the next Run analysis."""
    groups = {}
    for r in db.all_requests():
        if r["theme_id"] is not None:
            groups.setdefault(r["theme_id"], []).append(r)
    if not groups:
        db.update_theme_stats({})
        return
    mx = max(wsum(g) for g in groups.values())
    stats = {}
    for tid, g in groups.items():
        score = priority(g, mx)
        stats[tid] = (len(g), score, explain(g, score))
    db.update_theme_stats(stats)
