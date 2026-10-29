"""RAG layer: the tracker's database used as a searchable knowledge base for Aira.

INDEX  (table rag_chunks, inside the same voc.db)
  Every theme, every request (with its original feedback text, raiser, status and dates), every feedback entry that is
  still waiting for analysis and every section of the uploaded company policy document (policy.py) becomes one short
  "chunk" of plain text. A chunk is re-written only when its text changes
  (status change, new score ...), detected with a hash, and its embedding is created only then (incremental, cheap).

RETRIEVE  (per question)
  1. keyword search: local TF-IDF, no API call, always on.
  2. semantic search: only if AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME is set: the question is embedded (1 call) and compared
     with the stored chunk vectors.
  The two rankings are fused (reciprocal-rank fusion) and the top-k chunks go to the model together with a ranked overview.

Everything uses the existing budget guard (llm._spend), so MAX_CALLS_PER_RUN / MAX_CALLS_PER_DAY still apply.
Optional .env settings:  AIRA_RAG_TOP_K=8   AIRA_RAG_MAX_EMBED_BATCHES=5 (64 chunks per batch, per question)
                         AIRA_RAG_REBUILD_MAX_BATCHES=20
"""
import hashlib
from datetime import datetime
from functools import lru_cache

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

import db
import llm
from llm import log, setting

LEX_MIN_SCORE = 0.04        # keyword matches weaker than this are ignored
COVERAGE_MIN = 0.4          # a keyword hit must contain at least this share of the question's (rarity-weighted) words
RRF_K = 60                  # reciprocal-rank-fusion constant


def _hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _clip(text, n=400):
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def _day(ts):
    return (ts or "")[:10]


# ---------------------------------------------------------------- 1. database -> chunks
def build_chunks():
    """Turns the current database into chunks: [{chunk_id, kind, ref_id, title, text}]. Read-only."""
    max_attempts = setting("MAX_ATTEMPTS", 3)
    chunks = []
    with db.conn() as c:
        themes = c.execute("SELECT * FROM themes ORDER BY score DESC").fetchall()
        by_status = {}
        for r in c.execute("""SELECT theme_id, status, COUNT(*) AS n FROM requests
                              WHERE theme_id IS NOT NULL GROUP BY theme_id, status"""):
            by_status.setdefault(r["theme_id"], {})[r["status"]] = r["n"]
        names = {t["id"]: t["name"] for t in themes}

        for t in themes:
            n = by_status.get(t["id"], {})
            text = (f"Theme #{t['id']} “{t['name']}”. {t['description'] or ''} Priority score {t['score']}/100. "
                    f"{n.get('Pending', 0)} pending (waiting for client approval), {n.get('Approved', 0)} approved by clients "
                    f"(waiting for completion), {n.get('Completed', 0)} completed request(s). "
                    f"Why it ranks here: {t['reason'] or 'n/a'}")
            chunks.append({"chunk_id": f"T:{t['id']}", "kind": "theme", "ref_id": t["id"],
                           "title": f"Theme #{t['id']} · {t['name']}", "text": text})

        for r in c.execute("""SELECT r.id, r.summary, r.category, r.sentiment, r.urgency, r.status, r.approved_at,
                                     r.completed_at, r.model_priority, r.client_priority, r.theme_id, f.raised_by, f.source, f.created_at, f.text
                              FROM requests r JOIN feedback f ON f.id=r.feedback_id ORDER BY r.id"""):
            text = (f"Request #{r['id']}: {r['summary']} Category: {r['category']}. Sentiment: {r['sentiment']}. "
                    f"Urgency: {r['urgency']}/5. Raised by: {r['raised_by'] or 'unspecified'} via {r['source']} on "
                    f"{_day(r['created_at'])}. Status: {r['status']}.")
            if r["model_priority"] is not None or r["client_priority"] is not None:
                text += (f" Priority (0-100): model {'n/a' if r['model_priority'] is None else int(r['model_priority'])}, "
                         f"client {'n/a' if r['client_priority'] is None else int(r['client_priority'])}.")
            if r["approved_at"]:
                text += f" Approved by a client on {_day(r['approved_at'])}."
            if r["completed_at"]:
                text += f" Completed on {_day(r['completed_at'])}."
            if r["theme_id"] in names:
                text += f" Part of theme “{names[r['theme_id']]}”."
            text += f" Original feedback: {_clip(r['text'])}"
            chunks.append({"chunk_id": f"R:{r['id']}", "kind": "request", "ref_id": r["id"],
                           "title": f"Request #{r['id']} · {_clip(r['summary'], 48)}", "text": text})

        for f in c.execute("""SELECT id, raised_by, source, created_at, text, attempts FROM feedback
                              WHERE processed=0 ORDER BY id"""):
            state = "parked after repeated failures" if f["attempts"] >= max_attempts else "waiting for analysis"
            text = (f"Unanalyzed feedback #{f['id']} ({state}) from {f['raised_by'] or 'unspecified'} via {f['source']} "
                    f"on {_day(f['created_at'])}: {_clip(f['text'])}")
            chunks.append({"chunk_id": f"F:{f['id']}", "kind": "feedback", "ref_id": f["id"],
                           "title": f"Feedback #{f['id']} (unanalyzed)", "text": text})

        for s in c.execute("""SELECT s.doc_id, s.idx, s.heading, s.parent, s.text, d.name FROM policy_sections s
                              JOIN policy_docs d ON d.id=s.doc_id ORDER BY s.doc_id, s.idx"""):
            text = (f"Company policy document “{s['name']}”, section {s['heading']}"
                    + (f" (under {s['parent']})" if s["parent"] else "") + f": {s['text']}")
            chunks.append({"chunk_id": f"P:{s['doc_id']}:{s['idx']}", "kind": "policy", "ref_id": s["idx"],
                           "title": f"Policy · {s['heading']}", "text": text})
    return chunks


# ---------------------------------------------------------------- 2. incremental index in the database
def sync_index(full=False, max_batches=None):
    """Brings rag_chunks in line with the database. Only new/changed chunks are written and (if an embedding
    deployment is set) embedded. Returns counts; never raises for embedding problems (reports them in 'error')."""
    out = {"total": 0, "added": 0, "updated": 0, "removed": 0, "embedded": 0, "waiting": 0, "error": ""}
    chunks = build_chunks()
    now = datetime.now().isoformat(timespec="seconds")
    with db.conn() as c:
        if full:
            c.execute("DELETE FROM rag_chunks")
        have = {r["chunk_id"]: r["text_hash"] for r in c.execute("SELECT chunk_id, text_hash FROM rag_chunks")}
        keep = {ch["chunk_id"] for ch in chunks}
        gone = [(i,) for i in have if i not in keep]
        c.executemany("DELETE FROM rag_chunks WHERE chunk_id=?", gone)
        out["removed"] = len(gone)
        for ch in chunks:
            h, old = _hash(ch["text"]), have.get(ch["chunk_id"])
            if old is None:
                c.execute("""INSERT INTO rag_chunks(chunk_id, kind, ref_id, title, text, text_hash, updated_at)
                             VALUES(?,?,?,?,?,?,?)""",
                          (ch["chunk_id"], ch["kind"], ch["ref_id"], ch["title"], ch["text"], h, now))
                out["added"] += 1
            elif old != h:                                   # text changed -> old vector is stale
                c.execute("""UPDATE rag_chunks SET kind=?, ref_id=?, title=?, text=?, text_hash=?, embedding=NULL,
                             emb_model=NULL, updated_at=? WHERE chunk_id=?""",
                          (ch["kind"], ch["ref_id"], ch["title"], ch["text"], h, now, ch["chunk_id"]))
                out["updated"] += 1
        out["total"] = len(chunks)

    if llm.has_embeddings() and not llm.missing_config():     # vectors for new / changed / other-model chunks
        model = llm.embed_model_name()
        with db.conn() as c:
            todo = c.execute("SELECT chunk_id, text FROM rag_chunks WHERE embedding IS NULL OR emb_model IS NOT ?",
                             (model,)).fetchall()
        limit = (max_batches or setting("AIRA_RAG_MAX_EMBED_BATCHES", 5)) * 64
        batch = todo[:limit]
        out["waiting"] = len(todo) - len(batch)
        if batch:
            try:
                vecs = llm.embed([r["text"] for r in batch])
                with db.conn() as c:
                    c.executemany("UPDATE rag_chunks SET embedding=?, emb_model=? WHERE chunk_id=?",
                                  [(np.asarray(v, dtype=np.float32).tobytes(), model, r["chunk_id"])
                                   for v, r in zip(vecs, batch)])
                out["embedded"] = len(batch)
            except Exception as e:                            # budget cap, network ...: keyword search still works
                out["error"] = llm.explain(e)
                out["waiting"] = len(todo)
                log.warning("RAG embedding skipped: %s", out["error"])
    log.info("RAG index synced: %s", out)
    return out


def index_status():
    with db.conn() as c:
        total, emb = c.execute("SELECT COUNT(*), COALESCE(SUM(embedding IS NOT NULL), 0) FROM rag_chunks").fetchone()
        kinds = dict(c.execute("SELECT kind, COUNT(*) FROM rag_chunks GROUP BY kind").fetchall())
        last = c.execute("SELECT MAX(updated_at) FROM rag_chunks").fetchone()[0]
    return {"total": total, "embedded": int(emb), "kinds": kinds, "last": last}


# ---------------------------------------------------------------- 3. retrieval
@lru_cache(maxsize=64)
def _query_vec(query, model):
    """Embeds the question once (cached, so searching tracker data AND policy costs one call). Counts for the budget."""
    llm._spend("embed")
    return np.asarray(llm.embeddings_model().embed_query(query), dtype=np.float32)


def _keyword_hits(rows, query, allowed):
    """Keyword ranking over the allowed row indices. A row only counts if it contains enough of the question's meaningful
    words (weighted by how rare each word is in the WHOLE index, so 'client1' matters more than 'ask').
    Returns ({row_index: tf-idf score}, ranking best-first)."""
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        X = vec.fit_transform([r["text"] for r in rows])
    except ValueError:                                        # empty vocabulary
        return {}, []
    terms = {t for t in vec.build_analyzer()(query) if " " not in t}
    known = [t for t in terms if t in vec.vocabulary_]
    if not known or len(known) < COVERAGE_MIN * len(terms):   # the question is mostly about things the data never mentions
        return {}, []
    cols = [vec.vocabulary_[t] for t in known]
    idf = vec.idf_[cols]
    present = (X.tocsc()[:, cols] > 0).toarray()
    coverage = (present * idf).sum(axis=1) / idf.sum()        # share of the question's weight found in each row
    rarest = np.argsort(-idf)[: (len(cols) + 1) // 2]         # the distinctive half of the question's words
    has_distinctive = present[:, rarest].any(axis=1)          # a hit must contain at least one of them
    sc = (X @ vec.transform([query]).T).toarray().ravel()
    ok = [int(i) for i in np.argsort(-sc)
          if i in allowed and sc[i] >= LEX_MIN_SCORE and coverage[i] >= COVERAGE_MIN and has_distinctive[i]]
    return {i: float(sc[i]) for i in ok[:30]}, ok[:30]


def retrieve(query, k=None, kinds=None, exclude=None):
    """Top-k chunks for a question: [{chunk_id, kind, title, text, via}] (best first). Keyword + semantic, fused.
    kinds=("policy",) searches only those kinds; exclude=("policy",) skips them."""
    k = k or setting("AIRA_RAG_TOP_K", 8)
    with db.conn() as c:
        rows = c.execute("SELECT chunk_id, kind, title, text, embedding, emb_model FROM rag_chunks").fetchall()
    allowed = {i for i, r in enumerate(rows)
               if (not kinds or r["kind"] in kinds) and not (exclude and r["kind"] in exclude)}
    if not allowed or not query.strip():
        return []

    lex_score, lex = _keyword_hits(rows, query, allowed)      # keyword ranking (local, free)

    sem = []                                                  # semantic ranking (1 embedding call, cached per question)
    if llm.has_embeddings():
        model = llm.embed_model_name()
        idx = [i for i in sorted(allowed) if rows[i]["embedding"] is not None and rows[i]["emb_model"] == model]
        if idx:
            try:
                q = _query_vec(query, model)
                M = np.vstack([np.frombuffer(rows[i]["embedding"], dtype=np.float32) for i in idx])
                sims = (M @ q) / (np.linalg.norm(M, axis=1) * np.linalg.norm(q) + 1e-9)
                sem = [idx[j] for j in np.argsort(-sims)[:30]]
            except Exception as e:
                log.warning("RAG semantic search skipped: %s", e)

    score, via = {}, {}
    for name, ranking in (("keyword", lex), ("semantic", sem)):
        for rank, i in enumerate(ranking):
            score[i] = score.get(i, 0.0) + 1.0 / (RRF_K + rank)
            via.setdefault(i, []).append(name)
    best = sorted(score, key=score.get, reverse=True)[:k]
    return [{"chunk_id": rows[i]["chunk_id"], "kind": rows[i]["kind"], "title": rows[i]["title"],
             "text": rows[i]["text"], "via": "+".join(via[i]), "lex": round(lex_score.get(i, 0.0), 3)} for i in best]


# ---------------------------------------------------------------- 4. the database as one readable document
def export_markdown():
    """The whole knowledge base as a single Markdown document (also usable in any external RAG tool)."""
    chunks = build_chunks()
    parts = [f"# VOC Enhancement Tracker: knowledge document",
             f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from voc.db · {len(chunks)} entries_", ""]
    for kind, head in (("theme", "Themes (ranked tasks)"), ("request", "Requests"), ("feedback", "Unanalyzed feedback"),
                       ("policy", "Company policy document")):
        items = [c for c in chunks if c["kind"] == kind]
        if items:
            parts += [f"## {head}", ""] + [f"- **{c['title']}**: {c['text']}" for c in items] + [""]
    return "\n".join(parts)


# ---------------------------------------------------------------- 5. admin panel (sidebar)
def render_admin_panel():
    st.subheader("AI knowledge index (RAG)")
    if st.button("Rebuild AI index", key="rag_rebuild",
                 help="Re-reads the whole database into the index Aira searches. Normally not needed: it updates itself."):
        with st.spinner("Indexing…"):
            try:
                llm.begin_run()
                res = sync_index(full=True, max_batches=setting("AIRA_RAG_REBUILD_MAX_BATCHES", 20))
                if res["error"]:
                    st.warning("Indexed for keyword search; embeddings skipped: " + res["error"][:200])
                else:
                    st.success(f"Indexed {res['total']} entries" +
                               (f" ({res['embedded']} embedded, {res['waiting']} still queued)" if res["embedded"] or res["waiting"] else "") + ".")
            except Exception as e:
                st.error(f"Could not rebuild the index: {e}")
    try:
        s = index_status()
    except Exception as e:
        st.error(f"Could not read the index: {e}")
        return
    k = s["kinds"]
    st.caption(f"{s['total']} entries indexed ({k.get('theme', 0)} themes · {k.get('request', 0)} requests · "
               f"{k.get('feedback', 0)} unanalyzed · {k.get('policy', 0)} policy sections) · {s['embedded']} with embeddings"
               + (f" · updated {s['last'].replace('T', ' ')}" if s["last"] else ""))
    st.caption("Search: " + (f"keyword + semantic ({llm.embed_model_name()})" if llm.has_embeddings() else
               "keyword only (local). Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME in .env for semantic search."))
    try:
        st.download_button("Download knowledge document (.md)", export_markdown(), file_name="voc-knowledge.md",
                           mime="text/markdown", key="rag_download")
    except Exception as e:
        st.caption(f"Export unavailable: {e}")
