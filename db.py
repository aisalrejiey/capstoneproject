import hashlib
import os
import re
import sqlite3
from datetime import date, datetime
import numpy as np
import pandas as pd

import roles

DB_PATH = "voc.db"


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _migrate_if_old():
    """Older versions had a different schema: back the file up and start fresh."""
    if not os.path.exists(DB_PATH):
        return
    with sqlite3.connect(DB_PATH) as c:
        cols = [r[1] for r in c.execute("PRAGMA table_info(feedback)")]
    if cols and "text_hash" not in cols:
        backup = f"voc_backup_{datetime.now():%Y%m%d_%H%M%S}.db"
        os.replace(DB_PATH, backup)
        print(f"[voc] Old database schema found; moved to {backup}")


def init():
    _migrate_if_old()
    with conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT, text TEXT, text_hash TEXT UNIQUE,
            created_at TEXT, processed INTEGER DEFAULT 0,
            attempts INTEGER DEFAULT 0, last_error TEXT, raised_by TEXT DEFAULT '');
        CREATE TABLE IF NOT EXISTS requests(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER, summary TEXT, category TEXT,
            sentiment TEXT, urgency INTEGER,
            embedding BLOB, emb_model TEXT, theme_id INTEGER);
        CREATE TABLE IF NOT EXISTS themes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, description TEXT, score REAL, n_requests INTEGER, reason TEXT);
        CREATE TABLE IF NOT EXISTS theme_cache(
            sig TEXT PRIMARY KEY, name TEXT, description TEXT);
        CREATE TABLE IF NOT EXISTS api_calls(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, day TEXT, kind TEXT);
        """)
        cols = [r[1] for r in c.execute("PRAGMA table_info(feedback)")]
        if "raised_by" not in cols:          # upgrade databases from the previous version
            c.execute("ALTER TABLE feedback ADD COLUMN raised_by TEXT DEFAULT ''")
        if "reason" not in [r[1] for r in c.execute("PRAGMA table_info(themes)")]:
            c.execute("ALTER TABLE themes ADD COLUMN reason TEXT")
        # task workflow (Pending -> Approved by the client -> Completed by an admin): lives on requests, because themes
        # are rebuilt on every analysis
        rcols = [r[1] for r in c.execute("PRAGMA table_info(requests)")]
        if "status" not in rcols:
            c.execute("ALTER TABLE requests ADD COLUMN status TEXT NOT NULL DEFAULT 'Pending'")
        if "approved_at" not in rcols:                  # "Task Approved At": when the client clicked Approved
            c.execute("ALTER TABLE requests ADD COLUMN approved_at TEXT")
            if "completed_at" in rcols and "verified_at" in rcols:
                # one-time conversion of a database from the OLD workflow (Pending -> Completed -> Verified):
                # the old "Completed" step is the client's approval, the old "Verified" step is the real completion
                c.execute("UPDATE requests SET approved_at=completed_at, completed_at=verified_at")
                c.execute("""UPDATE requests SET status = CASE status WHEN 'Completed' THEN 'Approved'
                             WHEN 'Verified' THEN 'Completed' ELSE status END""")
                c.execute("UPDATE requests SET approved_at=NULL WHERE status='Pending'")   # reposted tasks were never approved
                try:
                    c.execute("ALTER TABLE requests DROP COLUMN verified_at")
                except sqlite3.OperationalError:        # very old SQLite: the unused column simply stays
                    pass
            elif "completed_at" not in rcols:
                c.execute("ALTER TABLE requests ADD COLUMN completed_at TEXT")   # "Task Completed At": admin clicked Complete
        c.execute("UPDATE requests SET status='Pending' WHERE status IS NULL OR status NOT IN ('Pending','Approved','Completed')")
        # priorities (0-100) kept separately: model_priority = what the model suggested when the client first saw/saved it
        # (never overwritten afterwards), client_priority = what the client chose in the Feedback Tracker
        rcols = [r[1] for r in c.execute("PRAGMA table_info(requests)")]
        for col in ("model_priority", "client_priority"):
            if col not in rcols:
                c.execute(f"ALTER TABLE requests ADD COLUMN {col} REAL")
        c.execute("CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status)")
        # company policy document -> policy fit score per theme (see policy.py)
        tcols = [r[1] for r in c.execute("PRAGMA table_info(themes)")]
        for col, typ in (("base_score", "REAL"), ("policy_score", "REAL"), ("policy_level", "TEXT"), ("policy_reason", "TEXT")):
            if col not in tcols:
                c.execute(f"ALTER TABLE themes ADD COLUMN {col} {typ}")
        c.execute("UPDATE themes SET base_score=score WHERE base_score IS NULL")   # score before any policy blend
        c.execute("""CREATE TABLE IF NOT EXISTS policy_docs(
                       id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, doc_hash TEXT, uploaded_at TEXT,
                       n_sections INTEGER, n_chars INTEGER)""")
        c.execute("""CREATE TABLE IF NOT EXISTS policy_sections(
                       id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id INTEGER, idx INTEGER, heading TEXT, parent TEXT,
                       text TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS policy_scores(
                       sig TEXT PRIMARY KEY, doc_hash TEXT, applies INTEGER, score REAL, level TEXT, section TEXT,
                       reason TEXT, scored_at TEXT)""")
        # RAG knowledge index: one searchable chunk per theme / request / unanalyzed feedback (rebuilt by rag.py)
        c.execute("""CREATE TABLE IF NOT EXISTS rag_chunks(
                       chunk_id TEXT PRIMARY KEY, kind TEXT NOT NULL, ref_id INTEGER, title TEXT, text TEXT NOT NULL,
                       text_hash TEXT NOT NULL, embedding BLOB, emb_model TEXT, updated_at TEXT)""")


# ---------- feedback ----------
def _hash(text):
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def next_number(role):
    """Next unused number for a role: highest existing + 1 (client1, client2 exist -> 3)."""
    with conn() as c:
        rows = c.execute("SELECT DISTINCT raised_by FROM feedback WHERE raised_by LIKE ?",
                         (role + "%",)).fetchall()
    nums = [int(m.group(1)) for r in rows if (m := re.fullmatch(role + r"(\d+)", r[0] or ""))]
    return max(nums, default=0) + 1


def raiser_counts():
    """{'client1': 3, 'user2': 1, ...}: entries per person."""
    with conn() as c:
        return {r[0]: r[1] for r in c.execute(
            "SELECT raised_by, COUNT(*) FROM feedback WHERE raised_by!='' GROUP BY raised_by")}


def _dup_hash(text, key):
    text = (text or "").strip()
    return _hash(text + "|" + key) if key else _hash(text)


def redate(text, key, when):
    """A duplicate found in a CSV that carries a date: make the stored entry use the CSV date."""
    with conn() as c:
        return c.execute("UPDATE feedback SET created_at=? WHERE text_hash=? AND created_at!=?",
                         (when, _dup_hash(text, key), when)).rowcount


def add_feedback(source, text, raised_by="", auto_role=None, created_at=None):
    """Adds an entry. Returns the raiser string (may be '') if added, or None if empty/duplicate.
    - raised_by='client2' -> that exact person (same number = same person).
    - auto_role='client'  -> the next free number for that role is assigned automatically.
    created_at: ISO date-time of the entry (default: now). Older entries gain a little priority.
    Duplicate = same text from the same person (for auto: same text from the same role).
    The same text from different people is kept: it is extra evidence for the theme."""
    text = (text or "").strip()
    if not text:
        return None
    key = auto_role or raised_by
    if auto_role:
        raised_by = f"{auto_role}{next_number(auto_role)}"
    h = _dup_hash(text, key)
    try:
        with conn() as c:
            c.execute("INSERT INTO feedback(source,text,text_hash,created_at,raised_by) VALUES(?,?,?,?,?)",
                      (source, text, h, created_at or datetime.now().isoformat(timespec="seconds"), raised_by))
        return raised_by
    except sqlite3.IntegrityError:
        return None


ALIASES = {"text": ["text", "feedback", "comment", "comments", "description"],
           "source": ["source", "channel"],
           "user": ["user", "raised_by", "raised by", "who", "raiser"],
           "date": ["date", "datetime", "date_time", "date and time", "date/time", "timestamp", "created_at", "created"]}


def normalize_columns(df):
    """Tolerant CSV headers: ' Date ', 'DATE', 'Date and Time', BOM... -> text/source/user/date.
    Separate 'date' and 'time' columns are combined."""
    df = df.rename(columns={c: str(c).strip().lstrip("\ufeff").lower() for c in df.columns})
    for canon, alts in ALIASES.items():
        if canon not in df.columns:
            for alt in alts:
                if alt in df.columns:
                    df = df.rename(columns={alt: canon})
                    break
    if "date" in df.columns and "time" in df.columns:
        df["date"] = df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip()
    return df


def _parse_dt(v):
    """-> (iso string | None, ok). Blank -> (None, True) = use now. Unparseable -> (None, False).
    Day-first dates (31/12/2026): set DATE_DAYFIRST=1 in .env."""
    if v is None or str(v).strip() == "" or str(v).strip().lower() in ("nan", "nat", "none"):
        return None, True
    first = os.getenv("DATE_DAYFIRST", "0") == "1"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = pd.to_datetime(v, errors="coerce", dayfirst=first)
        if pd.isna(ts):
            ts = pd.to_datetime(v, errors="coerce", dayfirst=not first)
    if pd.isna(ts):
        return None, False
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)             # store naive (UTC) time
    return ts.to_pydatetime().isoformat(timespec="seconds"), True


def import_rows(df, limit=500):
    """df needs 'text'; optional 'source', 'user' (client2 = exact person, 'client' = next free number)
    and 'date' (e.g. 2026-10-01 14:30; blank = now). The CSV date becomes the entry's date.
    Returns a dict: added, dup, bad_user, auto, bad_date, dated, redated."""
    df = normalize_columns(df)
    out = dict(added=0, dup=0, bad_user=0, auto=0, bad_date=0, dated=0, redated=0)
    role_only = []

    def store(src, text, user, when, auto_role=None):
        res = add_feedback(src, text, user, auto_role=auto_role, created_at=when)
        if res is not None:
            out["added"] += 1
            out["auto"] += bool(auto_role)
        else:
            out["dup"] += 1
            if when and text.strip():           # already stored: take the date from the CSV
                out["redated"] += redate(text, auto_role or user, when)

    for _, r in df.head(limit).iterrows():
        user = roles.parse_user(r.get("user"))
        src, text = str(r.get("source", "csv")), str(r["text"])
        when, ok_date = _parse_dt(r.get("date"))
        out["bad_date"] += (not ok_date)
        out["dated"] += (when is not None)
        if user is None:                      # unrecognised value -> import as unspecified
            out["bad_user"] += 1
            user = ""
        if user in roles.ROLES:               # role without a number: numbered after explicit ones
            role_only.append((src, text, user, when))
        else:
            store(src, text, user, when)
    for src, text, role, when in role_only:   # explicit numbers are already stored, so no clashes
        store(src, text, "", when, auto_role=role)
    return out


def pending_feedback(max_attempts, limit):
    with conn() as c:
        return c.execute("""SELECT * FROM feedback WHERE processed=0 AND attempts<?
                            ORDER BY id LIMIT ?""", (max_attempts, limit)).fetchall()


def counts(max_attempts):
    with conn() as c:
        total = c.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        done = c.execute("SELECT COUNT(*) FROM feedback WHERE processed=1").fetchone()[0]
        failed = c.execute("SELECT COUNT(*) FROM feedback WHERE processed=0 AND attempts>=?",
                           (max_attempts,)).fetchone()[0]
    return {"total": total, "done": done, "failed": failed, "pending": total - done - failed}


def mark_failed(feedback_id, err):
    with conn() as c:
        c.execute("UPDATE feedback SET attempts=attempts+1, last_error=? WHERE id=?",
                  (str(err)[:500], feedback_id))


def reset_failed():
    with conn() as c:
        c.execute("UPDATE feedback SET attempts=0 WHERE processed=0")


def save_requests(feedback_id, items):
    with conn() as c:
        for it in items:
            c.execute("""INSERT INTO requests(feedback_id,summary,category,sentiment,urgency)
                         VALUES(?,?,?,?,?)""",
                      (feedback_id, it["summary"], it["category"], it["sentiment"], it["urgency"]))
        c.execute("UPDATE feedback SET processed=1, last_error=NULL WHERE id=?", (feedback_id,))


def delete_feedback(ids):
    """Deletes feedback entries AND the requests extracted from them. Returns number removed."""
    ids = [int(i) for i in ids]
    if not ids:
        return 0
    q = ",".join("?" * len(ids))
    with conn() as c:
        c.execute(f"DELETE FROM requests WHERE feedback_id IN ({q})", ids)
        n = c.execute(f"DELETE FROM feedback WHERE id IN ({q})", ids).rowcount
        # Table now empty? Restart its numbering so new ids begin at 1 again.
        for table in ("feedback", "requests"):
            if c.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0:
                c.execute("DELETE FROM sqlite_sequence WHERE name=?", (table,))
        return n


# ---------- requests / embeddings / themes ----------
def all_requests():
    with conn() as c:
        return c.execute("""SELECT r.*, f.created_at, f.source, f.raised_by
                            FROM requests r JOIN feedback f ON f.id=r.feedback_id""").fetchall()


def set_embeddings(pairs, model):
    with conn() as c:
        for rid, vec in pairs:
            c.execute("UPDATE requests SET embedding=?, emb_model=? WHERE id=?",
                      (np.asarray(vec, dtype=np.float32).tobytes(), model, rid))


def cache_get(sig):
    with conn() as c:
        r = c.execute("SELECT name, description FROM theme_cache WHERE sig=?", (sig,)).fetchone()
    return (r["name"], r["description"]) if r else None


def cache_put(sig, name, desc):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO theme_cache(sig,name,description) VALUES(?,?,?)",
                  (sig, name, desc))


def replace_themes(themes, assignments):
    themes = sorted(themes, key=lambda t: -t["score"])      # theme #1 = highest priority
    with conn() as c:
        c.execute("DELETE FROM themes")
        c.execute("DELETE FROM sqlite_sequence WHERE name='themes'")   # ids restart at 1
        key_to_id = {}
        for t in themes:
            cur = c.execute("INSERT INTO themes(name,description,score,base_score,n_requests,reason) VALUES(?,?,?,?,?,?)",
                            (t["name"], t["description"], t["score"], t["score"], t["n_requests"], t.get("reason", "")))
            key_to_id[t["key"]] = cur.lastrowid
        for rid, key in assignments.items():
            c.execute("UPDATE requests SET theme_id=? WHERE id=?", (key_to_id[key], rid))


def reorder_themes():
    """Renumbers themes by score (theme #1 = highest priority) and moves the requests with them.
    Needed after the policy blend changes the order. Returns True if anything moved."""
    with conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM themes ORDER BY score DESC, id")]
        mapping = {r["id"]: i for i, r in enumerate(rows, 1)}
        if not rows or all(o == n for o, n in mapping.items()):
            return False
        cols = [k for k in rows[0] if k != "id"]
        c.execute("UPDATE requests SET theme_id=theme_id+1000000 WHERE theme_id IS NOT NULL")
        c.execute("DELETE FROM themes")
        c.execute("DELETE FROM sqlite_sequence WHERE name='themes'")
        for r in rows:
            c.execute(f"INSERT INTO themes(id,{','.join(cols)}) VALUES(?{',?' * len(cols)})",
                      [mapping[r["id"]]] + [r[k] for k in cols])
            c.execute("UPDATE requests SET theme_id=? WHERE theme_id=?", (mapping[r["id"]], r["id"] + 1000000))
    return True


# ---------- task workflow: Pending -> Approved (client) -> Completed (admin) ----------
STATUSES = ("Pending", "Approved", "Completed")


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _move(ids, to_status, from_status, **cols):
    """Moves the given requests from one status to another in ONE transaction. Only rows currently in
    `from_status` are touched, so a double click or a stale page changes nothing twice (no duplicate or
    overwritten timestamps). Returns how many rows changed."""
    ids = sorted({int(i) for i in ids})
    if not ids:
        return 0
    sets = ",".join(["status=?"] + [f"{k}=?" for k in cols])
    q = ",".join("?" * len(ids))
    with conn() as c:
        return c.execute(f"UPDATE requests SET {sets} WHERE status=? AND id IN ({q})",
                         [to_status, *cols.values(), from_status, *ids]).rowcount


def task_counts():
    """Number of TASKS (a task = the requests of one theme approved / completed together) that are Approved (waiting for an
    admin to complete them) and that are Completed. Matches the task cards shown in Admin Access."""
    with conn() as c:
        a = c.execute("SELECT COUNT(*) FROM (SELECT DISTINCT theme_id, approved_at FROM requests WHERE status='Approved')").fetchone()[0]
        d = c.execute("SELECT COUNT(*) FROM (SELECT DISTINCT theme_id, completed_at FROM requests WHERE status='Completed')").fetchone()[0]
    return {"approved": a, "completed": d}


def priority_int(score):
    """Priority shown to clients: the 0-100 priority score rounded half-up to a whole number (used everywhere)."""
    return int(float(score) + 0.5)


def saved_client_priority(ids):
    """The client's saved priority for these (pending) requests when they all share one value, else None."""
    ids = sorted({int(i) for i in ids})
    if not ids:
        return None
    with conn() as c:
        rows = c.execute(f"SELECT client_priority FROM requests WHERE status='Pending' AND id IN ({','.join('?' * len(ids))})",
                         ids).fetchall()
    vals = {r[0] for r in rows}
    return int(vals.pop()) if len(vals) == 1 and None not in vals else None


def save_client_priority(ids, model_priority, client_priority):
    """Client clicked Save Priority: stores their value on the still-Pending requests. model_priority is only filled in
    when empty (COALESCE), so the original model value is never overwritten. Returns how many rows changed."""
    ids = sorted({int(i) for i in ids})
    if not ids:
        return 0
    with conn() as c:
        return c.execute(f"""UPDATE requests SET client_priority=?, model_priority=COALESCE(model_priority, ?)
                             WHERE status='Pending' AND id IN ({','.join('?' * len(ids))})""",
                         [float(client_priority), float(model_priority), *ids]).rowcount


def approve_requests(ids, model_priority=None, client_priority=None):
    """Client clicked Approved: Pending -> Approved, records the approval time (completed_at stays empty). Also stores the
    priorities: client_priority = the value approved (saved, or the model's when left unchanged); model_priority only if
    still empty, so it is never overwritten."""
    ids = sorted({int(i) for i in ids})
    if not ids:
        return 0
    with conn() as c:
        return c.execute(f"""UPDATE requests SET status='Approved', approved_at=?, completed_at=NULL,
                             model_priority=COALESCE(model_priority, ?), client_priority=COALESCE(?, client_priority)
                             WHERE status='Pending' AND id IN ({','.join('?' * len(ids))})""",
                         [_now(), model_priority, client_priority, *ids]).rowcount


def complete_requests(ids):
    """Admin clicked Complete: Approved -> Completed, records the completion time. approved_at is never touched."""
    return _move(ids, "Completed", "Approved", completed_at=_now())


def reset_data():
    """Clears feedback/requests/themes. The API call log is KEPT so the daily cap can't be bypassed."""
    with conn() as c:
        c.executescript("""DELETE FROM requests; DELETE FROM themes; DELETE FROM theme_cache; DELETE FROM feedback; DELETE FROM rag_chunks;
                           DELETE FROM sqlite_sequence WHERE name IN ('requests','themes','feedback');""")


# ---------- API usage log ----------
def log_call(kind):
    now = datetime.now()
    with conn() as c:
        c.execute("INSERT INTO api_calls(ts,day,kind) VALUES(?,?,?)",
                  (now.isoformat(), date.today().isoformat(), kind))


def calls_today():
    with conn() as c:
        return c.execute("SELECT COUNT(*) FROM api_calls WHERE day=?",
                         (date.today().isoformat(),)).fetchone()[0]


def update_theme_stats(stats):
    """stats: {theme_id: (n_requests, score, reason)}. Themes not listed (now empty) are removed."""
    with conn() as c:
        if stats:
            q = ",".join("?" * len(stats))
            c.execute(f"DELETE FROM themes WHERE id NOT IN ({q})", list(stats))
        else:
            c.execute("DELETE FROM themes")
        for tid, (n, score, reason) in stats.items():
            c.execute("UPDATE themes SET n_requests=?, score=?, base_score=?, reason=? WHERE id=?",
                      (n, score, score, reason, tid))
