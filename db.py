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
        # task workflow (Pending -> Completed -> Verified): lives on requests, because themes are rebuilt on every analysis
        rcols = [r[1] for r in c.execute("PRAGMA table_info(requests)")]
        if "status" not in rcols:
            c.execute("ALTER TABLE requests ADD COLUMN status TEXT NOT NULL DEFAULT 'Pending'")
        if "completed_at" not in rcols:                 # "Task Submitted At": when the user clicked Completed
            c.execute("ALTER TABLE requests ADD COLUMN completed_at TEXT")
        if "verified_at" not in rcols:                  # "Task Verified At": when an admin clicked Verify
            c.execute("ALTER TABLE requests ADD COLUMN verified_at TEXT")
        c.execute("UPDATE requests SET status='Pending' WHERE status IS NULL OR status NOT IN ('Pending','Completed','Verified')")
        c.execute("CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status)")


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
            cur = c.execute("INSERT INTO themes(name,description,score,n_requests,reason) VALUES(?,?,?,?,?)",
                            (t["name"], t["description"], t["score"], t["n_requests"], t.get("reason", "")))
            key_to_id[t["key"]] = cur.lastrowid
        for rid, key in assignments.items():
            c.execute("UPDATE requests SET theme_id=? WHERE id=?", (key_to_id[key], rid))


# ---------- task workflow: Pending -> Completed -> Verified ----------
STATUSES = ("Pending", "Completed", "Verified")


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


def complete_requests(ids):
    """User clicked Completed: Pending -> Completed, records the submission time (verified_at stays empty)."""
    return _move(ids, "Completed", "Pending", completed_at=_now(), verified_at=None)


def verify_requests(ids):
    """Admin clicked Verify: Completed -> Verified, records the verification time."""
    return _move(ids, "Verified", "Completed", verified_at=_now())


def repost_requests(ids):
    """Admin clicked Repost: Completed -> Pending. completed_at is NOT touched, verified_at stays empty,
    and the feedback's own created_at is never modified."""
    return _move(ids, "Pending", "Completed")


def reset_data():
    """Clears feedback/requests/themes. The API call log is KEPT so the daily cap can't be bypassed."""
    with conn() as c:
        c.executescript("""DELETE FROM requests; DELETE FROM themes; DELETE FROM theme_cache; DELETE FROM feedback;
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
            c.execute("UPDATE themes SET n_requests=?, score=?, reason=? WHERE id=?", (n, score, reason, tid))
