"""Results display shared by the Feedback Tracker page and the Admin Access page.

Everything is read from the existing SQLite tables (themes, requests, feedback) that the analysis pipeline fills,
so results survive page refreshes and navigation. Nothing here calls the AI.
Task workflow (all saved in the database, on the requests):  Pending -> Approved (a client approves the change)
-> Completed (an admin completes it). The client's list shows only Pending tasks.
"""
import pandas as pd
import streamlit as st

import db
import policy
import roles
import theme


def add_urgency_cols(df, n_in_theme):
    """Adds adj_urgency (role weight), age/repeat boosts and the final urgency % (max 100)."""
    n = list(n_in_theme)
    df["adj_urgency"] = [roles.adj_urgency(u, b) for u, b in zip(df["urgency"], df["raised_by"])]
    df["age_boost_pct"] = [round(roles.age_boost(c) * 100) for c in df["created_at"]]
    df["repeat_boost_pct"] = [round(roles.repeat_boost(k) * 100) for k in n]
    df["urgency_pct"] = [roles.pct(roles.final_urgency(u, b, c, k))
                         for u, b, c, k in zip(df["urgency"], df["raised_by"], df["created_at"], n)]
    df["added"] = df.pop("created_at").str.slice(0, 16).str.replace("T", " ")
    return df


def load_results():
    """Reads saved results. Raises on database errors (the caller shows the error state).
    Returns {'themes': DataFrame, 'requests': DataFrame, 'empty': bool}."""
    with db.conn() as c:
        themes = pd.read_sql("SELECT * FROM themes ORDER BY score DESC", c)
        reqs = pd.read_sql("""SELECT r.id, r.summary, f.raised_by, f.source, r.category, r.sentiment,
                              r.urgency, r.theme_id, f.created_at,
                              r.status, r.approved_at, r.completed_at, r.model_priority, r.client_priority
                              FROM requests r JOIN feedback f ON f.id=r.feedback_id ORDER BY r.id""", c)
    reqs["raised_by"] = reqs["raised_by"].fillna("")
    reqs["status"] = reqs["status"].fillna("Pending")
    return {"themes": themes, "requests": reqs, "empty": themes.empty and reqs.empty}


# ---------------------------------------------------------------- workflow callbacks
def _flash(kind, msg):
    st.session_state["flash"] = (kind, msg)


def show_flash():
    """Shows (once) the result of the last button click: success, nothing-to-do or error."""
    f = st.session_state.pop("flash", None)
    if f:
        {"success": st.success, "warning": st.warning, "error": st.error}[f[0]](f[1])


def _act(fn, ids, done, label):
    try:
        n = fn(ids)
    except Exception as e:                                   # database problem: report, change nothing
        _flash("error", f"Could not {label}: {e}")
        return
    if n:
        _flash("success", done)
    else:                                                    # already changed (double click / another session)
        _flash("warning", "That task was already updated by someone else, so nothing changed. The list has been refreshed.")


def _save_priority(ids, key, model_p):
    """Feedback Tracker: Save Priority. Stores the slider value as the client's priority (model value is never overwritten).
    The result is shown inside the same control (key + '_msg'), so only that control refreshes."""
    v = st.session_state.get(key)
    if v is None:
        return
    mk = key + "_msg"
    try:
        n = db.save_client_priority(ids, model_p, int(v))
    except Exception as e:
        st.session_state[mk] = ("error", f"Could not save the priority: {e}")
        return
    st.session_state[mk] = (("success", f"Priority saved: {int(v)}. You can approve this change now.") if n else
                            ("warning", "That change is no longer open, so the priority was not saved."))


def _approve(ids, name, model_p, key):
    """Feedback Tracker: the client approves this change. Refused (server side) if the slider holds an unsaved value."""
    try:
        saved = db.saved_client_priority(ids)
        baseline = saved if saved is not None else model_p        # what an unchanged slider means
        v = st.session_state.get(key)
        if v is not None and int(v) != baseline:
            _flash("warning", "You changed the priority. Please click Save Priority before approving.")
            return
        _act(lambda i: db.approve_requests(i, model_p, baseline), ids,
             f"“{name}” approved with priority {baseline}. An admin will complete it.", "approve this task")
    except Exception as e:
        _flash("error", f"Could not approve this task: {e}")


def _complete(ids, name):
    """Admin Access: the approved change has been delivered."""
    _act(db.complete_requests, ids, f"“{name}” marked as Completed.", "complete this task")


def _prio_col(series):
    """Priority column for tables: whole numbers, empty when not set."""
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _stamp(series):
    """Timestamps for display: 'YYYY-MM-DD HH:MM:SS', empty when not set."""
    return series.fillna("").astype(str).str.replace("T", " ").str.slice(0, 19)


# ---------------------------------------------------------------- admin: approved tasks waiting for completion
def render_approved_review(data):
    """Admin Access: tasks clients approved, each with a Complete button (in a pale-blue panel)."""
    with st.container(key="review_panel"):
        _approved_review_body(data)


def _approved_review_body(data):
    reqs, themes = data["requests"], data["themes"]
    st.subheader("Approved tasks waiting for completion")
    todo = reqs[reqs["status"] == "Approved"]
    n_done = int((reqs["status"] == "Completed").sum())
    if todo.empty:
        st.caption(f"No approved tasks are waiting. Completed so far: {n_done} request(s).")
        return
    names = dict(zip(themes["id"], themes["name"])) if not themes.empty else {}
    st.caption(f"{todo['approved_at'].nunique()} task(s) approved by clients, highest client priority first. "
               f"Completed so far: {n_done} request(s).")
    tid = todo["theme_id"].map(lambda x: "none" if pd.isna(x) else str(int(x)))       # a request without a theme still shows
    todo = todo.assign(_k=tid + "|" + todo["approved_at"].fillna("").astype(str))
    groups = []
    for _, d in todo.groupby("_k"):                       # one card per task; most important (client priority) first
        cp = pd.to_numeric(d["client_priority"], errors="coerce").max()
        mp = pd.to_numeric(d["model_priority"], errors="coerce").max()
        groups.append((cp if pd.notna(cp) else -1, mp if pd.notna(mp) else -1, d["approved_at"].iloc[0], d))
    groups.sort(key=lambda g: (-g[0], -g[1], g[2]))      # ties: model priority, then the task that has waited longest
    for cp, mp, _, d in groups:
        ids = [int(i) for i in d["id"]]
        name = names.get(d["theme_id"].iloc[0]) or d["summary"].iloc[0][:60]
        with st.container(border=True):
            st.markdown(f"**{name}** {theme.pill('Approved')}  ·  client priority "
                        f"{int(round(cp)) if cp >= 0 else 'not set'}  ·  {len(d)} request(s)  ·  "
                        f"approved {_stamp(d['approved_at']).iloc[0]}", unsafe_allow_html=True)
            show = d[["summary", "raised_by", "category", "sentiment"]].copy()
            show["Model Priority"] = _prio_col(d["model_priority"])         # what the model suggested
            show["Client Priority"] = _prio_col(d["client_priority"])       # what the client chose
            st.dataframe(show, width="stretch", hide_index=True)
            st.button("Complete", key=f"complete_{ids[0]}", type="primary", on_click=_complete, args=(ids, name),
                      help="Mark this approved task as completed.")


# ---------------------------------------------------------------- ranked themes + all requests
def _saved_of(series):
    """The client's saved priority when every pending request of the theme shares one value, else None."""
    vals = series.dropna()
    return int(vals.iloc[0]) if len(series) and len(vals) == len(series) and vals.nunique() == 1 else None


def render_ranked_themes(data, empty_text="No themes yet. Add feedback and click Run analysis."):
    """Feedback Tracker: the active (Pending) tasks ranked by priority (the client's saved priority when there is one,
    otherwise the model's), each with a priority slider and an Approved button."""
    themes, reqs = data["themes"], data["requests"]
    active = reqs[reqs["status"] == "Pending"]
    st.subheader("Ranked themes")

    shown = themes[themes["id"].isin(active["theme_id"].dropna())] if not themes.empty else themes
    if themes.empty:
        st.caption(empty_text)
    elif shown.empty:
        st.caption("No active tasks. Every task has been completed.")
    else:
        rows = []                                                   # (priority, model score, theme, its requests, model, saved)
        for _, t in shown.iterrows():
            grp = active[active["theme_id"] == t["id"]]
            model_p, saved = db.priority_int(t["score"]), _saved_of(grp["client_priority"])
            rows.append((saved if saved is not None else model_p, float(t["score"]), t, grp, model_p, saved))
        rows.sort(key=lambda r: (-r[0], -r[1]))                     # highest priority first; ties by the model's score
        with st.container(key="chart_card"):                      # white card behind the chart
            st.bar_chart(pd.Series([r[0] for r in rows], index=[r[2]["name"] for r in rows]), color=theme.TEXT2,
                         horizontal=True, x_label="Priority (0-100)", y_label="")
        st.caption("Ranked by your priority where you set one, otherwise by the suggested priority. "
                   "Urgency % = model urgency × role weight × (1 + age boost + repeat boost), capped at 100%. "
                   "adj_urgency = after the role weight only.")
        for rank, (eff, _, t, grp, model_p, saved) in enumerate(rows, 1):
            d = grp[["id", "summary", "raised_by", "created_at", "source", "category", "sentiment", "urgency"]].copy()
            mark = "  ·  your priority" if saved is not None else ""
            with st.expander(f"#{rank}  ·  {eff}  ·  {t['name']}  ({len(d)} requests){mark}"):
                if isinstance(t["reason"], str) and t["reason"]:
                    st.write(f"**Why:** {t['reason']}")
                if t["description"]:
                    st.write(t["description"])
                if pd.notna(t.get("policy_score")):                       # company-policy fit (see policy.py)
                    pct = round(policy.weight() * 100)
                    lvl = f" · {t['policy_level']}" if t["policy_level"] else ""
                    st.write(f"**Policy fit:** {t['policy_score']:.0f}/100{lvl}. {t['policy_reason']}")
                    st.caption(f"Priority score = {100 - pct}% base score ({t['base_score']}) + {pct}% policy score.")
                ids = [int(i) for i in d.pop("id")]
                who = sorted({u for u in d["raised_by"] if u})
                if who:
                    st.caption("Raised by: " + ", ".join(who))
                rb = round(roles.repeat_boost(len(d)) * 100)
                if rb:
                    st.caption(f"Repeat boost: +{rb}% urgency because {len(d)} requests asked for this.")
                d = add_urgency_cols(d, [len(d)] * len(d))
                st.dataframe(d[["summary", "raised_by", "added", "source", "category", "sentiment", "urgency",
                                "adj_urgency", "age_boost_pct", "repeat_boost_pct", "urgency_pct"]],
                             width="stretch", hide_index=True)
                _priority_controls(ids, t["name"], model_p)


@st.fragment
def _priority_controls(ids, name, model_p):
    """Priority slider (defaults to the model's suggestion) + Save Priority + Approved.
    Unchanged slider -> Approved works directly. Changed slider -> Save Priority first; Approved stays disabled until the
    latest value is saved. A priority saved earlier becomes the new 'unchanged' value.
    This is a Streamlit fragment: moving the slider or saving re-runs ONLY this control, not the whole page. The saved value
    is read from the database on every run, so the buttons always reflect what is really stored."""
    key = f"prio_{ids[0]}"
    try:
        saved = db.saved_client_priority(ids)
    except Exception as e:
        st.error(f"Could not read the saved priority: {e}")
        return
    baseline = saved if saved is not None else model_p
    # Streamlit drops a slider's value while its page is not shown (e.g. after visiting Admin Access), so the value must be
    # restored whenever the key is missing, not only when the baseline changed. Otherwise the slider falls back to 0.
    if key not in st.session_state or st.session_state.get(key + "_base") != baseline:
        st.session_state[key] = baseline
        st.session_state[key + "_base"] = baseline
    v = st.slider("Priority (0-100)", 0, 100, key=key,
                  help="Starts at the model's suggestion. Change it if this matters more or less to you.")
    changed = int(v) != baseline
    st.caption(f"Model suggestion: {model_p}" + (f" · Your saved priority: {saved}" if saved is not None else ""))
    if changed:
        st.caption("You changed the priority. Click Save Priority to enable approval.")
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        st.button("Save Priority", key=f"saveprio_{ids[0]}", disabled=not changed, on_click=_save_priority,
                  args=(ids, key, model_p), help="Store your priority for this change.")
    with c2:
        approve = st.button("✓ Approved", key=f"approve_{ids[0]}", disabled=changed, type="primary",
                            help="Approve this change. It leaves your list and goes to the team to be completed.")
    msg = st.session_state.pop(key + "_msg", None)                  # result of Save Priority, shown right here
    if msg:
        {"success": st.success, "warning": st.warning, "error": st.error}[msg[0]](msg[1])
    if approve:
        _approve(ids, name, model_p, key)
        st.rerun()                                                  # the approved task leaves the list: refresh the whole page


def render_all_requests(data):
    """Admin Access: every extracted request with its status and the approval / completion timestamps (light-gray panel)."""
    with st.container(key="requests_panel"):
        _all_requests_body(data)


def _all_requests_body(data):
    reqs = data["requests"]
    with st.expander("All extracted requests"):
        allr = reqs.rename(columns={"theme_id": "theme"}).copy()
        allr["id"] = range(1, len(allr) + 1)                # display numbering always runs 1..N
        n_theme = allr.groupby("theme")["id"].transform("count").reindex(allr.index).fillna(1).astype(int)
        pol = dict(zip(data["themes"]["id"], data["themes"]["policy_score"])) if "policy_score" in data["themes"] else {}
        score_of = dict(zip(data["themes"]["id"], data["themes"]["score"].map(db.priority_int))) if not data["themes"].empty else {}
        live = allr["theme"].map(score_of).where(allr["status"] == "Pending")  # open requests: the model's current suggestion
        allr["Model Priority"] = _prio_col(allr["model_priority"].where(allr["model_priority"].notna(), live))
        allr["Client Priority"] = _prio_col(allr["client_priority"])           # latest value saved in the Feedback Tracker
        allr["Policy score"] = allr["theme"].map(pol)                      # the theme's policy fit (empty = no policy rule)
        allr["Task Approved At"] = _stamp(allr.pop("approved_at"))       # when the client clicked Approved
        allr["Task Completed At"] = _stamp(allr.pop("completed_at"))     # when an admin clicked Complete (else empty)
        allr = add_urgency_cols(allr, n_theme)
        allr["status"] = allr["status"].map(lambda x: f"{theme.STATUS_DOT.get(x, '')} {x}".strip())
        st.dataframe(allr[["id", "summary", "raised_by", "added", "source", "category", "sentiment", "urgency",
                           "adj_urgency", "age_boost_pct", "repeat_boost_pct", "urgency_pct", "theme",
                           "Policy score", "Model Priority", "Client Priority", "status", "Task Approved At", "Task Completed At"]],
                     width="stretch", hide_index=True,
                     column_config={                                # wide enough for 'YYYY-MM-DD HH:MM:SS'
                         "Task Approved At": st.column_config.TextColumn("Task Approved At", width=190),
                         "Task Completed At": st.column_config.TextColumn("Task Completed At", width=190)})
