"""Results display shared by the User Access page (read-only) and the Admin Access page.

Everything is read from the existing SQLite tables (themes, requests, feedback) that the analysis pipeline fills,
so results survive page refreshes and navigation. Nothing here calls the AI.
Task workflow (all saved in the database, on the requests):  Pending -> Completed (user) -> Verified (admin),
or Completed -> Repost -> Pending. The user's list shows only Pending tasks.
"""
import pandas as pd
import streamlit as st

import db
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
                              r.status, r.completed_at, r.verified_at
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


def _complete(ids, name):
    _act(db.complete_requests, ids, f"“{name}” marked as Completed. An admin will verify it.", "complete this task")


def _verify(ids, name):
    _act(db.verify_requests, ids, f"“{name}” verified.", "verify this task")


def _repost(ids, name):
    _act(db.repost_requests, ids, f"“{name}” reposted to the active list.", "repost this task")


def _stamp(series):
    """Timestamps for display: 'YYYY-MM-DD HH:MM:SS', empty when not set."""
    return series.fillna("").astype(str).str.replace("T", " ").str.slice(0, 19)


# ---------------------------------------------------------------- admin: tasks waiting for review
def render_completed_review(data):
    """Admin Access: tasks users marked Completed, with Verify / Repost (in a pale-blue panel)."""
    with st.container(key="review_panel"):
        _completed_review_body(data)


def _completed_review_body(data):
    reqs, themes = data["requests"], data["themes"]
    st.subheader("Completed tasks · awaiting verification")
    done = reqs[reqs["status"] == "Completed"]
    n_ver = int((reqs["status"] == "Verified").sum())
    if done.empty:
        st.caption(f"No tasks are waiting for verification. Verified so far: {n_ver} request(s).")
        return
    names = dict(zip(themes["id"], themes["name"])) if not themes.empty else {}
    st.caption(f"{done['completed_at'].nunique()} task(s) submitted by users. Verified so far: {n_ver} request(s).")
    done = done.assign(_k=done["theme_id"].astype(str) + "|" + done["completed_at"].astype(str))
    for k, d in sorted(done.groupby("_k"), key=lambda kv: kv[1]["completed_at"].iloc[0], reverse=True):
        ids = [int(i) for i in d["id"]]
        name = names.get(d["theme_id"].iloc[0]) or d["summary"].iloc[0][:60]
        with st.container(border=True):
            st.markdown(f"**{name}** {theme.pill('Completed')}  ·  {len(d)} request(s)  ·  "
                        f"submitted {_stamp(d['completed_at']).iloc[0]}", unsafe_allow_html=True)
            st.dataframe(d[["summary", "raised_by", "category", "sentiment"]],
                         width="stretch", hide_index=True)
            b1, b2, _ = st.columns([1, 1, 4])
            b1.button("Verify", key=f"verify_{ids[0]}", type="primary", on_click=_verify, args=(ids, name))
            b2.button("Repost", key=f"repost_{ids[0]}", on_click=_repost, args=(ids, name))


# ---------------------------------------------------------------- ranked themes + all requests
def render_ranked_themes(data, empty_text="No themes yet. Add feedback and click Run analysis."):
    """User Access: ranked chart + the active (Pending) tasks, each with a Completed button."""
    themes, reqs = data["themes"], data["requests"]
    active = reqs[reqs["status"] == "Pending"]
    st.subheader("Ranked themes")

    shown = themes[themes["id"].isin(active["theme_id"].dropna())] if not themes.empty else themes
    if themes.empty:
        st.caption(empty_text)
    elif shown.empty:
        st.caption("No active tasks. Every task has been completed.")
    else:
        with st.container(key="chart_card"):                      # white card behind the chart
            st.bar_chart(shown.set_index("name")["score"], color=theme.TEXT2, horizontal=True,
                         x_label="Priority score (0-100)", y_label="")
        st.caption("Urgency % = model urgency × role weight × (1 + age boost + repeat boost), capped at 100%. "
                   "adj_urgency = after the role weight only.")
        for _, t in shown.iterrows():
            d = active[active["theme_id"] == t["id"]][["id", "summary", "raised_by", "created_at", "source",
                                                       "category", "sentiment", "urgency"]].copy()
            with st.expander(f"#{t['id']}  ·  {t['score']}  ·  {t['name']}  ({len(d)} requests)"):
                if isinstance(t["reason"], str) and t["reason"]:
                    st.write(f"**Why:** {t['reason']}")
                if t["description"]:
                    st.write(t["description"])
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
                st.button("✓ Completed", key=f"done_{ids[0]}", on_click=_complete, args=(ids, t["name"]), type="primary",
                          help="Mark this task as completed. It leaves your list and goes to an admin for verification.")


def render_all_requests(data):
    """Admin Access: every extracted request with its status and the two task timestamps (in a light-gray panel)."""
    with st.container(key="requests_panel"):
        _all_requests_body(data)


def _all_requests_body(data):
    reqs = data["requests"]
    with st.expander("All extracted requests"):
        allr = reqs.rename(columns={"theme_id": "theme"}).copy()
        allr["id"] = range(1, len(allr) + 1)                # display numbering always runs 1..N
        n_theme = allr.groupby("theme")["id"].transform("count").reindex(allr.index).fillna(1).astype(int)
        allr["Task Submitted At"] = _stamp(allr.pop("completed_at"))     # when the user clicked Completed
        allr["Task Verified At"] = _stamp(allr.pop("verified_at"))       # when an admin clicked Verify (else empty)
        allr = add_urgency_cols(allr, n_theme)
        allr["status"] = allr["status"].map(lambda x: f"{theme.STATUS_DOT.get(x, '')} {x}".strip())
        st.dataframe(allr[["id", "summary", "raised_by", "added", "source", "category", "sentiment", "urgency",
                           "adj_urgency", "age_boost_pct", "repeat_boost_pct", "urgency_pct", "theme",
                           "status", "Task Submitted At", "Task Verified At"]],
                     width="stretch", hide_index=True,
                     column_config={                                # wide enough for 'YYYY-MM-DD HH:MM:SS'
                         "Task Submitted At": st.column_config.TextColumn("Task Submitted At", width=190),
                         "Task Verified At": st.column_config.TextColumn("Task Verified At", width=190)})
