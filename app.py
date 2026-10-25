import re
from datetime import date, datetime

import pandas as pd
import streamlit as st

import aira
import db
import llm
import pipeline
import results
import roles
import ui
from llm import setting

st.set_page_config(page_title="VOC Enhancement Tracker", page_icon="⬛", layout="wide")
db.init()

# Black & white look: neutral alert boxes, black primary buttons
st.markdown("""
<style>
div[data-testid="stAlert"]{background:#F4F4F4;border:1px solid #111;border-radius:6px;}
div[data-testid="stAlert"] *{color:#111 !important;}
div[data-testid="stAlert"] svg{fill:#111 !important;}
.stButton>button[kind="primary"]{background:#111;color:#fff;border:1px solid #111;}
.stButton>button[kind="primary"]:hover{background:#333;border-color:#333;color:#fff;}
h1{font-weight:700;letter-spacing:-0.02em;}
</style>
""", unsafe_allow_html=True)

MAX_ATTEMPTS = setting("MAX_ATTEMPTS", 3)
W = roles.weights()


def add_cb():
    """Runs before the page reloads: add the entry, then empty the text box."""
    role, num = st.session_state.get("role_sel", "user"), st.session_state.get("num_sel")
    src, text = st.session_state.get("src_sel", "ticket"), st.session_state.get("paste_text", "")
    created = None                             # default: the moment it is added
    if st.session_state.get("manual_dt") and st.session_state.get("date_sel") and st.session_state.get("time_sel"):
        created = datetime.combine(st.session_state["date_sel"], st.session_state["time_sel"]).isoformat(timespec="seconds")
    if num:
        res = db.add_feedback(src, text, roles.make(role, num), created_at=created)
        how = ""
    else:                                      # number left blank -> next free number for the role
        res = db.add_feedback(src, text, auto_role=role, created_at=created)
        how = " (number assigned automatically)"
    if res is not None:
        st.session_state["paste_text"] = ""
        st.session_state["flash"] = ("success", f"Added, raised by {res}{how}.")
    else:
        st.session_state["flash"] = ("warning", "Empty or duplicate (same text from the same person), not added.")


def delete_cb():
    n = db.delete_feedback(st.session_state.get("del_ids", []))
    pipeline.refresh_theme_stats()
    st.session_state["del_ids"] = []
    st.session_state["stale"] = True
    st.session_state["flash"] = ("success", f"Deleted {n} entr{'y' if n == 1 else 'ies'}. "
                                 "Click Run analysis to re-cluster what is left.")



def admin_page():
    """Admin Access: input forms, CSV import, analysis, entries, then the results (unchanged workflow)."""
    if not ui.admin_allowed():
        return
    results.show_flash()
    lim, used = llm.limits(), db.calls_today()

    # ---------- sidebar: connection + API budget ----------
    with st.sidebar:
        st.subheader("Azure OpenAI")
        miss = llm.missing_config()
        if miss:
            st.error("Missing in .env: " + ", ".join(miss))
            st.caption(llm.env_hint())
        else:
            st.write(f"**Deployment:** {llm.deployment_name()}")
            st.caption(f"Host: {llm.endpoint_host()}")
            st.caption("Embeddings: " + (llm.embed_model_name() if llm.has_embeddings()
                                         else "none (local TF-IDF clustering)"))
        st.metric("API calls today", f"{used} / {lim['day']}")
        st.progress(min(used / max(lim["day"], 1), 1.0))
        st.caption(f"Safety limits: {lim['run']} calls per run · {setting('MAX_ITEMS_PER_RUN', 30)} items per run")
        if not miss and st.button("Test connection"):
            with st.spinner("Testing…"):
                ok, msg = llm.test_connection()
            (st.success if ok else st.error)(msg)
        st.divider()
        st.subheader("Priority weights")
        st.caption("How much each role counts in the theme score (edit ROLE_WEIGHTS in .env):")
        st.caption(" · ".join(f"{r} ×{W[r]:g}" for r in sorted(W, key=lambda r: -W[r])))
        st.caption(f"Age boost: +{setting('AGE_BOOST_PCT_PER_STEP', 5)}% urgency per {setting('AGE_STEP_DAYS', 10)} days "
                   f"(max +{setting('AGE_BOOST_MAX_PCT', 30)}%). Repeat boost: +{setting('REPEAT_BOOST_PCT_EACH', 5)}% per "
                   f"extra request on a theme (max +{setting('REPEAT_BOOST_MAX_PCT', 20)}%). Urgency never exceeds 100%.")

    # ---------- header + counters ----------
    cn = db.counts(MAX_ATTEMPTS)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Feedback items", cn["total"])
    m2.metric("Analyzed", cn["done"])
    m3.metric("Pending", cn["pending"])
    m4.metric("Parked (errors)", cn["failed"])

    flash = st.session_state.pop("flash", None)
    if flash:
        (st.success if flash[0] == "success" else st.warning)(flash[1])

    st.divider()

    # ---------- one page: add feedback (left) + analyze (right) ----------
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Add feedback")
        c_src, c_role, c_num = st.columns([2, 2, 1])
        c_src.selectbox("Source", ["ticket", "email", "survey", "form"], key="src_sel")
        role = c_role.selectbox("Raised by", roles.ROLES, key="role_sel")
        num = c_num.number_input("No.", min_value=1, max_value=999, value=None, step=1, key="num_sel",
                                 placeholder="auto")
        counts = db.raiser_counts()
        mine = sorted((u for u in counts if roles.role_of(u) == role),
                      key=lambda u: int(re.sub(r"\D", "", u) or 0))
        if num:
            who = roles.make(role, num)
            n = counts.get(who, 0)
            st.caption(f"{who} already has {n} entr{'y' if n == 1 else 'ies'}: this one is added to the same person."
                       if n else f"{who} is a new person.")
        else:
            st.caption(f"Number left blank → this will be {role}{db.next_number(role)}. "
                       f"Existing {role}s: {', '.join(mine) if mine else 'none'}. Type a number to reuse one.")
        st.checkbox("Set date and time manually (default: now)", key="manual_dt")
        if st.session_state.get("manual_dt"):
            c_d, c_t = st.columns(2)
            c_d.date_input("Date", value=date.today(), max_value=date.today(), key="date_sel")
            c_t.time_input("Time", value=datetime.now().replace(second=0, microsecond=0).time(), key="time_sel", step=60)
        st.text_area("Paste feedback", height=120, key="paste_text",
                     placeholder="Paste one ticket, email, survey or form response…")
        st.button("Add item", on_click=add_cb)

        st.markdown("**Or import a CSV** (columns: `source`, `user`, `date`, `text`; `user` like client2 = that person, "
                    "`client` alone = next free number; `date` like 2026-10-01 14:30 is used as the entry's date, blank = now; max 500 rows)")
        up = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
        if up and st.button("Import CSV"):
            df = db.normalize_columns(pd.read_csv(up))
            if "text" not in df.columns:
                st.error(f"CSV needs a `text` column. Columns found: {', '.join(map(str, df.columns))}")
            else:
                res = db.import_rows(df)
                st.session_state["stale"] = True
                st.success(f"Imported {res['added']} new row(s); skipped {res['dup']} empty/duplicate.")
                if "date" in df.columns:
                    st.info(f"Dates read from the CSV for {res['dated']} row(s).")
                else:
                    st.warning("No date column found (accepted names: date, datetime, date and time, timestamp). "
                               f"Entries were dated now. Columns found: {', '.join(map(str, df.columns))}")
                if res["redated"]:
                    st.info(f"{res['redated']} already-stored entr(y/ies) matched by text; their date was updated from the CSV.")
                if res["bad_date"]:
                    st.warning(f"{res['bad_date']} row(s) had a date that could not be read (use e.g. 2026-10-01 14:30, "
                               "or set DATE_DAYFIRST=1 in .env for 31/12/2026 style); they were dated now.")
                if res["auto"]:
                    st.info(f"{res['auto']} row(s) had a role without a number and got the next free number automatically.")
                if "user" not in df.columns:
                    st.info("No `user` column found: entries were added without a raiser (weight ×1).")
                if bad_user_note := res["bad_user"]:
                    st.warning(f"{bad_user_note} row(s) had an unrecognised `user` value (use user/client/developer/tester + "
                               "number) and were imported without a raiser.")
                if len(df) > 500:
                    st.warning("Only the first 500 rows were read.")

    with right:
        st.subheader("Analyze")
        cn = db.counts(MAX_ATTEMPTS)
        this_run = min(cn["pending"], setting("MAX_ITEMS_PER_RUN", 30))
        st.write(f"**{cn['pending']}** pending. This run processes up to **{this_run}** item(s), "
                 f"about {this_run} API calls plus a few for theme naming (hard cap {lim['run']}).")
        if st.session_state.get("stale") and not cn["pending"]:
            st.info("Entries changed since the last analysis. Click Run analysis to re-cluster what is left.")
        if cn["failed"]:
            st.warning(f"{cn['failed']} item(s) failed {MAX_ATTEMPTS} times and are parked so they can't use more calls.")
            if st.button("Retry failed items"):
                db.reset_failed()
                st.rerun()

        if st.button("Run analysis", type="primary", disabled=bool(miss) or cn["pending"] + cn["done"] == 0):
            if st.session_state.get("running"):
                st.warning("A run is already in progress.")
            else:
                st.session_state.running = True
                try:
                    bar = st.progress(0.0, "Extracting requests…")
                    res = pipeline.run_analysis(lambda x: bar.progress(x))
                    bar.empty()
                    st.session_state["stale"] = bool(res["stopped"])
                    st.success(f"Analyzed {res['done']} new item(s) using {res['calls']} API call(s)"
                               + (f" → {res['themes']} themes (all current entries re-clustered)."
                                  if res["themes"] is not None else "."))
                    if res["empty"]:
                        st.info(f"{res['empty']} of {res['done']} item(s) contained no enhancement request "
                                "(praise, comments or questions only), so they create no themes.")
                    if res["failed"]:
                        st.warning(f"{res['failed']} item(s) failed (retried up to {MAX_ATTEMPTS} times in total).")
                    if res["stopped"]:
                        st.error(f"Run stopped early to protect your API quota: {res['stopped']}")
                    elif res["first_error"]:
                        st.error(f"First error: {res['first_error']}")
                    if res["remaining"]:
                        st.info(f"{res['remaining']} item(s) still pending. Click Run analysis again to continue.")
                finally:
                    st.session_state.running = False

        with st.expander("Reset data"):
            st.caption("Clears feedback, requests and themes. The API usage counter is kept.")
            if st.button("Clear all data"):
                db.reset_data()
                st.session_state["stale"] = False
                st.rerun()

    st.divider()

    # ---------- manage entries ----------
    st.subheader("Feedback entries")
    with db.conn() as c:
        fb = pd.read_sql("""SELECT f.id, f.raised_by, f.created_at, f.source, f.text, f.processed, f.attempts,
                            (SELECT COUNT(*) FROM requests r WHERE r.feedback_id=f.id) AS requests
                            FROM feedback f ORDER BY f.id""", c)
    if fb.empty:
        st.caption("No entries yet.")
    else:
        fb["raised_by"] = fb["raised_by"].fillna("")
        fb["status"] = fb.apply(lambda r: "analyzed" if r["processed"] else
                                ("parked" if r["attempts"] >= MAX_ATTEMPTS else "pending"), axis=1)
        fb["added"] = fb["created_at"].str.slice(0, 16).str.replace("T", " ")
        fb["no"] = range(1, len(fb) + 1)                    # display numbering always runs 1..N
        st.dataframe(fb[["no", "raised_by", "added", "source", "text", "status", "requests"]].rename(columns={"no": "id"}),
                     width="stretch", hide_index=True)
        label_of = {r["id"]: f"#{r['no']} {r['raised_by'] or '-'}  {r['text'][:60]}" for _, r in fb.iterrows()}
        st.multiselect("Select entries to delete (resolved or entered by mistake)", list(fb["id"]),
                       key="del_ids", format_func=lambda i: label_of[i])
        n_sel = len(st.session_state.get("del_ids", []))
        st.button(f"Delete selected ({n_sel})", disabled=n_sel == 0, on_click=delete_cb)
        st.caption("Deleting removes the entry and the requests extracted from it. "
                   "Then click Run analysis: remaining entries (plus any new ones) are re-clustered "
                   "using stored data, with no new extraction calls.")

    st.divider()


    # ---------- results (saved in the database; same display the User Access page shows) ----------
    try:
        data = results.load_results()
    except Exception as e:
        st.error(f"Could not read the saved results: {e}")
    else:
        results.render_completed_review(data)              # tasks users marked Completed: Verify / Repost
        st.divider()
        results.render_all_requests(data)                  # Admin Access: requests table only (no ranked themes)


def user_page():
    """User Access: read-only view of results already saved by Admin Access. Never starts an analysis."""
    st.subheader("Results")
    results.show_flash()
    try:
        with st.spinner("Loading results…"):
            data = results.load_results()
    except Exception as e:                                   # error state
        st.error(f"Could not load results: {e}")
        st.button("Try again", key="user_retry")             # any click reruns the page
        return
    if data["empty"]:                                        # empty state
        st.info("No results generated. Please upload data.")
        ui.link_button("Go to Admin Access", "admin", "user_to_admin", type="primary")
        return
    pend = data["requests"][data["requests"]["status"] == "Pending"]       # loaded state
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active themes", pend["theme_id"].nunique())
    m2.metric("Active requests", len(pend))
    m3.metric("Completed (awaiting verification)", int((data["requests"]["status"] == "Completed").sum()))
    m4.metric("Verified", int((data["requests"]["status"] == "Verified").sum()))
    results.render_ranked_themes(data)                  # User Access: ranked themes only


# ---------- router: Home / User Access / Admin Access ----------
view = ui.current_view()
ui.render_chrome(view)
if view != "home":
    with ui.page(view):
        user_page() if view == "user" else admin_page()

# ---------- Aira chat bubble (bottom-right, on every view) ----------
aira.render(on_dark=(view == "home"))
