"""Landing-page analytics: two bar charts built from the real database (no mock data).

1. Requests Raised vs Requests Resolved per month / quarter of a financial year (April-March).
     Raised   = requests by their creation timestamp (the feedback entry's created_at, the same date used everywhere).
     Resolved = requests whose status is 'Verified', placed by their verification timestamp (verified_at).
2. Requests by category (existing `category` field), same filters, one series per category (by creation timestamp).
Periods without requests show 0. Both charts share one financial-year dropdown and one Monthly/Quarterly toggle.
Charts are drawn with Altair (the engine behind Streamlit's own charts).
"""
import math
from datetime import date

import pandas as pd
import streamlit as st

import db
import theme

MONTHS = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
QUARTERS = ["Q1 (Apr–Jun)", "Q2 (Jul–Sep)", "Q3 (Oct–Dec)", "Q4 (Jan–Mar)"]
RAISED, RESOLVED = "Requests Raised", "Requests Resolved"
SERIES_COLORS = [theme.RAISED, theme.RESOLVED]            # blue / green
CATEGORY_ORDER = ["UI/UX", "Performance", "Integration", "Reporting", "Mobile", "Security", "Other"]
CATEGORY_COLORS = theme.CATEGORY_COLORS                    # fixed colour per category, same everywhere in the app
EXTRA_COLORS = theme.EXTRA_COLORS                          # only for categories that are not listed above


# ---------------------------------------------------------------- financial years and periods
def fy_of(d):
    """Financial year (April-March) a date falls in, named by its starting year (Apr 2025 - Mar 2026 -> 2025)."""
    return d.year if d.month >= 4 else d.year - 1


def fy_label(start):
    return f"FY {start}–{str(start + 1)[-2:]}"


def financial_years(today=None, n=10):
    """The latest n financial years, newest first (start years)."""
    cur = fy_of(today or date.today())
    return [cur - i for i in range(n)]


def period_labels(freq):
    return MONTHS if freq == "Monthly" else QUARTERS


def _fy_series(ts):
    return ts.dt.year.where(ts.dt.month >= 4, ts.dt.year - 1)


def _per_period(ts, fy, freq):
    """Counts of timestamps per period of financial year `fy` (zeros included). ts = datetime Series."""
    ts = ts.dropna()
    ts = ts[_fy_series(ts) == fy]
    idx = (ts.dt.month - 4) % 12                       # Apr=0 ... Mar=11
    if freq == "Quarterly":
        idx = idx // 3
    n = len(period_labels(freq))
    return [int(x) for x in idx.value_counts().reindex(range(n), fill_value=0).tolist()]


# ---------------------------------------------------------------- data
def load_requests():
    """One row per request: category, status, creation time (from its feedback) and verification time."""
    with db.conn() as c:
        df = pd.read_sql("""SELECT r.id, r.category, r.status, r.verified_at, f.created_at
                            FROM requests r JOIN feedback f ON f.id=r.feedback_id""", c)
    df["created"] = pd.to_datetime(df["created_at"], format="ISO8601", errors="coerce")
    df["verified"] = pd.to_datetime(df["verified_at"], format="ISO8601", errors="coerce")
    df.loc[df["status"] != "Verified", "verified"] = pd.NaT          # resolved = Verified only
    df["category"] = df["category"].fillna("").replace("", "Other")
    return df


def overview_rows(df, fy, freq):
    labels = period_labels(freq)
    raised = _per_period(df["created"], fy, freq)
    resolved = _per_period(df["verified"], fy, freq)
    return ([{"period": p, "series": RAISED, "count": n} for p, n in zip(labels, raised)] +
            [{"period": p, "series": RESOLVED, "count": n} for p, n in zip(labels, resolved)])


def categories_in(df):
    present = set(df["category"])
    return [c for c in CATEGORY_ORDER if c in present] + sorted(present - set(CATEGORY_ORDER))


def category_rows(df, fy, freq, cats):
    labels = period_labels(freq)
    rows = []
    for cat in cats:
        counts = _per_period(df.loc[df["category"] == cat, "created"], fy, freq)
        rows += [{"period": p, "series": cat, "count": n} for p, n in zip(labels, counts)]
    return rows


def category_palette(cats):
    extra = iter(EXTRA_COLORS * 3)
    return [CATEGORY_COLORS.get(c) or next(extra) for c in cats]


# ---------------------------------------------------------------- charts (Altair)
def _grouped_bars(rows, series, colors, freq, labels_on):
    import altair as alt
    top = max([r["count"] for r in rows] + [0])
    ymax = max(4, math.ceil(top * 1.18))                              # headroom for the value labels
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("period:N", sort=period_labels(freq), axis=alt.Axis(title=None, labelAngle=0, labelLimit=140)),
        xOffset=alt.XOffset("series:N", sort=series),
        y=alt.Y("count:Q", scale=alt.Scale(domain=[0, ymax], nice=False),
                axis=alt.Axis(title="Requests", tickMinStep=1, format="d")),
        color=alt.Color("series:N", sort=series, scale=alt.Scale(domain=series, range=colors),
                        legend=alt.Legend(title=None, orient="top", symbolType="square", columns=0)),
        tooltip=[alt.Tooltip("period:N", title="Period"), alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("count:Q", title="Requests")])
    chart = base.mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
    if labels_on:                                                     # numbers above bars, including 0
        chart = chart + base.mark_text(dy=-7, fontSize=11, color=theme.TEXT).encode(text="count:Q")
    return (chart.properties(height=320)
            .configure(background=theme.CARD, font="Inter, 'Segoe UI', Helvetica, Arial, sans-serif")
            .configure_view(stroke=None)
            .configure_axis(labelColor=theme.TEXT2, titleColor=theme.TEXT2, gridColor=theme.BORDER, domainColor=theme.BORDER,
                            tickColor=theme.BORDER, labelFontSize=12, titleFontSize=12)
            .configure_legend(labelColor=theme.TEXT2, labelFontSize=12, symbolSize=110))


def _show(chart):
    try:
        st.altair_chart(chart, theme=None, width="stretch")
    except TypeError:                                                 # older Streamlit
        st.altair_chart(chart, theme=None, use_container_width=True)


# ---------------------------------------------------------------- UI
CSS = """
<style>
.st-key-voc_charts{background:__CARD__;border:1px solid __BORDER__;border-radius:14px;padding:1.5rem 1.7rem 1.2rem;
  margin:7vh auto 0;width:min(1080px,100%);box-shadow:0 1px 3px rgba(17,24,39,.06);color:__TEXT__;}
.voc-card-title{font-size:1.3rem;font-weight:700;color:__TEXT__;letter-spacing:-.01em;}
.voc-card-sub{color:__TEXT2__;font-size:.9rem;margin-top:.15rem;}
.voc-chart-h{font-weight:650;color:__TEXT__;font-size:1.02rem;margin:1.1rem 0 .1rem;}
.voc-note{color:__MUTED__;font-size:.8rem;margin-top:.6rem;}
.st-key-voc_charts label p{color:__TEXT2__;font-weight:600;font-size:.85rem;}
.st-key-voc_charts [data-baseweb="select"]>div{background:__CARD__;border:1px solid __BORDER__;border-radius:8px;color:__TEXT__;}
.st-key-voc_charts [data-baseweb="select"]>div:hover,.st-key-voc_charts [data-baseweb="select"]>div:focus-within{border-color:__PRIMARY__;}
ul[role="listbox"] li[aria-selected="true"]{background:__TINT_BLUE__ !important;}
.st-key-voc_charts [class*="st-key-freq_"] button{border-radius:999px;padding:.2rem 1.1rem;min-height:2.3rem;
  border:1px solid __BORDER__;background:__CARD__;color:__TEXT2__;transition:background-color .2s,color .2s,border-color .2s;}
.st-key-voc_charts [class*="st-key-freq_"] button p{color:inherit;font-size:.9rem;}
.st-key-voc_charts [class*="st-key-freq_"] button:hover{background:__TINT_BLUE__;color:__PRIMARY__;border-color:__PRIMARY__;}
.st-key-voc_charts [class*="st-key-freq_"] button:focus-visible{outline:2px solid __PRIMARY__;outline-offset:2px;}
@media (max-width:640px){.st-key-voc_charts{padding:1.1rem 1rem;margin-top:5vh;}}
@media (prefers-reduced-motion:reduce){.st-key-voc_charts [class*="st-key-freq_"] button{transition:none;}}
</style>
"""
for _k, _v in {"CARD": theme.CARD, "BORDER": theme.BORDER, "TEXT": theme.TEXT, "TEXT2": theme.TEXT2, "MUTED": theme.MUTED,
               "PRIMARY": theme.PRIMARY, "TINT_BLUE": theme.TINT_BLUE}.items():
    CSS = CSS.replace(f"__{_k}__", _v)


def _set_freq(freq):
    st.session_state["home_freq"] = freq


def render():
    """Landing page: the white card with both charts (below the title and navigation)."""
    st.session_state.setdefault("home_freq", "Monthly")
    freq = st.session_state["home_freq"]
    active = (f"<style>.st-key-voc_charts .st-key-freq_{freq} button{{background:{theme.PRIMARY} !important;"
              f"color:#fff !important;border-color:{theme.PRIMARY} !important;}}</style>")
    st.markdown(CSS + active, unsafe_allow_html=True)
    with st.container(key="voc_charts"):
        st.markdown('<div class="voc-card-title">Request overview</div>'
                    '<div class="voc-card-sub">Live data from your tracker · financial year runs April to March</div>',
                    unsafe_allow_html=True)
        c1, c2, _ = st.columns([2, 2, 3])
        with c1:
            fy = st.selectbox("Financial year", financial_years(), format_func=fy_label, key="home_fy")
        with c2:
            st.markdown('<div style="font-size:.85rem;font-weight:600;color:#333;margin-bottom:.35rem">View</div>',
                        unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            b1.button("Monthly", key="freq_Monthly", on_click=_set_freq, args=("Monthly",))
            b2.button("Quarterly", key="freq_Quarterly", on_click=_set_freq, args=("Quarterly",))

        try:
            df = load_requests()
        except Exception as e:
            st.error(f"Could not load chart data: {e}")
            return
        st.markdown(f'<div class="voc-chart-h">Requests raised vs resolved · {fy_label(fy)} · {freq.lower()}</div>',
                    unsafe_allow_html=True)
        rows = overview_rows(df, fy, freq)
        _show(_grouped_bars(rows, [RAISED, RESOLVED], SERIES_COLORS, freq, True))

        st.markdown(f'<div class="voc-chart-h">Requests by category · {fy_label(fy)} · {freq.lower()}</div>',
                    unsafe_allow_html=True)
        cats = categories_in(df)
        if cats:
            _show(_grouped_bars(category_rows(df, fy, freq, cats), cats, category_palette(cats), freq, False))
        else:
            st.caption("No categorised requests yet.")
        if not any(r["count"] for r in rows):
            st.caption(f"No requests were raised or resolved in {fy_label(fy)}, so every value is 0.")
        st.markdown('<div class="voc-note">Raised = request creation date. Resolved = status Verified, counted in the '
                    'period of its verification date. Categories are counted by creation date.</div>',
                    unsafe_allow_html=True)
