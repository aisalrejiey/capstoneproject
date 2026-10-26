"""Single source of truth for the app's look (EY-inspired, light enterprise theme).

Every colour used by the pages, charts, badges and the Aira widget comes from here, so a change in this file
restyles the landing page, User Access and Admin Access together. Streamlit's own widgets pick up the same
palette from .streamlit/config.toml (restart Streamlit after changing that file).
"""

# ---- surfaces and text ----
BG = "#F8F8FC"          # page background
CARD = "#FFFFFF"        # cards and chart backgrounds
BORDER = "#E5E7EB"      # grid lines and borders
TEXT = "#111827"        # primary text
TEXT2 = "#374151"       # secondary text, chart labels, dark-gray bars
MUTED = "#6B7280"       # inactive navigation, captions
TINT_BLUE = "#EFF6FF"   # pale blue section tint
TINT_GRAY = "#F3F4F6"   # light gray section tint

# ---- brand ----
PRIMARY = "#2563EB"         # primary buttons and active elements
PRIMARY_HOVER = "#1D4ED8"   # hover state of PRIMARY
YELLOW = "#FFE600"          # EY accent: key KPIs and highlights only

# ---- functional colours (used identically in charts, legends, badges, indicators) ----
CATEGORY_COLORS = {
    "UI/UX": "#2563EB",
    "Performance": "#10B981",
    "Integration": "#F59E0B",
    "Reporting": "#8B5CF6",
    "Mobile": "#14B8A6",
    "Security": "#EF4444",      # not in the brief: red = critical/protective, kept distinct from the five above
    "Other": "#9CA3AF",         # not in the brief: neutral gray
}
EXTRA_COLORS = ["#6B7280", "#4B5563", "#D1D5DB"]     # only for categories that are not listed above
RAISED, RESOLVED = "#2563EB", "#10B981"              # Requests Raised / Requests Resolved
STATUS_COLORS = {"Pending": "#F59E0B", "Completed": "#2563EB", "Verified": "#10B981"}
STATUS_DOT = {"Pending": "🟠", "Completed": "🔵", "Verified": "🟢"}   # same colours, usable inside tables


def _contrast(hex_a, hex_b):
    def lum(h):
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
        f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
    hi, lo = sorted((lum(hex_a), lum(hex_b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def on_color(bg):
    """Readable text colour (white or primary text) for a filled badge of colour bg."""
    return "#FFFFFF" if _contrast("#FFFFFF", bg) >= _contrast(TEXT, bg) else TEXT


def pill(status):
    """Small colored status badge (HTML) for Pending / Completed / Verified."""
    bg = STATUS_COLORS.get(status, MUTED)
    return f'<span class="voc-pill" style="background:{bg};color:{on_color(bg)}">{status}</span>'


GLOBAL_CSS = f"""
<style>
/* ---------- base ---------- */
.stApp{{background:{BG};color:{TEXT};}}
[data-testid="stHeader"]{{background:transparent;}}
.block-container{{padding-top:2.2rem !important;padding-bottom:6.5rem !important;max-width:1240px;}}
h1,h2,h3{{color:{TEXT};letter-spacing:-.01em;}}
h3{{font-weight:650;}}
[data-testid="stCaptionContainer"]{{color:{TEXT2};}}
hr{{border-color:{BORDER} !important;}}
[data-testid="stSidebar"]{{background:{TINT_GRAY};border-right:1px solid {BORDER};}}

/* ---------- buttons: white/outlined by default, blue for primary actions ---------- */
.stButton>button,[data-testid="stFormSubmitButton"] button{{border-radius:8px;border:1px solid {BORDER};
  background:{CARD};color:{TEXT};font-weight:500;transition:border-color .15s,color .15s,background-color .15s;}}
.stButton>button:hover,[data-testid="stFormSubmitButton"] button:hover{{border-color:{PRIMARY};color:{PRIMARY};}}
.stButton>button[kind="primary"],[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]{{
  background:{PRIMARY};border-color:{PRIMARY};color:#fff;}}
.stButton>button[kind="primary"]:hover,[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover{{
  background:{PRIMARY_HOVER};border-color:{PRIMARY_HOVER};color:#fff;}}
.stButton>button:focus-visible{{outline:2px solid {PRIMARY};outline-offset:2px;}}

/* ---------- KPI cards (st.metric) ---------- */
[data-testid="stMetric"]{{background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:.9rem 1.1rem;}}
[data-testid="stMetricLabel"] p{{color:{TEXT2};font-size:.85rem;}}
[data-testid="stMetricValue"]{{color:{TEXT};font-weight:700;}}
.st-key-kpi_band,.st-key-kpi_band_admin{{background:{TINT_BLUE};border:1px solid {BORDER};border-radius:14px;
  padding:1rem 1.1rem;margin-bottom:1.2rem;}}
.st-key-kpi_band [data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"],
.st-key-kpi_band_admin [data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"]{{border-top:3px solid {YELLOW};}}
.st-key-kpi_band [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"],
.st-key-kpi_band_admin [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"]{{border-top:3px solid {PRIMARY};}}
.st-key-kpi_band [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"],
.st-key-kpi_band_admin [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"]{{border-top:3px solid {STATUS_COLORS["Pending"]};}}
.st-key-kpi_band [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"]{{border-top:3px solid {STATUS_COLORS["Verified"]};}}
.st-key-kpi_band_admin [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"]{{border-top:3px solid #EF4444;}}

/* ---------- panels ---------- */
.st-key-chart_card{{background:{CARD};border:1px solid {BORDER};border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:1rem;}}
.st-key-review_panel{{background:{TINT_BLUE};border:1px solid {BORDER};border-radius:14px;padding:1rem 1.2rem;}}
.st-key-requests_panel{{background:{TINT_GRAY};border:1px solid {BORDER};border-radius:14px;padding:1rem 1.2rem;}}
[data-testid="stExpander"] details{{background:{CARD};border:1px solid {BORDER};border-radius:12px;}}
[data-testid="stExpander"] summary:hover{{color:{PRIMARY};}}
[data-testid="stAlert"]{{border-radius:10px;}}

/* ---------- status pill ---------- */
.voc-pill{{display:inline-block;font-size:.72rem;font-weight:600;padding:.1rem .6rem;border-radius:999px;
  vertical-align:middle;margin-left:.4rem;letter-spacing:.01em;}}
</style>
"""
