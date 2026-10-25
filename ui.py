"""Navigation chrome for the three views: Home, User Access, Admin Access.

- Router: st.session_state["view"] mirrored to the URL (?view=user / ?view=admin), so a browser refresh or a
  bookmarked link stays on the same page. Home has no query string.
- Home: dark cherry red background, centered title + two muted text options (hover = grow slightly + white).
  Other pages: white background (animated transition).
- Other views: title slides to the top-left, the options become a horizontal nav bar (active = black, larger),
  a thin divider, then the page content fades/slides up.
- All animation is plain CSS and is switched off for users who prefer reduced motion.
- Optional: set ADMIN_PASSWORD in .env to require a password for Admin Access (off by default).
"""
import hmac
import os

import streamlit as st

import analytics

TITLE = "VOC Enhancement Tracker"
SUBTITLE = "Customer feedback → enhancement requests → ranked themes"
VIEWS = {"user": "User Access", "admin": "Admin Access"}
GRAY, BLACK = "#9AA0A6", "#111111"

CSS = """
<style>
.block-container{padding-top:2.2rem !important;padding-bottom:6.5rem !important;}   /* room for the Aira bubble */

/* ---- text-style buttons (nav options + title) ---- */
.st-key-voc_home button,.st-key-voc_top button{background:transparent !important;border:none !important;
  box-shadow:none !important;color:__GRAY__ !important;padding:.3rem .6rem !important;min-height:0 !important;
  width:auto;transition:transform .2s ease,color .2s ease;transform-origin:center;}
.st-key-voc_home button p,.st-key-voc_top button p{color:inherit !important;font-size:inherit;font-weight:inherit;}
.st-key-voc_home button{font-size:1.4rem;font-weight:500;}
.st-key-voc_top button{font-size:1.05rem;font-weight:500;}
.st-key-voc_home button:hover,.st-key-voc_home button:focus-visible,
.st-key-voc_top button:hover,.st-key-voc_top button:focus-visible{color:__BLACK__ !important;transform:scale(1.08);}
.st-key-voc_home button:focus-visible,.st-key-voc_top button:focus-visible{outline:2px solid __BLACK__;outline-offset:3px;}

/* ---- home ---- */
.st-key-voc_home{margin-top:12vh;}
.voc-hero{text-align:center;animation:voc-fade .5s ease both;}
.voc-title-home{font-size:3rem;font-weight:700;letter-spacing:-.03em;color:__BLACK__;line-height:1.15;}
.voc-sub{color:__GRAY__;margin:.6rem 0 2.2rem;font-size:1.05rem;}
.st-key-voc_home [data-testid="stHorizontalBlock"]{justify-content:center;gap:2.5rem;flex-wrap:wrap;}

/* ---- top bar (non-home views) ---- */
.st-key-voc_top [data-testid="stHorizontalBlock"]{align-items:center;flex-wrap:wrap;gap:.25rem 1.5rem;}
.st-key-voc_top [data-testid="stColumn"],.st-key-voc_home [data-testid="stColumn"]{flex:0 0 auto !important;
  width:auto !important;min-width:0 !important;}
.st-key-voc_top [data-testid="stColumn"]:first-child{margin-right:auto;}
.st-key-title_home{animation:voc-title-slide .5s cubic-bezier(.2,.7,.2,1) both;transform-origin:left center;}
.st-key-title_home button{font-size:1.5rem !important;font-weight:700 !important;letter-spacing:-.02em;
  color:__BLACK__ !important;padding-left:0 !important;}
.st-key-nav_user,.st-key-nav_admin{animation:voc-nav-in .45s ease both;}
.voc-divider{border:0;border-top:1px solid #E3E5E8;margin:.4rem 0 1.4rem;animation:voc-fade .5s ease both;}

/* ---- page content ---- */
[class*="st-key-voc_page_"]{animation:voc-rise .35s ease both;}

@keyframes voc-title-slide{from{transform:translateX(calc(50vw - 50% - 4rem)) scale(1.7);opacity:.5}
  to{transform:none;opacity:1}}
@keyframes voc-nav-in{from{opacity:0;transform:translateY(28px)}to{opacity:1;transform:none}}
@keyframes voc-rise{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes voc-fade{from{opacity:0}to{opacity:1}}

/* ---- phones / tablets ---- */
@media (max-width:900px){.voc-title-home{font-size:2.3rem}.st-key-voc_home{margin-top:10vh}}
@media (max-width:640px){.voc-title-home{font-size:1.8rem}.voc-sub{font-size:.95rem}
  .st-key-voc_home button{font-size:1.2rem}.st-key-title_home button{font-size:1.2rem !important}
  .st-key-voc_home [data-testid="stHorizontalBlock"]{gap:1rem}}

@media (prefers-reduced-motion:reduce){
  .voc-hero,.voc-divider,.st-key-title_home,.st-key-nav_user,.st-key-nav_admin,[class*="st-key-voc_page_"]{animation:none !important;}
  .st-key-voc_home button,.st-key-voc_top button{transition:none !important;transform:none !important;}
}
</style>
""".replace("__GRAY__", GRAY).replace("__BLACK__", BLACK)


CHERRY_DARK = "#A50A1A"      # dark cherry red (a true red, not purple-leaning burgundy)

# page background: white everywhere, dark cherry on the landing page; .stApp transitions between them
BASE_THEME = """
<style>
.stApp{background:#FFFFFF;transition:background-color .45s ease;}
[data-testid="stHeader"]{background:transparent;}
@media (prefers-reduced-motion:reduce){.stApp{transition:none !important;}}
</style>
"""
HOME_THEME = f"""
<style>
.stApp{{background:{CHERRY_DARK} !important;}}
[data-testid="stHeader"] *{{color:#fff !important;}}
.voc-title-home{{color:#fff;}}
.voc-sub{{color:rgba(255,255,255,.8);}}
.st-key-voc_home button{{color:rgba(255,255,255,.62) !important;}}
.st-key-voc_home button:hover,.st-key-voc_home button:focus-visible{{color:#fff !important;}}
.st-key-voc_home button:focus-visible{{outline-color:#fff;}}
</style>
"""


# ---------------------------------------------------------------- router
def current_view():
    """Home / user / admin. Taken from session state, falling back to the URL (refresh, bookmarks)."""
    if "view" not in st.session_state:
        v = st.query_params.get("view", "")
        st.session_state["view"] = v if v in VIEWS else "home"
    return st.session_state["view"]


def go(view):
    """Button callback: switch page and mirror it in the URL."""
    st.session_state["view"] = view
    if view in VIEWS:
        st.query_params["view"] = view
    elif "view" in st.query_params:
        del st.query_params["view"]


def link_button(label, view, key, **kw):
    """A real button (keyboard accessible, keeps the chat/session) that navigates to a view."""
    return st.button(label, key=key, on_click=go, args=(view,), **kw)


# ---------------------------------------------------------------- chrome
def render_chrome(view):
    """Title + navigation. Home = centered; other views = top bar with divider."""
    st.markdown(CSS + BASE_THEME + (HOME_THEME if view == "home" else ""), unsafe_allow_html=True)
    if view == "home":
        with st.container(key="voc_home"):
            st.markdown(f'<div class="voc-hero"><div role="heading" aria-level="1" class="voc-title-home">{TITLE}</div>'
                        f'<p class="voc-sub">{SUBTITLE}</p></div>', unsafe_allow_html=True)
            cols = st.columns(2)
            with cols[0]:
                link_button("User Access", "user", "nav_user")
            with cols[1]:
                link_button("Admin Access", "admin", "nav_admin")
        analytics.render()                      # request charts: a short scroll below the title
        return

    # active option: black + slightly larger (rules are generated for the current view only)
    st.markdown(f"<style>.st-key-nav_{view} button{{color:{BLACK} !important;font-weight:650 !important;"
                f"transform:scale(1.12) !important;}}</style>", unsafe_allow_html=True)
    with st.container(key="voc_top"):
        c_title, c_user, c_admin = st.columns(3)
        with c_title:
            link_button(TITLE, "home", "title_home", help="Back to Home")
        with c_user:
            link_button("User Access", "user", "nav_user")
        with c_admin:
            link_button("Admin Access", "admin", "nav_admin")
    st.markdown('<hr class="voc-divider">', unsafe_allow_html=True)


def page(view):
    """Container for the selected page's content (fades/slides up when the page changes)."""
    return st.container(key=f"voc_page_{view}")


# ---------------------------------------------------------------- optional admin gate
def admin_allowed():
    """True if Admin Access may be shown. With ADMIN_PASSWORD set in .env a password is required
    (checked on the server; the page content is not rendered until it matches). Without it: open, as before."""
    pw = os.getenv("ADMIN_PASSWORD", "")
    if not pw or st.session_state.get("admin_ok"):
        return True
    st.subheader("Admin sign-in")
    with st.form("admin_login"):
        entered = st.text_input("Admin password", type="password")
        ok = st.form_submit_button("Sign in")
    if ok:
        if hmac.compare_digest(entered.encode(), pw.encode()):
            st.session_state["admin_ok"] = True
            st.rerun()
        st.error("Wrong password.")
    return False
