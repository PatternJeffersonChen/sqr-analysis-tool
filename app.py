"""
SQR Analysis Tool — Pattern
Search Query Report Analyser
"""

from __future__ import annotations

import io
import math
from html import escape as esc

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sqr_engine import (
    AnalysisResult,
    NegativeMatchType,
    QueryPath,
    Severity,
    analyse_sqr,
)

# ═════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Pattern — SQR Analysis",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 30'><path d='M23.1 0.3L0.3 22.8c-0.4 0.4-0.4 1 0 1.3l5.6 5.6c0.4 0.4 1 0.4 1.4 0L30.1 7.2c0.4-0.4 0.4-1 0-1.3L24.5 0.3C24.1-0.1 23.5-0.1 23.1 0.3z' fill='%23009BFF'/><path d='M32.5 9.6L19.1 22.8c-0.4 0.4-0.4 1 0 1.3l5.6 5.6c0.4 0.4 1 0.4 1.4 0l13.4-13.3c0.4-0.4 0.4-1 0-1.3l-5.6-5.6C33.5 9.2 32.9 9.2 32.5 9.6z' fill='%23009BFF'/></svg>",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═════════════════════════════════════════════════════════════════════════════
# 2. BRAND CONSTANTS (matching feed-audit-app)
# ═════════════════════════════════════════════════════════════════════════════
BLUE = "#009BFF"
VIOLET = "#770BFF"
SEA = "#4CC3AE"
DARK = "#090A0F"
LIGHT = "#FCFCFC"
GRADIENT = f"linear-gradient(135deg, {VIOLET}, {BLUE})"
AMBER = "#FFC107"
ORANGE = "#FF9800"
RED = "#FF5252"

LOGO_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="191" height="38" viewBox="0 0 191 38" fill="none"><g clip-path="url(#clip0)"><path d="M23.1323 0.277129L0.336866 22.8367C-0.0379489 23.2076 -0.0379491 23.809 0.336866 24.1799L5.95075 29.7357C6.32557 30.1067 6.93326 30.1067 7.30808 29.7357L30.1035 7.1762C30.4783 6.80526 30.4783 6.20385 30.1035 5.83292L24.4896 0.277128C24.1148 -0.0938081 23.5071 -0.0938066 23.1323 0.277129Z" fill="#009BFF"/><path d="M32.5193 9.56975L19.1171 22.8332C18.7423 23.2042 18.7423 23.8056 19.1171 24.1765L24.731 29.7323C25.1058 30.1032 25.7135 30.1032 26.0884 29.7323L39.4905 16.4688C39.8654 16.0979 39.8654 15.4965 39.4905 15.1255L33.8767 9.56975C33.5018 9.19881 32.8942 9.19881 32.5193 9.56975Z" fill="#009BFF"/><path d="M72.0318 17.9793C72.0318 24.7983 66.8064 30.0154 60.5177 30.0154C56.9126 30.0154 54.1845 28.5504 52.4273 26.1703V37.4408C52.4273 37.5892 52.3677 37.7314 52.2618 37.8363C52.1558 37.9411 52.0121 38 51.8622 38H47.9977C47.8478 38 47.7041 37.9411 47.5981 37.8363C47.4921 37.7314 47.4326 37.5892 47.4326 37.4408V7.09944C47.4326 6.95113 47.4921 6.8089 47.5981 6.70403C47.7041 6.59916 47.8478 6.54025 47.9977 6.54025H51.8622C52.0121 6.54025 52.1558 6.59916 52.2618 6.70403C52.3677 6.8089 52.4273 6.95113 52.4273 7.09944V9.83398C54.1781 7.40976 56.9126 5.94482 60.5177 5.94482C66.8064 5.94482 72.0318 11.2075 72.0318 17.9793ZM67.0388 17.9793C67.0388 13.7263 63.8936 10.6578 59.733 10.6578C55.5724 10.6578 52.4273 13.7247 52.4273 17.9793C52.4273 22.2339 55.5708 25.3008 59.733 25.3008C63.8952 25.3008 67.0388 22.2355 67.0388 17.9793Z" fill="#FCFCFC"/><path d="M98.4814 7.09944V28.8608C98.4814 29.0091 98.4219 29.1513 98.3159 29.2562C98.2099 29.361 98.0662 29.42 97.9164 29.42H94.0534C93.9035 29.42 93.7598 29.361 93.6538 29.2562C93.5479 29.1513 93.4883 29.0091 93.4883 28.8608V26.1246C91.7375 28.5504 89.003 30.0154 85.3963 30.0154C79.1076 30.0154 73.8838 24.7527 73.8838 17.9793C73.8838 11.1619 79.1076 5.94482 85.3963 5.94482C89.003 5.94482 91.7311 7.40976 93.4883 9.7883V7.09944C93.4883 6.95113 93.5479 6.8089 93.6538 6.70403C93.7598 6.59916 93.9035 6.54025 94.0534 6.54025H97.9164C98.0662 6.54025 98.2099 6.59916 98.3159 6.70403C98.4219 6.8089 98.4814 6.95113 98.4814 7.09944ZM93.4883 17.9793C93.4883 13.7263 90.3432 10.6578 86.1826 10.6578C82.022 10.6578 78.8768 13.7247 78.8768 17.9793C78.8768 22.2339 82.022 25.3008 86.1826 25.3008C90.3432 25.3008 93.4883 22.2355 93.4883 17.9793Z" fill="#FCFCFC"/><path d="M139.785 25.4851C142.343 25.4851 144.31 24.4345 145.472 23.0168C145.557 22.9161 145.675 22.8489 145.806 22.8272C145.937 22.8055 146.071 22.8309 146.185 22.8986L149.33 24.718C149.398 24.7567 149.458 24.8092 149.504 24.8721C149.551 24.9351 149.584 25.007 149.6 25.0833C149.617 25.1596 149.617 25.2384 149.6 25.3147C149.584 25.391 149.551 25.463 149.505 25.5261C147.356 28.3378 144.023 30.0154 139.74 30.0154C132.11 30.0154 127.166 24.844 127.166 17.9793C127.166 11.206 132.113 5.94482 139.373 5.94482C146.263 5.94482 150.979 11.436 150.979 18.025C150.965 18.7152 150.904 19.4036 150.794 20.0853H132.387C133.173 23.6547 136.087 25.4851 139.785 25.4851ZM145.935 16.0576C145.241 12.1196 142.328 10.4294 139.323 10.4294C135.578 10.4294 133.034 12.6252 132.342 16.0576H145.935Z" fill="#FCFCFC"/><path d="M191.055 15.3712V28.8612C191.055 29.0095 190.996 29.1517 190.89 29.2566C190.784 29.3615 190.64 29.4204 190.49 29.4204H186.627C186.477 29.4204 186.334 29.3615 186.228 29.2566C186.122 29.1517 186.062 29.0095 186.062 28.8612V15.8753C186.062 12.3973 184.026 10.5669 180.883 10.5669C177.599 10.5669 175.011 12.4886 175.011 17.1559V28.8612C175.011 28.9348 174.997 29.0076 174.968 29.0756C174.94 29.1435 174.898 29.2052 174.845 29.2572C174.793 29.3091 174.73 29.3503 174.661 29.3783C174.593 29.4063 174.519 29.4206 174.445 29.4204H170.582C170.432 29.4204 170.288 29.3615 170.182 29.2566C170.076 29.1517 170.017 29.0095 170.017 28.8612V7.09988C170.017 6.95158 170.076 6.80934 170.182 6.70447C170.288 6.5996 170.432 6.54069 170.582 6.54069H174.446C174.521 6.54048 174.594 6.55479 174.663 6.5828C174.732 6.61081 174.794 6.65197 174.847 6.70392C174.899 6.75586 174.941 6.81758 174.97 6.88552C174.998 6.95347 175.013 7.02632 175.013 7.09988V9.46267C176.538 7.08256 179.035 5.93896 182.175 5.93896C187.356 5.94527 191.055 9.4233 191.055 15.3712Z" fill="#FCFCFC"/><path d="M165.186 6.12744C162.274 6.12744 159.456 7.27261 158.065 10.3805V7.09934C158.065 6.95103 158.006 6.8088 157.9 6.70393C157.794 6.59906 157.65 6.54014 157.5 6.54014H153.637C153.487 6.54014 153.344 6.59906 153.238 6.70393C153.132 6.8088 153.072 6.95103 153.072 7.09934V28.8607C153.072 29.009 153.132 29.1512 153.238 29.2561C153.344 29.3609 153.487 29.4198 153.637 29.4198H157.5C157.65 29.4198 157.794 29.3609 157.9 29.2561C158.006 29.1512 158.065 29.009 158.065 28.8607V17.8878C158.065 12.7637 161.823 11.4815 165.186 11.4815H166.926C167.076 11.4815 167.22 11.4226 167.326 11.3177C167.432 11.2129 167.491 11.0706 167.491 10.9223V6.68821C167.491 6.61464 167.477 6.54176 167.449 6.47373C167.42 6.40571 167.379 6.34387 167.326 6.29178C167.274 6.23969 167.211 6.19836 167.143 6.17016C167.074 6.14196 167 6.12744 166.926 6.12744H165.186Z" fill="#FCFCFC"/><path d="M112.505 11.2989C112.579 11.2991 112.653 11.2848 112.721 11.2568C112.79 11.2287 112.853 11.1876 112.905 11.1356C112.958 11.0837 113 11.022 113.028 10.954C113.057 10.8861 113.071 10.8132 113.071 10.7397V7.0994C113.071 7.02584 113.057 6.95299 113.028 6.88504C113 6.8171 112.958 6.75538 112.905 6.70343C112.853 6.65149 112.79 6.61033 112.721 6.58232C112.653 6.55431 112.579 6.54 112.505 6.54021H106.561V0.554469C106.56 0.408335 106.501 0.268486 106.397 0.164857C106.293 0.0612278 106.152 0.00205319 106.004 0H102.133C101.984 0 101.84 0.0589149 101.734 0.163784C101.628 0.268653 101.568 0.410887 101.568 0.559194V22.6718C101.568 27.369 103.857 29.6247 108.634 29.6247H112.506C112.656 29.6243 112.799 29.5652 112.905 29.4604C113.011 29.3557 113.071 29.2137 113.071 29.0655V25.6001C113.071 25.5265 113.057 25.4537 113.028 25.3857C113 25.3178 112.958 25.256 112.905 25.2041C112.853 25.1521 112.79 25.111 112.721 25.083C112.653 25.055 112.579 25.0407 112.505 25.0409H109.533C107.255 25.0409 106.561 24.4549 106.561 22.2827V11.2989H112.505Z" fill="#FCFCFC"/><path d="M126.061 11.2989C126.211 11.2989 126.354 11.24 126.46 11.1351C126.566 11.0302 126.626 10.888 126.626 10.7397V7.0994C126.626 6.9511 126.566 6.80886 126.46 6.70399C126.354 6.59912 126.211 6.54021 126.061 6.54021H120.124V0.554469C120.123 0.406981 120.063 0.265959 119.957 0.16211C119.851 0.0582605 119.708 -5.26628e-06 119.559 3.57006e-10H115.696C115.546 3.57006e-10 115.402 0.0589149 115.296 0.163784C115.19 0.268653 115.131 0.410887 115.131 0.559194V22.6718C115.131 27.369 117.418 29.6247 122.196 29.6247H126.069C126.218 29.6239 126.361 29.5646 126.467 29.4599C126.572 29.3552 126.632 29.2134 126.632 29.0655V25.6001C126.632 25.5264 126.617 25.4536 126.588 25.3858C126.559 25.318 126.516 25.2565 126.463 25.205C126.41 25.1535 126.347 25.1128 126.278 25.0855C126.209 25.0581 126.135 25.0446 126.061 25.0456H123.089C120.812 25.0456 120.118 24.4596 120.118 22.2874V11.2989H126.061Z" fill="#FCFCFC"/></g><defs><clipPath id="clip0"><rect width="191" height="38" fill="white"/></clipPath></defs></svg>'''

SEV_CLR = {Severity.CRITICAL: RED, Severity.HIGH: ORANGE, Severity.MEDIUM: AMBER, Severity.LOW: SEA}
PATH_CLR = {
    QueryPath.NEGATIVE.value: RED,
    QueryPath.GROWTH.value: SEA,
    QueryPath.TARGETING.value: ORANGE,
    QueryPath.MONITOR.value: "rgba(255,255,255,0.12)",
}

# ═════════════════════════════════════════════════════════════════════════════
# 3. CSS INJECTION
# ═════════════════════════════════════════════════════════════════════════════

def _inject_css():
    st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Wix+Madefor+Display:wght@400;500;600;700;800&display=swap');

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to   {{ opacity: 1; }}
    }}
    @keyframes slideInLeft {{
        from {{ opacity: 0; transform: translateX(-20px); }}
        to   {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes drawCircle {{
        from {{ stroke-dashoffset: 440; }}
    }}
    @keyframes barGrow {{
        from {{ width: 0%; }}
    }}
    @keyframes pulse {{
        0%, 100% {{ transform: scale(1); }}
        50%      {{ transform: scale(1.03); }}
    }}

    /* ── Base ─────────────────────────────────────────── */
    .stApp {{
        background: {DARK};
        font-family: 'Wix Madefor Display', 'Source Sans Pro', sans-serif;
    }}
    .stApp, .stApp p, .stApp span, .stApp label, .stApp li {{ color: {LIGHT}; }}
    section[data-testid="stSidebar"] {{ background: #0E1117; }}

    /* ── Buttons ──────────────────────────────────────── */
    .stButton > button {{
        background: {GRADIENT};
        color: white; border: none; border-radius: 10px;
        font-weight: 600; font-size: 15px; padding: 0.6rem 1.6rem;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    }}
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,155,255,0.25);
        color: white; border: none;
    }}

    /* ── Download button ─────────────────────────────── */
    .stDownloadButton > button {{
        background: transparent; border: 1px solid rgba(255,255,255,0.1);
        color: {LIGHT}; border-radius: 10px; transition: all 0.3s ease;
    }}
    .stDownloadButton > button:hover {{
        border-color: {BLUE}; color: {BLUE};
        background: rgba(0,155,255,0.04);
    }}

    /* ── Metric cards ─────────────────────────────────── */
    [data-testid="stMetric"] {{
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 1.2rem;
        animation: fadeInUp 0.5s ease-out both;
        transition: all 0.3s ease;
    }}
    [data-testid="stMetric"]:hover {{
        border-color: rgba(0,155,255,0.15);
        transform: translateY(-2px);
    }}
    [data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stMetric"] {{ animation-delay: 0.05s; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] {{ animation-delay: 0.10s; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] {{ animation-delay: 0.15s; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] {{ animation-delay: 0.20s; }}
    [data-testid="stHorizontalBlock"] > div:nth-child(5) [data-testid="stMetric"] {{ animation-delay: 0.25s; }}

    /* ── Section headers ──────────────────────────────── */
    .sh {{
        border-left: 3px solid {BLUE}; padding-left: 12px;
        margin: 1.8rem 0 0.6rem; font-size: 1.05rem;
        font-weight: 700; color: {LIGHT};
        animation: slideInLeft 0.4s ease-out both;
    }}

    /* ── Glass card ────────────────────────────────────── */
    .glass {{
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 1.4rem;
        animation: fadeInUp 0.5s ease-out both;
        transition: all 0.3s ease;
    }}
    .glass:hover {{
        border-color: rgba(0,155,255,0.12);
        transform: translateY(-2px);
    }}

    /* ── Hero card ─────────────────────────────────────── */
    .hero-card {{
        background: linear-gradient(135deg, rgba(119,11,255,0.08), rgba(0,155,255,0.08));
        border: 1px solid rgba(0,155,255,0.12);
        border-radius: 20px; padding: 2rem; margin-top: 0.5rem;
        animation: fadeInUp 0.6s ease-out both;
        position: relative; overflow: hidden;
    }}
    .hero-card::before {{
        content: ''; position: absolute; top: -50%; right: -20%;
        width: 300px; height: 300px; border-radius: 50%;
        background: radial-gradient(circle, rgba(0,155,255,0.06) 0%, transparent 70%);
        pointer-events: none;
    }}

    /* ── Severity badges ──────────────────────────────── */
    .badge {{
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-weight: 600; font-size: 0.72rem;
    }}
    .badge-critical {{ background: rgba(255,82,82,0.12); color: {RED}; }}
    .badge-high {{ background: rgba(255,152,0,0.12); color: {ORANGE}; }}
    .badge-medium {{ background: rgba(255,193,7,0.12); color: {AMBER}; }}
    .badge-low {{ background: rgba(76,195,174,0.12); color: {SEA}; }}

    /* ── Neg row ──────────────────────────────────────── */
    .neg-row {{
        display: flex; align-items: center; gap: 12px;
        padding: 10px 14px; border-radius: 10px; margin-bottom: 3px;
        animation: fadeInUp 0.35s ease-out both;
        transition: background 0.2s ease;
        border-left: 3px solid transparent;
    }}
    .neg-row:hover {{ background: rgba(255,255,255,0.03); }}
    .neg-row.critical {{ border-left-color: {RED}; }}
    .neg-row.high {{ border-left-color: {ORANGE}; }}
    .neg-row.medium {{ border-left-color: {AMBER}; }}
    .neg-row.low {{ border-left-color: {SEA}; }}
    .neg-term {{ flex: 1; font-size: 0.88rem; color: {LIGHT}; font-weight: 500; }}
    .neg-meta {{ font-size: 0.76rem; color: #555; text-align: right; }}
    .neg-waste {{ font-size: 0.84rem; font-weight: 700; }}

    /* ── Upload hero ───────────────────────────────────── */
    .upload-hero {{
        text-align: center; padding: 4rem 2rem 3rem;
        animation: fadeInUp 0.7s ease-out both;
    }}
    .upload-hero .icon {{
        font-size: 3.5rem; opacity: 0.15;
        animation: pulse 3s ease-in-out infinite;
    }}
    .upload-hero h2 {{
        font-size: 1.5rem; font-weight: 700; color: {LIGHT};
        margin: 0.8rem 0 0.3rem;
    }}
    .upload-hero p {{ color: #444; font-size: 0.88rem; max-width: 480px; margin: 0 auto; line-height: 1.6; }}
    .upload-tags {{
        display: flex; gap: 7px; justify-content: center;
        margin-top: 1rem; flex-wrap: wrap;
    }}
    .upload-tags span {{
        font-size: 0.68rem; font-weight: 600; padding: 3px 11px;
        border-radius: 20px; background: rgba(0,155,255,0.07); color: {BLUE};
    }}

    /* ── File info bar ─────────────────────────────────── */
    .file-info {{
        display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
        padding: 0.7rem 1.2rem; border-radius: 12px;
        background: rgba(0,155,255,0.04);
        border: 1px solid rgba(0,155,255,0.08);
        margin-bottom: 1rem;
        animation: fadeIn 0.4s ease both;
        font-size: 0.8rem; color: #888;
    }}
    .file-info .fi-item {{ display: flex; align-items: center; gap: 6px; }}
    .file-info .fi-val {{ color: {LIGHT}; font-weight: 600; }}

    /* ── Narrative card ────────────────────────────────── */
    .narrative {{
        background: rgba(0,155,255,0.04);
        border: 1px solid rgba(0,155,255,0.1);
        border-radius: 16px; padding: 1.4rem 1.6rem;
        margin-top: 0.8rem;
        animation: fadeInUp 0.6s ease-out 0.2s both;
        line-height: 1.7; font-size: 0.88rem; color: #ccc;
    }}
    .narrative strong {{ color: {LIGHT}; }}
    .narrative .highlight {{ color: {BLUE}; font-weight: 700; }}
    .narrative .warn {{ color: {AMBER}; font-weight: 600; }}
    .narrative .bad {{ color: {RED}; font-weight: 600; }}
    .narrative .good {{ color: {SEA}; font-weight: 600; }}

    /* ── Quick wins ────────────────────────────────────── */
    .quick-wins {{
        background: linear-gradient(135deg, rgba(76,195,174,0.06), rgba(0,155,255,0.06));
        border: 1px solid rgba(76,195,174,0.12);
        border-radius: 16px; padding: 1.2rem 1.4rem;
        animation: fadeInUp 0.5s ease-out 0.3s both;
        margin-top: 0.8rem;
    }}
    .quick-wins .qw-title {{
        font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; color: {SEA}; margin-bottom: 0.6rem;
    }}
    .quick-wins .qw-item {{
        display: flex; align-items: flex-start; gap: 10px;
        padding: 0.4rem 0; font-size: 0.84rem; color: #ccc;
    }}
    .quick-wins .qw-num {{
        width: 22px; height: 22px; border-radius: 50%; flex-shrink: 0;
        background: rgba(76,195,174,0.12); color: {SEA};
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem; font-weight: 700;
    }}

    /* ── Footer ────────────────────────────────────────── */
    .footer {{
        text-align: center; padding: 2.5rem 1rem 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.04);
        margin-top: 3rem; animation: fadeIn 0.5s ease 0.5s both;
    }}
    .footer-text {{ font-size: 0.72rem; color: #333; letter-spacing: 0.5px; }}

    /* ── Chrome ────────────────────────────────────────── */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    [data-testid="manage-app-button"] {{display: none;}}
    .stAppDeployButton {{display: none;}}
    ._profileContainer_gzau3_53 {{display: none;}}
    [data-testid="stStatusWidget"] {{display: none;}}
    ._container_gzau3_1 {{display: none;}}

    .stTabs [data-baseweb="tab-highlight"] {{ background: {GRADIENT}; border-radius: 3px; }}
    .stTabs [data-baseweb="tab"] {{
        color: {LIGHT}; padding: 12px 18px; font-weight: 500;
        transition: color 0.2s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {BLUE}; }}
    .stTabs [data-baseweb="tab-border"] {{ background: rgba(255,255,255,0.05); }}

    .stApp h1,.stApp h2,.stApp h3 {{
        font-family: 'Wix Madefor Display', sans-serif; color: {LIGHT};
    }}
    .stApp h1 {{ font-weight: 800; }}
    .stApp h2 {{ font-weight: 700; }}
    .stApp h3 {{ font-weight: 600; }}

    [data-testid="stExpander"] {{
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px; background: rgba(255,255,255,0.015);
    }}
    hr {{ border: none; border-top: 1px solid rgba(255,255,255,0.05); }}

    /* stagger animations for neg rows */
    .neg-row:nth-child(1) {{ animation-delay: 0s; }}
    .neg-row:nth-child(2) {{ animation-delay: .03s; }}
    .neg-row:nth-child(3) {{ animation-delay: .06s; }}
    .neg-row:nth-child(4) {{ animation-delay: .09s; }}
    .neg-row:nth-child(5) {{ animation-delay: .12s; }}
    .neg-row:nth-child(6) {{ animation-delay: .15s; }}
    .neg-row:nth-child(7) {{ animation-delay: .18s; }}
    .neg-row:nth-child(8) {{ animation-delay: .21s; }}
    .neg-row:nth-child(9) {{ animation-delay: .24s; }}
    .neg-row:nth-child(10) {{ animation-delay: .27s; }}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 4. HTML HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def badge(label: str) -> str:
    cls = label.lower().replace(" ", "-")
    return f'<span class="badge badge-{cls}">{esc(label)}</span>'


def svg_ring(pct, size=140, stroke=9, color=BLUE):
    r = (size - stroke) / 2
    circ = 2 * math.pi * r
    offset = circ * (1 - pct / 100)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="{stroke}"/>'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"'
        f' stroke-dasharray="{circ}" stroke-dashoffset="{offset}" stroke-linecap="round"'
        f' transform="rotate(-90 {size/2} {size/2})"'
        f' style="animation:drawCircle 1.2s cubic-bezier(0.4,0,0.2,1) both;"/>'
        f'<text x="{size/2}" y="{size/2+1}" text-anchor="middle" dominant-baseline="middle"'
        f' fill="{LIGHT}" font-family="Wix Madefor Display,sans-serif" font-weight="800"'
        f' font-size="{size*0.22}px">{pct:.0f}%</text></svg>'
    )


def _mt_label(mt: NegativeMatchType) -> str:
    return {NegativeMatchType.BROAD: "Broad", NegativeMatchType.PHRASE: "Phrase", NegativeMatchType.EXACT: "Exact"}.get(mt, str(mt.value))


def _currency(v: float) -> str:
    return f"${v:,.0f}" if abs(v) >= 1000 else f"${v:,.2f}"


def _load(f) -> pd.DataFrame:
    name = f.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(f)
        except UnicodeDecodeError:
            f.seek(0)
            return pd.read_csv(f, encoding="latin-1")
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    raise ValueError("Upload a .csv or .xlsx file.")


def _csv(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


CL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="Wix Madefor Display, Source Sans Pro, sans-serif", color="#888"))


# ═════════════════════════════════════════════════════════════════════════════
# 5. MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

_inject_css()

# Header
st.markdown(
    f'<div style="display:flex;align-items:center;margin-bottom:0.4rem;animation:fadeIn 0.5s ease both;">'
    f'<div>{LOGO_SVG}</div></div>',
    unsafe_allow_html=True)
st.markdown('<h1 style="margin:0 0 0.15rem;animation:fadeInUp 0.5s ease 0.1s both;">Search Query Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#555;margin-bottom:1.2rem;animation:fadeInUp 0.5s ease 0.15s both;">Upload a search terms report for categorised negatives, growth candidates, n-gram insights, and targeting recommendations.</p>', unsafe_allow_html=True)

# Upload + settings
col_up, _, col_cfg = st.columns([5, 0.3, 2])
with col_up:
    uploaded = st.file_uploader("Upload search terms report", type=["csv", "xlsx", "xls"],
                                help="Google Ads > Campaigns > Insights & Reports > Search Terms",
                                label_visibility="collapsed")
with col_cfg:
    target_cpa = st.number_input("Target CPA ($)", min_value=1.0, value=50.0, step=5.0,
                                  help="Account or campaign target cost per acquisition")
    min_clicks = st.number_input("Min clicks", min_value=1, value=5, step=1,
                                  help="Queries below this threshold are categorised as Monitor")

if uploaded is None:
    st.markdown(
        f'<div class="upload-hero">'
        f'<div class="icon">&#x2B06;</div>'
        f'<h2>Drop your search terms report above</h2>'
        f'<p>Get categorised negative keywords, growth candidates, n-gram analysis, targeting issues, and an actionable implementation list.</p>'
        f'<div class="upload-tags"><span>CSV</span><span>XLSX</span><span>XLS</span></div>'
        f'</div>', unsafe_allow_html=True)
    st.stop()

# ── Parse ──
try:
    df = _load(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

if df.empty:
    st.warning("The uploaded file contains no data.")
    st.stop()

try:
    result = analyse_sqr(df, target_cpa=target_cpa, min_clicks=min_clicks)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Analysis error: {e}")
    st.stop()

# File info bar
file_size = uploaded.size
size_str = f"{file_size/1024:.0f} KB" if file_size < 1024*1024 else f"{file_size/(1024*1024):.1f} MB"
st.markdown(
    f'<div class="file-info">'
    f'<div class="fi-item"><span class="fi-val">{esc(uploaded.name)}</span></div>'
    f'<div class="fi-item">Queries: <span class="fi-val">{result.total_queries:,}</span></div>'
    f'<div class="fi-item">Size: <span class="fi-val">{size_str}</span></div>'
    f'</div>', unsafe_allow_html=True)

# Escalation alerts
for alert in result.escalation_alerts:
    st.error(alert)

# ── Hero card with waste ring ──
waste_pct = result.waste_percentage
waste_color = SEA if waste_pct < 10 else AMBER if waste_pct < 20 else ORANGE if waste_pct < 30 else RED

col_ring, col_summary = st.columns([1, 2])
with col_ring:
    st.markdown(
        f'<div class="ring-wrap">'
        f'{svg_ring(100 - waste_pct, size=160, stroke=10, color=waste_color)}'
        f'<div style="font-size:0.75rem;color:#555;margin-top:0.6rem;text-transform:uppercase;letter-spacing:1.5px;font-weight:500;">Efficiency</div>'
        f'<div style="font-size:1rem;font-weight:700;color:{waste_color};">{_currency(result.wasted_spend)} wasted</div>'
        f'</div>',
        unsafe_allow_html=True)

with col_summary:
    # Narrative summary
    neg_count = len(result.negatives)
    growth_count = len(result.growth_candidates)
    bullets = []
    if waste_pct >= 20:
        bullets.append(f'<span class="bad">{waste_pct:.1f}% waste</span> detected across {result.total_queries:,} queries. Immediate action recommended.')
    elif waste_pct >= 10:
        bullets.append(f'<span class="warn">{waste_pct:.1f}% waste</span> across {result.total_queries:,} queries. Room for improvement.')
    else:
        bullets.append(f'<span class="good">{waste_pct:.1f}% waste</span> across {result.total_queries:,} queries. Healthy efficiency.')

    if neg_count > 0:
        recoverable = sum(n.estimated_waste for n in result.negatives)
        bullets.append(f'<strong>{neg_count} negative keywords</strong> identified. Potential recovery: <span class="highlight">{_currency(recoverable)}</span>/month.')

    if growth_count > 0:
        bullets.append(f'<span class="good">{growth_count} growth candidates</span> converting below target CPA. Promote to dedicated keywords.')

    if result.ten_percent_rule_triggered:
        bullets.append(f'<span class="bad">10% rule triggered</span> on one or more keywords. The problem is targeting, not queries.')

    narrative = "<ul style='margin:0;padding-left:1.2rem;'>" + "".join(f"<li style='margin-bottom:0.4rem;'>{b}</li>" for b in bullets) + "</ul>"
    st.markdown(f'<div class="narrative">{narrative}</div>', unsafe_allow_html=True)

# ── KPI strip ──
st.markdown("")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Queries", f"{result.total_queries:,}")
k2.metric("Total Spend", _currency(result.total_cost))
k3.metric("Wasted Spend", _currency(result.wasted_spend))
k4.metric("Waste %", f"{result.waste_percentage:.1f}%",
          delta=f"-{result.waste_percentage:.1f}%" if result.waste_percentage > 0 else None,
          delta_color="inverse")
k5.metric("Negatives Found", f"{len(result.negatives):,}")

st.markdown("---")

# ── Tabs ──
tab_neg, tab_ngram, tab_growth, tab_target, tab_all = st.tabs([
    "Negatives", "N-Gram Analysis", "Growth", "Targeting", "Full Data",
])

# ═══ NEGATIVES TAB ═══
with tab_neg:
    if result.negatives:
        total_rec = sum(n.estimated_waste for n in result.negatives)
        crit = sum(1 for n in result.negatives if n.severity == Severity.CRITICAL)
        high = sum(1 for n in result.negatives if n.severity == Severity.HIGH)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recoverable Spend", _currency(total_rec))
        m2.metric("Total Negatives", str(len(result.negatives)))
        m3.metric("Critical", str(crit))
        m4.metric("High", str(high))

        # Quick wins
        top_3 = result.negatives[:3]
        if top_3:
            wins_html = '<div class="quick-wins"><div class="qw-title">Top Actions</div>'
            for i, n in enumerate(top_3, 1):
                wins_html += (
                    f'<div class="qw-item">'
                    f'<div class="qw-num">{i}</div>'
                    f'<div>Add <strong>{esc(n.term)}</strong> as {_mt_label(n.match_type)} negative. '
                    f'Saves ~{_currency(n.estimated_waste)}/month. {esc(n.reason)}</div>'
                    f'</div>'
                )
            wins_html += '</div>'
            st.markdown(wins_html, unsafe_allow_html=True)

        st.markdown("")

        # Filters
        fc1, fc2 = st.columns(2)
        with fc1:
            sev_f = st.multiselect("Severity", [s.value for s in Severity], default=[s.value for s in Severity])
        with fc2:
            src_f = st.multiselect("Source", ["Query", "Ngram"], default=["Query", "Ngram"])

        filtered = [n for n in result.negatives if n.severity.value in sev_f and n.source.title() in src_f]

        # Custom HTML rows
        if filtered:
            rows_html = ""
            for n in filtered:
                sev_cls = n.severity.value.lower()
                rows_html += (
                    f'<div class="neg-row {sev_cls}">'
                    f'<div class="neg-term">{esc(n.term)}</div>'
                    f'<div>{badge(n.severity.value)}</div>'
                    f'<div class="neg-meta">{_mt_label(n.match_type)}</div>'
                    f'<div class="neg-waste" style="color:{SEV_CLR.get(n.severity, LIGHT)}">{_currency(n.estimated_waste)}</div>'
                    f'</div>'
                )
            st.markdown(f'<div class="glass" style="padding:0.6rem 0;">{rows_html}</div>', unsafe_allow_html=True)

            # Also provide dataframe for export
            neg_df = pd.DataFrame([{
                "Negative Keyword": n.term, "Match Type": _mt_label(n.match_type),
                "Severity": n.severity.value, "Est. Waste": n.estimated_waste,
                "Queries": n.query_count, "Reason": n.reason, "Source": n.source.title(),
            } for n in filtered])

            with st.expander(f"View as table ({len(filtered)} rows)"):
                st.dataframe(neg_df, use_container_width=True, hide_index=True,
                             column_config={"Est. Waste": st.column_config.NumberColumn(format="$%.2f")})

            st.download_button("Export negatives CSV", data=_csv(neg_df),
                               file_name="sqr_negatives.csv", mime="text/csv")
    else:
        st.markdown('<div class="glass" style="text-align:center;padding:2rem;color:#555;">No negative keyword recommendations at current thresholds.</div>', unsafe_allow_html=True)

# ═══ N-GRAM TAB ═══
with tab_ngram:
    if result.ngram_data is not None and not result.ngram_data.empty:
        ng = result.ngram_data.copy()
        n_size = st.radio("N-gram size", [1, 2, 3], format_func=lambda x: f"{x}-gram", horizontal=True)
        fng = ng[ng["n"] == n_size].head(50)

        if not fng.empty:
            waste_ng = fng[fng["conversions"] == 0].head(15)
            if not waste_ng.empty:
                st.markdown(f'<div class="sh">Top Wasted {n_size}-grams</div>', unsafe_allow_html=True)
                fig = go.Figure(go.Bar(
                    x=waste_ng["cost"], y=waste_ng["ngram"], orientation="h",
                    marker=dict(color=RED, line=dict(width=0)),
                    hovertemplate="%{y}: $%{x:,.2f}<extra></extra>",
                ))
                fig.update_layout(**CL, height=max(260, len(waste_ng)*26+60), bargap=0.3,
                                  xaxis=dict(title_text="Cost ($)", gridcolor="rgba(255,255,255,0.03)"),
                                  yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.03)"),
                                  margin=dict(l=120, r=20, t=10, b=40))
                st.plotly_chart(fig, use_container_width=True)

            perf = fng[fng["conversions"] > 0].sort_values("conversions", ascending=False).head(15)
            if not perf.empty:
                st.markdown(f'<div class="sh">Top Converting {n_size}-grams</div>', unsafe_allow_html=True)
                fig2 = go.Figure(go.Bar(
                    x=perf["conversions"], y=perf["ngram"], orientation="h",
                    marker=dict(color=SEA, line=dict(width=0)),
                    hovertemplate="%{y}: %{x} conversions<extra></extra>",
                ))
                fig2.update_layout(**CL, height=max(260, len(perf)*26+60), bargap=0.3,
                                   xaxis=dict(title_text="Conversions", gridcolor="rgba(255,255,255,0.03)"),
                                   yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.03)"),
                                   margin=dict(l=120, r=20, t=10, b=40))
                st.plotly_chart(fig2, use_container_width=True)

            with st.expander(f"All {n_size}-gram data ({len(fng)} rows)"):
                cols = [c for c in ["ngram","cost","clicks","impressions","conversions","query_count","cpa","cvr"] if c in fng.columns]
                st.dataframe(fng[cols], use_container_width=True, hide_index=True,
                             column_config={"cost": st.column_config.NumberColumn(format="$%.2f"),
                                            "cpa": st.column_config.NumberColumn(format="$%.2f"),
                                            "cvr": st.column_config.NumberColumn(format="%.1f%%")})
    else:
        st.markdown('<div class="glass" style="text-align:center;padding:2rem;color:#555;">Not enough data for n-gram analysis.</div>', unsafe_allow_html=True)

# ═══ GROWTH TAB ═══
with tab_growth:
    if result.growth_candidates:
        st.markdown(f'<div class="sh">Promotion Candidates</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="narrative">These queries have <span class="good">3+ conversions below target CPA</span>. '
            f'Promote them to dedicated keywords for more control and better bid allocation.</div>',
            unsafe_allow_html=True)
        gdf = pd.DataFrame([{
            "Search Term": g.term, "Conversions": g.conversions,
            "CPA": g.cpa, "Cost": g.cost, "Clicks": g.clicks,
        } for g in result.growth_candidates])
        st.dataframe(gdf, use_container_width=True, hide_index=True,
                     column_config={"CPA": st.column_config.NumberColumn(format="$%.2f"),
                                    "Cost": st.column_config.NumberColumn(format="$%.2f")})
        st.download_button("Export growth candidates", data=_csv(gdf),
                           file_name="sqr_growth.csv", mime="text/csv")
    else:
        st.markdown('<div class="glass" style="text-align:center;padding:2rem;color:#555;">No growth candidates at current thresholds.</div>', unsafe_allow_html=True)

# ═══ TARGETING TAB ═══
with tab_target:
    if result.targeting_issues:
        if result.ten_percent_rule_triggered:
            st.markdown(
                f'<div class="narrative"><span class="bad">10% rule triggered.</span> '
                f'One or more keywords have >10% of queries flagged for negation. '
                f'The problem is the targeting, not the queries. Consider changing match types or replacing keywords.</div>',
                unsafe_allow_html=True)
        st.markdown(f'<div class="sh">Keywords Exceeding 10% Threshold</div>', unsafe_allow_html=True)
        for issue in result.targeting_issues:
            with st.expander(f"{issue.keyword}  -  {issue.negative_rate}% negative rate  -  ${issue.waste_amount:,.0f} waste"):
                st.markdown(issue.recommendation)
    else:
        has_kw = "keyword" in (result.categorised_df.columns if result.categorised_df is not None else [])
        if not has_kw:
            st.markdown(
                '<div class="glass" style="text-align:center;padding:2rem;color:#555;">'
                'Include the <strong>Keyword</strong> column in your export to enable targeting analysis and the 10% rule check.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="glass" style="text-align:center;padding:2rem;color:{SEA};">'
                f'All keywords below the 10% threshold. No targeting issues detected.</div>',
                unsafe_allow_html=True)

# ═══ FULL DATA TAB ═══
with tab_all:
    if result.categorised_df is not None:
        cat = result.categorised_df.copy()

        # Donut chart
        st.markdown('<div class="sh">Query Distribution</div>', unsafe_allow_html=True)
        pc = cat["path"].value_counts()
        fig3 = go.Figure(go.Pie(
            labels=pc.index, values=pc.values, hole=0.65,
            marker=dict(colors=[PATH_CLR.get(p, "#27272A") for p in pc.index],
                        line=dict(color=DARK, width=2.5)),
            textinfo="label+percent",
            textfont=dict(size=12, family="Wix Madefor Display, sans-serif", color="#aaa"),
            hovertemplate="%{label}<br>%{value} queries (%{percent})<extra></extra>",
        ))
        fig3.update_layout(**CL, height=380, showlegend=False,
                           margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig3, use_container_width=True)

        # Table
        show = [c for c in ["search_term","keyword","campaign","match_type",
                            "cost","clicks","impressions","conversions",
                            "conversion_value","path"] if c in cat.columns]
        st.dataframe(cat[show].sort_values("cost", ascending=False),
                     use_container_width=True, hide_index=True,
                     column_config={"cost": st.column_config.NumberColumn(format="$%.2f"),
                                    "conversion_value": st.column_config.NumberColumn(format="$%.2f")})

# ── Footer ──
st.markdown(
    f'<div class="footer">'
    f'<div class="footer-text">Built by Pattern AI  ·  SQR Analysis v1.0</div>'
    f'</div>', unsafe_allow_html=True)
