"""
SQR Analysis Tool — Pattern AI
Search Query Report Analyser

Upload a Google Ads search terms export. Get categorised negatives,
growth candidates, n-gram insights, and targeting recommendations.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sqr_engine import (
    AnalysisResult,
    NegativeMatchType,
    QueryPath,
    Severity,
    analyse_sqr,
    compute_ngrams,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SQR Analysis — Pattern AI",
    page_icon="https://raw.githubusercontent.com/PatternJeffersonChen/sqr-analysis-tool/main/.streamlit/favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design system
#
# References: Stripe, Linear, Vercel developer aesthetic
# Near-black canvas. One accent. Monospace for data. Tight letter-spacing.
# Borders that barely whisper. Engineered, not friendly.
# ─────────────────────────────────────────────────────────────────────────────

# Colours
BG          = "#09090B"
SURFACE     = "#0F0F11"
SURFACE_2   = "#161618"
BORDER      = "rgba(255,255,255,0.06)"
BORDER_HVR  = "rgba(255,255,255,0.10)"
TEXT_1      = "#FAFAFA"
TEXT_2      = "rgba(255,255,255,0.50)"
TEXT_3      = "rgba(255,255,255,0.28)"
ACCENT      = "#3ECFB4"
ACCENT_DIM  = "rgba(62,207,180,0.12)"
RED         = "#EF4444"
ORANGE      = "#F97316"
YELLOW      = "#EAB308"
GREEN       = ACCENT

# Plotly base
PLOTLY_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="Inter, -apple-system, sans-serif", color="#71717A", size=11),
    margin=dict(l=0, r=24, t=40, b=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.03)", zerolinecolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.03)", zerolinecolor="rgba(255,255,255,0.05)"),
)

# Severity → colour
SEV_CLR = {Severity.CRITICAL: RED, Severity.HIGH: ORANGE, Severity.MEDIUM: YELLOW, Severity.LOW: GREEN}

# Path → colour
PATH_CLR = {
    QueryPath.NEGATIVE.value: RED,
    QueryPath.GROWTH.value: GREEN,
    QueryPath.TARGETING.value: ORANGE,
    QueryPath.MONITOR.value: "rgba(255,255,255,0.12)",
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS injection
#
# Font loading uses <link> tags (more reliable on Streamlit Cloud than @import).
# Every value here is intentional. If it looks like a lot of CSS, that's
# because Streamlit ships with heavy defaults that need surgical overrides.
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

st.markdown(f"""
<style>
/* ═══ RESET & CANVAS ═══ */
.stApp, [data-testid="stAppViewContainer"], .main,
[data-testid="stHeader"], section[data-testid="stSidebar"] {{
    background-color: {BG} !important;
}}
header[data-testid="stHeader"] {{ display: none !important; }}
#MainMenu, footer, [data-testid="stToolbar"] {{ display: none !important; }}
[data-testid="stSidebar"] {{ background: {SURFACE} !important; border-right: 1px solid {BORDER} !important; }}

/* ═══ TYPOGRAPHY ═══ */
html, body, [class*="css"], .stMarkdown, p, span, li, label, div {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}
h1 {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 300 !important;
    font-size: 2rem !important;
    letter-spacing: -0.045em !important;
    color: {TEXT_1} !important;
    line-height: 1.15 !important;
    margin-bottom: 0 !important;
}}
h2 {{
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: {TEXT_3} !important;
}}
h3 {{
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: {TEXT_3} !important;
}}
p, li, span {{ color: {TEXT_2} !important; line-height: 1.65; font-size: 0.88rem; }}

/* ═══ LAYOUT ═══ */
.block-container {{
    padding: 2.5rem 3rem 3rem 3rem !important;
    max-width: 1180px !important;
}}

/* ═══ METRIC CARDS ═══ */
[data-testid="stMetric"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    padding: 1.1rem 1.3rem !important;
}}
[data-testid="stMetricLabel"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: {TEXT_3} !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 300 !important;
    font-size: 1.65rem !important;
    letter-spacing: -0.035em !important;
    color: {TEXT_1} !important;
}}
[data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
}}

/* ═══ DATAFRAMES ═══ */
[data-testid="stDataFrame"], .stDataFrame {{
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}}
[data-testid="stDataFrame"] th {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    background: {SURFACE} !important;
}}
[data-testid="stDataFrame"] td {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
}}

/* ═══ TABS ═══ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 0 !important;
    border-bottom: 1px solid {BORDER} !important;
    background: transparent !important;
}}
[data-testid="stTabs"] button[data-baseweb="tab"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: {TEXT_3} !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.8rem 1.4rem !important;
    border-radius: 0 !important;
    transition: color 0.2s ease !important;
}}
[data-testid="stTabs"] button[data-baseweb="tab"]:hover {{
    color: {TEXT_2} !important;
}}
[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {TEXT_1} !important;
    border-bottom-color: {ACCENT} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
    background-color: {ACCENT} !important;
}}

/* ═══ FILE UPLOADER ═══ */
[data-testid="stFileUploader"] section {{
    background: {SURFACE} !important;
    border: 1px dashed rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    padding: 2.5rem 2rem !important;
    transition: border-color 0.2s ease, background 0.2s ease !important;
}}
[data-testid="stFileUploader"] section:hover {{
    border-color: rgba(62,207,180,0.25) !important;
    background: rgba(62,207,180,0.02) !important;
}}
[data-testid="stFileUploader"] button {{
    background: {ACCENT} !important;
    color: {BG} !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.2rem !important;
}}
[data-testid="stFileUploader"] small {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    color: {TEXT_3} !important;
    letter-spacing: 0.04em !important;
}}

/* ═══ INPUTS ═══ */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 4px !important;
    color: {TEXT_1} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    padding: 0.55rem 0.75rem !important;
}}
[data-testid="stNumberInput"] input:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 1px {ACCENT_DIM} !important;
}}
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: {TEXT_3} !important;
}}
/* +/- buttons */
[data-testid="stNumberInput"] button {{
    background: {SURFACE_2} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_2} !important;
}}

/* ═══ BUTTONS ═══ */
.stButton > button, .stDownloadButton > button {{
    background: {ACCENT} !important;
    color: {BG} !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity 0.15s ease !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    opacity: 0.85 !important;
    background: {ACCENT} !important;
    color: {BG} !important;
}}

/* ═══ MULTISELECT ═══ */
[data-testid="stMultiSelect"] [data-baseweb="select"] {{
    background: {SURFACE} !important;
    border-color: {BORDER} !important;
    border-radius: 4px !important;
}}
[data-testid="stMultiSelect"] label {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: {TEXT_3} !important;
}}

/* ═══ RADIO ═══ */
[data-testid="stRadio"] > label {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: {TEXT_3} !important;
}}

/* ═══ EXPANDERS ═══ */
[data-testid="stExpander"] {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.74rem !important;
    color: {TEXT_2} !important;
}}

/* ═══ ALERTS ═══ */
[data-testid="stAlert"] {{
    border-radius: 6px !important;
    font-size: 0.82rem !important;
}}

/* ═══ DIVIDER ═══ */
hr {{ border-color: {BORDER} !important; margin: 1.5rem 0 !important; }}

/* ═══ SCROLLBAR ═══ */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.08); border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: rgba(255,255,255,0.14); }}

/* ═══ SELECTBOX DROPDOWN ═══ */
[data-baseweb="popover"] {{
    background: {SURFACE_2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
}}
[data-baseweb="menu"] {{
    background: {SURFACE_2} !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def _neg_df(negatives: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "Negative Keyword": n.term,
        "Match Type": _mt_label(n.match_type),
        "Severity": n.severity.value,
        "Est. Waste": n.estimated_waste,
        "Queries": n.query_count,
        "Reason": n.reason,
        "Source": n.source.title(),
    } for n in negatives])


def _growth_df(candidates: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "Search Term": g.term,
        "Conversions": g.conversions,
        "CPA": g.cpa,
        "Cost": g.cost,
        "Clicks": g.clicks,
    } for g in candidates])


def _csv(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _bar_chart(data, x_col, y_col, colour, title, x_title):
    """Horizontal bar chart in the Pattern aesthetic."""
    fig = go.Figure(go.Bar(
        x=data[x_col],
        y=data[y_col],
        orientation="h",
        marker_color=colour,
        marker_line_width=0,
    ))
    fig.update_layout(
        **PLOTLY_BASE,
        title=dict(text=title, font=dict(size=11, color=TEXT_3, family="JetBrains Mono, monospace")),
        height=max(260, len(data) * 26 + 60),
        yaxis=dict(
            autorange="reversed",
            gridcolor="rgba(255,255,255,0.03)",
            tickfont=dict(family="JetBrains Mono, monospace", size=10, color=TEXT_2),
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=10, color=TEXT_3)),
            gridcolor="rgba(255,255,255,0.03)",
        ),
        bargap=0.35,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f'<div style="margin-bottom:2.5rem;">'
    f'<p style="font-family:JetBrains Mono,monospace;font-size:0.58rem;'
    f'font-weight:500;letter-spacing:0.18em;text-transform:uppercase;'
    f'color:{TEXT_3};margin:0 0 0.6rem 0;">Pattern AI</p>'
    f'<h1 style="font-family:Inter,sans-serif;font-weight:300;font-size:2rem;'
    f'letter-spacing:-0.045em;color:{TEXT_1};margin:0 0 0.5rem 0;'
    f'line-height:1.15;">Search Query Analysis</h1>'
    f'<p style="font-size:0.84rem;color:{TEXT_3};font-weight:400;'
    f'margin:0;line-height:1.6;max-width:520px;">'
    f'Upload a search terms report. Get categorised negatives, growth '
    f'candidates, and n-gram insights ready to implement.</p>'
    f'</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Upload + config
# ─────────────────────────────────────────────────────────────────────────────

col_up, _, col_cfg = st.columns([5, 0.3, 2])

with col_up:
    uploaded = st.file_uploader(
        "Drop your search terms report",
        type=["csv", "xlsx", "xls"],
        help="Google Ads > Campaigns > Insights & Reports > Search Terms",
        label_visibility="collapsed",
    )

with col_cfg:
    st.markdown("## Settings")
    target_cpa = st.number_input("Target CPA ($)", min_value=1.0, value=50.0, step=5.0,
                                  help="Account or campaign target cost per acquisition")
    min_clicks = st.number_input("Min clicks", min_value=1, value=5, step=1,
                                  help="Queries below this are categorised as Monitor")

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

if uploaded is not None:
    try:
        df = _load(uploaded)
    except Exception as e:
        st.error(f"Could not read file — {e}")
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
        st.error(f"Analysis error — {e}")
        st.stop()

    # Escalation alerts
    for alert in result.escalation_alerts:
        st.error(alert)

    # ── KPI strip ──────────────────────────────────────────────────────────
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

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_neg, tab_ngram, tab_growth, tab_target, tab_all = st.tabs([
        "Negatives", "N-Gram Analysis", "Growth", "Targeting", "Full Data",
    ])

    # ═══ NEGATIVES ═══
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

            st.markdown("")

            fc1, fc2 = st.columns(2)
            with fc1:
                sev_f = st.multiselect("Severity", [s.value for s in Severity], default=[s.value for s in Severity])
            with fc2:
                src_f = st.multiselect("Source", ["Query", "Ngram"], default=["Query", "Ngram"])

            filtered = [n for n in result.negatives if n.severity.value in sev_f and n.source.title() in src_f]
            ndf = _neg_df(filtered)

            if not ndf.empty:
                st.dataframe(ndf, use_container_width=True, hide_index=True,
                             column_config={"Est. Waste": st.column_config.NumberColumn(format="$%.2f")})
                st.download_button("Export negatives", data=_csv(ndf),
                                   file_name="sqr_negatives.csv", mime="text/csv")

    # ═══ N-GRAM ANALYSIS ═══
    with tab_ngram:
        if result.ngram_data is not None and not result.ngram_data.empty:
            ng = result.ngram_data.copy()
            n_size = st.radio("N-gram size", [1, 2, 3], format_func=lambda x: f"{x}-gram", horizontal=True)
            fng = ng[ng["n"] == n_size].head(50)

            if not fng.empty:
                # Wasters
                waste = fng[fng["conversions"] == 0].head(15)
                if not waste.empty:
                    st.plotly_chart(
                        _bar_chart(waste, "cost", "ngram", RED,
                                   f"Top wasted {n_size}-grams  ·  0 conversions", "Cost ($)"),
                        use_container_width=True,
                    )

                # Performers
                perf = fng[fng["conversions"] > 0].sort_values("conversions", ascending=False).head(15)
                if not perf.empty:
                    st.plotly_chart(
                        _bar_chart(perf, "conversions", "ngram", ACCENT,
                                   f"Top converting {n_size}-grams", "Conversions"),
                        use_container_width=True,
                    )

                with st.expander(f"All {n_size}-gram data  ·  {len(fng)} rows"):
                    cols = [c for c in ["ngram", "cost", "clicks", "impressions", "conversions", "query_count", "cpa", "cvr"] if c in fng.columns]
                    st.dataframe(fng[cols], use_container_width=True, hide_index=True,
                                 column_config={
                                     "cost": st.column_config.NumberColumn(format="$%.2f"),
                                     "cpa": st.column_config.NumberColumn(format="$%.2f"),
                                     "cvr": st.column_config.NumberColumn(format="%.1f%%"),
                                 })

    # ═══ GROWTH ═══
    with tab_growth:
        if result.growth_candidates:
            gdf = _growth_df(result.growth_candidates)
            st.dataframe(gdf, use_container_width=True, hide_index=True,
                         column_config={
                             "CPA": st.column_config.NumberColumn(format="$%.2f"),
                             "Cost": st.column_config.NumberColumn(format="$%.2f"),
                         })
            st.download_button("Export growth candidates", data=_csv(gdf),
                               file_name="sqr_growth.csv", mime="text/csv")

    # ═══ TARGETING ═══
    with tab_target:
        if result.targeting_issues:
            if result.ten_percent_rule_triggered:
                st.warning(
                    "10% rule triggered — one or more keywords have >10% of queries "
                    "flagged for negation. The problem is the targeting, not the queries."
                )
            for issue in result.targeting_issues:
                with st.expander(f"{issue.keyword}  ·  {issue.negative_rate}% negative rate  ·  ${issue.waste_amount:,.0f} waste"):
                    st.markdown(issue.recommendation)
        else:
            has_kw = "keyword" in (result.categorised_df.columns if result.categorised_df is not None else [])
            if not has_kw:
                st.markdown(
                    f'<p style="font-size:0.8rem;color:{TEXT_3};margin-top:1rem;">'
                    'Include the <strong>Keyword</strong> column in your export to enable targeting analysis.</p>',
                    unsafe_allow_html=True,
                )

    # ═══ FULL DATA ═══
    with tab_all:
        if result.categorised_df is not None:
            cat = result.categorised_df.copy()

            # Donut
            pc = cat["path"].value_counts()
            donut = go.Figure(go.Pie(
                labels=pc.index, values=pc.values, hole=0.72,
                marker=dict(colors=[PATH_CLR.get(p, "#27272A") for p in pc.index],
                            line=dict(color=BG, width=2.5)),
                textinfo="label+percent",
                textfont=dict(size=10, family="JetBrains Mono, monospace", color=TEXT_2),
                hoverinfo="label+value+percent",
            ))
            donut.update_layout(**PLOTLY_BASE, height=340, showlegend=False,
                                title=dict(text="Query distribution", font=dict(size=11, color=TEXT_3, family="JetBrains Mono, monospace")))
            st.plotly_chart(donut, use_container_width=True)

            # Table
            show = [c for c in ["search_term", "keyword", "campaign", "match_type",
                                "cost", "clicks", "impressions", "conversions",
                                "conversion_value", "path"] if c in cat.columns]
            st.dataframe(cat[show].sort_values("cost", ascending=False),
                         use_container_width=True, hide_index=True,
                         column_config={
                             "cost": st.column_config.NumberColumn(format="$%.2f"),
                             "conversion_value": st.column_config.NumberColumn(format="$%.2f"),
                         })

else:
    # ── Empty state ────────────────────────────────────────────────────────
    st.markdown("")
    st.markdown("")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            f"""
            <div style="text-align:center;padding:6rem 0 4rem;">
                <div style="font-family:'Inter',sans-serif;font-weight:200;
                     font-size:4.5rem;color:rgba(255,255,255,0.04);
                     letter-spacing:-0.06em;margin-bottom:2rem;
                     user-select:none;">P</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                     font-weight:500;letter-spacing:0.16em;text-transform:uppercase;
                     color:rgba(255,255,255,0.15);margin-bottom:0.8rem;">
                    Upload a search terms export</div>
                <div style="font-size:0.78rem;color:rgba(255,255,255,0.18);
                     line-height:2;max-width:360px;margin:0 auto;
                     font-family:'Inter',sans-serif;font-weight:300;">
                    Google Ads &rsaquo; Campaigns &rsaquo; Search Terms<br>
                    CSV or Excel &middot; Include cost, clicks, conversions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<div style="text-align:center;padding:0.5rem 0;">'
    f'<span style="font-family:JetBrains Mono,monospace;font-size:0.5rem;'
    f'font-weight:500;letter-spacing:0.16em;text-transform:uppercase;'
    f'color:rgba(255,255,255,0.10);">Pattern AI</span>'
    f'<span style="color:rgba(255,255,255,0.06);margin:0 0.6rem;">·</span>'
    f'<span style="font-family:JetBrains Mono,monospace;font-size:0.5rem;'
    f'color:rgba(255,255,255,0.08);letter-spacing:0.06em;">v1.0</span>'
    f'</div>',
    unsafe_allow_html=True,
)
