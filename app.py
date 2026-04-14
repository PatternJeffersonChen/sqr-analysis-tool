"""
SQR Analysis Tool
Pattern AI - Search Query Report Analyser

Upload a Google Ads search terms export. Get categorised negatives,
growth candidates, n-gram insights, and targeting recommendations
ready to implement.
"""

from __future__ import annotations

import io
import traceback
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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SQR Analysis - Pattern",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Pattern brand CSS
#
# Design philosophy: Stripe/Linear/Vercel developer aesthetic.
# Near-black canvas (#09090B). One accent (#3ECFB4). Monospace for data.
# Tight letter-spacing. Extreme whitespace. Borders that barely whisper.
# Zero or 2px border-radius. Engineered, not friendly.
# ---------------------------------------------------------------------------

_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Foundation ── */
    :root {
        --bg-primary: #09090B;
        --bg-surface: #111113;
        --bg-elevated: #18181B;
        --border-subtle: rgba(255,255,255,0.06);
        --border-hover: rgba(255,255,255,0.12);
        --text-primary: #FAFAFA;
        --text-secondary: rgba(255,255,255,0.55);
        --text-tertiary: rgba(255,255,255,0.30);
        --accent: #3ECFB4;
        --accent-muted: rgba(62,207,180,0.15);
        --red: #F87171;
        --orange: #FB923C;
        --yellow: #FACC15;
        --radius: 2px;
    }

    /* ── Canvas ── */
    .stApp,
    [data-testid="stAppViewContainer"],
    .main,
    section[data-testid="stSidebar"],
    [data-testid="stHeader"] {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--border-subtle) !important;
    }

    /* ── Typography: Inter body, JetBrains data ── */
    html, body, [class*="css"], .stMarkdown, p, span, li, label, div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        letter-spacing: -0.01em;
    }
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
    }
    h1 {
        font-weight: 300 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.04em !important;
        margin-bottom: 0.25rem !important;
        line-height: 1.1 !important;
    }
    h2 {
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: var(--text-secondary) !important;
        margin-top: 2rem !important;
        margin-bottom: 0.75rem !important;
    }
    h3 {
        font-weight: 500 !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: var(--text-tertiary) !important;
        margin-bottom: 0.5rem !important;
    }
    p, li, span {
        color: var(--text-secondary) !important;
        line-height: 1.6;
    }

    /* ── Block container ── */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius) !important;
        padding: 1.25rem 1.5rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.65rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--text-tertiary) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 300 !important;
        font-size: 1.8rem !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.03em !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
    }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"],
    .stDataFrame {
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stDataFrame"] table,
    .stDataFrame table {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0 !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        background: transparent !important;
    }
    [data-testid="stTabs"] button[data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: var(--text-tertiary) !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 0.75rem 1.25rem !important;
        border-radius: 0 !important;
        transition: color 0.15s ease !important;
    }
    [data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        color: var(--text-secondary) !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--text-primary) !important;
        border-bottom-color: var(--accent) !important;
        background: transparent !important;
    }
    /* Tab highlight bar */
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
    }

    /* ── Inputs ── */
    input, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius) !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
    }
    input:focus, [data-testid="stNumberInput"] input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px rgba(62,207,180,0.15) !important;
    }
    /* Number input labels */
    [data-testid="stNumberInput"] label,
    [data-testid="stTextInput"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: var(--text-tertiary) !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] section {
        background: var(--bg-surface) !important;
        border: 1px dashed rgba(255,255,255,0.10) !important;
        border-radius: var(--radius) !important;
        padding: 2.5rem 2rem !important;
        transition: border-color 0.15s ease !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(62,207,180,0.3) !important;
    }
    [data-testid="stFileUploader"] small {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        color: var(--text-tertiary) !important;
    }

    /* ── Buttons ── */
    .stButton > button,
    .stDownloadButton > button {
        background: var(--accent) !important;
        color: #09090B !important;
        border: none !important;
        border-radius: var(--radius) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1.5rem !important;
        transition: opacity 0.15s ease !important;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        opacity: 0.85 !important;
        background: var(--accent) !important;
        color: #09090B !important;
    }

    /* ── Multiselect ── */
    [data-testid="stMultiSelect"] {
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stMultiSelect"] [data-baseweb="select"] {
        background: var(--bg-surface) !important;
        border-color: var(--border-subtle) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stMultiSelect"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: var(--text-tertiary) !important;
    }

    /* ── Radio buttons ── */
    [data-testid="stRadio"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.04em !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.02em !important;
        color: var(--text-secondary) !important;
    }

    /* ── Alerts ── */
    [data-testid="stAlert"] {
        border-radius: var(--radius) !important;
        font-size: 0.82rem !important;
        border: 1px solid var(--border-subtle) !important;
    }

    /* ── Divider ── */
    hr {
        border-color: var(--border-subtle) !important;
        margin: 2rem 0 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.18); }

    /* ── Hide Streamlit branding ── */
    #MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }

    /* ── Reduce default gap between elements ── */
    .stVerticalBlock > div:has(> [data-testid="stMetric"]) {
        gap: 0 !important;
    }

    /* ── Toast / success / warning override ── */
    .stSuccess, .stWarning, .stError, .stInfo {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
    }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly theme
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#09090B",
    plot_bgcolor="#09090B",
    font=dict(family="Inter, -apple-system, sans-serif", color="#A1A1AA", size=12),
    margin=dict(l=0, r=0, t=36, b=0),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zerolinecolor="rgba(255,255,255,0.06)",
    ),
    colorway=["#3ECFB4", "#F87171", "#FACC15", "#818CF8", "#FB923C"],
)

SEVERITY_COLOURS = {
    Severity.CRITICAL: "#F87171",
    Severity.HIGH: "#FB923C",
    Severity.MEDIUM: "#FACC15",
    Severity.LOW: "#3ECFB4",
}

PATH_COLOURS = {
    QueryPath.NEGATIVE.value: "#F87171",
    QueryPath.GROWTH.value: "#3ECFB4",
    QueryPath.TARGETING.value: "#FB923C",
    QueryPath.MONITOR.value: "rgba(255,255,255,0.15)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def match_type_label(mt: NegativeMatchType) -> str:
    return {
        NegativeMatchType.BROAD: "Broad",
        NegativeMatchType.PHRASE: "Phrase",
        NegativeMatchType.EXACT: "Exact",
    }.get(mt, str(mt.value))


def fmt_currency(val: float) -> str:
    if abs(val) >= 1000:
        return f"${val:,.0f}"
    return f"${val:,.2f}"


def load_data(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel, with error handling for common issues."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            # Try UTF-8 first, fall back to latin-1
            try:
                return pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding="latin-1")
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError("Unsupported file type. Upload a .csv or .xlsx file.")
    except Exception as e:
        raise ValueError(f"Could not parse file: {e}")


def build_negatives_df(negatives: list) -> pd.DataFrame:
    rows = []
    for neg in negatives:
        rows.append({
            "Negative Keyword": neg.term,
            "Match Type": match_type_label(neg.match_type),
            "Severity": neg.severity.value,
            "Est. Waste": neg.estimated_waste,
            "Queries": neg.query_count,
            "Reason": neg.reason,
            "Source": neg.source.title(),
        })
    return pd.DataFrame(rows)


def build_growth_df(candidates: list) -> pd.DataFrame:
    rows = []
    for g in candidates:
        rows.append({
            "Search Term": g.term,
            "Conversions": g.conversions,
            "CPA": g.cpa,
            "Cost": g.cost,
            "Clicks": g.clicks,
        })
    return pd.DataFrame(rows)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    '<p style="font-family: JetBrains Mono, monospace; font-size: 0.6rem; '
    'font-weight: 500; letter-spacing: 0.16em; text-transform: uppercase; '
    'color: rgba(255,255,255,0.20); margin-bottom: 0.25rem;">Pattern AI</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h1 style="font-family: Inter, sans-serif; font-weight: 300; '
    'font-size: 2.2rem; letter-spacing: -0.04em; color: #FAFAFA; '
    'margin-bottom: 0.25rem; line-height: 1.1;">Search Query Analysis</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size: 0.88rem; color: rgba(255,255,255,0.40); '
    'font-weight: 300; margin-bottom: 2rem; line-height: 1.6;">'
    "Upload a search terms report. Get categorised negatives, growth "
    "candidates, and n-gram insights ready to implement.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Upload + settings
# ---------------------------------------------------------------------------

col_upload, col_spacer, col_config = st.columns([5, 0.5, 2], gap="small")

with col_upload:
    uploaded = st.file_uploader(
        "Search Terms Report",
        type=["csv", "xlsx", "xls"],
        help="Google Ads > Campaigns > Insights & Reports > Search Terms",
        label_visibility="collapsed",
    )

with col_config:
    st.markdown("### Configuration")
    target_cpa = st.number_input(
        "Target CPA ($)",
        min_value=1.0,
        value=50.0,
        step=5.0,
        help="Account or campaign target cost per acquisition",
    )
    min_clicks = st.number_input(
        "Min clicks",
        min_value=1,
        value=5,
        step=1,
        help="Queries below this threshold are categorised as Monitor",
    )

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

if uploaded is not None:
    # Load
    try:
        df = load_data(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    if df.empty:
        st.warning("The uploaded file is empty.")
        st.stop()

    # Analyse
    try:
        result = analyse_sqr(df, target_cpa=target_cpa, min_clicks=min_clicks)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

    # ── Escalation alerts ──
    for alert in result.escalation_alerts:
        st.error(alert)

    # ── KPI strip ──
    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Queries", f"{result.total_queries:,}")
    k2.metric("Total Spend", fmt_currency(result.total_cost))
    k3.metric("Wasted Spend", fmt_currency(result.wasted_spend))
    k4.metric(
        "Waste %",
        f"{result.waste_percentage:.1f}%",
        delta=f"-{result.waste_percentage:.1f}%" if result.waste_percentage > 0 else None,
        delta_color="inverse",
    )
    k5.metric("Negatives Found", f"{len(result.negatives):,}")
    st.markdown("---")

    # ── Tabs ──
    tab_neg, tab_ngram, tab_growth, tab_targeting, tab_overview = st.tabs([
        "Negatives",
        "N-Gram Analysis",
        "Growth Candidates",
        "Targeting Issues",
        "Full Data",
    ])

    # ═══════ TAB: Negatives ═══════
    with tab_neg:
        if not result.negatives:
            pass  # Jil Sander silence
        else:
            total_recoverable = sum(n.estimated_waste for n in result.negatives)
            crit_count = sum(1 for n in result.negatives if n.severity == Severity.CRITICAL)
            high_count = sum(1 for n in result.negatives if n.severity == Severity.HIGH)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Recoverable Spend", fmt_currency(total_recoverable))
            s2.metric("Total Negatives", str(len(result.negatives)))
            s3.metric("Critical", str(crit_count))
            s4.metric("High", str(high_count))

            st.markdown("")

            # Filters
            fc1, fc2 = st.columns(2)
            with fc1:
                severity_filter = st.multiselect(
                    "Severity",
                    options=[s.value for s in Severity],
                    default=[s.value for s in Severity],
                )
            with fc2:
                source_filter = st.multiselect(
                    "Source",
                    options=["Query", "Ngram"],
                    default=["Query", "Ngram"],
                )

            filtered = [
                n for n in result.negatives
                if n.severity.value in severity_filter
                and n.source.title() in source_filter
            ]

            neg_df = build_negatives_df(filtered)

            if not neg_df.empty:
                st.dataframe(
                    neg_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Est. Waste": st.column_config.NumberColumn(format="$%.2f"),
                    },
                )

                st.download_button(
                    "Export Negatives",
                    data=to_csv_bytes(neg_df),
                    file_name="sqr_negatives.csv",
                    mime="text/csv",
                )

    # ═══════ TAB: N-Gram Analysis ═══════
    with tab_ngram:
        if result.ngram_data is None or result.ngram_data.empty:
            pass
        else:
            ngram_df = result.ngram_data.copy()

            ngram_type = st.radio(
                "N-gram size",
                options=[1, 2, 3],
                format_func=lambda x: f"{x}-gram",
                horizontal=True,
            )

            filtered_ngrams = ngram_df[ngram_df["n"] == ngram_type].head(50)

            if filtered_ngrams.empty:
                st.info(f"No {ngram_type}-grams found.")
            else:
                # Top wasters
                top_waste = filtered_ngrams[filtered_ngrams["conversions"] == 0].head(15)
                if not top_waste.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=top_waste["cost"],
                        y=top_waste["ngram"],
                        orientation="h",
                        marker_color="#F87171",
                        marker_line_width=0,
                    ))
                    fig.update_layout(
                        **PLOTLY_LAYOUT,
                        title=dict(
                            text=f"Top wasted {ngram_type}-grams  /  0 conversions",
                            font=dict(
                                size=12,
                                color="rgba(255,255,255,0.35)",
                                family="JetBrains Mono, monospace",
                            ),
                        ),
                        height=max(280, len(top_waste) * 28 + 60),
                        yaxis=dict(
                            autorange="reversed",
                            gridcolor="rgba(255,255,255,0.04)",
                            tickfont=dict(
                                family="JetBrains Mono, monospace",
                                size=11,
                                color="rgba(255,255,255,0.5)",
                            ),
                        ),
                        xaxis=dict(
                            title=dict(
                                text="Cost ($)",
                                font=dict(size=10, color="rgba(255,255,255,0.3)"),
                            ),
                            gridcolor="rgba(255,255,255,0.04)",
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Top performers
                top_perf = (
                    filtered_ngrams[filtered_ngrams["conversions"] > 0]
                    .sort_values("conversions", ascending=False)
                    .head(15)
                )
                if not top_perf.empty:
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=top_perf["conversions"],
                        y=top_perf["ngram"],
                        orientation="h",
                        marker_color="#3ECFB4",
                        marker_line_width=0,
                    ))
                    fig2.update_layout(
                        **PLOTLY_LAYOUT,
                        title=dict(
                            text=f"Top converting {ngram_type}-grams",
                            font=dict(
                                size=12,
                                color="rgba(255,255,255,0.35)",
                                family="JetBrains Mono, monospace",
                            ),
                        ),
                        height=max(280, len(top_perf) * 28 + 60),
                        yaxis=dict(
                            autorange="reversed",
                            gridcolor="rgba(255,255,255,0.04)",
                            tickfont=dict(
                                family="JetBrains Mono, monospace",
                                size=11,
                                color="rgba(255,255,255,0.5)",
                            ),
                        ),
                        xaxis=dict(
                            title=dict(
                                text="Conversions",
                                font=dict(size=10, color="rgba(255,255,255,0.3)"),
                            ),
                            gridcolor="rgba(255,255,255,0.04)",
                        ),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Full table
                with st.expander(f"Full {ngram_type}-gram data  /  {len(filtered_ngrams)} rows"):
                    display_cols = [
                        c for c in [
                            "ngram", "cost", "clicks", "impressions",
                            "conversions", "query_count", "cpa", "cvr",
                        ]
                        if c in filtered_ngrams.columns
                    ]
                    st.dataframe(
                        filtered_ngrams[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "cost": st.column_config.NumberColumn(format="$%.2f"),
                            "cpa": st.column_config.NumberColumn(format="$%.2f"),
                            "cvr": st.column_config.NumberColumn(format="%.1f%%"),
                        },
                    )

    # ═══════ TAB: Growth Candidates ═══════
    with tab_growth:
        if not result.growth_candidates:
            pass
        else:
            growth_df = build_growth_df(result.growth_candidates)
            st.dataframe(
                growth_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "CPA": st.column_config.NumberColumn(format="$%.2f"),
                    "Cost": st.column_config.NumberColumn(format="$%.2f"),
                },
            )

            st.download_button(
                "Export Growth Candidates",
                data=to_csv_bytes(growth_df),
                file_name="sqr_growth_candidates.csv",
                mime="text/csv",
            )

    # ═══════ TAB: Targeting Issues ═══════
    with tab_targeting:
        if not result.targeting_issues:
            has_keyword = "keyword" in (
                result.categorised_df.columns
                if result.categorised_df is not None
                else []
            )
            if not has_keyword:
                st.markdown(
                    '<p style="font-size: 0.82rem; color: rgba(255,255,255,0.35);">'
                    "Include the Keyword column in your export to enable "
                    "targeting analysis and the 10% rule check.</p>",
                    unsafe_allow_html=True,
                )
        else:
            if result.ten_percent_rule_triggered:
                st.warning(
                    "The 10% rule has been triggered. One or more keywords have "
                    "more than 10% of their queries flagged for negation. "
                    "The problem is the targeting, not the queries."
                )

            for issue in result.targeting_issues:
                with st.expander(
                    f"{issue.keyword}  -  {issue.negative_rate}% negative rate  "
                    f"-  ${issue.waste_amount:,.0f} waste"
                ):
                    st.markdown(issue.recommendation)

    # ═══════ TAB: Full Data ═══════
    with tab_overview:
        if result.categorised_df is not None:
            cat_df = result.categorised_df.copy()

            # Donut chart
            path_counts = cat_df["path"].value_counts()
            fig3 = go.Figure(data=[go.Pie(
                labels=path_counts.index,
                values=path_counts.values,
                hole=0.7,
                marker=dict(
                    colors=[
                        PATH_COLOURS.get(p, "#3A3A3C")
                        for p in path_counts.index
                    ],
                    line=dict(color="#09090B", width=2),
                ),
                textinfo="label+percent",
                textfont=dict(
                    size=11,
                    family="JetBrains Mono, monospace",
                    color="rgba(255,255,255,0.6)",
                ),
                hoverinfo="label+value+percent",
            )])
            fig3.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(
                    text="Query distribution by path",
                    font=dict(
                        size=12,
                        color="rgba(255,255,255,0.35)",
                        family="JetBrains Mono, monospace",
                    ),
                ),
                height=360,
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Full table
            display_cols = [
                c for c in [
                    "search_term", "keyword", "campaign", "match_type",
                    "cost", "clicks", "impressions", "conversions",
                    "conversion_value", "path",
                ]
                if c in cat_df.columns
            ]

            st.dataframe(
                cat_df[display_cols].sort_values("cost", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "cost": st.column_config.NumberColumn(format="$%.2f"),
                    "conversion_value": st.column_config.NumberColumn(format="$%.2f"),
                },
            )

else:
    # ── Empty state: Jil Sander silence ──
    st.markdown("")
    st.markdown("")
    st.markdown("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            """
            <div style="text-align: center; padding: 5rem 0;">
                <div style="font-family: 'Inter', sans-serif; font-weight: 200;
                     font-size: 4rem; color: rgba(255,255,255,0.06);
                     letter-spacing: -0.05em; margin-bottom: 2rem;">P</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
                     font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase;
                     color: rgba(255,255,255,0.18); margin-bottom: 1rem;">
                    Upload a search terms export
                </div>
                <div style="font-size: 0.8rem; color: rgba(255,255,255,0.22);
                     line-height: 2; max-width: 380px; margin: 0 auto;
                     font-family: 'Inter', sans-serif; font-weight: 300;">
                    Google Ads &rsaquo; Campaigns &rsaquo; Insights &amp; Reports &rsaquo; Search Terms<br>
                    CSV or Excel. Include cost, clicks, conversions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ──
st.markdown("---")
st.markdown(
    '<div style="text-align: center; padding: 0.75rem 0;">'
    '<span style="font-family: JetBrains Mono, monospace; font-size: 0.55rem; '
    'font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase; '
    'color: rgba(255,255,255,0.12);">Pattern AI</span>'
    '<span style="color: rgba(255,255,255,0.08); margin: 0 0.75rem;">|</span>'
    '<span style="font-family: JetBrains Mono, monospace; font-size: 0.55rem; '
    'color: rgba(255,255,255,0.10); letter-spacing: 0.06em;">'
    "SQR Analysis v1.0"
    "</span>"
    "</div>",
    unsafe_allow_html=True,
)
