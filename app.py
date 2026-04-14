"""
SQR Analysis Tool
Pattern AI - Search Query Report Analyser

Upload a Google Ads search terms export. Get categorised negatives,
growth candidates, n-gram insights, and targeting recommendations
ready to implement.
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

# ---------------------------------------------------------------------------
# Page config and theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SQR Analysis - Pattern",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Pattern brand + fashion-UX CSS
#
# Philosophy: Helmut Lang restraint meets Swiss International precision.
# Near-black canvas. One accent (Pattern teal). Monospace for data.
# Generous negative space. Weight contrast for hierarchy.
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* --- Foundation: near-black canvas --- */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0A0A0B;
        color: #E8E8E8;
    }
    header[data-testid="stHeader"] {
        background-color: #0A0A0B;
    }
    [data-testid="stSidebar"] {
        background-color: #111113;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* --- Typography: Inter + JetBrains Mono --- */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        letter-spacing: -0.01em;
    }
    h1 {
        font-weight: 300 !important;
        font-size: 2.4rem !important;
        letter-spacing: -0.03em !important;
        color: #FFFFFF !important;
        margin-bottom: 0.25rem !important;
    }
    h2 {
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        letter-spacing: -0.02em !important;
        color: #FFFFFF !important;
        text-transform: uppercase !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    h3 {
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        color: rgba(255,255,255,0.5) !important;
        margin-bottom: 0.75rem !important;
    }
    p, li, span, div {
        color: #C8C8C8;
        line-height: 1.6;
    }

    /* --- Metric cards --- */
    [data-testid="stMetric"] {
        background: #111113;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 2px;
        padding: 1.25rem 1.5rem;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: rgba(255,255,255,0.4) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 300 !important;
        font-size: 2rem !important;
        color: #FFFFFF !important;
        letter-spacing: -0.03em !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
    }

    /* --- Tables --- */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 2px;
    }
    .stDataFrame table {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* --- Tabs --- */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        background: transparent;
    }
    [data-testid="stTabs"] button[data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: rgba(255,255,255,0.4) !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 0 !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom-color: #3ECFB4 !important;
        background: transparent !important;
    }

    /* --- Inputs --- */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background: #111113 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 2px !important;
        color: #FFFFFF !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: #3ECFB4 !important;
        box-shadow: 0 0 0 1px rgba(62,207,180,0.2) !important;
    }

    /* --- File uploader --- */
    [data-testid="stFileUploader"] {
        background: transparent;
    }
    [data-testid="stFileUploader"] section {
        background: #111113 !important;
        border: 1px dashed rgba(255,255,255,0.12) !important;
        border-radius: 2px !important;
        padding: 2rem !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(62,207,180,0.4) !important;
    }

    /* --- Buttons --- */
    .stButton button, .stDownloadButton button {
        background: #3ECFB4 !important;
        color: #0A0A0B !important;
        border: none !important;
        border-radius: 2px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 1.5rem !important;
        transition: opacity 0.15s ease !important;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        opacity: 0.85 !important;
        background: #3ECFB4 !important;
    }

    /* --- Expander --- */
    [data-testid="stExpander"] {
        background: #111113;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 2px;
    }
    [data-testid="stExpander"] summary {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.04em !important;
        color: rgba(255,255,255,0.6) !important;
    }

    /* --- Alerts --- */
    [data-testid="stAlert"] {
        border-radius: 2px !important;
        font-size: 0.85rem !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }

    /* --- Divider --- */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 2rem 0 !important;
    }

    /* --- Scrollbar --- */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0A0A0B; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    /* --- Pattern logo text --- */
    .pattern-brand {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.25);
        margin-bottom: 0.5rem;
    }
    .pattern-subtitle {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.45);
        font-weight: 300;
        margin-bottom: 2.5rem;
        line-height: 1.5;
    }

    /* --- Severity badges --- */
    .badge-critical {
        background: rgba(239,68,68,0.15); color: #F87171;
        padding: 0.15rem 0.5rem; border-radius: 1px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
    }
    .badge-high {
        background: rgba(251,146,60,0.15); color: #FB923C;
        padding: 0.15rem 0.5rem; border-radius: 1px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
    }
    .badge-medium {
        background: rgba(250,204,21,0.15); color: #FACC15;
        padding: 0.15rem 0.5rem; border-radius: 1px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
    }
    .badge-low {
        background: rgba(62,207,180,0.15); color: #3ECFB4;
        padding: 0.15rem 0.5rem; border-radius: 1px;
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
        font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
    }

    /* --- Hide streamlit branding --- */
    #MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }

    /* --- Reduce top padding --- */
    .block-container {
        padding-top: 3rem !important;
        max-width: 1100px !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly theme (consistent with Pattern dark aesthetic)
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0A0A0B",
    plot_bgcolor="#0A0A0B",
    font=dict(family="Inter, sans-serif", color="#C8C8C8", size=12),
    margin=dict(l=0, r=0, t=32, b=0),
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def severity_badge(sev: Severity) -> str:
    css_class = f"badge-{sev.value.lower()}"
    return f'<span class="{css_class}">{sev.value}</span>'


def match_type_label(mt: NegativeMatchType) -> str:
    labels = {
        NegativeMatchType.BROAD: "Broad",
        NegativeMatchType.PHRASE: "Phrase",
        NegativeMatchType.EXACT: "Exact",
    }
    return labels.get(mt, str(mt.value))


def format_currency(val: float) -> str:
    if val >= 1000:
        return f"${val:,.0f}"
    return f"${val:,.2f}"


def load_data(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel file into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Upload a .csv or .xlsx file.")


def build_negatives_export(result: AnalysisResult) -> pd.DataFrame:
    """Build a clean export DataFrame of negative keyword recommendations."""
    rows = []
    for neg in result.negatives:
        rows.append({
            "Negative Keyword": neg.term,
            "Match Type": match_type_label(neg.match_type),
            "Severity": neg.severity.value,
            "Est. Monthly Waste": neg.estimated_waste,
            "Query Count": neg.query_count,
            "Reason": neg.reason,
            "Source": neg.source.title(),
        })
    return pd.DataFrame(rows)


def build_growth_export(result: AnalysisResult) -> pd.DataFrame:
    rows = []
    for g in result.growth_candidates:
        rows.append({
            "Search Term": g.term,
            "Conversions": g.conversions,
            "CPA": g.cpa,
            "Cost": g.cost,
            "Clicks": g.clicks,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Application layout
# ---------------------------------------------------------------------------

# Header
st.markdown('<div class="pattern-brand">Pattern AI</div>', unsafe_allow_html=True)
st.title("Search Query Analysis")
st.markdown(
    '<div class="pattern-subtitle">'
    "Upload a search terms report. Get categorised negatives, growth "
    "candidates, and n-gram insights ready to implement."
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Upload + config
# ---------------------------------------------------------------------------

col_upload, col_config = st.columns([3, 1], gap="large")

with col_upload:
    uploaded = st.file_uploader(
        "Search Terms Report",
        type=["csv", "xlsx", "xls"],
        help="Export from Google Ads: Campaigns > Insights & Reports > Search Terms",
        label_visibility="collapsed",
    )

with col_config:
    st.markdown("### Settings")
    target_cpa = st.number_input(
        "Target CPA ($)",
        min_value=1.0,
        value=50.0,
        step=5.0,
        help="Your account or campaign target cost per acquisition",
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
    try:
        df = load_data(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    try:
        result = analyse_sqr(df, target_cpa=target_cpa, min_clicks=min_clicks)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # --- Escalation alerts ---
    for alert in result.escalation_alerts:
        st.error(alert)

    # --- KPI strip ---
    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Queries", f"{result.total_queries:,}")
    k2.metric("Total Spend", format_currency(result.total_cost))
    k3.metric("Wasted Spend", format_currency(result.wasted_spend))
    k4.metric(
        "Waste %",
        f"{result.waste_percentage:.1f}%",
        delta=f"-{result.waste_percentage:.1f}%" if result.waste_percentage > 0 else None,
        delta_color="inverse",
    )
    k5.metric("Negatives Found", f"{len(result.negatives):,}")

    st.markdown("---")

    # --- Main tabs ---
    tab_neg, tab_ngram, tab_growth, tab_targeting, tab_overview = st.tabs([
        "Negatives",
        "N-Gram Analysis",
        "Growth Candidates",
        "Targeting Issues",
        "Full Data",
    ])

    # ===== TAB: Negatives =====
    with tab_neg:
        if not result.negatives:
            st.info("No negative keyword recommendations at current thresholds.")
        else:
            # Summary row
            total_recoverable = sum(n.estimated_waste for n in result.negatives)
            crit_count = sum(1 for n in result.negatives if n.severity == Severity.CRITICAL)
            high_count = sum(1 for n in result.negatives if n.severity == Severity.HIGH)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Recoverable Spend", format_currency(total_recoverable))
            s2.metric("Total Negatives", str(len(result.negatives)))
            s3.metric("Critical", str(crit_count))
            s4.metric("High", str(high_count))

            st.markdown("")

            # Filter controls
            fc1, fc2 = st.columns(2)
            with fc1:
                severity_filter = st.multiselect(
                    "Filter by severity",
                    options=[s.value for s in Severity],
                    default=[s.value for s in Severity],
                )
            with fc2:
                source_filter = st.multiselect(
                    "Filter by source",
                    options=["Query", "Ngram"],
                    default=["Query", "Ngram"],
                )

            filtered = [
                n for n in result.negatives
                if n.severity.value in severity_filter
                and n.source.title() in source_filter
            ]

            # Build display table
            neg_export = build_negatives_export(
                AnalysisResult(
                    total_queries=0, total_cost=0, total_conversions=0,
                    wasted_spend=0, waste_percentage=0,
                    negatives=filtered,
                )
            )

            if not neg_export.empty:
                st.dataframe(
                    neg_export,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Est. Monthly Waste": st.column_config.NumberColumn(
                            format="$%.2f",
                        ),
                    },
                )

                # Download button
                csv_buf = io.BytesIO()
                neg_export.to_csv(csv_buf, index=False)
                st.download_button(
                    "Export Negatives CSV",
                    data=csv_buf.getvalue(),
                    file_name="sqr_negatives.csv",
                    mime="text/csv",
                )
            else:
                st.info("No negatives match current filters.")

    # ===== TAB: N-Gram Analysis =====
    with tab_ngram:
        if result.ngram_data is None or result.ngram_data.empty:
            st.info("Not enough data for n-gram analysis.")
        else:
            ngram_df = result.ngram_data.copy()

            # N-gram type selector
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
                # Top wasters chart
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
                            text=f"Top Wasted {ngram_type}-grams (0 conversions)",
                            font=dict(size=13, color="rgba(255,255,255,0.5)"),
                        ),
                        height=max(300, len(top_waste) * 28 + 60),
                        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.04)"),
                        xaxis=dict(title="Cost ($)", gridcolor="rgba(255,255,255,0.04)"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Top performers chart
                top_perf = filtered_ngrams[filtered_ngrams["conversions"] > 0].sort_values(
                    "conversions", ascending=False
                ).head(15)
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
                            text=f"Top Converting {ngram_type}-grams",
                            font=dict(size=13, color="rgba(255,255,255,0.5)"),
                        ),
                        height=max(300, len(top_perf) * 28 + 60),
                        yaxis=dict(autorange="reversed", gridcolor="rgba(255,255,255,0.04)"),
                        xaxis=dict(title="Conversions", gridcolor="rgba(255,255,255,0.04)"),
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Full n-gram table
                with st.expander(f"Full {ngram_type}-gram data ({len(filtered_ngrams)} rows)"):
                    display_cols = [
                        "ngram", "cost", "clicks", "impressions",
                        "conversions", "query_count", "cpa", "cvr",
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

    # ===== TAB: Growth Candidates =====
    with tab_growth:
        if not result.growth_candidates:
            st.info(
                "No growth candidates found. Queries need 3+ conversions "
                "below target CPA to qualify."
            )
        else:
            growth_df = build_growth_export(result)
            st.dataframe(
                growth_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "CPA": st.column_config.NumberColumn(format="$%.2f"),
                    "Cost": st.column_config.NumberColumn(format="$%.2f"),
                },
            )

            csv_buf = io.BytesIO()
            growth_df.to_csv(csv_buf, index=False)
            st.download_button(
                "Export Growth Candidates",
                data=csv_buf.getvalue(),
                file_name="sqr_growth_candidates.csv",
                mime="text/csv",
            )

    # ===== TAB: Targeting Issues =====
    with tab_targeting:
        if not result.targeting_issues:
            if "keyword" not in (result.categorised_df.columns if result.categorised_df is not None else []):
                st.info(
                    "No keyword column detected in your export. "
                    "Include the 'Keyword' column to enable targeting analysis "
                    "and the 10% rule check."
                )
            else:
                st.success("No targeting issues detected. All keywords are below the 10% threshold.")
        else:
            if result.ten_percent_rule_triggered:
                st.warning(
                    "The 10% rule has been triggered. One or more keywords have "
                    "more than 10% of their queries flagged for negation. "
                    "The problem is the targeting, not the queries."
                )

            for issue in result.targeting_issues:
                with st.expander(f"{issue.keyword}  -  {issue.negative_rate}% negative rate  -  ${issue.waste_amount:,.0f} waste"):
                    st.markdown(issue.recommendation)

    # ===== TAB: Full Data =====
    with tab_overview:
        if result.categorised_df is not None:
            cat_df = result.categorised_df.copy()

            # Path distribution chart
            path_counts = cat_df["path"].value_counts()
            path_colours = {
                QueryPath.NEGATIVE.value: "#F87171",
                QueryPath.GROWTH.value: "#3ECFB4",
                QueryPath.TARGETING.value: "#FB923C",
                QueryPath.MONITOR.value: "rgba(255,255,255,0.2)",
            }
            fig3 = go.Figure(data=[go.Pie(
                labels=path_counts.index,
                values=path_counts.values,
                hole=0.65,
                marker=dict(
                    colors=[path_colours.get(p, "#555") for p in path_counts.index],
                    line=dict(color="#0A0A0B", width=2),
                ),
                textinfo="label+percent",
                textfont=dict(size=11, family="JetBrains Mono, monospace"),
            )])
            fig3.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(
                    text="Query Distribution by Path",
                    font=dict(size=13, color="rgba(255,255,255,0.5)"),
                ),
                height=380,
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Full table
            display_cols = [c for c in [
                "search_term", "keyword", "campaign", "match_type",
                "cost", "clicks", "impressions", "conversions",
                "conversion_value", "path",
            ] if c in cat_df.columns]

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
    # --- Empty state (Jil Sander silence) ---
    st.markdown("")
    st.markdown("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            """
            <div style="text-align:center; padding: 4rem 0;">
                <div style="font-size: 3rem; margin-bottom: 1.5rem; opacity: 0.15;">P</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
                     letter-spacing: 0.1em; text-transform: uppercase;
                     color: rgba(255,255,255,0.25); margin-bottom: 0.75rem;">
                    Upload a search terms export
                </div>
                <div style="font-size: 0.85rem; color: rgba(255,255,255,0.3);
                     line-height: 1.8; max-width: 400px; margin: 0 auto;">
                    Google Ads > Campaigns > Insights & Reports > Search Terms<br>
                    Export as CSV or Excel. Include cost, clicks, conversions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- Footer ---
st.markdown("---")
st.markdown(
    '<div style="text-align:center; padding: 1rem 0;">'
    '<span class="pattern-brand">Pattern AI</span>'
    '<span style="color:rgba(255,255,255,0.15); margin: 0 0.75rem;">|</span>'
    '<span style="font-family: JetBrains Mono, monospace; font-size: 0.6rem; '
    'color: rgba(255,255,255,0.15); letter-spacing: 0.06em;">'
    "SQR Analysis Tool v1.0"
    "</span>"
    "</div>",
    unsafe_allow_html=True,
)
