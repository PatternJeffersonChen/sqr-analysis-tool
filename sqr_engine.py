"""
SQR Analysis Engine
Pure logic module - no UI dependencies. Handles all search query report
analysis: categorisation, n-gram decomposition, negative keyword
recommendations, and wasted spend calculation.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class QueryPath(Enum):
    """The three-path decision framework for every search query."""
    NEGATIVE = "Negative Path"
    GROWTH = "Growth Path"
    TARGETING = "Targeting Path"
    MONITOR = "Monitor"


class NegativeMatchType(Enum):
    BROAD = "Broad Match Negative"
    PHRASE = "Phrase Match Negative"
    EXACT = "Exact Match Negative"


class Severity(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


@dataclass
class NegativeRecommendation:
    term: str
    match_type: NegativeMatchType
    reason: str
    severity: Severity
    estimated_waste: float
    query_count: int
    source: str  # "query" or "ngram"


@dataclass
class GrowthCandidate:
    term: str
    conversions: int
    cost: float
    cpa: float
    clicks: int
    reason: str


@dataclass
class TargetingIssue:
    keyword: str
    negative_rate: float  # percentage of queries needing negation
    waste_amount: float
    recommendation: str


@dataclass
class AnalysisResult:
    """Complete output of an SQR analysis run."""
    total_queries: int
    total_cost: float
    total_conversions: int
    wasted_spend: float
    waste_percentage: float
    negatives: list[NegativeRecommendation] = field(default_factory=list)
    growth_candidates: list[GrowthCandidate] = field(default_factory=list)
    targeting_issues: list[TargetingIssue] = field(default_factory=list)
    ngram_data: Optional[pd.DataFrame] = None
    categorised_df: Optional[pd.DataFrame] = None
    ten_percent_rule_triggered: bool = False
    escalation_alerts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

_COLUMN_ALIASES = {
    "search_term": [
        "search term", "search_term", "query", "search query",
        "search terms", "matched search term",
    ],
    "cost": [
        "cost", "spend", "total cost", "cost (usd)", "cost ($)",
    ],
    "conversions": [
        "conversions", "conv", "conv.", "total conversions", "all conv.",
        "all conversions", "all conv",
    ],
    "clicks": [
        "clicks", "total clicks",
    ],
    "impressions": [
        "impressions", "impr", "impr.", "total impressions",
    ],
    "conversion_value": [
        "conversion value", "conv. value", "total conversion value",
        "all conv. value", "all conv value", "revenue", "value",
    ],
    "keyword": [
        "keyword", "keyword text", "matched keyword",
    ],
    "campaign": [
        "campaign", "campaign name",
    ],
    "match_type": [
        "match type", "keyword match type", "search match type",
    ],
    "ad_group": [
        "ad group", "ad group name", "adgroup",
    ],
    "ctr": [
        "ctr", "click-through rate", "click through rate",
    ],
    "avg_cpc": [
        "avg. cpc", "avg cpc", "average cpc",
    ],
    "conv_rate": [
        "conv. rate", "conv rate", "conversion rate",
    ],
    "cost_per_conv": [
        "cost / conv.", "cost / conv", "cost per conv.", "cost per conversion",
        "cost/conv", "cost/conv.",
    ],
    "currency_code": [
        "currency code", "currency",
    ],
}


def _detect_header_row(df: pd.DataFrame) -> int:
    """
    Google Ads exports often have metadata rows (report title, date range)
    before the actual column headers. Scan the first 10 rows for one that
    looks like a header (contains known column names like 'Search term',
    'Cost', 'Clicks', etc.).
    """
    known_headers = {
        "search term", "cost", "clicks", "impressions", "impr.",
        "conversions", "keyword", "campaign", "query", "search query",
    }
    for idx in range(min(10, len(df))):
        row_values = {str(v).lower().strip() for v in df.iloc[idx] if pd.notna(v)}
        matches = row_values & known_headers
        if len(matches) >= 3:
            return idx
    return -1


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map whatever column names the user's export uses to canonical names."""
    # Strip BOM and whitespace from column names
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    # Check if the first column has the expected search term header;
    # if not, the real headers might be in a data row (Google Ads metadata rows).
    lower_first_cols = {str(c).lower().strip() for c in df.columns}
    known_search = {"search term", "query", "search query", "search terms"}
    if not (lower_first_cols & known_search):
        header_idx = _detect_header_row(df)
        if header_idx >= 0:
            new_headers = [str(v).strip() if pd.notna(v) else f"col_{i}"
                          for i, v in enumerate(df.iloc[header_idx])]
            df.columns = new_headers
            df = df.iloc[header_idx + 1:].reset_index(drop=True)

    # Strip BOM/whitespace again after potential header reassignment
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    # Filter out Google Ads summary/total rows
    for col in df.columns:
        if col.lower().strip() in ("search term", "search_term", "query", "search query"):
            mask = df[col].astype(str).str.startswith("Total:")
            df = df[~mask].reset_index(drop=True)
            break

    rename_map: dict[str, str] = {}
    lower_cols = {c.lower().strip(): c for c in df.columns}

    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower_cols:
                rename_map[lower_cols[alias]] = canonical
                break

    df = df.rename(columns=rename_map)
    return df


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip currency symbols, commas, percent signs and coerce to float."""
    return (
        series.astype(str)
        .str.replace(r"[$,£€%]", "", regex=True)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


# ---------------------------------------------------------------------------
# N-gram decomposition
# ---------------------------------------------------------------------------

def compute_ngrams(df: pd.DataFrame, max_n: int = 3) -> pd.DataFrame:
    """Break search terms into 1/2/3-grams and aggregate metrics."""
    rows: list[dict] = []

    for _, row in df.iterrows():
        term = str(row.get("search_term", "")).lower().strip()
        words = term.split()
        if not words:
            continue

        cost = float(row.get("cost", 0))
        clicks = int(row.get("clicks", 0))
        impressions = int(row.get("impressions", 0))
        conversions = float(row.get("conversions", 0))
        conv_value = float(row.get("conversion_value", 0))

        for n in range(1, min(max_n + 1, len(words) + 1)):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i : i + n])
                rows.append({
                    "ngram": gram,
                    "n": n,
                    "cost": cost,
                    "clicks": clicks,
                    "impressions": impressions,
                    "conversions": conversions,
                    "conversion_value": conv_value,
                    "query_count": 1,
                })

    if not rows:
        return pd.DataFrame()

    ngram_df = pd.DataFrame(rows)
    agg = (
        ngram_df.groupby(["ngram", "n"])
        .agg(
            cost=("cost", "sum"),
            clicks=("clicks", "sum"),
            impressions=("impressions", "sum"),
            conversions=("conversions", "sum"),
            conversion_value=("conversion_value", "sum"),
            query_count=("query_count", "sum"),
        )
        .reset_index()
    )

    agg["cpa"] = agg.apply(
        lambda r: r["cost"] / r["conversions"] if r["conversions"] > 0 else float("inf"),
        axis=1,
    )
    agg["cvr"] = agg.apply(
        lambda r: r["conversions"] / r["clicks"] * 100 if r["clicks"] > 0 else 0,
        axis=1,
    )
    agg["ctr"] = agg.apply(
        lambda r: r["clicks"] / r["impressions"] * 100 if r["impressions"] > 0 else 0,
        axis=1,
    )

    return agg.sort_values("cost", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Query categorisation (three-path framework)
# ---------------------------------------------------------------------------

def categorise_query(
    row: pd.Series,
    account_avg_cpa: float,
    min_clicks_threshold: int = 10,
) -> str:
    """
    Assign a query to one of the three paths.

    Instead of relying on a user-supplied target CPA, we derive thresholds
    from the account's own average CPA (computed from the data).  This makes
    the tool zero-config for accounts that don't track a CPA target.

    Rules
    -----
    - < min_clicks  ->  Monitor (not enough data)
    - 0 conversions + any spend  ->  Negative
    - conversions >= 2 and CPA <= account_avg_cpa  ->  Growth
    - conversions > 0  but CPA > 3x account_avg_cpa  ->  Negative (poor ROI)
    - everything else  ->  Monitor
    """
    cost = float(row.get("cost", 0))
    conversions = float(row.get("conversions", 0))
    clicks = int(row.get("clicks", 0))

    if clicks < min_clicks_threshold:
        return QueryPath.MONITOR.value

    if conversions == 0 and cost > 0:
        return QueryPath.NEGATIVE.value

    if conversions >= 2 and (cost / max(conversions, 1)) <= account_avg_cpa:
        return QueryPath.GROWTH.value

    if conversions > 0 and account_avg_cpa > 0 and (cost / max(conversions, 1)) > account_avg_cpa * 3:
        return QueryPath.NEGATIVE.value

    return QueryPath.MONITOR.value


# ---------------------------------------------------------------------------
# Negative keyword recommendation logic
# ---------------------------------------------------------------------------

_ALWAYS_IRRELEVANT_PATTERNS = [
    (r"\bfree\b", "free", NegativeMatchType.BROAD, "Irrelevant to paid products"),
    (r"\bjobs?\b", "jobs", NegativeMatchType.BROAD, "Job-seeker intent, not buyer"),
    (r"\bhow to\b", "how to", NegativeMatchType.PHRASE, "Informational / DIY intent"),
    (r"\bwhat is\b", "what is", NegativeMatchType.PHRASE, "Informational intent"),
    (r"\brecipe\b", "recipe", NegativeMatchType.BROAD, "Non-commercial content"),
    (r"\bdiy\b", "diy", NegativeMatchType.BROAD, "DIY intent, not buyer"),
    (r"\bsalary\b", "salary", NegativeMatchType.BROAD, "Employment intent"),
    (r"\binternship\b", "internship", NegativeMatchType.BROAD, "Employment intent"),
    (r"\bcareers?\b", "career", NegativeMatchType.BROAD, "Employment intent"),
    (r"\brepair\b", "repair", NegativeMatchType.BROAD, "Service intent, not product"),
    (r"\bused\b", "used", NegativeMatchType.BROAD, "Second-hand intent"),
    (r"\bcheap\b", "cheap", NegativeMatchType.BROAD, "Low-value / bargain intent"),
    (r"\bcraigslist\b", "craigslist", NegativeMatchType.BROAD, "Marketplace redirect"),
    (r"\bebay\b", "ebay", NegativeMatchType.BROAD, "Marketplace redirect"),
]


def _suggest_match_type_for_ngram(ngram: str, n: int) -> NegativeMatchType:
    """Heuristic: single words get broad, multi-word gets phrase."""
    if n == 1:
        return NegativeMatchType.BROAD
    return NegativeMatchType.PHRASE


def _severity_from_waste(waste: float, total_cost: float) -> Severity:
    pct = (waste / total_cost * 100) if total_cost > 0 else 0
    if pct >= 5:
        return Severity.CRITICAL
    if pct >= 2:
        return Severity.HIGH
    if pct >= 0.5:
        return Severity.MEDIUM
    return Severity.LOW


# ---------------------------------------------------------------------------
# Main analysis orchestrator
# ---------------------------------------------------------------------------

def analyse_sqr(
    df: pd.DataFrame,
    min_clicks: int = 5,
    min_cost_threshold: float = 10.0,
) -> AnalysisResult:
    """
    Run the full SQR analysis pipeline.

    Parameters
    ----------
    df : DataFrame with search term data (any standard Google Ads export format)
    min_clicks : Minimum clicks before a query is evaluated (noise filter)
    min_cost_threshold : Minimum cost for a query to appear in recommendations
    """
    df = _normalise_columns(df.copy())

    # Clean numeric columns
    for col in ["cost", "conversions", "clicks", "impressions", "conversion_value"]:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])
        else:
            df[col] = 0

    if "search_term" not in df.columns:
        raise ValueError(
            "Could not find a search term column. "
            "Expected one of: " + ", ".join(_COLUMN_ALIASES["search_term"])
        )

    total_cost = df["cost"].sum()
    total_conversions = df["conversions"].sum()
    total_queries = len(df)

    # Derive account average CPA from the data itself (no user input needed)
    account_avg_cpa = (
        total_cost / total_conversions if total_conversions > 0 else total_cost * 0.1
    )

    # Step 1: Categorise every query
    df["path"] = df.apply(
        lambda r: categorise_query(r, account_avg_cpa, min_clicks), axis=1
    )

    # Step 2: Calculate wasted spend
    negative_mask = df["path"] == QueryPath.NEGATIVE.value
    wasted_spend = df.loc[negative_mask, "cost"].sum()
    waste_pct = (wasted_spend / total_cost * 100) if total_cost > 0 else 0

    # Step 3: Build negative recommendations from individual queries
    negatives: list[NegativeRecommendation] = []
    wasteful = df[negative_mask & (df["cost"] >= min_cost_threshold)].sort_values(
        "cost", ascending=False
    )

    for _, row in wasteful.iterrows():
        term = str(row["search_term"]).lower().strip()
        match_type = NegativeMatchType.EXACT
        reason = "Zero or poor conversions with significant spend"

        # Check known irrelevant patterns
        for pattern, neg_term, mt, desc in _ALWAYS_IRRELEVANT_PATTERNS:
            if re.search(pattern, term, re.IGNORECASE):
                match_type = mt
                reason = desc
                break

        negatives.append(NegativeRecommendation(
            term=term,
            match_type=match_type,
            reason=reason,
            severity=_severity_from_waste(row["cost"], total_cost),
            estimated_waste=round(row["cost"], 2),
            query_count=1,
            source="query",
        ))

    # Step 4: N-gram analysis for pattern-level negatives
    ngram_df = compute_ngrams(df, max_n=3)
    ngram_negatives: list[NegativeRecommendation] = []

    if not ngram_df.empty:
        # High cost, zero conversion n-grams
        wasteful_ngrams = ngram_df[
            (ngram_df["conversions"] == 0)
            & (ngram_df["cost"] >= min_cost_threshold)
            & (ngram_df["query_count"] >= 3)
        ].head(50)

        seen_terms = {n.term for n in negatives}
        for _, row in wasteful_ngrams.iterrows():
            gram = row["ngram"]
            if gram in seen_terms or len(gram) <= 2:
                continue
            seen_terms.add(gram)

            mt = _suggest_match_type_for_ngram(gram, int(row["n"]))
            ngram_negatives.append(NegativeRecommendation(
                term=gram,
                match_type=mt,
                reason=f"N-gram pattern: ${row['cost']:.0f} across {int(row['query_count'])} queries, 0 conversions",
                severity=_severity_from_waste(row["cost"], total_cost),
                estimated_waste=round(row["cost"], 2),
                query_count=int(row["query_count"]),
                source="ngram",
            ))

    all_negatives = negatives + ngram_negatives
    all_negatives.sort(key=lambda n: n.estimated_waste, reverse=True)

    # Step 5: Growth candidates
    growth_candidates: list[GrowthCandidate] = []
    growth_mask = df["path"] == QueryPath.GROWTH.value
    growers = df[growth_mask].sort_values("conversions", ascending=False)

    for _, row in growers.iterrows():
        cpa = row["cost"] / max(row["conversions"], 1)
        growth_candidates.append(GrowthCandidate(
            term=str(row["search_term"]),
            conversions=int(row["conversions"]),
            cost=round(row["cost"], 2),
            cpa=round(cpa, 2),
            clicks=int(row["clicks"]),
            reason="Consistent conversions below target CPA - promote to dedicated keyword",
        ))

    # Step 6: Targeting issues (10% rule)
    targeting_issues: list[TargetingIssue] = []
    ten_pct_triggered = False

    if "keyword" in df.columns:
        keyword_groups = df.groupby("keyword")
        for kw, group in keyword_groups:
            total_kw_queries = len(group)
            neg_queries = len(group[group["path"] == QueryPath.NEGATIVE.value])
            if total_kw_queries >= 10:
                neg_rate = neg_queries / total_kw_queries * 100
                if neg_rate > 10:
                    ten_pct_triggered = True
                    kw_waste = group.loc[
                        group["path"] == QueryPath.NEGATIVE.value, "cost"
                    ].sum()
                    targeting_issues.append(TargetingIssue(
                        keyword=str(kw),
                        negative_rate=round(neg_rate, 1),
                        waste_amount=round(kw_waste, 2),
                        recommendation=(
                            f"{neg_rate:.0f}% of queries need negation. "
                            "This exceeds the 10% threshold - consider changing "
                            "match type or replacing this keyword with more "
                            "specific variants."
                        ),
                    ))
        targeting_issues.sort(key=lambda t: t.waste_amount, reverse=True)

    # Step 7: Escalation alerts
    escalation_alerts: list[str] = []
    if waste_pct >= 15:
        escalation_alerts.append(
            f"Wasted spend is {waste_pct:.1f}% of total - "
            "immediate structural intervention needed."
        )
    if ten_pct_triggered:
        escalation_alerts.append(
            "10% rule triggered on one or more keywords - "
            "targeting changes required, not just negatives."
        )

    # Check for single high-waste queries (>10% of total cost)
    if not df.empty:
        max_single = df.loc[negative_mask, "cost"].max() if negative_mask.any() else 0
        if max_single > total_cost * 0.10:
            offender = df.loc[df["cost"] == max_single, "search_term"].iloc[0]
            escalation_alerts.append(
                f'Single query "{offender}" is spending '
                f">{max_single / total_cost * 100:.0f}% of budget with no conversions."
            )

    return AnalysisResult(
        total_queries=total_queries,
        total_cost=round(total_cost, 2),
        total_conversions=int(total_conversions),
        wasted_spend=round(wasted_spend, 2),
        waste_percentage=round(waste_pct, 1),
        negatives=all_negatives,
        growth_candidates=growth_candidates,
        targeting_issues=targeting_issues,
        ngram_data=ngram_df,
        categorised_df=df,
        ten_percent_rule_triggered=ten_pct_triggered,
        escalation_alerts=escalation_alerts,
    )
