#!/usr/bin/env python3
"""
Comprehensive analysis of RAG evaluation data for findings report v3.
Focuses on:
1. Search type dimension clarity (keyword/hybrid/semantic)
2. Single-concept vs cross-domain separation
3. Retrieval quality metrics (context_recall, context_precision)
4. Statistical significance with variance consideration
5. Answer correctness for end-to-end evaluation
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Load data
data_path = Path("/home/fliperbaker/projects/raglab/data/evaluation/ragas_results/comprehensive_20260101_164236.json")
with open(data_path) as f:
    data = json.load(f)

# Extract leaderboard (all 102 configurations)
all_runs = data["leaderboard"]

# Filter out semantic_0.75 (edge case)
runs = [r for r in all_runs if "semantic_0_75" not in r["collection"]]
print(f"Total runs after filtering semantic_0.75: {len(runs)}")

# Map collection names to chunking strategies
def get_chunking(collection):
    if "contextual" in collection:
        return "contextual"
    elif "raptor" in collection:
        return "raptor"
    elif "section" in collection:
        return "section"
    elif "semantic_0_3" in collection:
        return "semantic_0.3"
    return "unknown"

# Map search configuration to unified search type
def get_search_type(search_type, alpha):
    """
    Keyword (BM25 only) = alpha 0.0
    Hybrid (balanced) = alpha 0.5
    Semantic (vector-heavy) = alpha 1.0
    """
    if search_type == "keyword":
        return "keyword (α=0.0)"
    elif alpha == 0.5:
        return "hybrid (α=0.5)"
    elif alpha == 1.0:
        return "semantic (α=1.0)"
    return f"hybrid (α={alpha})"

# Enrich each run
for r in runs:
    r["chunking"] = get_chunking(r["collection"])
    r["unified_search"] = get_search_type(r["search_type"], r["alpha"])
    r["preprocessing"] = r["strategy"]

# Dimension values
CHUNKINGS = ["section", "contextual", "raptor", "semantic_0.3"]
SEARCHES = ["keyword (α=0.0)", "hybrid (α=0.5)", "semantic (α=1.0)"]
PREPROCESSINGS = ["none", "hyde", "decomposition", "graphrag"]
TOP_KS = [10, 20]

# Metrics to analyze
METRICS = ["context_recall", "context_precision", "answer_correctness", "relevancy"]
QUESTION_TYPES = ["single_concept", "cross_domain"]

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def analyze_dimension(runs, dimension_name, dimension_values, metric, question_type=None):
    """Analyze a single dimension for a specific metric and question type."""
    results = {}
    for val in dimension_values:
        if dimension_name == "chunking":
            subset = [r for r in runs if r["chunking"] == val]
        elif dimension_name == "search":
            subset = [r for r in runs if r["unified_search"] == val]
        elif dimension_name == "preprocessing":
            subset = [r for r in runs if r["preprocessing"] == val]
        elif dimension_name == "top_k":
            subset = [r for r in runs if r["top_k"] == val]
        else:
            continue

        if question_type:
            scores = [r["difficulty_breakdown"][question_type][metric] for r in subset
                      if question_type in r.get("difficulty_breakdown", {})]
        else:
            scores = [r["scores"][metric] for r in subset]

        results[val] = {
            "mean": np.mean(scores),
            "std": np.std(scores, ddof=1),
            "n": len(scores),
            "scores": scores
        }
    return results

def print_dimension_analysis(runs, dimension_name, dimension_values, metric, question_type=None):
    """Print analysis for a dimension."""
    results = analyze_dimension(runs, dimension_name, dimension_values, metric, question_type)

    q_label = f" ({question_type})" if question_type else ""
    print(f"\n{'='*60}")
    print(f"{dimension_name.upper()} dimension - {metric}{q_label}")
    print(f"{'='*60}")

    # Sort by mean descending
    sorted_vals = sorted(results.keys(), key=lambda x: results[x]["mean"], reverse=True)

    print(f"\n{'Value':<25} {'Mean':>8} {'Std':>8} {'N':>6}")
    print("-" * 50)
    for val in sorted_vals:
        r = results[val]
        print(f"{str(val):<25} {r['mean']:>8.4f} {r['std']:>8.4f} {r['n']:>6}")

    # Pairwise effect sizes for significant differences
    print(f"\nPairwise Cohen's d (row vs column):")
    print(f"{'':15}", end="")
    for v in sorted_vals[:4]:
        print(f"{str(v)[:12]:>14}", end="")
    print()

    for i, v1 in enumerate(sorted_vals[:4]):
        print(f"{str(v1)[:14]:<15}", end="")
        for j, v2 in enumerate(sorted_vals[:4]):
            if i < j:
                d = cohens_d(results[v1]["scores"], results[v2]["scores"])
                sig = "***" if abs(d) >= 0.8 else "**" if abs(d) >= 0.5 else "*" if abs(d) >= 0.2 else ""
                print(f"{d:>+8.2f}{sig:>4}", end="")
            else:
                print(f"{'':>14}", end="")
        print()

    return results

# MAIN ANALYSIS
print("\n" + "="*80)
print("RAG EVALUATION ANALYSIS V3")
print("="*80)
print(f"\nConfiguration count: {len(runs)} (excluding semantic_0.75)")
print("\nDimensions:")
print(f"  - Chunking: {CHUNKINGS}")
print(f"  - Search Type: {SEARCHES}")
print(f"  - Preprocessing: {PREPROCESSINGS}")
print(f"  - Top-K: {TOP_KS}")

# Section 1: Overall dimension impact (all questions)
print("\n" + "#"*80)
print("# SECTION 1: OVERALL DIMENSION IMPACT (ALL 15 QUESTIONS)")
print("#"*80)

for metric in ["context_recall", "context_precision", "answer_correctness"]:
    print(f"\n{'='*70}")
    print(f"METRIC: {metric.upper()}")
    print(f"{'='*70}")

    for dim_name, dim_values in [
        ("chunking", CHUNKINGS),
        ("search", SEARCHES),
        ("preprocessing", PREPROCESSINGS),
        ("top_k", TOP_KS)
    ]:
        print_dimension_analysis(runs, dim_name, dim_values, metric)

# Section 2: Single-concept vs Cross-domain comparison
print("\n" + "#"*80)
print("# SECTION 2: SINGLE-CONCEPT VS CROSS-DOMAIN ANALYSIS")
print("#"*80)

for question_type in QUESTION_TYPES:
    print(f"\n{'='*70}")
    print(f"QUESTION TYPE: {question_type.upper()}")
    print(f"{'='*70}")

    for metric in ["context_recall", "context_precision", "answer_correctness"]:
        for dim_name, dim_values in [
            ("chunking", CHUNKINGS),
            ("search", SEARCHES),
            ("preprocessing", PREPROCESSINGS)
        ]:
            print_dimension_analysis(runs, dim_name, dim_values, metric, question_type)

# Section 3: Best configurations
print("\n" + "#"*80)
print("# SECTION 3: TOP CONFIGURATIONS BY METRIC")
print("#"*80)

for metric in ["context_recall", "context_precision", "answer_correctness"]:
    print(f"\n{'='*60}")
    print(f"TOP 10 BY {metric.upper()}")
    print(f"{'='*60}")

    sorted_runs = sorted(runs, key=lambda r: r["scores"][metric], reverse=True)[:10]

    print(f"{'Chunking':<12} {'Search':<18} {'Strategy':<13} {'K':>3} {'Score':>8}")
    print("-" * 60)
    for r in sorted_runs:
        print(f"{r['chunking']:<12} {r['unified_search']:<18} {r['preprocessing']:<13} {r['top_k']:>3} {r['scores'][metric]:>8.4f}")

# Section 4: Cross-domain penalty analysis
print("\n" + "#"*80)
print("# SECTION 4: CROSS-DOMAIN PENALTY ANALYSIS")
print("#"*80)

print("\nThe 'cross-domain penalty' measures how much worse cross-domain questions")
print("perform compared to single-concept questions.\n")

for metric in ["context_recall", "context_precision", "answer_correctness"]:
    print(f"\n{metric.upper()} - Cross-Domain Penalty by Preprocessing:")
    print("-" * 60)

    for preprocessing in PREPROCESSINGS:
        subset = [r for r in runs if r["preprocessing"] == preprocessing]
        single = [r["difficulty_breakdown"]["single_concept"][metric] for r in subset]
        cross = [r["difficulty_breakdown"]["cross_domain"][metric] for r in subset]

        single_mean = np.mean(single)
        cross_mean = np.mean(cross)
        gap = cross_mean - single_mean
        gap_pct = (gap / single_mean) * 100 if single_mean > 0 else 0

        print(f"  {preprocessing:<15} Single: {single_mean:.4f}  Cross: {cross_mean:.4f}  Gap: {gap:+.4f} ({gap_pct:+.1f}%)")

# Section 5: Statistical summary for key findings
print("\n" + "#"*80)
print("# SECTION 5: STATISTICALLY SIGNIFICANT FINDINGS (d >= 0.8)")
print("#"*80)

findings = []

for metric in ["context_recall", "context_precision", "answer_correctness"]:
    for dim_name, dim_values in [
        ("chunking", CHUNKINGS),
        ("search", SEARCHES),
        ("preprocessing", PREPROCESSINGS),
        ("top_k", TOP_KS)
    ]:
        results = analyze_dimension(runs, dim_name, dim_values, metric)
        sorted_vals = sorted(results.keys(), key=lambda x: results[x]["mean"], reverse=True)

        for i, v1 in enumerate(sorted_vals):
            for j, v2 in enumerate(sorted_vals):
                if i < j:
                    d = cohens_d(results[v1]["scores"], results[v2]["scores"])
                    if abs(d) >= 0.8:
                        findings.append({
                            "dimension": dim_name,
                            "metric": metric,
                            "comparison": f"{v1} vs {v2}",
                            "winner": v1,
                            "d": d,
                            "winner_mean": results[v1]["mean"],
                            "winner_std": results[v1]["std"],
                            "loser_mean": results[v2]["mean"],
                            "loser_std": results[v2]["std"],
                            "gap": results[v1]["mean"] - results[v2]["mean"]
                        })

# Sort by effect size
findings.sort(key=lambda x: abs(x["d"]), reverse=True)

print(f"\nFound {len(findings)} statistically significant comparisons (d >= 0.8):\n")
print(f"{'Metric':<20} {'Dimension':<14} {'Winner':<15} {'vs':<15} {'Gap':>7} {'d':>7}")
print("-" * 85)
for f in findings:
    loser = f["comparison"].split(" vs ")[1]
    print(f"{f['metric']:<20} {f['dimension']:<14} {f['winner']:<15} {loser:<15} {f['gap']:>+7.3f} {f['d']:>+7.2f}")

# Section 6: Variance analysis
print("\n" + "#"*80)
print("# SECTION 6: VARIANCE ANALYSIS (RELIABILITY)")
print("#"*80)

print("\nCoefficient of Variation (CV = std/mean) by dimension:")
print("Lower CV = more consistent results across configurations\n")

for metric in ["context_recall", "context_precision", "answer_correctness"]:
    print(f"\n{metric.upper()}:")
    print("-" * 50)

    for dim_name, dim_values in [
        ("chunking", CHUNKINGS),
        ("search", SEARCHES),
        ("preprocessing", PREPROCESSINGS)
    ]:
        results = analyze_dimension(runs, dim_name, dim_values, metric)
        print(f"  {dim_name:<15}")
        for val in dim_values:
            r = results[val]
            cv = r["std"] / r["mean"] if r["mean"] > 0 else 0
            print(f"    {str(val):<20} CV={cv:.3f} (mean={r['mean']:.3f}, std={r['std']:.3f})")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
