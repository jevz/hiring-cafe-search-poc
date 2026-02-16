#!/usr/bin/env python3
"""
Profile the jobs.jsonl dataset.
Outputs data quality summary to DATA_PROFILE.md and stdout.

Usage:
    python profile_data.py [path/to/jobs.jsonl]
"""

import sys
from collections import Counter
from pathlib import Path

from src.data_loader import JobDataset


def profile(dataset: JobDataset) -> str:
    jobs = dataset.jobs
    n = len(jobs)

    # ── Null Rates ──────────────────────────────────────────────────────
    fields = {
        "title": [j.title for j in jobs],
        "company_name": [j.company_name for j in jobs],
        "location": [j.location for j in jobs],
        "description_text": [j.description_text for j in jobs],
        "seniority_level (v7)": [j.seniority_level for j in jobs],
        "seniority_level_new (v7_new)": [j.seniority_level_new for j in jobs],
        "remote_type": [j.remote_type for j in jobs],
        "employment_type": [j.employment_type for j in jobs],
        "salary_min": [j.salary_min for j in jobs],
        "salary_max": [j.salary_max for j in jobs],
        "required_skills": [j.required_skills for j in jobs],
        "company_type": [j.company_type for j in jobs],
        "industries": [j.industries for j in jobs],
        "latitude": [j.latitude for j in jobs],
        "longitude": [j.longitude for j in jobs],
    }

    null_table = []
    for name, values in fields.items():
        if name in ("required_skills", "industries"):
            null_count = sum(1 for v in values if not v)
        else:
            null_count = sum(1 for v in values if v is None)
        pct = null_count / n * 100
        null_table.append((name, null_count, pct))

    # ── Distributions ───────────────────────────────────────────────────
    def top_values(values, top_n=10):
        counts = Counter(v for v in values if v is not None)
        return counts.most_common(top_n)

    distributions = {
        "seniority_level (effective)": top_values([j.effective_seniority for j in jobs]),
        "remote_type": top_values([j.remote_type for j in jobs]),
        "employment_type": top_values([j.employment_type for j in jobs]),
        "company_type": top_values([j.company_type for j in jobs]),
        "industries (flattened)": top_values([ind for j in jobs for ind in j.industries], 15),
    }

    # ── Embedding Coverage ──────────────────────────────────────────────
    has_explicit = int(dataset._has_explicit.sum())
    has_inferred = int(dataset._has_inferred.sum())
    has_company = int(dataset._has_company.sum())
    has_all_3 = int((dataset._has_explicit & dataset._has_inferred & dataset._has_company).sum())
    has_none = int((~dataset._has_explicit & ~dataset._has_inferred & ~dataset._has_company).sum())

    # ── Salary Stats ────────────────────────────────────────────────────
    salaries_min = [j.salary_min for j in jobs if j.salary_min is not None]
    salaries_max = [j.salary_max for j in jobs if j.salary_max is not None]

    # ── Duplicates ──────────────────────────────────────────────────────
    url_counts = Counter(j.apply_url for j in jobs if j.apply_url)
    dup_urls = sum(1 for c in url_counts.values() if c > 1)
    title_co_counts = Counter((j.title, j.company_name) for j in jobs if j.title and j.company_name)
    dup_title_co = sum(1 for c in title_co_counts.values() if c > 1)

    # ── Build Report ────────────────────────────────────────────────────
    lines = [
        "# Data Profile\n",
        f"**Total jobs:** {n:,}\n",
        "## Null / Missing Rates\n",
        "| Field | Missing | % |",
        "|-------|---------|---|",
    ]
    for name, null_count, pct in null_table:
        lines.append(f"| {name} | {null_count:,} | {pct:.1f}% |")

    lines.extend([
        "\n## Embedding Coverage\n",
        f"| Embedding | Has Vector | % |",
        f"|-----------|-----------|---|",
        f"| explicit | {has_explicit:,} | {has_explicit/n*100:.1f}% |",
        f"| inferred | {has_inferred:,} | {has_inferred/n*100:.1f}% |",
        f"| company | {has_company:,} | {has_company/n*100:.1f}% |",
        f"| **all 3** | **{has_all_3:,}** | **{has_all_3/n*100:.1f}%** |",
        f"| none | {has_none:,} | {has_none/n*100:.1f}% |",
    ])

    lines.extend(["\n## Field Distributions\n"])
    for field_name, top in distributions.items():
        lines.append(f"### {field_name}\n")
        lines.append("| Value | Count | % |")
        lines.append("|-------|-------|---|")
        for val, count in top:
            lines.append(f"| {val} | {count:,} | {count/n*100:.1f}% |")
        lines.append("")

    lines.extend([
        "## Salary Statistics\n",
        f"**Jobs with salary_min:** {len(salaries_min):,} ({len(salaries_min)/n*100:.1f}%)",
        f"**Jobs with salary_max:** {len(salaries_max):,} ({len(salaries_max)/n*100:.1f}%)",
    ])
    if salaries_min:
        import statistics
        lines.append(f"**salary_min range:** ${min(salaries_min):,} - ${max(salaries_min):,} (median ${statistics.median(salaries_min):,.0f})")
    if salaries_max:
        import statistics
        lines.append(f"**salary_max range:** ${min(salaries_max):,} - ${max(salaries_max):,} (median ${statistics.median(salaries_max):,.0f})")

    lines.extend([
        "\n## Duplicates\n",
        f"**Duplicate apply_urls:** {dup_urls:,} URLs appear more than once",
        f"**Duplicate title+company:** {dup_title_co:,} combos appear more than once",
    ])

    return "\n".join(lines) + "\n"


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/jobs.jsonl"
    print(f"Loading dataset from {data_path}...")
    dataset = JobDataset.load(data_path)

    print("Profiling...")
    report = profile(dataset)

    out_path = Path("DATA_PROFILE.md")
    out_path.write_text(report)
    print(f"\nProfile written to {out_path}")
    print(report)


if __name__ == "__main__":
    main()
