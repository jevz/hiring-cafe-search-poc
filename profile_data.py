#!/usr/bin/env python3
"""
Profile the jobs.jsonl dataset using streaming (no full dataset in memory).
Outputs data quality summary to DATA_PROFILE.md and stdout.

Usage:
    python profile_data.py [path/to/jobs.jsonl]
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from src.data_loader import _safe_float, _normalize_str, _derive_company_type


def profile(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    n = 0

    # ── Null counters ────────────────────────────────────────────────────
    null_counts = {
        "title": 0,
        "company_name": 0,
        "location": 0,
        "description": 0,
        "seniority_level": 0,
        "remote_type": 0,
        "employment_type": 0,
        "salary_min": 0,
        "salary_max": 0,
        "required_skills": 0,
        "company_type": 0,
        "industries": 0,
        "latitude": 0,
        "longitude": 0,
    }

    # ── Distribution counters ────────────────────────────────────────────
    seniority_counter = Counter()
    remote_counter = Counter()
    employment_counter = Counter()
    company_type_counter = Counter()
    industry_counter = Counter()

    # ── Embedding counters ───────────────────────────────────────────────
    has_explicit = 0
    has_inferred = 0
    has_company = 0
    has_all_3 = 0
    has_none = 0

    # ── Salary collection (ints only, ~800KB for 100k) ───────────────────
    salaries_min = []
    salaries_max = []

    # ── Duplicate tracking ───────────────────────────────────────────────
    url_counts = Counter()
    title_company_counts = Counter()

    # ── Stream through the file ──────────────────────────────────────────
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            n += 1
            if n % 10000 == 0:
                print(f"  Processed {n:,} jobs...", file=sys.stderr)

            job_info = raw.get("job_information") or {}
            v7 = raw.get("v7_processed_job_data") or {}
            v5_job = raw.get("v5_processed_job_data") or {}
            v5_co = raw.get("v5_processed_company_data") or {}
            geoloc = raw.get("_geoloc") or []

            # Nested v7 sub-objects
            work_arr = v7.get("work_arrangement") or {}
            exp_req = v7.get("experience_requirements") or {}
            comp_ben = v7.get("compensation_and_benefits") or {}
            salary_info = comp_ben.get("salary") or {}
            v7_skills = v7.get("skills") or {}

            # ── Extract fields (same logic as data_loader) ───────────────
            title = job_info.get("title")
            company_info = job_info.get("company_info") or {}
            company_name = v5_co.get("name") or company_info.get("name")
            location = v5_job.get("formatted_workplace_location")
            if not location and work_arr.get("workplace_locations"):
                loc = work_arr["workplace_locations"][0]
                parts = [loc.get("city"), loc.get("state"), loc.get("country_code")]
                location = ", ".join(p for p in parts if p)
            description = job_info.get("description")
            seniority = _normalize_str(exp_req.get("seniority_level"))
            remote_type = _normalize_str(work_arr.get("workplace_type"))
            commitment = work_arr.get("commitment") or []
            employment_type = _normalize_str(commitment[0]) if commitment else None
            salary_min = _safe_float(v5_job.get("yearly_min_compensation"))
            salary_max = _safe_float(v5_job.get("yearly_max_compensation"))
            if salary_min is None:
                salary_min = _safe_float(salary_info.get("low"))
            if salary_max is None:
                salary_max = _safe_float(salary_info.get("high"))
            explicit_skills = v7_skills.get("explicit") or []
            required_skills = [s["value"] for s in explicit_skills if isinstance(s, dict) and "value" in s]
            company_type = _derive_company_type(v5_co)
            industries = v5_co.get("industries") or []
            geo_point = geoloc[0] if geoloc else {}
            lat = _safe_float(geo_point.get("lat"))
            lng = _safe_float(geo_point.get("lon"))

            # ── Null checks ──────────────────────────────────────────────
            if title is None:
                null_counts["title"] += 1
            if company_name is None:
                null_counts["company_name"] += 1
            if location is None:
                null_counts["location"] += 1
            if description is None:
                null_counts["description"] += 1
            if seniority is None:
                null_counts["seniority_level"] += 1
            if remote_type is None:
                null_counts["remote_type"] += 1
            if employment_type is None:
                null_counts["employment_type"] += 1
            if salary_min is None:
                null_counts["salary_min"] += 1
            if salary_max is None:
                null_counts["salary_max"] += 1
            if not required_skills:
                null_counts["required_skills"] += 1
            if company_type is None:
                null_counts["company_type"] += 1
            if not industries:
                null_counts["industries"] += 1
            if lat is None:
                null_counts["latitude"] += 1
            if lng is None:
                null_counts["longitude"] += 1

            # ── Distributions ────────────────────────────────────────────
            if seniority is not None:
                seniority_counter[seniority] += 1
            if remote_type is not None:
                remote_counter[remote_type] += 1
            if employment_type is not None:
                employment_counter[employment_type] += 1
            if company_type is not None:
                company_type_counter[company_type] += 1
            for ind in industries:
                industry_counter[ind] += 1

            # ── Embeddings ───────────────────────────────────────────────
            e_explicit = bool(v7.get("embedding_explicit_vector"))
            e_inferred = bool(v7.get("embedding_inferred_vector"))
            e_company = bool(v7.get("embedding_company_vector"))

            if e_explicit:
                has_explicit += 1
            if e_inferred:
                has_inferred += 1
            if e_company:
                has_company += 1
            if e_explicit and e_inferred and e_company:
                has_all_3 += 1
            if not e_explicit and not e_inferred and not e_company:
                has_none += 1

            # ── Salary ───────────────────────────────────────────────────
            if salary_min is not None:
                salaries_min.append(salary_min)
            if salary_max is not None:
                salaries_max.append(salary_max)

            # ── Duplicates ───────────────────────────────────────────────
            apply_url = raw.get("apply_url")
            if apply_url:
                url_counts[apply_url] += 1
            if title and company_name:
                title_company_counts[(title, company_name)] += 1

    print(f"  Processed {n:,} jobs total.", file=sys.stderr)

    # ── Duplicate stats ──────────────────────────────────────────────────
    dup_urls = sum(1 for c in url_counts.values() if c > 1)
    dup_title_co = sum(1 for c in title_company_counts.values() if c > 1)

    # ── Build Report ─────────────────────────────────────────────────────
    lines = [
        "# Data Profile\n",
        f"**Total jobs:** {n:,}\n",
        "## Null / Missing Rates\n",
        "| Field | Missing | % |",
        "|-------|---------|---|",
    ]
    for name, count in null_counts.items():
        display_name = name
        if name == "description":
            display_name = "description_text"
        pct = count / n * 100 if n else 0
        lines.append(f"| {display_name} | {count:,} | {pct:.1f}% |")

    lines.extend([
        "\n## Embedding Coverage\n",
        "| Embedding | Has Vector | % |",
        "|-----------|-----------|---|",
        f"| explicit | {has_explicit:,} | {has_explicit/n*100:.1f}% |" if n else "| explicit | 0 | 0% |",
        f"| inferred | {has_inferred:,} | {has_inferred/n*100:.1f}% |" if n else "| inferred | 0 | 0% |",
        f"| company | {has_company:,} | {has_company/n*100:.1f}% |" if n else "| company | 0 | 0% |",
        f"| **all 3** | **{has_all_3:,}** | **{has_all_3/n*100:.1f}%** |" if n else "| **all 3** | **0** | **0%** |",
        f"| none | {has_none:,} | {has_none/n*100:.1f}% |" if n else "| none | 0 | 0% |",
    ])

    distributions = {
        "seniority_level (effective)": seniority_counter.most_common(10),
        "remote_type": remote_counter.most_common(10),
        "employment_type": employment_counter.most_common(10),
        "company_type": company_type_counter.most_common(10),
        "industries (flattened)": industry_counter.most_common(15),
    }

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
        f"**Jobs with salary_min:** {len(salaries_min):,} ({len(salaries_min)/n*100:.1f}%)" if n else "**Jobs with salary_min:** 0",
        f"**Jobs with salary_max:** {len(salaries_max):,} ({len(salaries_max)/n*100:.1f}%)" if n else "**Jobs with salary_max:** 0",
    ])
    if salaries_min:
        lines.append(f"**salary_min range:** ${min(salaries_min):,} - ${max(salaries_min):,} (median ${statistics.median(salaries_min):,.0f})")
    if salaries_max:
        lines.append(f"**salary_max range:** ${min(salaries_max):,} - ${max(salaries_max):,} (median ${statistics.median(salaries_max):,.0f})")

    lines.extend([
        "\n## Duplicates\n",
        f"**Duplicate apply_urls:** {dup_urls:,} URLs appear more than once",
        f"**Duplicate title+company:** {dup_title_co:,} combos appear more than once",
    ])

    return "\n".join(lines) + "\n"


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "src/data/jobs.jsonl"
    print(f"Profiling dataset from {data_path}...")

    report = profile(data_path)

    out_path = Path("DATA_PROFILE.md")
    out_path.write_text(report)
    print(f"\nProfile written to {out_path}")
    print(report)


if __name__ == "__main__":
    main()
