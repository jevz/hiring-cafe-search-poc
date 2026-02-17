#!/usr/bin/env python3
"""
Pre-process jobs.jsonl into memory-mappable index files.

Streams the JSONL once and writes:
  - src/data/jobs.pkl      — list[Job] metadata (no embeddings)
  - src/data/explicit.npy  — (N, 1536) float32 embedding matrix
  - src/data/inferred.npy  — (N, 1536) float32 embedding matrix
  - src/data/company.npy   — (N, 1536) float32 embedding matrix

Usage:
    python build_index.py [path/to/jobs.jsonl] [--max-jobs N]
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

from src.data_loader import Job, strip_html, _safe_float, _normalize_str, _derive_company_type

DATA_DIR = Path("src/data")


def build_index(jsonl_path: str | Path, max_jobs: int | None = None):
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

    jobs: list[Job] = []
    explicit_vecs: list[list[float]] = []
    inferred_vecs: list[list[float]] = []
    company_vecs: list[list[float]] = []
    zero_vec: list[float] = []  # populated from first record's embedding length

    loaded = 0
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if max_jobs and loaded >= max_jobs:
                break

            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            job_info = raw.get("job_information") or {}
            v7 = raw.get("v7_processed_job_data") or {}
            v5_job = raw.get("v5_processed_job_data") or {}
            v5_co = raw.get("v5_processed_company_data") or {}
            geoloc = raw.get("_geoloc") or []

            work_arr = v7.get("work_arrangement") or {}
            exp_req = v7.get("experience_requirements") or {}
            comp_ben = v7.get("compensation_and_benefits") or {}
            salary_info = comp_ben.get("salary") or {}
            v7_skills = v7.get("skills") or {}

            company_info = job_info.get("company_info") or {}
            company_name = v5_co.get("name") or company_info.get("name")

            location = v5_job.get("formatted_workplace_location")
            if not location and work_arr.get("workplace_locations"):
                loc = work_arr["workplace_locations"][0]
                parts = [loc.get("city"), loc.get("state"), loc.get("country_code")]
                location = ", ".join(p for p in parts if p)

            salary_min = _safe_float(v5_job.get("yearly_min_compensation"))
            salary_max = _safe_float(v5_job.get("yearly_max_compensation"))
            if salary_min is None:
                salary_min = _safe_float(salary_info.get("low"))
            if salary_max is None:
                salary_max = _safe_float(salary_info.get("high"))

            commitment = work_arr.get("commitment") or []
            employment_type = _normalize_str(commitment[0]) if commitment else None

            explicit_skills = v7_skills.get("explicit") or []
            required_skills = [s["value"] for s in explicit_skills if isinstance(s, dict) and "value" in s]

            company_type = _derive_company_type(v5_co)
            geo_point = geoloc[0] if geoloc else {}

            job = Job(
                id=raw.get("id", f"unknown_{i}"),
                apply_url=raw.get("apply_url"),
                title=job_info.get("title"),
                company_name=company_name,
                location=location,
                description_text=strip_html(job_info.get("description")),
                seniority_level=_normalize_str(exp_req.get("seniority_level")),
                remote_type=_normalize_str(work_arr.get("workplace_type")),
                employment_type=employment_type,
                salary_min=salary_min,
                salary_max=salary_max,
                required_skills=required_skills,
                company_type=company_type,
                industries=v5_co.get("industries") or [],
                latitude=_safe_float(geo_point.get("lat")),
                longitude=_safe_float(geo_point.get("lon")),
            )
            jobs.append(job)

            explicit = v7.get("embedding_explicit_vector")
            inferred = v7.get("embedding_inferred_vector")
            company = v7.get("embedding_company_vector")

            # Infer embedding dimension from the first real vector we encounter
            if not zero_vec:
                for v in (explicit, inferred, company):
                    if v:
                        zero_vec = [0.0] * len(v)
                        break

            explicit_vecs.append(explicit or zero_vec)
            inferred_vecs.append(inferred or zero_vec)
            company_vecs.append(company or zero_vec)

            loaded += 1
            if loaded % 10000 == 0:
                print(f"  Parsed {loaded:,} jobs...", file=sys.stderr)

    print(f"  Parsed {loaded:,} jobs total.", file=sys.stderr)

    # Build and normalize embedding matrices
    print("  Building embedding matrices...", file=sys.stderr)
    for name, vecs in [("explicit", explicit_vecs), ("inferred", inferred_vecs), ("company", company_vecs)]:
        mat = np.array(vecs, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        mat /= norms

        out_path = DATA_DIR / f"{name}.npy"
        np.save(out_path, mat)
        print(f"  Saved {out_path} ({mat.shape})", file=sys.stderr)

        # Free memory before next matrix
        del mat, vecs

    # Save job metadata (pickle is safe here — we only load our own Job dataclass)
    pkl_path = DATA_DIR / "jobs.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(jobs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {pkl_path} ({len(jobs):,} jobs)", file=sys.stderr)

    print("Done.", file=sys.stderr)


def main():
    max_jobs = None
    jsonl_path = "src/data/jobs.jsonl"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--max-jobs":
            max_jobs = int(args[i + 1])
            i += 2
        else:
            jsonl_path = args[i]
            i += 1

    print(f"Building index from {jsonl_path}...")
    if max_jobs:
        print(f"  (limited to {max_jobs:,} jobs)")
    build_index(jsonl_path, max_jobs=max_jobs)


if __name__ == "__main__":
    main()
