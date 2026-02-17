#!/usr/bin/env python3
"""
Pre-process jobs.jsonl into memory-mappable index files.

Two-pass approach to stay memory-friendly:
  Pass 1: Count jobs (and detect embedding dimension)
  Pass 2: Stream jobs into pre-allocated .npy files + jobs.pkl

Deduplicates on (title, company_name) — keeps the first occurrence.

Output:
  - src/data/jobs.pkl      — list[Job] metadata (no embeddings)
  - src/data/explicit.npy  — (N, dim) float32 embedding matrix
  - src/data/inferred.npy  — (N, dim) float32 embedding matrix
  - src/data/company.npy   — (N, dim) float32 embedding matrix

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
CHUNK_SIZE = 5000  # rows to accumulate before flushing to mmap


def _count_and_detect(jsonl_path: Path, max_jobs: int | None) -> tuple[int, int]:
    """Pass 1: count non-duplicate jobs and detect embedding dimension."""
    seen = set()
    embed_dim = 0
    count = 0

    with open(jsonl_path, "r") as f:
        for line in f:
            if max_jobs and count >= max_jobs:
                break

            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Dedup key
            job_info = raw.get("job_information") or {}
            v5_co = raw.get("v5_processed_company_data") or {}
            title = job_info.get("title") or ""
            company = v5_co.get("name") or (job_info.get("company_info") or {}).get("name") or ""
            key = (title.lower().strip(), company.lower().strip())
            if key in seen:
                continue
            seen.add(key)

            # Detect embed dim from first real vector
            if embed_dim == 0:
                v7 = raw.get("v7_processed_job_data") or {}
                for k in ("embedding_explicit_vector", "embedding_inferred_vector", "embedding_company_vector"):
                    v = v7.get(k)
                    if v:
                        embed_dim = len(v)
                        break

            count += 1

    return count, embed_dim


def _parse_job(raw: dict, i: int) -> tuple[Job, list, list, list]:
    """Parse a single raw JSON record into a Job + 3 embedding vectors."""
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

    explicit_vec = v7.get("embedding_explicit_vector")
    inferred_vec = v7.get("embedding_inferred_vector")
    company_vec = v7.get("embedding_company_vector")

    return job, explicit_vec, inferred_vec, company_vec


def build_index(jsonl_path: str | Path, max_jobs: int | None = None):
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset not found: {jsonl_path}")

    # Pass 1: count unique jobs and detect embedding dimension
    print("  Pass 1: counting jobs...", file=sys.stderr)
    total, embed_dim = _count_and_detect(jsonl_path, max_jobs)
    print(f"  Found {total:,} unique jobs (embed_dim={embed_dim})", file=sys.stderr)

    if total == 0 or embed_dim == 0:
        print("  No valid jobs found.", file=sys.stderr)
        return

    # Pre-allocate memory-mapped .npy files
    print("  Pass 2: writing index files...", file=sys.stderr)
    mmap_files = {}
    for name in ("explicit", "inferred", "company"):
        path = DATA_DIR / f"{name}.npy"
        # Create the .npy file with correct header, then open as writable mmap
        dummy = np.zeros((total, embed_dim), dtype=np.float32)
        np.save(path, dummy)
        del dummy
        mmap_files[name] = np.load(path, mmap_mode="r+")

    zero_vec = np.zeros(embed_dim, dtype=np.float32)
    seen: set[tuple[str, str]] = set()
    jobs: list[Job] = []
    row = 0
    dupes = 0

    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if max_jobs and row >= max_jobs:
                break

            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Dedup check
            job_info_raw = raw.get("job_information") or {}
            v5_co_raw = raw.get("v5_processed_company_data") or {}
            title = job_info_raw.get("title") or ""
            company = v5_co_raw.get("name") or (job_info_raw.get("company_info") or {}).get("name") or ""
            key = (title.lower().strip(), company.lower().strip())
            if key in seen:
                dupes += 1
                continue
            seen.add(key)

            job, explicit, inferred, company_vec = _parse_job(raw, i)
            jobs.append(job)

            # Write embeddings directly to mmap
            def _norm(v):
                if v is None:
                    return zero_vec
                arr = np.array(v, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr /= norm
                return arr

            mmap_files["explicit"][row] = _norm(explicit)
            mmap_files["inferred"][row] = _norm(inferred)
            mmap_files["company"][row] = _norm(company_vec)

            row += 1
            if row % 10000 == 0:
                print(f"  Written {row:,}/{total:,} jobs...", file=sys.stderr)

    # Flush mmaps
    for mmap in mmap_files.values():
        mmap.flush()
    del mmap_files

    print(f"  Written {row:,} jobs ({dupes:,} duplicates skipped)", file=sys.stderr)

    # Save job metadata (pickle is safe — only our own Job dataclass)
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
