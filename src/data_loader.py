"""
Load and index jobs from JSONL dataset.

Handles messy real-world data: missing fields, inconsistent formats,
HTML descriptions. Stores embeddings as normalized NumPy matrices
for fast dot-product similarity.
"""

import json
import re
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import numpy as np


# ── HTML Stripping ──────────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data):
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def strip_html(html: str | None) -> str | None:
    if not html:
        return html
    stripper = _HTMLStripper()
    try:
        stripper.feed(html)
        text = stripper.get_text()
    except Exception:
        # Fallback: regex strip
        text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


# ── Job Data Model ──────────────────────────────────────────────────────────

@dataclass
class Job:
    id: str
    apply_url: str | None

    # Core info
    title: str | None
    company_name: str | None
    location: str | None
    description_text: str | None  # HTML-stripped

    # Structured fields (from v7_processed_job_data + v5_processed_job_data)
    seniority_level: str | None  # v7.experience_requirements.seniority_level
    remote_type: str | None  # v7.work_arrangement.workplace_type
    employment_type: str | None  # v7.work_arrangement.commitment[0]
    salary_min: float | None  # v5_job.yearly_min_compensation
    salary_max: float | None  # v5_job.yearly_max_compensation
    required_skills: list[str] = field(default_factory=list)

    # Company data (from v5_processed_company_data)
    company_type: str | None = None  # derived from is_non_profit/is_public_company
    industries: list[str] = field(default_factory=list)

    # Geo (from _geoloc)
    latitude: float | None = None
    longitude: float | None = None

    @property
    def salary_display(self) -> str | None:
        if self.salary_min and self.salary_max:
            return f"${self.salary_min:,} - ${self.salary_max:,}"
        elif self.salary_min:
            return f"${self.salary_min:,}+"
        elif self.salary_max:
            return f"Up to ${self.salary_max:,}"
        return None


# ── Dataset ─────────────────────────────────────────────────────────────────

# TODO: JobDataset.load() loads all jobs + 3 embedding matrices into memory at once
# (~1.7GB for 100k jobs). Needs chunked/streaming redesign for Phase 2 search engine.
# Options: memory-mapped NumPy arrays, batch loading, or on-disk index (FAISS/annoy).

class JobDataset:
    """In-memory job dataset with embedding matrices for vector search."""

    def __init__(self):
        self.jobs: list[Job] = []
        self.explicit_embeddings: np.ndarray | None = None  # (N, 1536)
        self.inferred_embeddings: np.ndarray | None = None
        self.company_embeddings: np.ndarray | None = None

        # Tracks which indices have valid embeddings
        self._has_explicit: np.ndarray | None = None
        self._has_inferred: np.ndarray | None = None
        self._has_company: np.ndarray | None = None

    def __len__(self):
        return len(self.jobs)

    @staticmethod
    def load(path: str | Path, max_jobs: int | None = None) -> "JobDataset":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        dataset = JobDataset()
        explicit_vecs = []
        inferred_vecs = []
        company_vecs = []

        embed_dim = 1536
        zero_vec = [0.0] * embed_dim

        total_lines = 0
        with open(path, "r") as f:
            for line in f:
                total_lines += 1
        if max_jobs:
            total_lines = min(total_lines, max_jobs)

        loaded = 0
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if max_jobs and i >= max_jobs:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract nested data safely
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

                # Company name: prefer v5_co.name, fall back to job_info.company_info.name
                company_info = job_info.get("company_info") or {}
                company_name = v5_co.get("name") or company_info.get("name")

                # Location: from v5_job (pre-formatted) or first v7 workplace location
                location = v5_job.get("formatted_workplace_location")
                if not location and work_arr.get("workplace_locations"):
                    loc = work_arr["workplace_locations"][0]
                    parts = [loc.get("city"), loc.get("state"), loc.get("country_code")]
                    location = ", ".join(p for p in parts if p)

                # Salary: prefer v5_job yearly (already normalized), fall back to v7 salary
                salary_min = _safe_float(v5_job.get("yearly_min_compensation"))
                salary_max = _safe_float(v5_job.get("yearly_max_compensation"))
                if salary_min is None:
                    salary_min = _safe_float(salary_info.get("low"))
                if salary_max is None:
                    salary_max = _safe_float(salary_info.get("high"))

                # Employment type: from v7 work_arrangement.commitment (list)
                commitment = work_arr.get("commitment") or []
                employment_type = _normalize_str(commitment[0]) if commitment else None

                # Skills: extract .value from v7.skills.explicit objects
                explicit_skills = v7_skills.get("explicit") or []
                required_skills = [s["value"] for s in explicit_skills if isinstance(s, dict) and "value" in s]

                # Company type: derive from v5_co boolean flags
                company_type = _derive_company_type(v5_co)

                # Geo: from _geoloc array
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
                dataset.jobs.append(job)

                # Embeddings — use zero vector if missing
                explicit_vecs.append(v7.get("embedding_explicit_vector") or zero_vec)
                inferred_vecs.append(v7.get("embedding_inferred_vector") or zero_vec)
                company_vecs.append(v7.get("embedding_company_vector") or zero_vec)

                loaded += 1
                if loaded % 10000 == 0:
                    print(f"  Loaded {loaded:,}/{total_lines:,} jobs...", file=sys.stderr)

        print(f"  Loaded {loaded:,} jobs total.", file=sys.stderr)

        # Build embedding matrices and normalize
        dataset.explicit_embeddings = _build_matrix(explicit_vecs)
        dataset.inferred_embeddings = _build_matrix(inferred_vecs)
        dataset.company_embeddings = _build_matrix(company_vecs)

        # Track which jobs have real (non-zero) embeddings
        dataset._has_explicit = np.any(dataset.explicit_embeddings != 0, axis=1)
        dataset._has_inferred = np.any(dataset.inferred_embeddings != 0, axis=1)
        dataset._has_company = np.any(dataset.company_embeddings != 0, axis=1)

        return dataset


# ── Helpers ─────────────────────────────────────────────────────────────────

def _derive_company_type(v5_co: dict) -> str | None:
    """Derive a company type label from v5_processed_company_data boolean flags."""
    if v5_co.get("is_non_profit"):
        return "non-profit"
    if v5_co.get("is_public_company"):
        return "public"
    # If we have a name but it's not public/non-profit, call it private
    if v5_co.get("name"):
        return "private"
    return None


def _build_matrix(vectors: list[list[float]]) -> np.ndarray:
    """Build a normalized embedding matrix from a list of vectors."""
    mat = np.array(vectors, dtype=np.float32)
    # Normalize rows to unit length (for dot product = cosine similarity)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero for missing embeddings
    mat /= norms
    return mat


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _normalize_str(val) -> str | None:
    if val is None:
        return None
    s = str(val).strip().lower()
    return s if s else None
