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

    # Structured fields (from v7_processed_job_data)
    seniority_level: str | None
    remote_type: str | None
    employment_type: str | None
    salary_min: int | None
    salary_max: int | None
    required_skills: list[str] = field(default_factory=list)

    # Normalized seniority (from v7_processed_job_data_new, preferred)
    seniority_level_new: str | None = None

    # Company data (from v5_processed_company_data)
    company_type: str | None = None
    industries: list[str] = field(default_factory=list)

    # Geo
    latitude: float | None = None
    longitude: float | None = None

    @property
    def effective_seniority(self) -> str | None:
        """Prefer v7_new normalized seniority, fall back to v7."""
        return self.seniority_level_new or self.seniority_level

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
                v7_new = raw.get("v7_processed_job_data_new") or {}
                v5 = raw.get("v5_processed_company_data") or {}
                geo = raw.get("geo") or {}

                # Parse salary safely
                salary_min = _safe_int(v7.get("salary_min"))
                salary_max = _safe_int(v7.get("salary_max"))

                job = Job(
                    id=raw.get("id", f"unknown_{i}"),
                    apply_url=raw.get("apply_url"),
                    title=job_info.get("title"),
                    company_name=job_info.get("company_name"),
                    location=job_info.get("location"),
                    description_text=strip_html(job_info.get("description")),
                    seniority_level=_normalize_str(v7.get("seniority_level")),
                    remote_type=_normalize_str(v7.get("remote_type")),
                    employment_type=_normalize_str(v7.get("employment_type")),
                    salary_min=salary_min,
                    salary_max=salary_max,
                    required_skills=v7.get("required_skills") or [],
                    seniority_level_new=_normalize_str(v7_new.get("seniority_level")),
                    company_type=_normalize_str(v5.get("company_type")),
                    industries=v5.get("industries") or [],
                    latitude=_safe_float(geo.get("lat") or geo.get("latitude")),
                    longitude=_safe_float(geo.get("lng") or geo.get("lon") or geo.get("longitude")),
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

def _build_matrix(vectors: list[list[float]]) -> np.ndarray:
    """Build a normalized embedding matrix from a list of vectors."""
    mat = np.array(vectors, dtype=np.float32)
    # Normalize rows to unit length (for dot product = cosine similarity)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid division by zero for missing embeddings
    mat /= norms
    return mat


def _safe_int(val) -> int | None:
    if val is None:
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


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
