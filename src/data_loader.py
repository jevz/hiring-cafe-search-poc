"""
Load and index jobs from JSONL dataset.

Handles messy real-world data: missing fields, inconsistent formats,
HTML descriptions. Stores embeddings as normalized NumPy matrices
for fast dot-product similarity.
"""

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

class JobDataset:
    """Job dataset backed by pre-built index files.

    Loads job metadata from pickle and memory-maps embedding matrices
    from .npy files. Run `python build_index.py` first to create the index.
    """

    def __init__(self):
        self.jobs: list[Job] = []
        self.explicit_embeddings: np.ndarray | None = None  # (N, 1536)
        self.inferred_embeddings: np.ndarray | None = None
        self.company_embeddings: np.ndarray | None = None

        self._has_explicit: np.ndarray | None = None
        self._has_inferred: np.ndarray | None = None
        self._has_company: np.ndarray | None = None

    def __len__(self):
        return len(self.jobs)

    @staticmethod
    def load(data_dir: str | Path = "src/data") -> "JobDataset":
        """Load from pre-built index files (jobs.pkl + *.npy).

        Embedding matrices are memory-mapped so they don't consume RAM
        until actually accessed.
        """
        import pickle  # safe: only loading our own Job dataclass

        data_dir = Path(data_dir)
        pkl_path = data_dir / "jobs.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Index not found at {data_dir}. Run `python build_index.py` first."
            )

        dataset = JobDataset()

        with open(pkl_path, "rb") as f:
            dataset.jobs = pickle.load(f)
        print(f"  Loaded {len(dataset.jobs):,} jobs from index.", file=sys.stderr)

        # Memory-map embedding matrices (read-only, OS pages in on demand)
        dataset.explicit_embeddings = np.load(data_dir / "explicit.npy", mmap_mode="r")
        dataset.inferred_embeddings = np.load(data_dir / "inferred.npy", mmap_mode="r")
        dataset.company_embeddings = np.load(data_dir / "company.npy", mmap_mode="r")

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
