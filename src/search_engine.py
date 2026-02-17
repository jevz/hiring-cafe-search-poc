"""
Search engine: weighted cosine similarity across 3 embedding spaces
with structured post-filtering.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from .data_loader import Job, JobDataset


@dataclass
class SearchResult:
    job: Job
    score: float
    rank: int


@dataclass
class SearchMeta:
    total_jobs: int
    matched_filters: int
    search_time_ms: float


@dataclass
class SearchFilters:
    remote_type: str | None = None
    seniority_level: str | None = None
    employment_type: str | None = None
    company_type: str | None = None
    min_salary: float | None = None
    industries: list[str] = field(default_factory=list)


@dataclass
class EmbeddingWeights:
    explicit: float = 0.50
    inferred: float = 0.30
    company: float = 0.20


class SearchEngine:
    def __init__(self, dataset: JobDataset):
        self.dataset = dataset

    def search(
        self,
        query_embedding: np.ndarray,
        filters: SearchFilters | None = None,
        weights: EmbeddingWeights | None = None,
        top_k: int = 10,
        exclusion_embeddings: list[np.ndarray] | None = None,
        semantic_query: str | None = None,
    ) -> tuple[list[SearchResult], SearchMeta]:
        t0 = time.perf_counter()

        if weights is None:
            weights = EmbeddingWeights()

        ds = self.dataset

        # Weighted cosine similarity (dot product on normalized vectors)
        scores = (
            weights.explicit * (ds.explicit_embeddings @ query_embedding)
            + weights.inferred * (ds.inferred_embeddings @ query_embedding)
            + weights.company * (ds.company_embeddings @ query_embedding)
        )

        # Penalize exclusions
        if exclusion_embeddings:
            for exc_vec in exclusion_embeddings:
                exc_scores = ds.explicit_embeddings @ exc_vec
                scores -= 0.3 * exc_scores

        # Apply structured filters, salary boost, and skills boost in one pass
        mask = np.ones(len(ds), dtype=bool)
        has_salary_filter = filters is not None and filters.min_salary is not None
        query_term_set = set(semantic_query.lower().split()) if semantic_query else set()

        for i, job in enumerate(ds.jobs):
            if filters and not _job_passes_filters(job, filters):
                mask[i] = False
                continue

            # Boost jobs with confirmed salary meeting threshold
            if has_salary_filter and job.salary_min is not None and job.salary_min >= filters.min_salary:
                scores[i] += 0.05

            # Boost jobs with exact skill matches (capped at 3 to avoid overwhelming similarity)
            if query_term_set and job.required_skills:
                matches = sum(1 for s in job.required_skills if s.lower() in query_term_set)
                if matches:
                    scores[i] += 0.02 * min(matches, 3)

        # Zero out filtered jobs
        scores = np.where(mask, scores, -np.inf)

        # Get top-k indices
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            # partial sort is faster than full sort for large N
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices[:top_k], start=1):
            if scores[idx] == -np.inf:
                break
            results.append(SearchResult(
                job=ds.jobs[idx],
                score=float(scores[idx]),
                rank=rank,
            ))

        meta = SearchMeta(
            total_jobs=len(ds),
            matched_filters=int(mask.sum()),
            search_time_ms=(time.perf_counter() - t0) * 1000,
        )

        return results, meta


_SENIORITY_ALIASES = {
    "entry level": {"entry level", "entry", "junior", "entry-level"},
    "mid level": {"mid level", "mid", "intermediate", "mid-level"},
    "senior level": {"senior level", "senior", "sr", "senior-level"},
    "manager": {"manager", "management"},
    "director": {"director"},
    "internship": {"internship", "intern"},
}


def _seniority_matches(job_seniority: str, filter_seniority: str) -> bool:
    """Fuzzy seniority matching — 'senior' matches 'senior level', etc."""
    if job_seniority == filter_seniority:
        return True
    aliases = _SENIORITY_ALIASES.get(filter_seniority, set())
    return job_seniority in aliases


def _job_passes_filters(job: Job, f: SearchFilters) -> bool:
    """Check if a job passes all active filters. Null-safe: missing fields never exclude."""
    if f.remote_type and job.remote_type is not None:
        if job.remote_type != f.remote_type:
            return False

    if f.seniority_level and job.seniority_level is not None:
        if not _seniority_matches(job.seniority_level, f.seniority_level):
            return False

    if f.employment_type and job.employment_type is not None:
        if job.employment_type != f.employment_type:
            return False

    if f.company_type and job.company_type is not None:
        if job.company_type != f.company_type:
            return False

    if f.min_salary is not None:
        # Only exclude if job has salary data AND it's below threshold
        if job.salary_max is not None and job.salary_max < f.min_salary:
            return False

    if f.industries:
        # Only exclude if job has industry data AND no overlap
        if job.industries and not set(job.industries) & set(f.industries):
            return False

    return True


def format_results(results: list[SearchResult], meta: SearchMeta | None = None) -> str:
    """Format search results for CLI display."""
    if not results:
        return "No results found."

    lines = []
    for r in results:
        j = r.job
        lines.append(f"{'─' * 60}")
        lines.append(f"  #{r.rank}  {j.title or 'Untitled'}")
        lines.append(f"       {j.company_name or 'Unknown Company'} — {j.location or 'Location N/A'}")

        details = []
        if j.remote_type:
            details.append(j.remote_type)
        if j.seniority_level:
            details.append(j.seniority_level)
        if j.employment_type:
            details.append(j.employment_type)
        if details:
            lines.append(f"       {' · '.join(details)}")

        if j.salary_display:
            lines.append(f"       Salary: {j.salary_display}")

        if j.required_skills:
            skills = ", ".join(j.required_skills[:8])
            if len(j.required_skills) > 8:
                skills += f" (+{len(j.required_skills) - 8} more)"
            lines.append(f"       Skills: {skills}")

        if j.apply_url:
            lines.append(f"       Apply: {j.apply_url}")

        lines.append(f"       Score: {r.score:.4f}")

    lines.append(f"{'─' * 60}")
    if meta:
        filter_info = ""
        if meta.matched_filters < meta.total_jobs:
            filter_info = f" ({meta.matched_filters:,} passed filters)"
        lines.append(f"  Showing {len(results)} of {meta.total_jobs:,} jobs{filter_info} in {meta.search_time_ms:.0f}ms")
    return "\n".join(lines)
