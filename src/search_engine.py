"""
Search engine: weighted cosine similarity across 3 embedding spaces
with structured post-filtering.
"""

from dataclasses import dataclass, field

import numpy as np

from .data_loader import Job, JobDataset


@dataclass
class SearchResult:
    job: Job
    score: float
    rank: int


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
    ) -> list[SearchResult]:
        """Search for jobs matching the query embedding.

        Args:
            query_embedding: Normalized 1536-dim query vector.
            filters: Structured filters to apply post-similarity.
            weights: Per-query embedding weights (default 0.5/0.3/0.2).
            top_k: Number of results to return.
            exclusion_embeddings: Vectors to penalize (for negation handling).
        """
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

        # Apply structured filters (build a boolean mask)
        mask = np.ones(len(ds), dtype=bool)

        if filters:
            for i, job in enumerate(ds.jobs):
                if not mask[i]:
                    continue
                if not _job_passes_filters(job, filters):
                    mask[i] = False

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

        return results


def _job_passes_filters(job: Job, f: SearchFilters) -> bool:
    """Check if a job passes all active filters. Null-safe: missing fields never exclude."""
    if f.remote_type and job.remote_type is not None:
        if job.remote_type != f.remote_type:
            return False

    if f.seniority_level and job.seniority_level is not None:
        if job.seniority_level != f.seniority_level:
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


def format_results(results: list[SearchResult]) -> str:
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
    return "\n".join(lines)
