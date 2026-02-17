"""
Search engine: hybrid retrieval combining weighted cosine similarity
across 3 embedding spaces with BM25 lexical search, fused via
Reciprocal Rank Fusion (RRF).
"""

import time
from dataclasses import dataclass, field

import numpy as np

from .data_loader import Job, JobDataset, tokenize


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
        bm25_weight: float = 0.4,
    ) -> tuple[list[SearchResult], SearchMeta]:
        t0 = time.perf_counter()

        if weights is None:
            weights = EmbeddingWeights()

        ds = self.dataset
        n = len(ds)

        # ── Semantic scores ────────────────────────────────────────────
        semantic_scores = (
            weights.explicit * (ds.explicit_embeddings @ query_embedding)
            + weights.inferred * (ds.inferred_embeddings @ query_embedding)
            + weights.company * (ds.company_embeddings @ query_embedding)
        )

        # Penalize exclusions
        if exclusion_embeddings:
            for exc_vec in exclusion_embeddings:
                exc_scores = ds.explicit_embeddings @ exc_vec
                semantic_scores -= 0.3 * exc_scores

        # ── BM25 scores ───────────────────────────────────────────────
        bm25_scores = np.zeros(n)
        if semantic_query and ds.bm25 is not None and bm25_weight > 0:
            query_tokens = tokenize(semantic_query)
            bm25_scores = ds.bm25.get_scores(query_tokens)

        # ── Apply structured filters + salary/skill boosts ────────────
        mask = np.ones(n, dtype=bool)
        has_salary_filter = filters is not None and filters.min_salary is not None
        query_term_set = set(semantic_query.lower().split()) if semantic_query else set()

        for i, job in enumerate(ds.jobs):
            if filters and not _job_passes_filters(job, filters):
                mask[i] = False
                continue

            if has_salary_filter and job.salary_min is not None and job.salary_min >= filters.min_salary:
                semantic_scores[i] += 0.05

            if query_term_set and job.required_skills:
                matches = sum(1 for s in job.required_skills if s.lower() in query_term_set)
                if matches:
                    semantic_scores[i] += 0.02 * min(matches, 3)

        # Zero out filtered jobs in both score arrays
        semantic_scores = np.where(mask, semantic_scores, -np.inf)
        bm25_scores = np.where(mask, bm25_scores, -np.inf)

        # ── Reciprocal Rank Fusion (RRF) ──────────────────────────────
        # Convert raw scores to ranks, then fuse: score = w_sem/(k+rank_sem) + w_bm25/(k+rank_bm25)
        RRF_K = 60  # standard constant from the RRF paper

        semantic_ranks = np.empty(n, dtype=np.float64)
        semantic_ranks[np.argsort(-semantic_scores)] = np.arange(1, n + 1)

        bm25_ranks = np.empty(n, dtype=np.float64)
        bm25_ranks[np.argsort(-bm25_scores)] = np.arange(1, n + 1)

        sem_weight = 1.0 - bm25_weight
        fused_scores = (
            sem_weight / (RRF_K + semantic_ranks)
            + bm25_weight / (RRF_K + bm25_ranks)
        )

        # Filtered-out jobs stay at bottom
        fused_scores = np.where(mask, fused_scores, -np.inf)

        # ── Top-k retrieval ───────────────────────────────────────────
        if top_k >= len(fused_scores):
            top_indices = np.argsort(fused_scores)[::-1]
        else:
            top_indices = np.argpartition(fused_scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(fused_scores[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices[:top_k], start=1):
            if fused_scores[idx] == -np.inf:
                break
            results.append(SearchResult(
                job=ds.jobs[idx],
                score=float(fused_scores[idx]),
                rank=rank,
            ))

        meta = SearchMeta(
            total_jobs=n,
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
