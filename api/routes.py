"""API routes wrapping the existing search engine."""

import logging
import time
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

from src.intent_parser import parse_intent
from src.token_tracker import tracker

from .models import (
    ClearSessionRequest,
    JobResult,
    ParsedIntentResponse,
    SearchMetaResponse,
    SearchRequest,
    SearchResponse,
)
from .session_store import store

router = APIRouter(prefix="/api")

# Simple per-IP rate limiter: max 30 requests per minute
_RATE_LIMIT = 30
_RATE_WINDOW = 60.0
_request_log: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> None:
    now = time.time()
    timestamps = _request_log[client_ip]
    # Prune old entries
    _request_log[client_ip] = [t for t in timestamps if now - t < _RATE_WINDOW]
    if len(_request_log[client_ip]) >= _RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    _request_log[client_ip].append(now)


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, request: Request):
    _check_rate_limit(request.client.host)

    engine = request.app.state.engine
    client = request.app.state.client

    # Session management
    session = store.get_or_create(req.session_id)
    session.history.append(req.query)

    # 1. Parse intent
    t0 = time.perf_counter()
    intent = parse_intent(
        req.query,
        conversation_history=session.history if len(session.history) > 1 else None,
    )
    t_intent = time.perf_counter()

    # 2. Embed query
    query_embedding = client.embed(intent.semantic_query)
    exclusion_embeddings = (
        [client.embed(t) for t in intent.exclusions] if intent.exclusions else None
    )
    t_embed = time.perf_counter()

    # 3. Search
    results, meta = engine.search(
        query_embedding=query_embedding,
        filters=intent.filters,
        weights=intent.weights,
        top_k=10,
        exclusion_embeddings=exclusion_embeddings,
        semantic_query=intent.semantic_query,
    )

    # Build response
    job_results = [
        JobResult(
            rank=r.rank,
            score=round(r.score, 4),
            id=r.job.id,
            title=r.job.title,
            company_name=r.job.company_name,
            location=r.job.location,
            remote_type=r.job.remote_type,
            seniority_level=r.job.seniority_level,
            employment_type=r.job.employment_type,
            salary_display=r.job.salary_display,
            salary_min=r.job.salary_min,
            salary_max=r.job.salary_max,
            required_skills=r.job.required_skills,
            industries=r.job.industries,
            apply_url=r.job.apply_url,
            company_type=r.job.company_type,
        )
        for r in results
    ]

    # Token cost from tracker
    summary = tracker.summary()

    f = intent.filters
    filters_dict = {}
    if f.remote_type:
        filters_dict["remote_type"] = f.remote_type
    if f.seniority_level:
        filters_dict["seniority_level"] = f.seniority_level
    if f.employment_type:
        filters_dict["employment_type"] = f.employment_type
    if f.company_type:
        filters_dict["company_type"] = f.company_type
    if f.min_salary is not None:
        filters_dict["min_salary"] = f.min_salary
    if f.max_salary is not None:
        filters_dict["max_salary"] = f.max_salary
    if f.industries:
        filters_dict["industries"] = f.industries

    w = intent.weights

    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "query=%r session=%s results=%d total=%.0fms (intent=%.0f embed=%.0f search=%.0f)",
        req.query, session.id[:8], len(job_results), total_ms,
        (t_intent - t0) * 1000, (t_embed - t_intent) * 1000, meta.search_time_ms,
    )

    return SearchResponse(
        results=job_results,
        meta=SearchMetaResponse(
            total_jobs=meta.total_jobs,
            matched_filters=meta.matched_filters,
            search_time_ms=round(meta.search_time_ms, 1),
            intent_time_ms=round((t_intent - t0) * 1000, 1),
            embed_time_ms=round((t_embed - t_intent) * 1000, 1),
        ),
        intent=ParsedIntentResponse(
            semantic_query=intent.semantic_query,
            filters=filters_dict,
            weights={"explicit": round(w.explicit, 2), "inferred": round(w.inferred, 2), "company": round(w.company, 2)},
            exclusions=intent.exclusions,
        ),
        conversation_history=list(session.history),
        session_id=session.id,
    )


@router.post("/session/clear")
def clear_session(req: ClearSessionRequest):
    store.clear(req.session_id)
    return {"status": "cleared"}


@router.get("/health")
def health(request: Request):
    engine = request.app.state.engine
    return {
        "status": "ready",
        "jobs_loaded": len(engine.dataset),
    }
