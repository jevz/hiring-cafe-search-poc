"""Pydantic request/response schemas for the search API."""

from pydantic import BaseModel


# ── Requests ───────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    session_id: str | None = None


class ClearSessionRequest(BaseModel):
    session_id: str


# ── Responses ──────────────────────────────────────────────────────────────

class JobResult(BaseModel):
    rank: int
    score: float
    id: str
    title: str | None
    company_name: str | None
    location: str | None
    remote_type: str | None
    seniority_level: str | None
    employment_type: str | None
    salary_display: str | None
    salary_min: float | None
    salary_max: float | None
    required_skills: list[str]
    industries: list[str]
    apply_url: str | None
    company_type: str | None


class ParsedIntentResponse(BaseModel):
    semantic_query: str
    filters: dict
    weights: dict
    exclusions: list[str]


class SearchMetaResponse(BaseModel):
    total_jobs: int
    matched_filters: int
    search_time_ms: float
    intent_time_ms: float
    embed_time_ms: float


class SearchResponse(BaseModel):
    results: list[JobResult]
    meta: SearchMetaResponse
    intent: ParsedIntentResponse
    conversation_history: list[str]
    session_id: str
