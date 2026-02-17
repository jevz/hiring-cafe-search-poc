#!/usr/bin/env python3
"""
FastAPI web server for the HiringCafe job search engine.

Usage:
    python server.py              # start on port 8000
    python server.py --port 3000  # custom port
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.data_loader import JobDataset
from src.embeddings import EmbeddingClient
from src.intent_parser import parse_intent
from src.search_engine import SearchEngine, format_results
from src.token_tracker import tracker

# ── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(title="HiringCafe Search", version="1.0.0")

# Global state — loaded on startup
dataset: JobDataset | None = None
engine: SearchEngine | None = None
embed_client: EmbeddingClient | None = None


@app.on_event("startup")
async def load_data():
    global dataset, engine, embed_client
    print("Loading dataset...", file=sys.stderr)
    dataset = JobDataset.load()
    engine = SearchEngine(dataset)
    embed_client = EmbeddingClient()
    print(f"Ready — {len(dataset):,} jobs indexed.", file=sys.stderr)


# ── API models ──────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    history: list[str] = []


class JobResult(BaseModel):
    rank: int
    score: float
    title: str | None
    company: str | None
    location: str | None
    remote_type: str | None
    seniority: str | None
    employment_type: str | None
    salary_display: str | None
    skills: list[str]
    apply_url: str | None


class IntentResponse(BaseModel):
    semantic_query: str
    filters: dict
    weights: dict
    exclusions: list[str]


class SearchMeta(BaseModel):
    total_jobs: int
    matched_filters: int
    search_time_ms: float


class SearchResponse(BaseModel):
    results: list[JobResult]
    meta: SearchMeta
    intent: IntentResponse


# ── API endpoints ───────────────────────────────────────────────────────────

@app.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    query = req.query.strip()
    history = req.history + [query] if req.history else [query]

    # Parse intent
    intent = parse_intent(query, conversation_history=history if len(history) > 1 else None)

    # Embed
    query_embedding = embed_client.embed(intent.semantic_query)

    # Embed exclusions
    exclusion_embeddings = None
    if intent.exclusions:
        exclusion_embeddings = [embed_client.embed(term) for term in intent.exclusions]

    # Search
    results, meta = engine.search(
        query_embedding=query_embedding,
        filters=intent.filters,
        weights=intent.weights,
        top_k=10,
        exclusion_embeddings=exclusion_embeddings,
        semantic_query=intent.semantic_query,
    )

    # Build response
    job_results = []
    for r in results:
        j = r.job
        job_results.append(JobResult(
            rank=r.rank,
            score=round(r.score, 4),
            title=j.title,
            company=j.company_name,
            location=j.location,
            remote_type=j.remote_type,
            seniority=j.seniority_level,
            employment_type=j.employment_type,
            salary_display=j.salary_display,
            skills=j.required_skills[:8] if j.required_skills else [],
            apply_url=j.apply_url,
        ))

    # Build intent response
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
    if f.industries:
        filters_dict["industries"] = f.industries

    w = intent.weights
    weights_dict = {
        "explicit": round(w.explicit, 2),
        "inferred": round(w.inferred, 2),
        "company": round(w.company, 2),
    }

    return SearchResponse(
        results=job_results,
        meta=SearchMeta(
            total_jobs=meta.total_jobs,
            matched_filters=meta.matched_filters,
            search_time_ms=round(meta.search_time_ms, 1),
        ),
        intent=IntentResponse(
            semantic_query=intent.semantic_query,
            filters=filters_dict,
            weights=weights_dict,
            exclusions=intent.exclusions,
        ),
    )


@app.post("/api/clear")
async def clear():
    return {"status": "ok"}


# ── Static files & SPA fallback ─────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run("server:app", host=args.host, port=args.port, reload=False)
