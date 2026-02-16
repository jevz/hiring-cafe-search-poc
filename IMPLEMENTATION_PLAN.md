# Implementation Plan

## Overview

Build a semantic job search engine with natural language search and multi-turn refinement, runnable via `python demo.py`. The plan is organized into phases that build on each other — each phase produces something runnable.

---

## Phase 0: Project Setup
**Goal:** Skeleton that runs, dependencies installed, token tracking from day one.

### Task 0.1 — Project scaffolding
- Create directory structure:
  ```
  hiringcafe/
  ├── demo.py
  ├── requirements.txt
  ├── README.md
  ├── TOKENS.md
  ├── data/            # jobs.jsonl goes here (gitignored)
  └── src/
      ├── __init__.py
      ├── data_loader.py
      ├── embeddings.py
      ├── search_engine.py
      ├── session.py
      └── token_tracker.py
  ```
- `requirements.txt`: numpy, openai, google-generativeai (for Gemini free tier), python-dotenv
- `.env.example` with `OPENAI_API_KEY`, `GEMINI_API_KEY`
- `.gitignore`: data/, .env, __pycache__, *.pyc

### Task 0.2 — Token tracker (`src/token_tracker.py`)
- Singleton that logs every LLM API call (model, tokens in/out, cost, timestamp)
- Writes cumulative report to TOKENS.md on demand
- Tracks both development usage and per-query runtime usage separately
- Start this FIRST so every subsequent API call is tracked

---

## Phase 1: Data Loading & Profiling
**Goal:** Load jobs.jsonl, understand what we're working with, document data quality.

### Task 1.1 — Data loader (`src/data_loader.py`)
- Stream-parse JSONL (don't load entire 2GB file into a string)
- Extract per job:
  - **Metadata**: id, apply_url, title, company_name, location, description (strip HTML)
  - **Structured fields from `v7_processed_job_data`**: seniority_level, remote_type, salary_min, salary_max, required_skills, employment_type
  - **Structured fields from `v7_processed_job_data_new`**: normalized seniority (prefer this over v7 when available)
  - **Company data from `v5_processed_company_data`**: company_type, industries, funding, employees
  - **Geo**: coordinates (if present)
  - **Embeddings**: 3 × 1536-dim vectors
- Store embeddings as 3 NumPy matrices (100k × 1536), normalize to unit length
- Store metadata as list of dicts (or dataclass) for filtering and display
- Handle missing fields gracefully: default to None, don't crash

### Task 1.2 — Data profiling script
- Run once over the dataset, output a summary:
  - Total jobs loaded
  - Null rates per field (salary_min, seniority_level, remote_type, required_skills, etc.)
  - Distribution of key fields (top 10 seniority levels, remote types, company types, industries)
  - Embedding coverage: how many jobs have all 3 embeddings? Any with 0?
  - Duplicate detection: how many duplicate apply_urls or title+company combos?
- Save results to `DATA_PROFILE.md` — this feeds into the README

---

## Phase 2: Core Search (Single-Turn)
**Goal:** `query → ranked results` working end-to-end with no refinement.

### Task 2.1 — Embedding client (`src/embeddings.py`)
- Wrap OpenAI `text-embedding-3-small` API
- Input: string → Output: 1536-dim numpy array (normalized)
- Integrate with token tracker (log every call)
- Simple retry with backoff on API errors
- Cache embeddings for repeated queries (in-memory dict)

### Task 2.2 — Search engine v1 (`src/search_engine.py`)
- Input: query embedding (1536-dim), optional filters dict
- Compute cosine similarity (dot product on normalized vectors) against all 3 embedding matrices
- **Query-dependent weighting** (key improvement over spec):
  - Default: 0.50 explicit / 0.30 inferred / 0.20 company
  - Provide an interface to accept custom weights per query
  - Phase 3 will supply dynamic weights; for now use defaults
- Combine weighted scores into final score per job
- Apply structured filters (post-similarity):
  - `remote_type`: exact match
  - `seniority_level`: exact match (check both v7 and v7_new)
  - `employment_type`: exact match
  - `company_type`: exact match
  - `min_salary`: job's salary_max >= user's min_salary (null salary = don't exclude)
  - `industries`: intersection match (job has at least one matching industry)
- Return top-k results (default 10) with scores

### Task 2.3 — Result formatting
- For each result, display:
  - Rank + similarity score
  - Job title, company name, location
  - Remote type, seniority level
  - Salary range (if available)
  - Required skills (truncated)
  - Apply URL
- Strip HTML from any displayed text
- Clean, readable CLI output

### Task 2.4 — Basic demo.py (single-turn only)
- Load data on startup (show progress: "Loading 100k jobs...")
- Accept a query string
- Embed it, search, display results
- Verify it works end-to-end before moving on

---

## Phase 3: Query Understanding (Hybrid Strategy)
**Goal:** Use an LLM to parse user intent into structured search parameters.

### Task 3.1 — Intent parser (`src/session.py` — parse function)
- Send the user's query (or conversation so far) to Gemini (free tier)
- Prompt the LLM to extract:
  ```json
  {
    "semantic_query": "cleaned/synthesized search query for embedding",
    "filters": {
      "remote_type": "remote" | "hybrid" | "onsite" | null,
      "seniority_level": "entry" | "junior" | "mid" | "senior" | "lead" | null,
      "employment_type": "full-time" | "part-time" | "contract" | null,
      "company_type": "startup" | "non-profit" | "enterprise" | null,
      "min_salary": int | null,
      "industries": ["list"] | null
    },
    "embedding_weights": {
      "explicit": 0.0-1.0,
      "inferred": 0.0-1.0,
      "company": 0.0-1.0
    },
    "exclusions": ["terms to penalize"]
  }
  ```
- The `embedding_weights` field is the key improvement: the LLM classifies whether the query is role-focused, company-focused, or mixed, and assigns weights accordingly
- The `exclusions` field handles negations ("not management" → penalize management-related results)
- Fallback: if Gemini fails, fall back to simple strategy (concatenation + regex)

### Task 3.2 — Simple strategy fallback
- Concatenate conversation queries with " | " separator
- Regex filter extraction with basic disambiguation:
  - Only match "remote" when it's a standalone word, not part of "remote sensing"
  - Only match "senior" when followed by role-like words or standalone
- This is the fallback, not the primary path

### Task 3.3 — Exclusion handling in search
- When the intent parser returns exclusions, embed the exclusion terms
- Penalize jobs that are similar to exclusion embeddings
- `final_score = weighted_similarity - 0.3 * max(exclusion_similarities)`

---

## Phase 4: Multi-Turn Refinement
**Goal:** Conversation that accumulates context across turns.

### Task 4.1 — Session manager (`src/session.py`)
- Maintain conversation history: list of (query, parsed_intent, result_ids)
- On each new turn:
  1. Append new query to history
  2. Send FULL conversation history to the LLM intent parser
  3. LLM synthesizes a combined intent (not just the latest turn)
  4. The `semantic_query` from the LLM is a SYNTHESIZED query incorporating all turns
     - e.g., turns ["data science", "non-profit", "remote"] → `"remote data science roles at mission-driven non-profits"`
  5. Filters accumulate: new filters override previous ones for the same field, add for new fields
  6. Run search with synthesized query + accumulated filters + dynamic weights
- This solves the concatenation problem: the LLM produces what a human would type if they knew their full intent upfront

### Task 4.2 — Session commands
- `new` or `reset`: clear session, start fresh
- `filters`: show currently accumulated filters
- `history`: show conversation turns
- `quit` / `exit`: end session

---

## Phase 5: Demo & Polish
**Goal:** Runnable demo showing 5+ queries, evaluation, deliverables.

### Task 5.1 — Define test queries (evaluation set)
Design 5+ queries that exercise different capabilities:

| # | Query | Tests | Expected Top Results |
|---|-------|-------|---------------------|
| 1 | "data scientist" | Basic role search | Data Scientist, ML Engineer roles |
| 2 | "python backend engineer at a startup" | Role + company filter | Backend/Python roles at startups |
| 3 | "remote machine learning jobs paying over 150k" | Role + filter combo | Remote ML roles with salary > 150k |
| 4 | Multi-turn: "frontend engineer" → "at a fintech" → "make it remote" | Refinement | Remote frontend roles at fintech companies |
| 5 | "non-profit healthcare" | Company-focused query | Roles at healthcare non-profits |
| 6 | "senior but not management" | Negation handling | Senior IC roles, not managers |
| 7 | "entry level software engineer" | Seniority + role | Junior/entry SWE roles |

Run each, inspect results, note quality. This becomes README content.

### Task 5.2 — Demo script (`demo.py`)
- Two modes:
  1. **Scripted demo**: runs the predefined test queries automatically, shows results (for evaluation and video)
  2. **Interactive REPL**: user types queries, refines, explores (for hands-on use)
- Both modes print token usage at the end
- Command: `python demo.py` (interactive) or `python demo.py --scripted` (automated)

### Task 5.3 — README.md
Write the README covering:
- How to set up and run (`pip install -r requirements.txt`, set API keys, `python demo.py`)
- Approach: three-embedding weighted search, LLM-powered intent parsing, multi-turn synthesis
- Key design decisions and trade-offs:
  - Why query-dependent weights instead of static
  - Why LLM synthesis instead of concatenation for multi-turn
  - Why Gemini free tier for parsing (cost)
  - In-memory NumPy vs vector DB
- Data quality observations (from profiling)
- What queries work well, what doesn't (from evaluation)
- What we'd improve with more time

### Task 5.4 — TOKENS.md
- Auto-generated from token tracker
- Development token usage (all API calls during building/testing)
- Per-query runtime cost breakdown
- Total project cost

---

## Phase 6 (Stretch): Enhancements
**Only if time permits.** Not required for submission.

### Task 6.1 — Geo-based filtering
- Parse location queries ("jobs in New York", "Bay Area")
- Use geo coordinates from dataset for radius-based filtering

### Task 6.2 — Deduplication
- Deduplicate by apply_url or (title + company_name) hash
- Show count of duplicates removed in data profile

### Task 6.3 — Result re-ranking with LLM
- Take top 20 results from vector search
- Send titles + brief descriptions to LLM
- Ask LLM to re-rank based on query intent
- Return re-ranked top 10

---

## Dependency Graph

```
Phase 0 (setup + token tracker)
    │
    ▼
Phase 1 (data loading + profiling)
    │
    ▼
Phase 2 (core search, single-turn)  ◄── This is our first runnable checkpoint
    │
    ▼
Phase 3 (LLM intent parsing + dynamic weights)
    │
    ▼
Phase 4 (multi-turn sessions)  ◄── This is our second runnable checkpoint
    │
    ▼
Phase 5 (demo, evaluation, deliverables)  ◄── Submission-ready
    │
    ▼
Phase 6 (stretch goals)
```

## Key Decisions Made

1. **Gemini for intent parsing** — free tier means zero cost for the hybrid strategy. Fallback to simple strategy if API fails.
2. **LLM-synthesized queries for multi-turn** — instead of concatenating strings, the LLM produces what a human would type knowing their full intent.
3. **LLM-assigned embedding weights** — the intent parser classifies query type and assigns explicit/inferred/company weights per query.
4. **Null-safe filtering** — missing salary, seniority, etc. never exclude a job from results. Filters only apply to jobs that have the relevant field.
5. **Profile data before building** — understand what we're working with, document it, design around the reality of the data.
