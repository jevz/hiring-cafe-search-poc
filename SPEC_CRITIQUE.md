# Spec Critique: AI Job Search Engine

## Context

This critiques our proposed technical spec against the original HiringCafe task brief. Several of my initial concerns are addressed by the brief itself; others remain valid and some new ones emerge.

---

## What the Spec Gets Right

- **Problem framing matches the brief exactly.** Two capabilities (search + refine), multi-turn flow, natural language input.
- **Three-embedding approach leverages the provided data well.** The brief explicitly provides three embedding types with clear purposes — the spec correctly uses all three.
- **Cost analysis is realistic and demonstrates budget awareness.** The brief doesn't set a budget, but being cost-conscious is pragmatic.

---

## Concerns Resolved by the Original Brief

### Embedding source text (previously "unspecified")

The brief actually documents this clearly:

| Embedding | Generated From |
|-----------|---------------|
| `embedding_explicit_vector` | Job title, listed skills, required majors, explicit keywords |
| `embedding_inferred_vector` | Related titles, inferred skills, likely relevant experience |
| `embedding_company_vector` | Company name, industry, org type, mission |

All use OpenAI `text-embedding-3-small` (1536 dimensions). **Our spec should include this detail** since it directly impacts how we construct query text. For example: if explicit embeddings were generated from structured field concatenation, our query phrasing should mirror that style.

### Interface specification

The brief says: "Runnable demo that shows 5+ queries with results. Include the refinement flow." and "Code should be runnable via a single command (e.g. `python demo.py`)." So a CLI REPL is the expected interface — our spec's `demo.py` is correct. But we should define the output format (what fields to display per result, how many results, etc.).

### "In-memory is fine for POC"

The brief is explicitly a POC/take-home. No need to caveat about production scaling. In-memory NumPy is the right call.

---

## Concerns That Remain Valid

### 1. Static embedding weights (50/30/20) — HIGH IMPACT

Still the biggest architectural weakness. The brief emphasizes "handling ambiguity" and "do the results feel right" as evaluation criteria. Static weights directly hurt both.

Consider the queries:
- `"python machine learning engineer"` → should be ~80% explicit, ~15% inferred, ~5% company
- `"mission-driven non-profit"` → should be ~10% explicit, ~10% inferred, ~80% company
- `"senior data science at a startup"` → mixed, roughly even

**Recommendation:** Have the hybrid LLM step output weights alongside filters. Even a simple heuristic (classify query as role-focused / company-focused / mixed) would help. This is low cost and high impact on result quality.

### 2. Query concatenation for multi-turn is semantically wrong — HIGH IMPACT

The simple strategy concatenates all turns into one string and embeds it. This fundamentally misrepresents the user's intent:

- Turn 1: `"data science jobs"` → embedded alone, this works great
- Turn 1+2: `"data science jobs non-profit social good"` → now the embedding drifts toward "non-profit data science" as a single concept
- Turn 1+2+3: `"data science jobs non-profit social good remote"` → even more drift, "remote" gets diluted by all the other terms

**Better approaches:**
1. Embed each turn separately, compute similarities independently, then combine scores (not embeddings)
2. Use the LLM (hybrid strategy) to synthesize a clean combined query: `"remote data scientist at a mission-driven non-profit"` — this is what a human would type if they knew what they wanted upfront
3. Weighted recency: give more weight to recent turns

### 3. Regex filter extraction will false-positive — MEDIUM IMPACT

The brief specifically mentions messy real-world data. "Remote sensing engineer" → triggers remote filter. "Senior Living Corp" → triggers senior filter. This matters more because the brief evaluates on "do the results feel right."

**Recommendation:** Even in the simple strategy, use basic NLP context (is "remote" modifying a noun like "sensing" or standing alone?). Or just use the hybrid strategy for filter extraction — it's free with Gemini.

### 4. No evaluation methodology — MEDIUM-HIGH IMPACT

The brief says: "Do the results feel right — run your queries, do the rankings make sense?" This is subjective evaluation, but we should still formalize it.

**Recommendation:** Define our 5+ demo queries upfront with what we expect to see in the top results. This serves as both our test suite and our README content. The brief requires a README that discusses "what queries work well, which don't" — we need evaluation to answer that.

### 5. Salary data sparsity

Still valid. No change.

### 6. HTML in descriptions

Still valid. Need to strip HTML for display in CLI output at minimum.

---

## New Concerns from the Original Brief

### 7. We're ignoring `v7_processed_job_data_new` and `geo`

The brief mentions two additional data fields our spec doesn't address:
- **`v7_processed_job_data_new`**: "Normalized structured job data (1-2 seniority levels)" — this is likely cleaner than `v7_processed_job_data` for seniority filtering
- **`geo`**: Geo coordinates for location-based search — enables "jobs near me" or "jobs in the Bay Area" queries

These aren't required, but using them would demonstrate thoroughness.

### 8. The brief emphasizes navigating messy data

> "A critical part of working at HiringCafe is navigating this messy real-world job data and learning how to make sense of it."

Our spec treats the data as clean and structured. We should:
- Profile the data first (how many jobs have null salary? null seniority? empty descriptions?)
- Document data quality issues in the README
- Handle missing fields gracefully rather than assuming they exist

### 9. Required deliverables our spec doesn't plan for

| Deliverable | Spec Coverage |
|-------------|---------------|
| Working code with 5+ queries | Covered (demo.py) |
| README with approach discussion | Not mentioned |
| Tokens report (development + per-query) | Mentioned but not structured |
| Demo video | Not mentioned |

The tokens report is interesting — they want to see both development cost AND per-query cost. Our spec tracks per-query cost but not development tokens. We should log all LLM API calls during development.

### 10. "Your solution must not be a wrapper for a 3rd party AI search solution"

Our spec is fine here — we're doing vector similarity from scratch. But worth noting: we should be careful that the hybrid strategy doesn't become "send everything to an LLM and return whatever it says." The core search must be our own vector math.

---

## Revised Recommendations (Priority Ordered)

1. **Profile the dataset first** — understand nulls, distributions, data quality before writing search logic
2. **Implement query-dependent embedding weights** — classify query type, adjust weights per search
3. **Fix multi-turn strategy** — use LLM synthesis or per-turn embedding with score combination
4. **Define 5+ test queries upfront** — these become both evaluation and demo content
5. **Use `v7_processed_job_data_new`** for cleaner structured filters where available
6. **Strip HTML from descriptions** for display
7. **Handle sparse data gracefully** — null-aware filtering, data quality notes in README
8. **Track all token usage** from the start (development + runtime)
9. **Plan README content** — approach, trade-offs, what works/doesn't, improvements
