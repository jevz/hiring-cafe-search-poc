# Spec Critique: AI Job Search Engine

## What's Strong

- **Problem framing is clear.** Multi-turn refinement (search → refine → refine) maps to real user behavior.
- **Three-embedding approach is clever.** Splitting into explicit/inferred/company lets you weight query aspects differently.
- **Cost analysis is realistic.** ~$0.000001/query means $10 budget is effectively unlimited.

---

## Architectural Concerns

### 1. Static weights (50/30/20) are a major weakness

The fixed weighting `0.5 * explicit + 0.3 * inferred + 0.2 * company` ignores query type. "Senior data scientist" should weight explicit heavily; "mission-driven non-profit" should weight company heavily. The hybrid strategy could output per-query weights for minimal additional cost.

### 2. In-memory search is fine for POC but spec should say so

~1.8GB RAM and brute-force dot product over 100k vectors works. But explicitly state this is a POC constraint and mention production alternatives (FAISS, pgvector, Pinecone).

### 3. Simple Strategy concatenation is semantically fragile

`"data science" + "non-profit" + "remote"` → `"data science non-profit remote"` — the embedding of a concatenated string is NOT the intersection of semantic spaces. Order, separators, and length all affect results non-obviously. The spec says it "works surprisingly well" with no evidence.

**Alternative:** Embed each turn separately and average the vectors, or compute separate similarity scores and combine them.

### 4. Hybrid strategy has unclear division of labor

The LLM extracts structured filters AND a semantic query, which then gets embedded. If the LLM is parsing intent, why not have it generate a better search query directly? The spec doesn't explain why both an LLM parse step and an embedding step are needed.

### 5. Regex filter extraction will produce false positives

"Remote sensing engineer" → incorrectly triggers remote filter. "Senior" in company names. No disambiguation strategy mentioned.

---

## Data Model Issues

### 6. No deduplication or freshness handling

100k jobs will contain duplicates (same role on multiple boards) and expired listings. No strategy mentioned.

### 7. HTML in descriptions

`"description": "<html>..."` — no mention of HTML stripping for display or processing.

### 8. Salary data is likely sparse

Most jobs don't list salary. Filtering on `min_salary` will dramatically shrink the candidate pool unpredictably. Need a strategy for null salary handling.

---

## Critical Gaps

### 9. No evaluation methodology (biggest gap)

No test queries with expected results, no relevance scoring, no A/B comparison between strategies, no quantitative success criteria. Without evaluation, you can't justify any design decision.

### 10. No error handling / degradation strategy

What happens when OpenAI API is down or rate-limited? No fallback defined.

### 11. Embedding source text is unspecified

What text was embedded to produce the three vector types? Raw title? Concatenated fields? LLM summary? This matters because query embeddings must be in the same semantic space as the document embeddings.

### 12. No interface specification

Entry point is `demo.py` but no definition of whether it's a CLI, web API, or REPL. No input/output contract.

---

## Recommendations

1. **Add query-dependent weighting** — detect query type and shift embedding weights accordingly.
2. **Define 10-20 test queries with expected results** — build evaluation first.
3. **Clarify embedding source text** — document what text was embedded for each vector type.
4. **Fix concatenation problem** — embed turns separately and average vectors instead of concatenating strings.
5. **Specify the interface** — define a FastAPI endpoint or CLI REPL upfront.
6. **Add deduplication** — at minimum by job URL or title+company.
7. **Handle sparse filters gracefully** — when salary is null, exclude from salary filtering rather than excluding the job.
