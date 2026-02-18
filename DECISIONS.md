# Design Decisions & Trade-offs

Documenting the reasoning behind key technical choices made during development.

---

## Memory Management

### Memory-mapped NumPy vs loading into RAM
**Decision:** Pre-build `.npy` index files, memory-map them at search time.

**Context:** The full dataset has 100k jobs × 3 embeddings × 1536 dimensions. Loading everything into RAM via the original `JobDataset.load()` caused the system to crash (~1.7GB of Python lists + NumPy conversion overhead).

**Alternatives considered:**
- **Chunked batch loading** — still ends up with ~1.7GB in RAM at runtime
- **FAISS on-disk index** — fastest search but adds a dependency and complexity
- **Memory-mapped NumPy** (chosen) — OS pages in data on demand, near-zero RSS for embeddings

**Trade-off:** Requires a one-time `python build_index.py` step before searching. Worth it — search startup is near-instant and doesn't crash.

### Two-pass index building
**Decision:** Pass 1 counts unique jobs, Pass 2 writes to pre-allocated mmap files.

**Context:** The single-pass approach accumulated 3 lists of 100k × 1536 floats in Python before converting to NumPy, which still caused memory spikes.

**Trade-off:** Two passes over the JSONL file is slower (reads the file twice) but never holds more than one row of embeddings in memory at a time.

---

## Data Quality

### Deduplication strategy: (title, company_name)
**Decision:** Deduplicate on `(title.lower(), company_name.lower())`, keeping the first occurrence.

**Context:** The dataset contains 28,828 duplicates out of 100k jobs (29%). These are the same role reposted or scraped multiple times.

**Why first occurrence:** The JSONL file is sorted newest-first (verified by comparing `estimated_publish_date` across duplicate groups). First occurrence = most recent posting.

**Alternatives considered:**
- **Deduplicate on `apply_url`** — too strict, 0 duplicates found in profiling (URLs are unique per scrape)
- **Fuzzy title matching** — would catch near-dupes like "Software Engineer" vs "Software Engineer (Chicago, IL, US, 60601)" but adds complexity and risk of false positives
- **Explicitly compare dates** — safer but requires buffering duplicates in memory; unnecessary since file ordering already guarantees newest-first

**Result:** 71,172 unique jobs indexed after dedup.

### Salary fields as floats, not ints
**Decision:** Store `salary_min` and `salary_max` as `float | None`.

**Context:** Original implementation used `int` and a `_safe_int` helper that cast floats to ints, losing precision on values like `$49,358.40`. Salary data naturally comes as floats from the API.

**Trade-off:** None — floats are strictly better here. Removed the unnecessary `_safe_int` function.

### Null-safe filtering
**Decision:** Missing fields never exclude a job from results.

**Context:** Many fields have incomplete coverage (salary ~52%, location varies). If a job doesn't have salary data, a salary filter should not exclude it — absence of data is not evidence of absence.

**Trade-off:** May surface jobs that don't actually meet the filter criteria. But the alternative (excluding all jobs without salary when the user asks for ">$100k") would eliminate ~48% of the dataset.

### Salary boost for confirmed matches
**Decision:** When a salary filter is active, boost scores (+0.05) for jobs with confirmed `salary_min >= threshold`.

**Context:** With null-safe filtering, a job with no salary data and a job paying $230k are treated identically — both pass the filter. But when the user explicitly asks for ">$150k", jobs that confirm they meet the threshold should rank higher.

**Alternatives considered:**
- **Penalize missing salary** — more aggressive, would bury good jobs that simply don't list salary
- **Proportional boost** — scale bonus by how much salary exceeds threshold. Adds complexity, and salary data quality is too noisy for fine-grained scaling

**Trade-off:** The +0.05 bonus is small enough (~10% of a typical 0.4-0.5 similarity score) to nudge confirmed-salary jobs up without overriding strong semantic matches.

---

## Search Architecture

### Python filter loop vs vectorized NumPy filtering
**Decision:** Keep filtering as a Python loop over Job objects rather than vectorizing with NumPy arrays.

**Context:** The similarity computation is fully vectorized (`matrix @ query_vec` via BLAS), but the post-filter step loops through all 71k jobs in Python to check structured fields (remote type, seniority, salary, etc.).

**Vectorized alternative considered:** Encode all filterable fields as integer/float NumPy arrays at build time (e.g. `remote_type_codes`, `salary_min`, `seniority_codes`). Then filtering becomes pure NumPy boolean operations — no Python loop, sub-millisecond on 71k rows.

**Why we didn't do it:** The filter loop takes ~20-50ms on 71k jobs. The actual latency bottleneck is the Claude CLI call (~2-3s) and OpenAI embedding API (~500ms). Vectorizing filters would add complexity (integer encoding tables, extra build artifacts, mapping dictionaries) for a speedup that's invisible to the user. At 1M+ jobs this decision should be revisited — the Python loop would become the bottleneck.

### Three-embedding weighted search
**Decision:** Weighted cosine similarity across explicit (0.5), inferred (0.3), and company (0.2) embedding spaces.

**Context:** The dataset provides 3 pre-computed embeddings per job that encode different aspects:
- **Explicit**: job title, listed skills, certifications
- **Inferred**: related titles, implied skills
- **Company**: company name, industry, org type

**Why dynamic weights:** The LLM intent parser adjusts weights per query. "Python engineer" is role-focused (explicit=0.6), "jobs at Google" is company-focused (company=0.6), "machine learning pytorch" is skill-focused (inferred=0.5). Static weights would under-serve company and skill queries.

### Pre-normalized embeddings
**Decision:** Normalize all embedding vectors to unit length during index build time.

**Context:** Cosine similarity = dot product when both vectors are unit length. By normalizing once at build time, search only needs `matrix @ query_vec` — a single BLAS operation — instead of computing norms per query.

**Trade-off:** The pre-computed embeddings from HiringCafe are almost certainly already normalized (OpenAI returns normalized vectors by default), so we normalize defensively. Costs nothing at query time.

---

## Intent Parsing

### Claude CLI over Gemini API
**Decision:** Use Claude Code CLI (`claude -p`) with Haiku for intent parsing instead of Gemini.

**Context:** Originally planned to use Gemini free tier. The Gemini CLI had heavy overhead (OAuth, retries, rate limiting). The Python SDK hit persistent 429 quota errors on a new API key. Claude CLI works reliably with existing auth.

**Trade-off:** Claude Haiku is not free (unlike Gemini free tier), but the cost per query is tiny (~$0.0006 per parse) and doesn't count toward the $10 OpenAI budget. For the final submission, this should be swapped back to Gemini once API key issues are resolved, or kept as-is with cost documented.

### LLM with regex fallback
**Decision:** Try LLM first, fall back to regex if the CLI is unavailable or fails.

**Context:** The regex fallback handles common patterns (remote, salary, seniority) but can't do synonym expansion, ambiguity resolution, or dynamic weight assignment. The LLM is better but adds latency (~2-3s per query).

**Trade-off:** Users always get results even if the LLM is down. Quality degrades gracefully.

### Synonym expansion in LLM prompt
**Decision:** Instruct the LLM to expand ambiguous technology names (e.g. "Go" → "Go Golang").

**Context:** Short or ambiguous terms like "Go", "JS", "ML" have weak embedding signal. The word "Go" is semantically close to generic programming, but "Go Golang" strongly signals the specific language.

**Trade-off:** Makes the embedding query longer, which slightly dilutes other terms in the query. But the improved recall for technology-specific searches is worth it.

---

## Token Tracking

### Persistent JSON store
**Decision:** Append every API call to `token_usage.json` immediately, generate `TOKENS.md` from cumulative data.

**Context:** The original tracker was in-memory only — totals reset every run. The spec asks for cumulative development + runtime usage.

**Alternatives considered:**
- **Append to TOKENS.md directly** — harder to parse back into structured data
- **SQLite** — overkill for a simple append log

**Trade-off:** `token_usage.json` grows unboundedly. For this project's scale (hundreds of API calls) this is fine.

### Estimated token counts for CLI calls
**Decision:** Estimate Claude CLI tokens as `word_count × 1.3` since the CLI doesn't report usage.

**Context:** The Claude CLI doesn't expose token counts. The 1.3 multiplier is a rough approximation of the token-to-word ratio for English text.

**Trade-off:** Costs reported for intent parsing are approximate, not exact. For a take-home project this is acceptable and transparent.

---

## Project Structure

### Single data_loader.py file
**Decision:** Keep all data loading code (Job dataclass, HTML stripper, helpers, JobDataset) in one file.

**Context:** At ~150 lines after the mmap redesign, splitting into separate files would add import overhead without improving clarity. Every piece is cohesive — they all serve the single purpose of loading and representing job data.

**Trade-off:** Will need splitting if the file grows beyond ~300 lines. Not there yet.

### build_index.py as a separate script
**Decision:** Index building is a separate CLI script, not part of the library.

**Context:** Building the index is a one-time operation that takes minutes and has different memory characteristics than searching. Keeping it separate makes the dependency clear: run `build_index.py` before `demo.py`.

---

## What we'd do differently with more time
- Fuzzy dedup (strip parenthesized locations from titles before comparing)
- Geo-distance scoring for location queries
- Soft salary scoring (bonus/penalty) instead of hard cutoff filter
- Skill keyword boost in post-scoring (literal match on `required_skills`)
- Swap back to Gemini free tier once API key quota is resolved
- LLM-based re-ranking of top results for quality
