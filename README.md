# HiringCafe Job Search Engine

Semantic job search engine over 100k US job postings with natural language queries, multi-turn refinement, and LLM-powered intent parsing.

## Quick Start

### Docker (recommended)

```bash
# Place the dataset in the data directory
cp /path/to/jobs.jsonl src/data/jobs.jsonl

# Set up API key
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Build and run (index is built automatically on first start)
docker compose up --build

# Open http://localhost:8000
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Set up API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Build the search index (one-time, ~2 min)
python build_index.py src/data/jobs.jsonl

# Run both backend and frontend
./run.sh

# Or run the scripted demo (CLI only, 5+ queries including multi-turn refinement)
python demo.py --scripted
```

## How It Works

### Architecture

```
User query
    │
    ▼
┌─────────────────┐
│  Intent Parser   │  Claude Haiku via CLI (regex fallback)
│  ─ semantic_query│  Extracts: search terms, filters, weights, exclusions
│  ─ filters       │  Expands ambiguous terms: "Go" → "Go Golang"
│  ─ weights       │  Classifies query type → adjusts embedding weights
│  ─ exclusions    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OpenAI Embed    │  text-embedding-3-small (1536-dim)
│  Query → vector  │  Cached to avoid duplicate API calls
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Search Engine   │  Weighted cosine similarity across 3 spaces
│  ─ explicit ×w₁  │  + structured post-filtering
│  ─ inferred ×w₂  │  + exclusion penalty
│  ─ company  ×w₃  │  → top-k results
└─────────────────┘
```

### Three-Embedding Weighted Search

The dataset provides 3 pre-computed embeddings per job, each encoding different aspects:

| Embedding | Generated From | Best For |
|-----------|---------------|----------|
| **Explicit** | Job title, listed skills, certifications | Role-specific queries |
| **Inferred** | Related titles, implied skills | Skill/technology queries |
| **Company** | Company name, industry, org type | Company-focused queries |

Instead of using fixed weights, the LLM intent parser assigns per-query weights based on query type:
- "python engineer" → explicit=0.6, inferred=0.3, company=0.1
- "jobs at Google" → explicit=0.2, inferred=0.2, company=0.6
- "machine learning pytorch" → explicit=0.4, inferred=0.5, company=0.1

### Multi-Turn Refinement

Queries accumulate in a conversation context. The LLM synthesizes all previous queries into a single coherent search intent:

```
search> data science jobs              → searches for data science roles
search> at non-profits                 → refines to data science at non-profits
search> make it remote                 → refines to remote data science at non-profits
```

The LLM produces what a human would type if they knew their full intent upfront, avoiding the semantic drift that comes from naive query concatenation. Type `/clear` to reset context.

### Intent Parsing

The LLM extracts structured parameters from natural language:

- **Filters**: remote type, seniority, employment type, company type, min salary, industries
- **Exclusions**: "not management" → penalizes management-related results
- **Synonym expansion**: "Go" → "Go Golang", "JS" → "JavaScript JS"
- **Fallback**: regex-based parser runs when the LLM is unavailable

## Data Processing

### Index Building

The raw `jobs.jsonl` (100k records, ~2GB) is pre-processed into memory-efficient index files:

1. **Two-pass build** — Pass 1 counts unique jobs, Pass 2 writes to pre-allocated memory-mapped files. Never holds more than one row of embeddings in memory.
2. **Deduplication** — Removes duplicate (title, company) pairs, keeping the first occurrence (newest, since the file is sorted newest-first). Removes 28,828 duplicates (29%).
3. **Pre-normalization** — Embedding vectors are normalized to unit length at build time, so search only needs dot products.

Result: 71,172 unique jobs indexed across 4 files (~1.6GB total).

### Memory Management

Embedding matrices are memory-mapped (`np.load(mmap_mode="r")`). The OS pages data in on demand — search startup is near-instant and RSS stays low. This avoids the ~1.7GB RAM spike that crashed the system when loading everything into Python lists.

### Data Quality

Key observations from profiling (see `DATA_PROFILE.md`):

- **Salary coverage**: ~52% of jobs have salary data. Jobs without salary are never excluded by salary filters.
- **Embedding coverage**: 100% of jobs have all 3 embeddings.
- **Duplicates**: 29% of the dataset was duplicate (title, company) pairs.
- **HTML descriptions**: Stripped to plain text for display.

## Design Decisions

See [DECISIONS.md](DECISIONS.md) for detailed trade-off documentation covering:

- Memory-mapped NumPy vs FAISS vs loading into RAM
- Two-pass index building to avoid memory spikes
- Deduplication strategy and which duplicate to keep
- Null-safe filtering (missing data never excludes)
- Claude CLI vs Gemini for intent parsing
- Estimated vs exact token counts for CLI calls

## What Works Well

- **Role queries**: "python backend engineer", "data scientist" — strong explicit embedding match
- **Compound queries**: "remote machine learning jobs paying over 150k" — LLM correctly separates semantic terms from filters
- **Company-focused**: "non-profit healthcare" — dynamic weights boost company embedding
- **Multi-turn refinement**: progressive narrowing works naturally via LLM synthesis
- **Negation**: "senior but not management" — exclusion embeddings penalize unwanted results

## What Could Be Better

- **Ambiguous short queries**: Single-word queries like "Go" need synonym expansion (implemented) but still have weaker signal than multi-word queries
- **Location queries**: No geo-distance scoring — "jobs in New York" relies on embedding similarity rather than coordinate math
- **Salary filtering**: Hard cutoff rather than soft scoring — a job paying $149k is completely excluded by a ">$150k" filter
- **Fuzzy dedup**: Doesn't catch near-duplicates like "Software Engineer" vs "Software Engineer (Chicago, IL)"

## Token Usage

Run `python demo.py --scripted` to generate a fresh `TOKENS.md` report. Token usage is tracked cumulatively across all sessions in `token_usage.json`.

- **Embedding cost**: ~$0.00000X per query (text-embedding-3-small at $0.02/1M tokens)
- **Intent parsing cost**: ~$0.0006 per query (Claude Haiku, estimated from word count)
- **Total project cost**: Well under the $10 OpenAI budget

## Project Structure

```
├── Dockerfile           # Multi-stage build (Node frontend + Python backend)
├── docker-compose.yml   # Single-command run with volume mount
├── run.sh               # Local dev launcher (backend + frontend)
├── demo.py              # Interactive REPL + scripted demo
├── build_index.py       # One-time index builder (two-pass, memory-efficient)
├── requirements.txt     # Python dependencies
├── .env.example         # API key template
├── api/
│   ├── main.py          # FastAPI app + static file serving
│   ├── routes.py        # API endpoints (/api/search, /api/health)
│   ├── models.py        # Pydantic request/response schemas
│   └── session_store.py # In-memory session management
├── frontend/            # React + TypeScript + Vite + TailwindCSS
│   ├── src/
│   │   ├── App.tsx      # Main app layout
│   │   ├── components/  # SearchBar, ResultCard, ChatPanel, etc.
│   │   ├── hooks/       # useSearch custom hook
│   │   ├── api/         # Fetch client
│   │   └── types/       # TypeScript interfaces
│   └── ...
└── src/
    ├── data_loader.py   # Job dataclass + memory-mapped dataset loader
    ├── embeddings.py    # OpenAI embedding client with caching
    ├── search_engine.py # Weighted cosine similarity + filtering
    ├── intent_parser.py # LLM intent parsing + regex fallback
    └── token_tracker.py # Persistent token tracking + reporting
```

## Time Spent

<!-- TODO: Update with actual hours -->
Approximately **X hours** over Y days.

## Requirements

- **Docker** (recommended) or Python 3.12+ and Node 22+
- OpenAI API key (for `text-embedding-3-small` embeddings and `gpt-4o-mini` intent parsing)
- ~2GB disk for the dataset + index files
