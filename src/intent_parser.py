"""
LLM-powered intent parser: extracts structured search parameters from natural language.

Uses OpenAI gpt-4o-mini for parsing, with a regex fallback if the API fails.
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field

from openai import OpenAI

from .search_engine import SearchFilters, EmbeddingWeights
from .token_tracker import tracker

SYSTEM_PROMPT = """\
You are a job search query parser. Given a user's natural language job search query,
extract structured search parameters as JSON.

Respond with ONLY valid JSON, no markdown fences, no explanation.

Schema:
{
  "semantic_query": "cleaned search query optimized for embedding similarity (job titles, skills, domain terms only — remove filter-like words such as 'remote', salary numbers, seniority terms)",
  "filters": {
    "remote_type": "onsite" | "hybrid" | "remote" | null,
    "seniority_level": "entry level" | "mid level" | "senior level" | "manager" | "director" | "internship" | null,
    "employment_type": "full time" | "part time" | "contract" | "temporary" | "seasonal" | null,
    "company_type": "public" | "private" | "non-profit" | null,
    "min_salary": number | null,
    "industries": ["list of industries"] | null
  },
  "embedding_weights": {
    "explicit": 0.0-1.0,
    "inferred": 0.0-1.0,
    "company": 0.0-1.0
  },
  "exclusions": ["terms to penalize in results"],
  "bm25_weight": 0.0-1.0
}

Guidelines for embedding_weights (must sum to 1.0):
- Role-focused queries (e.g. "python engineer"): explicit=0.6, inferred=0.3, company=0.1
- Company/industry-focused (e.g. "jobs at Google"): explicit=0.2, inferred=0.2, company=0.6
- Skill-focused (e.g. "machine learning pytorch"): explicit=0.4, inferred=0.5, company=0.1
- Balanced/general: explicit=0.5, inferred=0.3, company=0.2

Guidelines for semantic_query:
- Strip out filter terms (remote, salary, seniority) — those go in filters
- Focus on role, skills, and domain terms that will match well against job embeddings
- If the query mentions a company name, include it in semantic_query too
- Expand ambiguous or short technology names with common synonyms/aliases to improve embedding recall (e.g. "Go" → "Go Golang", "JS" → "JavaScript JS", "ML" → "machine learning ML", "k8s" → "Kubernetes k8s")

Guidelines for filters:
- Only set a filter if the user clearly indicates it
- "over 100k" or "paying 150k+" → min_salary: 100000 or 150000
- "senior" → seniority_level: "senior level"
- "entry level" or "junior" → seniority_level: "entry level"
- "startup" → company_type: "private" (startups are private companies)
- Map user's language to the exact enum values listed above

Guidelines for exclusions:
- "not management" → ["management", "manager"]
- "no sales" → ["sales", "sales representative"]
- Only include if the user explicitly excludes something

Guidelines for bm25_weight (0.0-1.0, controls keyword vs semantic blend):
- Queries with specific company names or niche technologies: 0.5-0.6 (keyword match matters)
- General role/skill queries: 0.3-0.4 (balanced)
- Abstract/conceptual queries (e.g. "creative roles"): 0.1-0.2 (semantic dominates)
- Default: 0.4
"""

MODEL = "gpt-4o-mini"


@dataclass
class ParsedIntent:
    semantic_query: str
    filters: SearchFilters
    weights: EmbeddingWeights
    exclusions: list[str] = field(default_factory=list)
    bm25_weight: float = 0.4


def parse_intent_llm(query: str, conversation_history: list[str] | None = None) -> ParsedIntent | None:
    """Parse user query using OpenAI gpt-4o-mini. Returns None on failure."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key)

        if conversation_history and len(conversation_history) > 1:
            context = "Previous queries in this session:\n"
            for prev in conversation_history[:-1]:
                context += f"- {prev}\n"
            context += f"\nCurrent query: {query}\n"
            context += "\nSynthesize ALL queries into a single coherent search intent."
            user_msg = context
        else:
            user_msg = query

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=500,
        )

        raw_text = response.choices[0].message.content.strip()

        # Track actual token usage from API response
        usage = response.usage
        tracker.log(
            model=MODEL,
            purpose="intent_parsing",
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
        )

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)

        data = json.loads(raw_text)
        return _parse_response(data, query)

    except Exception as e:
        print(f"  Intent parser error: {e}", file=sys.stderr)
        return None


def _parse_response(data: dict, original_query: str) -> ParsedIntent:
    """Convert raw LLM JSON response into ParsedIntent."""
    semantic_query = data.get("semantic_query") or original_query

    raw_filters = data.get("filters") or {}
    filters = SearchFilters(
        remote_type=raw_filters.get("remote_type"),
        seniority_level=raw_filters.get("seniority_level"),
        employment_type=raw_filters.get("employment_type"),
        company_type=raw_filters.get("company_type"),
        min_salary=raw_filters.get("min_salary"),
        industries=raw_filters.get("industries") or [],
    )

    raw_weights = data.get("embedding_weights") or {}
    w_explicit = raw_weights.get("explicit", 0.5)
    w_inferred = raw_weights.get("inferred", 0.3)
    w_company = raw_weights.get("company", 0.2)
    # Normalize to sum to 1.0
    total = w_explicit + w_inferred + w_company
    if total > 0:
        w_explicit /= total
        w_inferred /= total
        w_company /= total
    weights = EmbeddingWeights(explicit=w_explicit, inferred=w_inferred, company=w_company)

    exclusions = data.get("exclusions") or []
    bm25_weight = data.get("bm25_weight", 0.4)
    bm25_weight = max(0.0, min(1.0, float(bm25_weight)))

    return ParsedIntent(
        semantic_query=semantic_query,
        filters=filters,
        weights=weights,
        exclusions=exclusions,
        bm25_weight=bm25_weight,
    )


# ── Regex fallback ──────────────────────────────────────────────────────────

_REMOTE_RE = re.compile(r"\b(remote|work from home|wfh)\b", re.IGNORECASE)
_HYBRID_RE = re.compile(r"\bhybrid\b", re.IGNORECASE)
_ONSITE_RE = re.compile(r"\b(onsite|on-site|in-office)\b", re.IGNORECASE)
_SALARY_RE = re.compile(r"\b(\d{2,3})[kK]\b|\$(\d{4,7})\b")
_SENIORITY_MAP = {
    r"\b(entry[- ]?level|junior)\b": "entry level",
    r"\b(mid[- ]?level|intermediate)\b": "mid level",
    r"\b(senior|sr\.?)\b": "senior level",
    r"\b(manager|management)\b": "manager",
    r"\b(director)\b": "director",
    r"\b(intern(?:ship)?)\b": "internship",
}
_EMPLOYMENT_MAP = {
    r"\b(full[- ]?time)\b": "full time",
    r"\b(part[- ]?time)\b": "part time",
    r"\b(contract)\b": "contract",
}
_COMPANY_MAP = {
    r"\b(non[- ]?profit|nonprofit)\b": "non-profit",
    r"\b(startup|start-up)\b": "private",
    r"\b(public company)\b": "public",
}


def parse_intent_fallback(query: str) -> ParsedIntent:
    """Regex-based fallback when OpenAI API is unavailable."""
    filters = SearchFilters()

    # Remote type
    if _REMOTE_RE.search(query):
        filters.remote_type = "remote"
    elif _HYBRID_RE.search(query):
        filters.remote_type = "hybrid"
    elif _ONSITE_RE.search(query):
        filters.remote_type = "onsite"

    # Salary
    m = _SALARY_RE.search(query)
    if m:
        if m.group(1):
            filters.min_salary = float(m.group(1)) * 1000
        elif m.group(2):
            filters.min_salary = float(m.group(2))

    # Seniority
    for pattern, value in _SENIORITY_MAP.items():
        if re.search(pattern, query, re.IGNORECASE):
            filters.seniority_level = value
            break

    # Employment type
    for pattern, value in _EMPLOYMENT_MAP.items():
        if re.search(pattern, query, re.IGNORECASE):
            filters.employment_type = value
            break

    # Company type
    for pattern, value in _COMPANY_MAP.items():
        if re.search(pattern, query, re.IGNORECASE):
            filters.company_type = value
            break

    # Strip filter terms from semantic query
    clean = query
    for pattern in [_REMOTE_RE, _HYBRID_RE, _ONSITE_RE, _SALARY_RE]:
        clean = pattern.sub("", clean)
    for pattern in list(_SENIORITY_MAP.keys()) + list(_EMPLOYMENT_MAP.keys()) + list(_COMPANY_MAP.keys()):
        clean = re.sub(pattern, "", clean, flags=re.IGNORECASE)
    # Clean up residual words
    clean = re.sub(r"\b(paying|over|at least|minimum|salary|jobs?|roles?|positions?|at|a)\b", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"[,$+]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    return ParsedIntent(
        semantic_query=clean or query,
        filters=filters,
        weights=EmbeddingWeights(),
        exclusions=[],
    )


def parse_intent(query: str, conversation_history: list[str] | None = None) -> ParsedIntent:
    """Parse query intent — tries OpenAI first, falls back to regex."""
    result = parse_intent_llm(query, conversation_history)
    if result is not None:
        return result

    print("  (Using fallback parser — OpenAI API unavailable)", file=sys.stderr)
    return parse_intent_fallback(query)
