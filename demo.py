#!/usr/bin/env python3
"""
Job search demo — semantic search with LLM intent parsing.

Usage:
    python demo.py              # interactive REPL
    python demo.py --scripted   # run predefined test queries
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from src.data_loader import JobDataset
from src.embeddings import EmbeddingClient
from src.intent_parser import parse_intent
from src.search_engine import SearchEngine, format_results
from src.token_tracker import tracker


def _search_with_intent(query: str, engine: SearchEngine, client: EmbeddingClient,
                         conversation_history: list[str] | None = None):
    """Parse intent, embed, search, and print results."""
    intent = parse_intent(query, conversation_history)

    print(f"  Parsed: \"{intent.semantic_query}\"")
    active_filters = []
    f = intent.filters
    if f.remote_type:
        active_filters.append(f"remote={f.remote_type}")
    if f.seniority_level:
        active_filters.append(f"seniority={f.seniority_level}")
    if f.employment_type:
        active_filters.append(f"type={f.employment_type}")
    if f.company_type:
        active_filters.append(f"company={f.company_type}")
    if f.min_salary is not None:
        active_filters.append(f"salary>=${f.min_salary:,.0f}")
    if f.industries:
        active_filters.append(f"industries={f.industries}")
    if active_filters:
        print(f"  Filters: {', '.join(active_filters)}")
    w = intent.weights
    print(f"  Weights: explicit={w.explicit:.2f}, inferred={w.inferred:.2f}, company={w.company:.2f}")
    if intent.exclusions:
        print(f"  Excluding: {intent.exclusions}")
    print()

    # Embed the cleaned semantic query
    query_embedding = client.embed(intent.semantic_query)

    # Embed exclusions if any
    exclusion_embeddings = None
    if intent.exclusions:
        exclusion_embeddings = [client.embed(term) for term in intent.exclusions]

    results = engine.search(
        query_embedding=query_embedding,
        filters=intent.filters,
        weights=intent.weights,
        top_k=10,
        exclusion_embeddings=exclusion_embeddings,
    )
    print(format_results(results))


def run_interactive(engine: SearchEngine, client: EmbeddingClient):
    print("\nType a job search query (or 'quit' to exit):\n")

    while True:
        try:
            query = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        _search_with_intent(query, engine, client)
        print()


SCRIPTED_QUERIES = [
    "data scientist",
    "python backend engineer at a startup",
    "remote machine learning jobs paying over 150k",
    "non-profit healthcare",
    "entry level software engineer",
]


def run_scripted(engine: SearchEngine, client: EmbeddingClient):
    for i, query in enumerate(SCRIPTED_QUERIES, 1):
        print(f"\n{'═' * 60}")
        print(f"  Query {i}: \"{query}\"")
        print(f"{'═' * 60}")

        _search_with_intent(query, engine, client)


def main():
    scripted = "--scripted" in sys.argv

    print("Loading dataset...")
    dataset = JobDataset.load()
    engine = SearchEngine(dataset)
    client = EmbeddingClient()
    print(f"Ready — {len(dataset):,} jobs indexed.\n")

    if scripted:
        run_scripted(engine, client)
    else:
        run_interactive(engine, client)

    tracker.write_report()
    s = tracker.summary()
    print(f"\nToken usage: {s['total_calls']} API calls, ${s['total_cost_usd']:.6f} total cost")
    print("Report written to TOKENS.md")


if __name__ == "__main__":
    main()
