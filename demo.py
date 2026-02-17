#!/usr/bin/env python3
"""
Job search demo — single-turn semantic search.

Usage:
    python demo.py              # interactive REPL
    python demo.py --scripted   # run predefined test queries
"""

import sys

from dotenv import load_dotenv

load_dotenv()

from src.data_loader import JobDataset
from src.embeddings import EmbeddingClient
from src.search_engine import SearchEngine, format_results
from src.token_tracker import tracker


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

        embedding = client.embed(query)
        results = engine.search(embedding, top_k=10)
        print(format_results(results))
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

        embedding = client.embed(query)
        results = engine.search(embedding, top_k=5)
        print(format_results(results))


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
