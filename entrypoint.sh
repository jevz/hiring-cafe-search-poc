#!/bin/bash
set -e

DATA_DIR="src/data"

# Build index on first run if jobs.jsonl exists but index doesn't
if [ -f "$DATA_DIR/jobs.jsonl" ] && [ ! -f "$DATA_DIR/jobs.pkl" ]; then
    echo "Index not found. Building from jobs.jsonl (this takes ~2 min)..."
    python build_index.py "$DATA_DIR/jobs.jsonl"
fi

exec "$@"
