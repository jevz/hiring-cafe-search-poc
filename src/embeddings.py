"""
OpenAI embedding client.

Wraps text-embedding-3-small to match the dataset's embedding model.
Includes caching, retry, and token tracking.
"""

import os
import time

import numpy as np
from openai import OpenAI

from .token_tracker import tracker

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536


class EmbeddingClient:
    def __init__(self, api_key: str | None = None):
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        """Embed a text string, returning a normalized 1536-dim vector."""
        text = text.strip()
        if not text:
            return np.zeros(DIMENSIONS, dtype=np.float32)

        if text in self._cache:
            return self._cache[text]

        vec = self._call_api(text)
        self._cache[text] = vec
        return vec

    def _call_api(self, text: str, retries: int = 3) -> np.ndarray:
        for attempt in range(retries):
            try:
                response = self._client.embeddings.create(
                    model=MODEL,
                    input=text,
                )
                usage = response.usage
                tracker.log(
                    model=MODEL,
                    purpose="embedding",
                    input_tokens=usage.total_tokens,
                )

                vec = np.array(response.data[0].embedding, dtype=np.float32)
                # Normalize to unit length
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                return vec

            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"  Embedding API error: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
