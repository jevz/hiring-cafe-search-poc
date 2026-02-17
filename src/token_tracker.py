"""
Token usage tracker for all LLM API calls.
Persists usage to a JSON store so totals accumulate across runs.
Writes human-readable reports to TOKENS.md.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

STORE_PATH = Path("token_usage.json")

# Pricing per 1M tokens
PRICING = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.0, "output": 0.0},
    "gemini-1.5-flash": {"input": 0.0, "output": 0.0},
}


@dataclass
class APICall:
    timestamp: float
    model: str
    purpose: str  # "embedding", "intent_parsing"
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _load_store() -> list[dict]:
    if STORE_PATH.exists():
        try:
            return json.loads(STORE_PATH.read_text())
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _save_store(calls: list[dict]):
    STORE_PATH.write_text(json.dumps(calls, indent=2) + "\n")


class TokenTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._session_calls: list[APICall] = []
        return cls._instance

    def log(self, model: str, purpose: str, input_tokens: int, output_tokens: int = 0):
        pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        call = APICall(
            timestamp=time.time(),
            model=model,
            purpose=purpose,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self._session_calls.append(call)

        # Persist immediately
        store = _load_store()
        store.append(asdict(call))
        _save_store(store)

        return call

    @property
    def calls(self) -> list[APICall]:
        return list(self._session_calls)

    def summary(self) -> dict:
        """Summary of current session only."""
        return _summarize(self._session_calls)

    def cumulative_summary(self) -> dict:
        """Summary of all calls across all sessions."""
        store = _load_store()
        all_calls = [
            APICall(**entry) for entry in store
        ]
        return _summarize(all_calls)

    def write_report(self, path: str | Path = "TOKENS.md"):
        session = self.summary()
        cumulative = self.cumulative_summary()

        lines = [
            "# Token Usage Report\n",
            "## Cumulative (All Sessions)\n",
            f"**Total API calls:** {cumulative['total_calls']}",
            f"**Total cost:** ${cumulative['total_cost_usd']:.6f}\n",
            "| Purpose | Calls | Input Tokens | Output Tokens | Cost |",
            "|---------|-------|-------------|---------------|------|",
        ]
        for purpose, data in cumulative["by_purpose"].items():
            lines.append(
                f"| {purpose} | {data['count']} | {data['input_tokens']:,} | "
                f"{data['output_tokens']:,} | ${data['cost_usd']:.6f} |"
            )

        lines.extend([
            "",
            "## This Session\n",
            f"**API calls:** {session['total_calls']}",
            f"**Cost:** ${session['total_cost_usd']:.6f}\n",
        ])

        # Per-query average (from cumulative data)
        store = _load_store()
        runtime_calls = [c for c in store if c["purpose"] in ("embedding", "intent_parsing")]
        if runtime_calls:
            # Group by purpose for per-query breakdown
            embedding_calls = [c for c in runtime_calls if c["purpose"] == "embedding"]
            parsing_calls = [c for c in runtime_calls if c["purpose"] == "intent_parsing"]

            lines.append("## Per-Query Averages\n")
            if embedding_calls:
                avg = sum(c["cost_usd"] for c in embedding_calls) / len(embedding_calls)
                lines.append(f"**Avg embedding cost:** ${avg:.8f} ({len(embedding_calls)} calls)")
            if parsing_calls:
                avg = sum(c["cost_usd"] for c in parsing_calls) / len(parsing_calls)
                lines.append(f"**Avg intent parsing cost:** ${avg:.8f} ({len(parsing_calls)} calls)")

        Path(path).write_text("\n".join(lines) + "\n")

    def reset(self):
        self._session_calls.clear()


def _summarize(calls: list[APICall]) -> dict:
    by_purpose: dict[str, dict] = {}
    for call in calls:
        if call.purpose not in by_purpose:
            by_purpose[call.purpose] = {
                "count": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
            }
        s = by_purpose[call.purpose]
        s["count"] += 1
        s["input_tokens"] += call.input_tokens
        s["output_tokens"] += call.output_tokens
        s["cost_usd"] += call.cost_usd

    total_cost = sum(s["cost_usd"] for s in by_purpose.values())
    total_calls = sum(s["count"] for s in by_purpose.values())
    return {"by_purpose": by_purpose, "total_calls": total_calls, "total_cost_usd": total_cost}


# Module-level convenience
tracker = TokenTracker()
