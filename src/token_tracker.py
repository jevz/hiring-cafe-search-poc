"""
Token usage tracker for all LLM API calls.
Tracks both development and runtime usage, writes reports to TOKENS.md.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class APICall:
    timestamp: float
    model: str
    purpose: str  # "embedding", "intent_parsing", "development"
    input_tokens: int
    output_tokens: int
    cost_usd: float


# Pricing per 1M tokens (as of 2024)
PRICING = {
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "gemini-2.0-flash": {"input": 0.0, "output": 0.0},  # free tier
    "gemini-1.5-flash": {"input": 0.0, "output": 0.0},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class TokenTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._calls: list[APICall] = []
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
        self._calls.append(call)
        return call

    @property
    def calls(self) -> list[APICall]:
        return list(self._calls)

    def summary(self) -> dict:
        by_purpose: dict[str, dict] = {}
        for call in self._calls:
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

    def write_report(self, path: str | Path = "TOKENS.md"):
        s = self.summary()
        lines = [
            "# Token Usage Report\n",
            f"**Total API calls:** {s['total_calls']}",
            f"**Total cost:** ${s['total_cost_usd']:.6f}\n",
            "## Breakdown by Purpose\n",
            "| Purpose | Calls | Input Tokens | Output Tokens | Cost |",
            "|---------|-------|-------------|---------------|------|",
        ]
        for purpose, data in s["by_purpose"].items():
            lines.append(
                f"| {purpose} | {data['count']} | {data['input_tokens']:,} | "
                f"{data['output_tokens']:,} | ${data['cost_usd']:.6f} |"
            )

        lines.extend([
            "\n## Per-Query Average\n",
        ])
        runtime_purposes = ["embedding", "intent_parsing"]
        runtime_calls = [c for c in self._calls if c.purpose in runtime_purposes]
        if runtime_calls:
            avg_cost = sum(c.cost_usd for c in runtime_calls) / len(runtime_calls)
            lines.append(f"**Average runtime cost per API call:** ${avg_cost:.8f}")
        else:
            lines.append("No runtime calls recorded yet.")

        Path(path).write_text("\n".join(lines) + "\n")

    def reset(self):
        self._calls.clear()


# Module-level convenience
tracker = TokenTracker()
