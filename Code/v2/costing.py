"""Token accounting for v1 and v2 model calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tiktoken
import yaml


DEFAULT_PRICES_PATH = Path(__file__).resolve().parent / "config" / "prices.yaml"


def load_prices(path: str | Path = DEFAULT_PRICES_PATH) -> dict[str, dict[str, float]]:
    """Load the per-1K-token prices from the YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("prices", {})


class CostLedger:
    """Accumulate model-call token usage and USD costs."""

    def __init__(
        self,
        prices: dict[str, dict[str, float]] | None = None,
        prices_path: str | Path = DEFAULT_PRICES_PATH,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.prices = prices if prices is not None else load_prices(prices_path)
        self.encoder = tiktoken.get_encoding(encoding_name)
        self.calls: list[dict[str, Any]] = []

    def record(self, model: str, input_tokens: int, output_tokens: int) -> dict[str, Any]:
        """Record one call using already-counted input and output tokens."""
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts cannot be negative")
        try:
            price = self.prices[model]
        except KeyError as exc:
            raise KeyError(f"No token price configured for model: {model}") from exc

        usd = (
            input_tokens / 1000 * float(price["input_per_1k"])
            + output_tokens / 1000 * float(price["output_per_1k"])
        )
        call = {
            "model": model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "usd": round(usd, 12),
        }
        self.calls.append(call)
        return call

    def record_text(self, model: str, input_text: str, output_text: str) -> dict[str, Any]:
        """Count text with tiktoken and record the resulting call."""
        return self.record(
            model,
            input_tokens=len(self.encoder.encode(input_text)),
            output_tokens=len(self.encoder.encode(output_text)),
        )

    @property
    def usd_total(self) -> float:
        return round(sum(call["usd"] for call in self.calls), 12)

