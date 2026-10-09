"""Safe public entry point for the Phase 5 answer pipeline."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from .graph import Pipeline


class ClaimResult(BaseModel):
    text: str
    verdict: str
    chunk_ids: list[str] = Field(default_factory=list)


class AnswerResult(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    claims: list[ClaimResult] = Field(default_factory=list)
    model_tier: str = "UNKNOWN"
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    verified: bool = False
    abstained: bool = True


def _safe(message: str, latency_ms: float = 0.0) -> AnswerResult:
    return AnswerResult(answer=message, latency_ms=latency_ms, verified=False, abstained=True)


def _result_from_state(state: dict[str, Any]) -> AnswerResult:
    verdicts = {item.get("claim_id"): item for item in state.get("verdicts", [])}
    claims = []
    for claim in state.get("claims", []):
        verdict = verdicts.get(claim.get("claim_id"), {})
        if state.get("abstained") and verdict.get("verdict") != "SUPPORTED":
            continue
        claims.append(ClaimResult(text=claim.get("claim_text", claim.get("claim", "")), verdict=verdict.get("verdict", "UNSUPPORTED"), chunk_ids=verdict.get("evidence_chunk_ids", claim.get("citation_ids", []))))
    citations = state.get("citations") or sorted({cid for claim in claims for cid in claim.chunk_ids})
    draft = state.get("draft")
    shared_ledger = state.get("cost_ledger")
    cost = getattr(shared_ledger, "usd_total", 0.0)
    if not cost and draft:
        cost = getattr(getattr(draft, "ledger", None), "usd_total", 0.0)
    return AnswerResult(answer=state.get("answer") or getattr(draft, "answer", "I could not fully verify this answer."), citations=citations, claims=claims, model_tier=state.get("model_tier", "UNKNOWN"), cost_usd=cost, latency_ms=state.get("latency_ms", 0.0), verified=bool(state.get("verified", False)), abstained=bool(state.get("abstained", False)))


def answer(question: Any) -> AnswerResult:
    start = time.perf_counter()
    if not isinstance(question, str) or not question.strip():
        return _safe("I could not fully verify this answer because no question was provided.", (time.perf_counter() - start) * 1000)
    try:
        state = Pipeline().run(question.strip())
        return _result_from_state(state)
    except Exception:
        return _safe("I could not fully verify this answer from the available evidence.", (time.perf_counter() - start) * 1000)
