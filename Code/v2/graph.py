"""LangGraph orchestration for routing, retrieval, drafting, verification, and abstention."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .costing import CostLedger
from .drafter import Drafter
from .retrieval import Retriever
from .router import Router
from .verifier import Verifier

LOGGER = logging.getLogger(__name__)
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "regen.txt"


class AnswerState(TypedDict, total=False):
    question: str
    tier: str
    chunks: Any
    draft: Any
    claims: list[dict[str, Any]]
    verdicts: list[dict[str, Any]]
    cost_ledger: CostLedger
    latency_ms: float
    attempts: int
    model_tier: str
    verified: bool
    abstained: bool
    answer: str
    citations: list[str]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


class Pipeline:
    """Build and run the Phase 5 graph with injectable collaborators for tests."""

    def __init__(self, *, router=None, retriever=None, drafter=None, verifier=None, ledger=None):
        self.ledger = ledger or CostLedger()
        self.router = router or Router(ledger=self.ledger)
        self.retriever = retriever or Retriever()
        self.drafter = drafter or Drafter(ledger=self.ledger)
        self.verifier = verifier or Verifier(ledger=self.ledger)
        self.regen_template = PROMPT_PATH.read_text(encoding="utf-8")
        self.graph = self._build()

    def _build(self):
        builder = StateGraph(AnswerState)
        builder.add_node("router", self._router_node)
        builder.add_node("retriever", self._retriever_node)
        builder.add_node("drafter", self._drafter_node)
        builder.add_node("verifier", self._verifier_node)
        builder.add_node("abstain", self._abstain_node)
        builder.set_entry_point("router")
        builder.add_edge("router", "retriever")
        builder.add_edge("retriever", "drafter")
        builder.add_edge("drafter", "verifier")
        builder.add_conditional_edges(
            "verifier", self._after_verifier, {"deliver": END, "regen": "drafter", "abstain": "abstain"}
        )
        builder.add_edge("abstain", END)
        return builder.compile()

    def _router_node(self, state: AnswerState) -> dict[str, Any]:
        decision = self.router.classify(state["question"])
        return {"tier": decision.label, "model_tier": "TIER_A" if decision.label == "SIMPLE" else "TIER_B"}

    def _retriever_node(self, state: AnswerState) -> dict[str, Any]:
        return {"chunks": self.retriever.retrieve(state["question"])}

    def _regen_prompt(self, state: AnswerState) -> str:
        unsupported = [
            item for item in state.get("verdicts", []) if str(item.get("verdict", "")).upper() == "UNSUPPORTED"
        ]
        chunks = "\n".join(f"{item['chunk_id']}: {item.get('text', '')}" for item in state.get("chunks", []))
        return self.regen_template.format(
            question=state["question"], chunks=chunks, previous_draft=_get(state.get("draft"), "answer", ""), unsupported_claims=unsupported
        )

    def _drafter_node(self, state: AnswerState) -> dict[str, Any]:
        attempts = state.get("attempts", 0) + 1
        prompt_override = self._regen_prompt(state) if attempts > 1 else None
        draft = self.drafter.draft(state["question"], state["chunks"], state["tier"], prompt_override=prompt_override)
        return {"draft": draft, "claims": _get(draft, "claims", []), "attempts": attempts}

    def _verifier_node(self, state: AnswerState) -> dict[str, Any]:
        report = self.verifier.verify(state["draft"], state["chunks"])
        return {"verdicts": _get(report, "verdicts", []), "claims": _get(report, "claims", state.get("claims", [])), "verified": bool(_get(report, "passed", False))}

    def _after_verifier(self, state: AnswerState) -> str:
        verdicts = state.get("verdicts", [])
        if not verdicts or not state.get("claims"):
            return "abstain"
        if state.get("verified"):
            return "deliver"
        return "regen" if state.get("attempts", 0) < 3 else "abstain"

    def _abstain_node(self, state: AnswerState) -> dict[str, Any]:
        supported = [
            (claim, verdict)
            for claim in state.get("claims", [])
            for verdict in state.get("verdicts", [])
            if _get(claim, "claim_id") == _get(verdict, "claim_id") and _get(verdict, "verdict") == "SUPPORTED"
        ]
        texts = [f"{_get(claim, 'claim_text', _get(claim, 'claim', ''))} [{cid}]" for claim, verdict in supported for cid in (_get(verdict, "evidence_chunk_ids", []) or _get(claim, "citation_ids", []))]
        answer = "I could not fully verify this answer."
        if texts:
            answer += " " + " ".join(texts)
        return {"answer": answer, "citations": sorted({cid for claim, verdict in supported for cid in (_get(verdict, "evidence_chunk_ids", []) or _get(claim, "citation_ids", []))}), "verified": False, "abstained": True}

    def run(self, question: str) -> AnswerState:
        start = time.perf_counter()
        state = self.graph.invoke({"question": question, "attempts": 0, "cost_ledger": self.ledger})
        state["latency_ms"] = (time.perf_counter() - start) * 1000
        return state

    def answer(self, question: str):
        """Return the public Pydantic result while retaining injectable graph parts."""
        from .pipeline import _result_from_state

        return _result_from_state(self.run(question))
