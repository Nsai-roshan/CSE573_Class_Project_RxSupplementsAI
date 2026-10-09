"""Importable, non-streaming extraction of the frozen v1 answer path."""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
import requests
import yaml
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .costing import CostLedger, load_prices


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_PATH = Path(__file__).resolve().parent / "config" / "models.yaml"
load_dotenv(REPO_ROOT / ".env")
_V1_EMBEDDER: Any | None = None


@dataclass
class BaselineResult:
    answer: str
    citations: list[str]
    cost_usd: float
    ledger: CostLedger
    model: str
    provider: str
    prompt: str
    chunks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "cost_usd": self.cost_usd,
            "ledger": self.ledger.calls,
            "model": self.model,
            "provider": self.provider,
            "chunks": self.chunks,
        }


def _resolve_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    pattern = re.compile(r"\$\{([A-Z0-9_]+)(?::-(.*?))?\}")

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        return os.getenv(name, default or "")

    return pattern.sub(replace, value)


def _load_models() -> dict[str, Any]:
    with MODELS_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _configured_provider(config: dict[str, Any], provider: str | None) -> str:
    selected = provider or os.getenv("LLM_PROVIDER") or _resolve_env(
        config.get("LLM_PROVIDER", "groq")
    )
    if selected not in config.get("providers", {}):
        raise ValueError(f"Unsupported LLM_PROVIDER: {selected}")
    return selected


def _tier_b_model(config: dict[str, Any], provider: str) -> str:
    override = os.getenv(f"{provider.upper()}_TIER_B_MODEL")
    if override:
        return override
    return config.get("model_sets", {}).get(provider, {}).get(
        "tier_b", config.get("tier_b")
    )


def _retrieve_v1(question: str) -> dict[str, Any]:
    """Reproduce query_phi.py's FAISS top-3 plus BM25 top-2 retrieval."""
    global _V1_EMBEDDER
    if _V1_EMBEDDER is None:
        _V1_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    model = _V1_EMBEDDER
    corpus = REPO_ROOT / "Data" / "CorpusData"
    with (corpus / "natmed_documents.json").open("r", encoding="utf-8") as handle:
        documents = json.load(handle)
    index = faiss.read_index(str(corpus / "natmed_data.faiss"))

    q_emb = model.encode([question], convert_to_numpy=True)
    _distances, indices = index.search(q_emb, 3)
    results = [documents[i] for i in indices[0]]
    context = ""
    citations: list[str] = []
    chunks: list[dict[str, Any]] = []
    for result in results:
        context += result["id"] + ": " + result["text"] + "                     "
        citations.append(result["id"])
        source_id = str(result["id"])
        doc_id, _, chunk_index = source_id.rpartition("-")
        chunks.append({
            "chunk_id": f"{doc_id}:{chunk_index}" if chunk_index.isdigit() else source_id,
            "doc_id": doc_id or source_id,
            "text": result["text"],
            "source": source_id,
            "score": 0.0,
            "origin": "dense",
        })

    def tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    with (corpus / "bm25_index.pkl").open("rb") as handle:
        data = pickle.load(handle)
    passages = data["documents"]
    bm25: BM25Okapi = data["bm25"]
    scores = bm25.get_scores(tokenize(question))
    top_indices = np.asarray(scores).argsort()[::-1][:2]
    for index_value in top_indices:
        context += str(passages[index_value]) + "                   "
        citations.append(str(passages[index_value]).split(":", 1)[0])
        passage = str(passages[index_value])
        source, _, text = passage.partition(":")
        doc_id, _, chunk_index = source.rpartition("-")
        chunks.append({
            "chunk_id": f"{doc_id}:{chunk_index}" if chunk_index.isdigit() else source,
            "doc_id": doc_id or source,
            "text": text,
            "source": source,
            "score": float(scores[index_value]),
            "origin": "bm25",
        })

    return {
        "context": re.sub(r"\([\d,\s]+\)", "", context),
        "citations": citations,
        "chunks": chunks,
    }


def _build_v1_prompt(context: str, question: str) -> str:
    """Keep the prompt shape and safety/citation instructions from query_phi.py."""
    return f"""
                    You are a helpful medical assistant. 
                    ONLY use the information provided below only to answer the user's question. 
                    Do not mention the information, the source, or that you were given context. 
                    Do not say “according to the context,” “the document says,” or anything similar.

                    If the information is relevant, incorporate it naturally into your human-like answer. 
                    If it is not relevant, tell the user you do not know. 

                    Information:
                    "{context}"

                    User question:
                    {question}

                    Write a direct answer to the user and do not provide any information not necessary to answer their question. If some information does not relate to the user question, just ignore it.
                    Include citations to the sources of the information you use at the end of your answer. You can ONLY cite the sources found in the given context. ONLY the name of the source as provided.
                    DO NOT use any sources of information that are not explicitly provided to you.
                    """


def _generate_with_provider(prompt: str, model: str, provider: str) -> str:
    config = _load_models()
    provider_config = config["providers"][provider]
    base_url = _resolve_env(provider_config["base_url"]).rstrip("/")

    if provider == "ollama":
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]

    api_key_name = provider_config["api_key_env"]
    api_key = os.getenv(api_key_name or "")
    if not api_key:
        raise RuntimeError(f"{api_key_name} is required for provider {provider}")
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def _extract_citations(answer: str) -> list[str]:
    matches = re.findall(r"(?:Source:\s*)?NatMedPro-[A-Za-z0-9_.-]+", answer)
    return list(dict.fromkeys(matches))


def baseline_answer(
    question: str,
    *,
    retrieve_fn: Callable[[str], dict[str, Any]] | None = None,
    generate_fn: Callable[[str, str, str], str] | None = None,
    prices: dict[str, dict[str, float]] | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> BaselineResult:
    """Answer using the v1 retrieval and prompt path with one generation call."""
    config = _load_models()
    selected_provider = _configured_provider(config, provider)
    selected_model = model or _tier_b_model(config, selected_provider)
    retrieved = (retrieve_fn or _retrieve_v1)(question)
    prompt = _build_v1_prompt(retrieved["context"], question)
    answer = (generate_fn or _generate_with_provider)(
        prompt, selected_model, selected_provider
    )
    ledger = CostLedger(prices=prices if prices is not None else load_prices())
    ledger.record_text(selected_model, prompt, answer)
    return BaselineResult(
        answer=answer,
        citations=_extract_citations(answer),
        cost_usd=ledger.usd_total,
        ledger=ledger,
        model=selected_model,
        provider=selected_provider,
        prompt=prompt,
        chunks=list(retrieved.get("chunks", [])),
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise SystemExit("usage: python -m Code.v2.baseline \"question\"")
    result = baseline_answer(question)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
