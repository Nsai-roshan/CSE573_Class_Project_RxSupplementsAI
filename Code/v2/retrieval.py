"""FAISS + BM25 reciprocal-rank fusion with KG provenance context."""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .kg_store import KGStore


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "Data" / "CorpusData"


class RetrievalResult(list):
    """Six fused document chunks plus separately addressable KG context."""

    def __init__(self, chunks: list[dict[str, Any]], kg_context: list[str] | None = None):
        super().__init__(chunks)
        self.kg_context = kg_context or []

    @property
    def context_lines(self) -> list[str]:
        return [
            f"{item['source']}: {item['text']}" for item in self
        ] + self.kg_context

    def as_context(self) -> str:
        return " ".join(self.context_lines)


def rrf_fusion(
    rankings: Sequence[Sequence[str]], top_k: int = 6, rrf_k: int = 60
) -> list[tuple[str, float]]:
    """Fuse ranked IDs using reciprocal rank fusion."""
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    position = 0
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            if item_id not in first_seen:
                first_seen[item_id] = position
                position += 1
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores, key=lambda item_id: (-scores[item_id], first_seen[item_id]))
    return [(item_id, scores[item_id]) for item_id in ordered[:top_k]]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _doc_identity(document_id: str) -> tuple[str, int]:
    match = re.match(r"^(.*)-(\d+)$", document_id)
    if match:
        return match.group(1), int(match.group(2))
    return document_id, 0


class Retriever:
    """Load the v1 stores once and return provenance-preserving fused chunks."""

    def __init__(
        self,
        *,
        embedder: Any | None = None,
        dense_index: Any | None = None,
        documents: list[dict[str, Any]] | None = None,
        bm25: Any | None = None,
        bm25_passages: list[str] | None = None,
        kg_store: KGStore | None = None,
        corpus_path: str | Path = CORPUS_PATH,
    ) -> None:
        corpus = Path(corpus_path)
        self.embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")
        if documents is None:
            with (corpus / "natmed_documents.json").open("r", encoding="utf-8") as handle:
                documents = json.load(handle)
        self.documents = documents
        self.dense_index = dense_index or faiss.read_index(str(corpus / "natmed_data.faiss"))

        if bm25 is None or bm25_passages is None:
            with (corpus / "bm25_index.pkl").open("rb") as handle:
                data = pickle.load(handle)
            bm25 = data["bm25"]
            bm25_passages = data["documents"]
        self.bm25 = bm25
        self.bm25_passages = bm25_passages
        self.kg_store = kg_store or KGStore(backend="memory")

        self._document_by_id = {document["id"]: document for document in documents}
        self._document_by_source_text: dict[tuple[str, str], dict[str, Any]] = {}
        for document in documents:
            source = document["id"].removeprefix("Source: ")
            self._document_by_source_text[(source.rsplit("-", 1)[0], document["text"])] = document

    def _dense_candidates(self, question: str) -> list[str]:
        embedding = self.embedder.encode([question], convert_to_numpy=True)
        _distances, indices = self.dense_index.search(embedding, 8)
        return [self._chunk_id_for_document(self.documents[index]) for index in indices[0]]

    def _bm25_candidates(self, question: str) -> list[str]:
        scores = self.bm25.get_scores(_tokenize(question))
        indices = np.asarray(scores).argsort()[::-1][:8]
        return [self._chunk_id_for_passage(self.bm25_passages[index]) for index in indices]

    @staticmethod
    def _chunk_id_for_document(document: dict[str, Any]) -> str:
        doc_id, chunk_index = _doc_identity(document["id"])
        return f"{doc_id}:{chunk_index}"

    def _chunk_id_for_passage(self, passage: str) -> str:
        source, _, text = str(passage).partition(":")
        source_base, source_index = _doc_identity(source)
        document = self._document_by_source_text.get((source_base, text))
        if document is not None:
            return self._chunk_id_for_document(document)
        return f"Source: {source_base}:{source_index}"

    def _item_for_chunk(self, chunk_id: str, score: float, origins: dict[str, set[str]]) -> dict[str, Any]:
        doc_id, chunk_index = chunk_id.rsplit(":", 1)
        source_id = f"{doc_id}-{chunk_index}"
        document = self._document_by_id.get(source_id)
        if document is None:
            document = self._document_by_id.get(doc_id)
        text = document["text"] if document else ""
        return {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text,
            "source": source_id,
            "score": score,
            "origin": "+".join(sorted(origins.get(chunk_id, set()))),
        }

    def retrieve(self, question: str) -> RetrievalResult:
        dense = self._dense_candidates(question)
        bm25 = self._bm25_candidates(question)
        origins: dict[str, set[str]] = {}
        for chunk_id in dense:
            origins.setdefault(chunk_id, set()).add("dense")
        for chunk_id in bm25:
            origins.setdefault(chunk_id, set()).add("bm25")
        fused = rrf_fusion([dense, bm25], top_k=6)
        chunks = [self._item_for_chunk(chunk_id, score, origins) for chunk_id, score in fused]
        kg_context = self.kg_store.lookup(question)
        return RetrievalResult(chunks, kg_context)

