import logging

import numpy as np

from Code.v2.retrieval import Retriever, rrf_fusion
from Code.v2.kg_store import KGStore


class FakeEmbedder:
    def encode(self, _questions, convert_to_numpy=True):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class FakeFaissIndex:
    def search(self, _embedding, _k):
        return (
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], dtype=np.float32),
            np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64),
        )


class FakeBM25:
    def get_scores(self, _tokens):
        return np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])


def _documents():
    return [
        {
            "id": f"Source: NatMedPro-L-Theanine-Benefits-{i}",
            "text": (
                "L-Theanine benefits include calm focus."
                if i == 0
                else f"fixture text {i}"
            ),
        }
        for i in range(8)
    ]


def test_rrf_fusion_combines_rankings_and_returns_top_k():
    fused = rrf_fusion(
        [["a", "b", "c"], ["c", "b", "d"]], top_k=3, rrf_k=60
    )

    assert [item[0] for item in fused] == ["c", "b", "a"]
    assert fused[0][1] > fused[2][1]


def test_known_answer_fixture_is_in_fused_top_six():
    retriever = Retriever(
        embedder=FakeEmbedder(),
        dense_index=FakeFaissIndex(),
        documents=_documents(),
        bm25=FakeBM25(),
        bm25_passages=[f"NatMedPro-L-Theanine-Benefits:{i}" for i in range(8)],
        kg_store=KGStore(triples=[]),
    )

    results = retriever.retrieve("What are the benefits of L-Theanine?")

    assert len(results) == 6
    assert any("calm focus" in item["text"] for item in results)


def test_missing_kg_is_dense_bm25_only_with_warning(tmp_path, caplog):
    store = KGStore(backend="memory", path=tmp_path / "missing.json")
    retriever = Retriever(
        embedder=FakeEmbedder(),
        dense_index=FakeFaissIndex(),
        documents=_documents(),
        bm25=FakeBM25(),
        bm25_passages=[f"NatMedPro-L-Theanine-Benefits:{i}" for i in range(8)],
        kg_store=store,
    )

    with caplog.at_level(logging.WARNING):
        results = retriever.retrieve("What are the benefits of L-Theanine?")

    assert len(results) == 6
    assert not results.kg_context
    assert "KG" in caplog.text


def test_chunk_ids_are_unique_and_parse_to_document_and_index():
    retriever = Retriever(
        embedder=FakeEmbedder(),
        dense_index=FakeFaissIndex(),
        documents=_documents(),
        bm25=FakeBM25(),
        bm25_passages=[f"NatMedPro-L-Theanine-Benefits:{i}" for i in range(8)],
        kg_store=KGStore(triples=[]),
    )

    results = retriever.retrieve("L-Theanine")
    ids = [item["chunk_id"] for item in results]

    assert len(ids) == len(set(ids))
    for item in results:
        doc_id, chunk_index = item["chunk_id"].rsplit(":", 1)
        assert doc_id == item["doc_id"]
        assert chunk_index.isdigit()
