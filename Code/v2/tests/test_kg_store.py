import json
import logging

from Code.v2.kg_store import KGStore


def test_memory_kg_lookup_returns_matching_neighbor_facts(tmp_path):
    path = tmp_path / "kg.json"
    path.write_text(
        json.dumps(
            [
                {"subject": "creatine", "predicate": "may support", "object": "muscle"},
                {"subject": "creatine", "predicate": "has risk", "object": "stomach upset"},
            ]
        ),
        encoding="utf-8",
    )

    facts = KGStore(backend="memory", path=path).lookup("CREATINE supplement")

    assert len(facts) >= 1
    assert all(fact.startswith("kg:") for fact in facts)
    assert KGStore(backend="memory", path=path).lookup("unlisted herb") == []


def test_missing_kg_file_warns_and_returns_no_facts(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        store = KGStore(backend="memory", path=tmp_path / "missing.json")

    assert "KG" in caplog.text
    assert store.lookup("creatine") == []

