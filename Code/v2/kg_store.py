"""Knowledge-graph lookup with memory and optional Neo4j backends."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
DEFAULT_KG_PATH = Path(__file__).resolve().parents[2] / "Data" / "KnowledgeGraphData" / "KGRelationshipsA.json"


class KGStore:
    """Look up one-hop facts without importing the side-effectful v1 KG script."""

    def __init__(
        self,
        backend: str = "memory",
        path: str | Path = DEFAULT_KG_PATH,
        triples: list[dict[str, Any]] | None = None,
    ) -> None:
        self.backend = backend
        self.path = Path(path)
        self.triples: list[dict[str, Any]] = []
        self.driver = None
        self._load_warning: str | None = None

        if backend == "memory":
            if triples is not None:
                self.triples = triples
            else:
                self.triples = self._load_memory_triples()
        elif backend == "neo4j":
            self._connect_neo4j()
        else:
            raise ValueError(f"Unsupported KG backend: {backend}")

    def _load_memory_triples(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                raise ValueError("KG JSON must contain a list")
            return [triple for triple in data if isinstance(triple, dict)]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._load_warning = f"KG memory backend unavailable: {exc}"
            LOGGER.warning(self._load_warning)
            return []

    def _connect_neo4j(self) -> None:
        uri = os.getenv("NEO4J_URI")
        if not uri:
            self._load_warning = "KG Neo4j backend requested but NEO4J_URI is unset"
            LOGGER.warning(self._load_warning)
            return
        try:
            from neo4j import GraphDatabase

            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "")
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        except Exception as exc:  # pragma: no cover - exercised when Neo4j exists
            self._load_warning = f"KG Neo4j backend unavailable: {exc}"
            LOGGER.warning(self._load_warning)

    @staticmethod
    def _tokens(entity: str) -> list[str]:
        stop_words = {
            "a", "an", "and", "are", "can", "does", "for", "how", "i",
            "is", "it", "of", "on", "some", "the", "to", "what", "with",
        }
        return [
            token
            for token in re.findall(r"[a-z0-9]+", str(entity).lower())
            if token not in stop_words and len(token) >= 3
        ]

    @staticmethod
    def _format_fact(subject: Any, predicate: Any, object_: Any) -> str:
        return f"kg: {subject} {predicate} {object_}"

    def _lookup_memory(self, tokens: list[str]) -> list[str]:
        facts: list[str] = []
        seen: set[str] = set()
        for triple in self.triples:
            subject = triple.get("subject")
            predicate = triple.get("predicate")
            object_ = triple.get("object")
            subject_text = str(subject or "")
            object_text = str(object_ or "")
            haystack = f"{subject_text} {object_text}".lower()
            if tokens and any(token in haystack for token in tokens):
                fact = self._format_fact(subject_text, predicate or "", object_text)
                if fact not in seen:
                    facts.append(fact)
                    seen.add(fact)
        return facts

    def _lookup_neo4j(self, tokens: list[str]) -> list[str]:
        if self.driver is None:
            return []
        query = """
        MATCH (n)-[r]-(m)
        WHERE any(token IN $tokens WHERE
            toLower(coalesce(n.name, '')) CONTAINS token OR
            toLower(coalesce(m.name, '')) CONTAINS token)
        RETURN n.name AS subject, type(r) AS predicate, m.name AS object
        LIMIT 100
        """
        facts: list[str] = []
        with self.driver.session() as session:
            for record in session.run(query, tokens=tokens):
                facts.append(
                    self._format_fact(
                        record.get("subject"),
                        record.get("predicate"),
                        record.get("object"),
                    )
                )
        return list(dict.fromkeys(facts))

    def lookup(self, entity: str) -> list[str]:
        """Return matching one-hop facts as ``kg:`` context lines."""
        if self._load_warning:
            LOGGER.warning(self._load_warning)
        tokens = self._tokens(entity)
        if not tokens:
            return []
        if self.backend == "neo4j":
            return self._lookup_neo4j(tokens)
        return self._lookup_memory(tokens)

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()

