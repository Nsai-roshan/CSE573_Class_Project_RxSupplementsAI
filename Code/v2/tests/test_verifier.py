import json

import pytest

from Code.v2.verifier import Verifier


class FakeClient:
    def __init__(self, contents):
        self.contents = iter(contents)
        self.calls = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = next(self.contents)
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


CHUNK = {
    "chunk_id": "Source: NatMedPro-Turmeric-Interactions:2",
    "doc_id": "Source: NatMedPro-Turmeric-Interactions",
    "text": "Turmeric may increase bleeding risk with warfarin. The studied dose was 500 mg daily.",
    "source": "Source: NatMedPro-Turmeric-Interactions-2",
}
CHUNKS = [CHUNK]


def _draft(claim_text="Turmeric may increase bleeding risk with warfarin."):
    return {
        "answer": claim_text,
        "claims": [{"claim": claim_text, "citation_ids": [CHUNK["chunk_id"]]}],
    }


def _verify(decomposition, judgments, draft=None):
    client = FakeClient([json.dumps(decomposition), json.dumps(judgments)])
    report = Verifier(client=client).verify(draft or _draft(), CHUNKS)
    return report, client


def test_faithful_claim_is_supported():
    report, client = _verify(
        [{"claim_id": "c1", "claim_text": "Turmeric may increase bleeding risk with warfarin.", "citation_ids": [CHUNK["chunk_id"]]}],
        [{"claim_id": "c1", "verdict": "SUPPORTED", "evidence_chunk_ids": [CHUNK["chunk_id"]]}],
    )

    assert report.passed is True
    assert report.verdicts[0]["verdict"] == "SUPPORTED"
    assert len(client.calls) == 2
    assert len(report.ledger.calls) == 2


def test_wrong_dosage_is_unsupported():
    report, _ = _verify(
        [{"claim_id": "c1", "claim_text": "The studied dose was 5000 mg daily.", "citation_ids": [CHUNK["chunk_id"]]}],
        [{"claim_id": "c1", "verdict": "UNSUPPORTED", "evidence_chunk_ids": []}],
        _draft("The studied dose was 5000 mg daily."),
    )

    assert report.passed is False
    assert report.verdicts[0]["verdict"] == "UNSUPPORTED"


def test_partially_evidenced_claim_remains_partial():
    report, _ = _verify(
        [{"claim_id": "c1", "claim_text": "Turmeric may increase bleeding risk and cure cancer.", "citation_ids": [CHUNK["chunk_id"]]}],
        [{"claim_id": "c1", "verdict": "PARTIAL", "evidence_chunk_ids": [CHUNK["chunk_id"]]}],
        _draft("Turmeric may increase bleeding risk and cure cancer."),
    )

    assert report.passed is True
    assert report.verdicts[0]["verdict"] == "PARTIAL"


def test_non_entailing_citation_is_unsupported():
    report, _ = _verify(
        [{"claim_id": "c1", "claim_text": "Turmeric cures cancer.", "citation_ids": [CHUNK["chunk_id"]]}],
        [{"claim_id": "c1", "verdict": "UNSUPPORTED", "evidence_chunk_ids": []}],
        _draft("Turmeric cures cancer."),
    )

    assert report.passed is False
    assert report.verdicts[0]["verdict"] == "UNSUPPORTED"


def test_empty_claim_list_is_vacuously_passing_and_logged():
    report, client = _verify([], [], {"answer": "I don't know.", "claims": []})

    assert report.passed is True
    assert report.claims == []
    assert report.verdicts == []
    assert len(client.calls) == 2
    assert len(report.ledger.calls) == 2


def test_unknown_judge_verdict_is_conservatively_unsupported():
    report, _ = _verify(
        [{"claim_id": "c1", "claim_text": "Turmeric is safe.", "citation_ids": [CHUNK["chunk_id"]]}],
        [{"claim_id": "c1", "verdict": "MAYBE", "evidence_chunk_ids": []}],
    )

    assert report.passed is False
    assert report.verdicts[0]["verdict"] == "UNSUPPORTED"
