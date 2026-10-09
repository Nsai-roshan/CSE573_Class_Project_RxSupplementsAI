import json

from Code.v2.drafter import Drafter
from Code.v2.retrieval import RetrievalResult


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "Response",
            (),
            {"choices": [type("Choice", (), {"message": type("Message", (), {"content": self.content})()})()]},
        )()


class FakeClient:
    def __init__(self, content):
        self.chat = type("Chat", (), {"completions": FakeCompletions(content)})()


def retrieval_fixture():
    return RetrievalResult(
        [
            {
                "chunk_id": "Source: NatMedPro-L-Theanine-Benefits:0",
                "doc_id": "Source: NatMedPro-L-Theanine-Benefits",
                "text": "L-Theanine may support calm focus.",
                "source": "Source: NatMedPro-L-Theanine-Benefits-0",
                "score": 0.03,
                "origin": "dense+bm25",
            }
        ]
    )


def test_drafter_uses_model_for_forced_tier_and_claim_ids_are_grounded():
    result_json = json.dumps(
        {
            "answer": "L-Theanine may support calm focus. [Source: NatMedPro-L-Theanine-Benefits:0]",
            "claims": [
                {
                    "claim": "L-Theanine may support calm focus.",
                    "citation_ids": ["Source: NatMedPro-L-Theanine-Benefits:0"],
                }
            ],
        }
    )
    client = FakeClient(result_json)
    drafter = Drafter(client=client)

    result = drafter.draft("What are the benefits of L-Theanine?", retrieval_fixture(), "SIMPLE")

    assert result.claims
    allowed = {item["chunk_id"] for item in retrieval_fixture()}
    assert set(result.claims[0]["citation_ids"]) <= allowed
    assert client.chat.completions.calls[0]["model"] == drafter.tier_a


def test_drafter_complex_tier_uses_tier_b_model():
    result_json = json.dumps({"answer": "I don't know.", "claims": []})
    client = FakeClient(result_json)
    drafter = Drafter(client=client)

    drafter.draft("Is turmeric safe with warfarin?", retrieval_fixture(), "COMPLEX")

    assert client.chat.completions.calls[0]["model"] == drafter.tier_b


def test_off_topic_draft_is_dont_know_without_claims():
    result_json = json.dumps({"answer": "I don't know based on the retrieved context.", "claims": []})
    drafter = Drafter(client=FakeClient(result_json))

    result = drafter.draft("What is the capital of France?", retrieval_fixture(), "SIMPLE")

    assert "don't know" in result.answer.lower()
    assert result.claims == []


def test_drafter_adds_only_allowed_citation_to_uncited_factual_sentence():
    result_json = json.dumps(
        {
            "answer": "L-Theanine may support calm focus. [Source: NatMedPro-L-Theanine-Benefits:0] Discuss it with a clinician.",
            "claims": [
                {
                    "claim": "L-Theanine may support calm focus.",
                    "citation_ids": ["Source: NatMedPro-L-Theanine-Benefits:0"],
                }
            ],
        }
    )
    drafter = Drafter(client=FakeClient(result_json))

    result = drafter.draft("What are the benefits of L-Theanine?", retrieval_fixture(), "SIMPLE")

    assert result.answer.endswith(
        "Discuss it with a clinician. [Source: NatMedPro-L-Theanine-Benefits:0]"
    )
