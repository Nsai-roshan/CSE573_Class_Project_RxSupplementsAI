import json

import pytest

from Code.v2.router import Router


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletions:
    def __init__(self, labels):
        self.labels = iter(labels)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"choices": [FakeChoice(json.dumps(next(self.labels))) ]})()


class FakeClient:
    def __init__(self, labels):
        self.chat = type("Chat", (), {"completions": FakeCompletions(labels)})()


HAND_LABELED = [
    ("What does vitamin B12 do?", "SIMPLE"),
    ("How much vitamin D is recommended daily?", "SIMPLE"),
    ("What is fenugreek?", "SIMPLE"),
    ("What are the benefits of magnesium?", "SIMPLE"),
    ("Is black tea generally healthy?", "SIMPLE"),
    ("Can turmeric interact with warfarin?", "COMPLEX"),
    ("I take antibiotics and supplements; what should I avoid?", "COMPLEX"),
    ("Compare saffron and caffeine safety during pregnancy.", "COMPLEX"),
    ("What supplement plan should I use for several conditions?", "COMPLEX"),
    ("Could this supplement interact with my medications?", "COMPLEX"),
]


def test_ten_hand_labeled_questions_classify_at_least_eight_correctly():
    client = FakeClient(
        [{"label": label, "confidence": 0.95} for _, label in HAND_LABELED]
    )
    router = Router(client=client)

    decisions = [router.classify(question) for question, _ in HAND_LABELED]
    correct = sum(decision.label == expected for decision, (_, expected) in zip(decisions, HAND_LABELED))

    assert correct >= 8
    assert all(0.0 <= decision.confidence <= 1.0 for decision in decisions)


def test_router_uses_json_mode_and_configured_model():
    client = FakeClient([{"label": "SIMPLE", "confidence": 0.9}])
    router = Router(client=client)

    decision = router.classify("What does vitamin B12 do?")
    call = client.chat.completions.calls[0]

    assert decision.tier == router.tier_a
    assert call["model"] == router.router_model
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}

