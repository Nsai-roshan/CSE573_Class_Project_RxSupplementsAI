import pytest

from Code.v2.costing import CostLedger
from Code.v2.drafter import DraftResult
from Code.v2.graph import Pipeline
from Code.v2.pipeline import answer
from Code.v2.retrieval import RetrievalResult
from Code.v2.router import RouterDecision
from Code.v2.verifier import VerifyReport

CHUNK = {"chunk_id": "doc:0", "doc_id": "doc", "text": "L-Theanine may support calm focus.", "source": "doc-0", "score": 1.0, "origin": "dense+bm25"}

class FakeRouter:
    def __init__(self, label="SIMPLE"): self.label = label
    def classify(self, question):
        model = "tier-a-model" if self.label == "SIMPLE" else "tier-b-model"
        return RouterDecision(self.label, 0.99, model, "router-model")

class FakeRetriever:
    def retrieve(self, question): return RetrievalResult([CHUNK])

class FakeDrafter:
    def __init__(self, answer_text=None, fail_first=False): self.calls, self.answer_text, self.fail_first = [], answer_text, fail_first
    def draft(self, question, chunks, tier, prompt_override=None):
        self.calls.append(prompt_override)
        claim = "L-Theanine cures insomnia." if self.fail_first and len(self.calls) == 1 else "L-Theanine may support calm focus."
        if self.answer_text is not None: return DraftResult(self.answer_text, [], "model", tier, 0.1, CostLedger())
        return DraftResult(claim + " [doc:0]", [{"claim": claim, "citation_ids": ["doc:0"]}], "model", tier, 0.1, CostLedger())

class FakeVerifier:
    def __init__(self, fail_first=False, empty=False, always_fail=False): self.calls, self.fail_first, self.empty, self.always_fail = 0, fail_first, empty, always_fail
    def verify(self, draft, chunks):
        self.calls += 1
        claims = [] if self.empty else [{"claim_id": "c1", "claim_text": "L-Theanine may support calm focus.", "citation_ids": ["doc:0"]}]
        if self.always_fail or (self.fail_first and self.calls == 1):
            return VerifyReport(claims, [{"claim_id": "c1", "verdict": "UNSUPPORTED", "evidence_chunk_ids": [], "reason": "not entailed"}], False, CostLedger())
        verdicts = [{"claim_id": "c1", "verdict": "SUPPORTED", "evidence_chunk_ids": ["doc:0"]}] if claims else []
        return VerifyReport(claims, verdicts, True, CostLedger())

def make_pipeline(label="SIMPLE", drafter=None, verifier=None):
    return Pipeline(router=FakeRouter(label), retriever=FakeRetriever(), drafter=drafter or FakeDrafter(), verifier=verifier or FakeVerifier())

def test_simple_question_delivers_verified_tier_a_answer():
    result = make_pipeline().answer("What are the benefits of L-Theanine?")
    assert result.model_tier == "TIER_A" and result.verified and not result.abstained
    assert result.citations == ["doc:0"]

def test_complex_question_delivers_verified_tier_b_answer():
    result = make_pipeline("COMPLEX").answer("Is turmeric safe with warfarin?")
    assert result.model_tier == "TIER_B" and result.verified

def test_off_topic_question_abstains_without_crashing():
    result = make_pipeline(drafter=FakeDrafter("I don't know based on the retrieved context."), verifier=FakeVerifier(empty=True)).answer("What is the capital of France?")
    assert result.abstained and not result.verified and result.citations == []

def test_adversarial_unsupported_answer_has_no_confident_citations():
    result = make_pipeline(drafter=FakeDrafter(fail_first=True), verifier=FakeVerifier(always_fail=True)).answer("What supplement reverses every disease?")
    assert result.abstained and not result.verified and result.citations == []

def test_failed_verification_regenerates_with_stricter_prompt():
    drafter, verifier = FakeDrafter(fail_first=True), FakeVerifier(fail_first=True)
    result = make_pipeline(drafter=drafter, verifier=verifier).answer("What are the benefits of L-Theanine?")
    assert len(drafter.calls) == 2 and drafter.calls[0] is None and "Unsupported claims" in drafter.calls[1]
    assert result.verified

@pytest.mark.parametrize("question", ["", None])
def test_bad_input_returns_safe_abstention(question):
    result = answer(question)
    assert result.abstained and not result.verified and result.citations == []
