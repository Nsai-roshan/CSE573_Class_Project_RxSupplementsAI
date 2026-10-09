import pytest

from Code.v2.baseline import _tier_b_model, baseline_answer


@pytest.fixture
def fixture_retriever():
    def retrieve(_question):
        return {
            "context": (
                "Source: NatMedPro-L-Theanine-benefits-0: supports calm focus. "
                "NatMedPro-L-Theanine-benefits: supports relaxation."
            ),
            "citations": [
                "Source: NatMedPro-L-Theanine-benefits-0",
                "NatMedPro-L-Theanine-benefits",
            ],
        }

    return retrieve


@pytest.mark.parametrize(
    "question",
    [
        "What are the benefits of L-Theanine?",
        "Can L-Theanine support relaxation?",
        "What does L-Theanine do for focus?",
    ],
)
def test_baseline_fixture_questions_return_citations_and_cost(
    question, fixture_retriever
):
    def generate(_prompt, _model, _provider):
        return (
            "L-Theanine may support calm focus and relaxation. "
            "[Source: NatMedPro-L-Theanine-benefits-0]"
        )

    result = baseline_answer(
        question,
        retrieve_fn=fixture_retriever,
        generate_fn=generate,
        prices={
            "fixture-tier-b": {
                "input_per_1k": 0.002,
                "output_per_1k": 0.004,
            }
        },
        model="fixture-tier-b",
        provider="groq",
    )

    assert result.answer
    assert result.citations
    assert result.cost_usd > 0
    assert len(result.ledger.calls) == 1


def test_tier_b_model_can_be_overridden_per_provider(monkeypatch):
    monkeypatch.setenv("GROQ_TIER_B_MODEL", "openai/gpt-oss-20b")

    assert _tier_b_model(
        {"model_sets": {"groq": {"tier_b": "configured-default"}}},
        "groq",
    ) == "openai/gpt-oss-20b"
