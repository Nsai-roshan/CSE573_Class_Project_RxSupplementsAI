from Code.v2.costing import CostLedger


def test_cost_ledger_uses_per_1k_prices_and_totals_calls():
    ledger = CostLedger(
        prices={
            "fixture-model": {
                "input_per_1k": 0.002,
                "output_per_1k": 0.004,
            }
        }
    )

    ledger.record("fixture-model", input_tokens=1500, output_tokens=500)
    ledger.record("fixture-model", input_tokens=500, output_tokens=1000)

    assert ledger.calls == [
        {
            "model": "fixture-model",
            "input_tokens": 1500,
            "output_tokens": 500,
            "usd": 0.005,
        },
        {
            "model": "fixture-model",
            "input_tokens": 500,
            "output_tokens": 1000,
            "usd": 0.005,
        },
    ]
    assert ledger.usd_total == 0.01


def test_cost_ledger_counts_text_with_tiktoken():
    ledger = CostLedger(
        prices={"fixture-model": {"input_per_1k": 1.0, "output_per_1k": 1.0}}
    )

    call = ledger.record_text("fixture-model", "hello", "world")

    assert call["input_tokens"] > 0
    assert call["output_tokens"] > 0
    assert call["usd"] > 0

