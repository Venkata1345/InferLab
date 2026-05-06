"""Cost math tests — pure arithmetic, no network."""

import pytest

from eval.cost import (
    GEMINI_2_FLASH,
    OPENAI_GPT_4O_MINI,
    PRICING_BY_PREDICTOR,
    Pricing,
    lookup,
)


class TestPricing:
    def test_cost_for_input_only(self) -> None:
        p = Pricing(input_per_m=1.0, output_per_m=10.0, as_of="2026-05", source="test")
        # 1M input tokens @ $1/M = $1.00
        assert p.cost_for(1_000_000, 0) == pytest.approx(1.0)

    def test_cost_for_output_only(self) -> None:
        p = Pricing(input_per_m=1.0, output_per_m=10.0, as_of="2026-05", source="test")
        assert p.cost_for(0, 1_000_000) == pytest.approx(10.0)

    def test_cost_for_mixed(self) -> None:
        p = Pricing(input_per_m=1.0, output_per_m=10.0, as_of="2026-05", source="test")
        assert p.cost_for(500_000, 100_000) == pytest.approx(0.5 + 1.0)

    def test_cost_per_1k_invoices(self) -> None:
        p = Pricing(input_per_m=1.0, output_per_m=10.0, as_of="2026-05", source="test")
        # 100 invoices using 1M input + 100k output = $1 + $1 = $2 → $20 per 1k invoices
        assert p.cost_per_1k_invoices(1_000_000, 100_000, 100) == pytest.approx(20.0)

    def test_zero_invoices_returns_zero(self) -> None:
        p = Pricing(input_per_m=1.0, output_per_m=10.0, as_of="2026-05", source="test")
        assert p.cost_per_1k_invoices(1_000_000, 100_000, 0) == 0.0


class TestPricingConstants:
    def test_openai_gpt_4o_mini_dated(self) -> None:
        assert OPENAI_GPT_4O_MINI.as_of == "2026-05"
        assert "openai" in OPENAI_GPT_4O_MINI.source.lower()

    def test_gemini_dated(self) -> None:
        assert GEMINI_2_FLASH.as_of == "2026-05"

    def test_lookup_known_predictor(self) -> None:
        assert lookup("openai-gpt-4o-mini") is OPENAI_GPT_4O_MINI

    def test_lookup_unknown_predictor(self) -> None:
        assert lookup("unknown-model") is None

    def test_lookup_vllm_returns_self_hosted(self) -> None:
        # Any predictor name starting with "vllm-" gets the self-hosted Pricing.
        result = lookup("vllm-Qwen-Qwen2.5-3B-Instruct")
        assert result is not None
        assert result.input_per_m == 0.0
        assert result.output_per_m == 0.0

    def test_pricing_table_keys_match_predictor_naming(self) -> None:
        # If we rename a predictor, this catches the cost lookup silently breaking.
        for name in PRICING_BY_PREDICTOR:
            assert name.startswith(("openai-", "gemini-")), (
                f"unexpected predictor naming in PRICING_BY_PREDICTOR: {name}"
            )
