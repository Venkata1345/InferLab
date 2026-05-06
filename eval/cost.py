"""$/1k invoices for each predictor — date-tagged published pricing.

Pricing constants are explicit, dated, and sourced. When pricing changes,
update the constant + the `as_of` date; old result JSON files keep the old
cost numbers, which is the right behavior for reproducibility.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    """Per-million-token API pricing snapshot."""

    input_per_m: float    # USD per 1M input (prompt) tokens
    output_per_m: float   # USD per 1M output (completion) tokens
    as_of: str            # YYYY-MM — the date of this snapshot
    source: str           # short label / URL where the price was taken from

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_m + output_tokens * self.output_per_m
        ) / 1_000_000.0

    def cost_per_1k_invoices(
        self, total_input_tokens: int, total_output_tokens: int, n_invoices: int
    ) -> float:
        if n_invoices <= 0:
            return 0.0
        total = self.cost_for(total_input_tokens, total_output_tokens)
        return total / n_invoices * 1000.0


# OpenAI gpt-4o-mini, https://openai.com/api/pricing as of 2026-05.
OPENAI_GPT_4O_MINI = Pricing(
    input_per_m=0.15,
    output_per_m=0.60,
    as_of="2026-05",
    source="openai.com/api/pricing",
)

# Google Gemini 2.5 Flash, https://ai.google.dev/gemini-api/docs/pricing as of 2026-05.
# Standard paid tier, text input.
GEMINI_2_5_FLASH = Pricing(
    input_per_m=0.30,
    output_per_m=2.50,
    as_of="2026-05",
    source="ai.google.dev/gemini-api/docs/pricing",
)

# Gemini 2.0 Flash kept for users who already had access (gated for new keys).
GEMINI_2_FLASH = Pricing(
    input_per_m=0.10,
    output_per_m=0.40,
    as_of="2026-05",
    source="ai.google.dev/pricing",
)

# Self-hosted vLLM serving — no per-token API cost. Compute cost (GPU $/hr)
# is captured separately at the bench level (throughput / hourly rate).
VLLM_SELF_HOSTED = Pricing(
    input_per_m=0.0,
    output_per_m=0.0,
    as_of="2026-05",
    source="self-hosted (compute cost amortized via throughput)",
)


# Gemini 2.5 Flash-Lite, https://ai.google.dev/gemini-api/docs/pricing as of 2026-05.
# Standard paid tier, text input. Lite is the appropriate Flash variant for
# high-throughput structured extraction (no thinking by default).
GEMINI_2_5_FLASH_LITE = Pricing(
    input_per_m=0.10,
    output_per_m=0.40,
    as_of="2026-05",
    source="ai.google.dev/gemini-api/docs/pricing",
)


PRICING_BY_PREDICTOR: dict[str, Pricing] = {
    "openai-gpt-4o-mini": OPENAI_GPT_4O_MINI,
    "gemini-gemini-2.5-flash-lite": GEMINI_2_5_FLASH_LITE,
    "gemini-gemini-2.5-flash": GEMINI_2_5_FLASH,
    "gemini-gemini-2.0-flash": GEMINI_2_FLASH,
    "gemini-gemini-1.5-flash": Pricing(
        # gemini-1.5-flash, https://ai.google.dev/pricing as of 2026-05.
        input_per_m=0.075,
        output_per_m=0.30,
        as_of="2026-05",
        source="ai.google.dev/pricing",
    ),
}


def lookup(predictor_name: str) -> Pricing | None:
    """Return Pricing for `predictor_name`, or None if it's a self-hosted predictor."""
    if predictor_name.startswith("vllm-"):
        return VLLM_SELF_HOSTED
    return PRICING_BY_PREDICTOR.get(predictor_name)
