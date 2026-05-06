"""Sanity check: Gemini Flash produces schema-valid Invoice JSON for 2 eval examples.

Skipped when GEMINI_API_KEY is not set or eval.jsonl hasn't been built.
Network-bound; ~$0.005 per run. Run with:

    pytest tests/test_gemini_smoke.py -v
"""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from service.schema import Invoice

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "eval.jsonl"
N_SAMPLES = 2


pytestmark = [
    pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set"),
    pytest.mark.skipif(
        not EVAL_PATH.exists(),
        reason=f"{EVAL_PATH} missing — run `python -m data.build_dataset` first",
    ),
]


def _load_first_n(n: int) -> list[dict]:
    out = []
    with EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if len(out) >= n:
                break
    return out


def test_gemini_structured_output_smoke() -> None:
    """Two real calls; every response must parse as Invoice with at least one
    populated field. Validates the schema translator + Gemini's response_schema
    handling without burning many tokens."""
    from baselines.gemini_caller import GeminiPredictor

    predictor = GeminiPredictor()
    examples = _load_first_n(N_SAMPLES)
    assert len(examples) == N_SAMPLES

    for ex in examples:
        result = predictor.extract(ex["invoice_id"], ex["input_text"])
        assert result.error is None, f"invoice {ex['invoice_id']}: {result.error}"
        assert result.prediction is not None
        # Must round-trip through our Pydantic Invoice model.
        invoice = Invoice.model_validate(result.prediction)
        non_null = sum(
            v is not None
            for v in (
                invoice.vendor_name, invoice.invoice_number, invoice.invoice_date,
                invoice.total_amount, invoice.currency,
            )
        )
        assert non_null >= 1, (
            f"invoice {ex['invoice_id']}: all fields null — {invoice.model_dump()}"
        )
        assert result.input_tokens is not None and result.input_tokens > 0
        assert result.output_tokens is not None and result.output_tokens > 0
