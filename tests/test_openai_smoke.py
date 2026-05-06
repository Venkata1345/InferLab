"""Sanity check: GPT-4o-mini produces schema-valid Invoice JSON for 5 eval examples.

Skipped when OPENAI_API_KEY is not set or eval.jsonl hasn't been built.
Network-bound; ~$0.01 per run. Run with:

    pytest tests/test_openai_smoke.py -v
"""

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from service.prompt import build_messages
from service.schema import Invoice

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "eval.jsonl"
SMOKE_MODEL = os.environ.get("OPENAI_SMOKE_MODEL", "gpt-4o-mini")
N_SAMPLES = 5


pytestmark = [
    pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set"),
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


def test_openai_structured_output_smoke() -> None:
    """Every response must parse as Invoice AND have at least one populated
    field. We don't score accuracy here — just confirm schema + prompt produce
    usable output before we wire up the full eval harness."""
    from openai import OpenAI

    client = OpenAI()
    examples = _load_first_n(N_SAMPLES)
    assert len(examples) == N_SAMPLES, f"only {len(examples)} examples in eval.jsonl"

    for ex in examples:
        completion = client.beta.chat.completions.parse(
            model=SMOKE_MODEL,
            messages=build_messages(ex["input_text"]),
            response_format=Invoice,
        )
        msg = completion.choices[0].message
        assert msg.parsed is not None, (
            f"invoice {ex['invoice_id']}: refusal or parse failure ({msg.refusal!r})"
        )
        invoice: Invoice = msg.parsed
        non_null_scalar_count = sum(
            v is not None
            for v in (
                invoice.vendor_name,
                invoice.invoice_number,
                invoice.invoice_date,
                invoice.total_amount,
                invoice.currency,
            )
        )
        assert non_null_scalar_count >= 1, (
            f"invoice {ex['invoice_id']}: model returned all nulls — {invoice.model_dump()}"
        )
