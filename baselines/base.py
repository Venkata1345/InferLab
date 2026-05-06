"""Predictor protocol + result type. Every caller (OpenAI, Gemini, vLLM) implements this."""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class PredictionResult:
    """One model call's full bookkeeping. eval/runner.py consumes these."""

    invoice_id: str
    prediction: dict[str, Any] | None    # parsed Invoice as dict, or None on parse failure
    raw_output: str | None               # raw JSON string returned by the API
    latency_ms: float                    # end-to-end wall time for this single request
    input_tokens: int | None             # prompt tokens (None if API didn't report)
    output_tokens: int | None            # completion tokens (None if API didn't report)
    error: str | None                    # short error label if call failed; None on success


@runtime_checkable
class Predictor(Protocol):
    """All callers expose `name` (used as the results filename stem) and `extract`."""

    name: str

    def extract(self, invoice_id: str, ocr_text: str) -> PredictionResult: ...
