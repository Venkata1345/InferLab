"""vLLM caller — hits the local OpenAI-compatible endpoint with guided_json.

vLLM exposes an OpenAI-compatible HTTP API, so we use the openai SDK with
base_url pointed at the local server. For schema-constrained decoding we use
vLLM's `extra_body={"guided_json": ...}` rather than `response_format` — the
guided_json path is the documented vLLM-native one and works across vLLM
versions; OpenAI's strict-mode `response_format` adapter sometimes
mis-translates Pydantic's `anyOf`/null unions.

temperature=0.0 by default — deterministic outputs are non-negotiable for the
accuracy metrics; sampling defeats the point of structured extraction.
"""

import json
import os
import time

from baselines.base import PredictionResult
from service.prompt import build_messages
from service.schema import INVOICE_JSON_SCHEMA
from service.vllm_server import DEFAULT_CONFIG

DEFAULT_MODEL = DEFAULT_CONFIG.model
DEFAULT_BASE_URL = DEFAULT_CONFIG.base_url


def _safe_name(model: str) -> str:
    """HF model IDs contain '/'; turn them into something filesystem-safe."""
    return model.replace("/", "_")


class VLLMPredictor:
    name: str

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        max_retries: int = 1,
        timeout: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.name = f"vllm-{_safe_name(model)}"
        self.base_url = base_url or os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL)
        # api_key must be non-empty for the openai SDK; vLLM doesn't enforce auth.
        self.client = OpenAI(
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
            base_url=self.base_url,
            max_retries=0,
            timeout=timeout,
        )
        self.max_retries = max_retries
        self.temperature = temperature
        self._guided_json = INVOICE_JSON_SCHEMA

    def extract(self, invoice_id: str, ocr_text: str) -> PredictionResult:
        messages = build_messages(ocr_text)
        last_err: Exception | None = None
        t0 = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    extra_body={"guided_json": self._guided_json},
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                msg = completion.choices[0].message
                raw = msg.content
                pred: dict | None
                err: str | None
                if not raw:
                    pred, err = None, "empty_response"
                else:
                    try:
                        pred = json.loads(raw)
                        err = None
                    except json.JSONDecodeError as e:
                        pred, err = None, f"json_parse_failed: {e}"
                usage = completion.usage
                return PredictionResult(
                    invoice_id=invoice_id,
                    prediction=pred,
                    raw_output=raw,
                    latency_ms=latency_ms,
                    input_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                    error=err,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PredictionResult(
            invoice_id=invoice_id,
            prediction=None,
            raw_output=None,
            latency_ms=latency_ms,
            input_tokens=None,
            output_tokens=None,
            error=f"{type(last_err).__name__}: {last_err}" if last_err else "unknown_error",
        )


def from_env() -> VLLMPredictor:
    return VLLMPredictor(
        model=os.environ.get("VLLM_MODEL_NAME", DEFAULT_MODEL),
        base_url=os.environ.get("VLLM_BASE_URL"),
    )
