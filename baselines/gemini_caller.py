"""Gemini Flash caller using response_mime_type=application/json + response_schema.

Gemini's schema dialect (OpenAPI 3.0) doesn't accept Pydantic's $ref or anyOf
output, so we translate INVOICE_JSON_SCHEMA in-place:
  - inline $defs into where they're $ref'd
  - convert `anyOf: [{type: X}, {type: null}]` → `{type: X, nullable: true}`
  - drop title / additionalProperties / $defs (Gemini ignores or rejects them)

System prompt goes into `system_instruction` on the GenerativeModel; the
per-call payload is just the user message.
"""

import json
import os
import time
from typing import Any

from baselines.base import PredictionResult
from service.prompt import SYSTEM_PROMPT, build_user_message
from service.schema import INVOICE_JSON_SCHEMA

DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _gemini_compatible_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic-emitted JSON Schema → Gemini-compatible (OpenAPI 3.0) schema."""
    defs = schema.get("$defs", {})
    return _resolve(schema, defs)


def _resolve(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        # Inline $ref to a $def.
        if "$ref" in node:
            ref = node["$ref"]
            if not ref.startswith("#/$defs/"):
                raise ValueError(f"Unsupported $ref form: {ref}")
            name = ref.removeprefix("#/$defs/")
            if name not in defs:
                raise ValueError(f"$ref to unknown def: {name}")
            return _resolve(defs[name], defs)
        # Collapse anyOf [non-null, null] → {…, nullable: true}.
        if "anyOf" in node:
            branches = node["anyOf"]
            non_null = [b for b in branches if b.get("type") != "null"]
            has_null = any(b.get("type") == "null" for b in branches)
            if len(non_null) == 1 and has_null:
                merged = dict(non_null[0])
                merged["nullable"] = True
                # Carry description / title from the outer wrapper.
                for k in ("description", "title"):
                    if k in node and k not in merged:
                        merged[k] = node[k]
                return _resolve(merged, defs)
        # Recurse into remaining keys, dropping ones Gemini doesn't accept.
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in ("$defs", "title", "additionalProperties"):
                continue
            out[k] = _resolve(v, defs)
        return out
    if isinstance(node, list):
        return [_resolve(x, defs) for x in node]
    return node


GEMINI_INVOICE_SCHEMA: dict[str, Any] = _gemini_compatible_schema(INVOICE_JSON_SCHEMA)


def _is_retryable(exc: BaseException) -> bool:
    """Gemini SDK exception names for transient errors. Match by class name to
    avoid binding to specific google.api_core / google.generativeai paths that
    differ across SDK versions."""
    name = type(exc).__name__
    return name in {
        "ResourceExhausted",         # 429
        "DeadlineExceeded",          # 504
        "ServiceUnavailable",        # 503
        "InternalServerError",       # 500
        "RetryError",
    }


class GeminiPredictor:
    name: str

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        import google.generativeai as genai

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set (and no api_key argument provided)")
        genai.configure(api_key=api_key)
        self.model_name = model
        self.name = f"gemini-{model}"
        self.max_retries = max_retries
        self._model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )
        self._gen_config = {
            "response_mime_type": "application/json",
            "response_schema": GEMINI_INVOICE_SCHEMA,
        }
        # Note: gemini-2.5-flash with thinking enabled is the wrong tool for batch
        # structured extraction (slow + expensive output tokens). flash-lite is the
        # appropriate Flash variant — no thinking by default. To enable thinking
        # control, migrate to the `google-genai` SDK (the deprecated `google-
        # generativeai` rejects `thinking_config` in GenerationConfig).

    def extract(self, invoice_id: str, ocr_text: str) -> PredictionResult:
        last_err: Exception | None = None
        t0 = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                response = self._model.generate_content(
                    build_user_message(ocr_text),
                    generation_config=self._gen_config,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                raw = response.text  # JSON string per response_mime_type
                try:
                    pred = json.loads(raw)
                    parse_error = None
                except json.JSONDecodeError as e:
                    pred = None
                    parse_error = f"json_parse_failed: {e}"
                usage = getattr(response, "usage_metadata", None)
                return PredictionResult(
                    invoice_id=invoice_id,
                    prediction=pred,
                    raw_output=raw,
                    latency_ms=latency_ms,
                    input_tokens=getattr(usage, "prompt_token_count", None),
                    output_tokens=getattr(usage, "candidates_token_count", None),
                    error=parse_error,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.max_retries and _is_retryable(e):
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


def from_env() -> GeminiPredictor:
    return GeminiPredictor(model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
