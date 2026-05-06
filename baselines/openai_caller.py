"""GPT-4o-mini caller using OpenAI structured-output (response_format=Invoice).

Uses client.beta.chat.completions.parse so OpenAI does its own strict-mode
schema adaptation from the Pydantic model. We measure wall-time per call,
capture token usage from response.usage, and retry only on transient errors
(rate-limit, timeout) with exponential backoff.
"""

import os
import time

from baselines.base import PredictionResult
from service.prompt import build_messages
from service.schema import Invoice

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIPredictor:
    name: str

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
    ) -> None:
        # Import inside __init__ so module import doesn't fail when openai isn't installed.
        from openai import OpenAI

        self.model = model
        self.name = f"openai-{model}"
        self.max_retries = max_retries
        # max_retries=0 because we drive retries ourselves (we want to log + time them).
        self.client = OpenAI(api_key=api_key, max_retries=0, timeout=timeout)

    def extract(self, invoice_id: str, ocr_text: str) -> PredictionResult:
        from openai import APIError, APITimeoutError, RateLimitError

        messages = build_messages(ocr_text)
        last_err: Exception | None = None
        t0 = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    response_format=Invoice,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                msg = completion.choices[0].message
                usage = completion.usage
                if msg.parsed is not None:
                    return PredictionResult(
                        invoice_id=invoice_id,
                        prediction=msg.parsed.model_dump(),
                        raw_output=msg.content,
                        latency_ms=latency_ms,
                        input_tokens=usage.prompt_tokens if usage else None,
                        output_tokens=usage.completion_tokens if usage else None,
                        error=None,
                    )
                # Refusal path — strict structured output declined to answer.
                return PredictionResult(
                    invoice_id=invoice_id,
                    prediction=None,
                    raw_output=msg.content,
                    latency_ms=latency_ms,
                    input_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                    error=f"refusal: {msg.refusal}",
                )
            except (RateLimitError, APITimeoutError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
            except APIError as e:
                # Other 4xx/5xx — don't retry (likely a permanent issue).
                last_err = e
                break
            except Exception as e:  # noqa: BLE001 — final safety net
                last_err = e
                break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return PredictionResult(
            invoice_id=invoice_id,
            prediction=None,
            raw_output=None,
            latency_ms=latency_ms,
            input_tokens=None,
            output_tokens=None,
            error=f"{type(last_err).__name__}: {last_err}",
        )


def from_env() -> OpenAIPredictor:
    """Convenience constructor: model from OPENAI_MODEL env var, default gpt-4o-mini."""
    return OpenAIPredictor(model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
