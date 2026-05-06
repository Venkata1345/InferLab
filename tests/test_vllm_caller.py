"""vllm_caller unit tests — construction + config wiring, no network.

Real correctness validation happens via `python -m cli.main eval vllm --limit 5`
against a live vLLM server on Colab (step 6 sanity check).
"""

import pytest

from baselines.vllm_caller import (
    VLLMPredictor,
    _safe_name,
    _strip_markdown_fences,
    from_env,
)
from service.schema import INVOICE_JSON_SCHEMA
from service.vllm_server import DEFAULT_CONFIG


class TestSafeName:
    def test_replaces_slashes(self) -> None:
        assert _safe_name("Qwen/Qwen2.5-3B-Instruct") == "Qwen_Qwen2.5-3B-Instruct"

    def test_no_change_for_simple_name(self) -> None:
        assert _safe_name("llama") == "llama"


class TestVLLMPredictor:
    def test_name_derived_from_model(self) -> None:
        p = VLLMPredictor(model="Qwen/Qwen2.5-3B-Instruct")
        assert p.name == "vllm-Qwen_Qwen2.5-3B-Instruct"

    def test_default_base_url_matches_config(self) -> None:
        p = VLLMPredictor()
        assert p.base_url == DEFAULT_CONFIG.base_url
        assert p.base_url == "http://0.0.0.0:8000/v1"

    def test_explicit_base_url_overrides(self) -> None:
        p = VLLMPredictor(base_url="http://192.168.0.42:9999/v1")
        assert p.base_url == "http://192.168.0.42:9999/v1"

    def test_temperature_default_is_zero(self) -> None:
        # Determinism is non-negotiable for accuracy metrics — guard the default.
        assert VLLMPredictor().temperature == 0.0

    def test_response_format_uses_invoice_schema(self) -> None:
        p = VLLMPredictor()
        assert p._response_format["type"] == "json_schema"
        assert p._response_format["json_schema"]["name"] == "Invoice"
        assert p._response_format["json_schema"]["schema"] is INVOICE_JSON_SCHEMA


class TestStripMarkdownFences:
    def test_no_fences_pass_through(self) -> None:
        assert _strip_markdown_fences('{"a": 1}') == '{"a": 1}'

    def test_strips_json_fence(self) -> None:
        wrapped = '```json\n{"a": 1}\n```'
        assert _strip_markdown_fences(wrapped) == '{"a": 1}'

    def test_strips_bare_fence(self) -> None:
        wrapped = '```\n{"a": 1}\n```'
        assert _strip_markdown_fences(wrapped) == '{"a": 1}'

    def test_strips_with_surrounding_whitespace(self) -> None:
        wrapped = '   ```json\n{"a": 1}\n```   '
        assert _strip_markdown_fences(wrapped) == '{"a": 1}'

    def test_handles_unclosed_fence(self) -> None:
        # Defensive: opening fence but no closing one shouldn't blow up.
        assert _strip_markdown_fences('```json\n{"a": 1}') == '{"a": 1}'


class TestFromEnv:
    def test_uses_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_MODEL_NAME", "some/other-model")
        monkeypatch.setenv("VLLM_BASE_URL", "http://example.com:1234/v1")
        p = from_env()
        assert p.model == "some/other-model"
        assert p.base_url == "http://example.com:1234/v1"

    def test_falls_back_to_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VLLM_MODEL_NAME", raising=False)
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        p = from_env()
        assert p.model == DEFAULT_CONFIG.model


class TestPricingLookup:
    """Confirm the cost.lookup() name-prefix rule plays well with vLLM names."""

    def test_vllm_name_resolves_to_self_hosted(self) -> None:
        from eval.cost import VLLM_SELF_HOSTED, lookup

        p = VLLMPredictor(model="Qwen/Qwen2.5-3B-Instruct")
        assert lookup(p.name) is VLLM_SELF_HOSTED


class TestVLLMConfig:
    def test_base_url_is_well_formed(self) -> None:
        assert DEFAULT_CONFIG.base_url == f"http://{DEFAULT_CONFIG.host}:{DEFAULT_CONFIG.port}/v1"

    def test_model_is_qwen_3b(self) -> None:
        # If we ever change the default model, the README + cost table must be reviewed.
        assert DEFAULT_CONFIG.model == "Qwen/Qwen2.5-3B-Instruct"
