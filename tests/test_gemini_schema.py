"""Unit tests for the Gemini schema translator — pure function, no network.

(The full Gemini caller round-trip is exercised by tests/test_gemini_smoke.py
which is network-bound and skipped without GEMINI_API_KEY.)
"""

import pytest

from baselines.gemini_caller import GEMINI_INVOICE_SCHEMA, _gemini_compatible_schema, _resolve
from service.schema import INVOICE_JSON_SCHEMA


class TestResolve:
    def test_inlines_ref(self) -> None:
        defs = {"Foo": {"type": "object", "properties": {"x": {"type": "integer"}}}}
        node = {"$ref": "#/$defs/Foo"}
        out = _resolve(node, defs)
        assert out == {"type": "object", "properties": {"x": {"type": "integer"}}}

    def test_anyof_with_null_collapses_to_nullable(self) -> None:
        node = {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "d"}
        out = _resolve(node, {})
        assert out == {"type": "string", "nullable": True, "description": "d"}

    def test_anyof_without_null_kept_as_anyof(self) -> None:
        # If the union doesn't include null, we don't transform — let Gemini reject if it must.
        node = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
        out = _resolve(node, {})
        assert "anyOf" in out

    def test_drops_dollar_defs(self) -> None:
        node = {"type": "object", "$defs": {"Foo": {"type": "string"}}}
        out = _resolve(node, {})
        assert "$defs" not in out

    def test_drops_title(self) -> None:
        node = {"type": "object", "title": "Should be dropped"}
        out = _resolve(node, {})
        assert "title" not in out

    def test_drops_additional_properties(self) -> None:
        # Gemini ignores additionalProperties — drop it cleanly.
        node = {"type": "object", "additionalProperties": False, "properties": {}}
        out = _resolve(node, {})
        assert "additionalProperties" not in out

    def test_unknown_ref_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown def"):
            _resolve({"$ref": "#/$defs/Missing"}, {})

    def test_external_ref_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            _resolve({"$ref": "https://example.com/schema"}, {})

    def test_recurses_into_arrays(self) -> None:
        node = {"type": "array", "items": {"$ref": "#/$defs/Foo"}}
        defs = {"Foo": {"type": "string"}}
        out = _resolve(node, defs)
        assert out["items"] == {"type": "string"}


class TestGeminiInvoiceSchema:
    def test_no_dollar_refs_in_output(self) -> None:
        # Walk the translated schema; assert no $ref remains.
        def has_ref(node):
            if isinstance(node, dict):
                if "$ref" in node:
                    return True
                return any(has_ref(v) for v in node.values())
            if isinstance(node, list):
                return any(has_ref(x) for x in node)
            return False

        assert not has_ref(GEMINI_INVOICE_SCHEMA)

    def test_no_anyof_in_output(self) -> None:
        def has_anyof(node):
            if isinstance(node, dict):
                if "anyOf" in node:
                    return True
                return any(has_anyof(v) for v in node.values())
            if isinstance(node, list):
                return any(has_anyof(x) for x in node)
            return False

        assert not has_anyof(GEMINI_INVOICE_SCHEMA)

    def test_optional_fields_have_nullable_true(self) -> None:
        # Every scalar field in our Invoice is optional — should become nullable=true.
        for field in ("vendor_name", "invoice_number", "invoice_date", "total_amount", "currency"):
            assert GEMINI_INVOICE_SCHEMA["properties"][field].get("nullable") is True

    def test_line_items_schema_inlined(self) -> None:
        # LineItem was in $defs; should be inlined into the items schema.
        items_schema = GEMINI_INVOICE_SCHEMA["properties"]["line_items"]["items"]
        assert "$ref" not in items_schema
        assert items_schema["type"] == "object"
        assert set(items_schema["required"]) == {"description", "quantity", "unit_price", "amount"}

    def test_required_set_preserved(self) -> None:
        # The 6 top-level required keys should survive translation.
        assert set(GEMINI_INVOICE_SCHEMA["required"]) == {
            "vendor_name",
            "invoice_number",
            "invoice_date",
            "total_amount",
            "currency",
            "line_items",
        }

    def test_translator_is_pure_function(self) -> None:
        # Calling twice produces the same result; doesn't mutate INVOICE_JSON_SCHEMA.
        a = _gemini_compatible_schema(INVOICE_JSON_SCHEMA)
        b = _gemini_compatible_schema(INVOICE_JSON_SCHEMA)
        assert a == b
        # Source schema still has $defs (we didn't mutate it).
        assert "$defs" in INVOICE_JSON_SCHEMA
