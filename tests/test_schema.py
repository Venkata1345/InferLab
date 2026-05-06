"""Pydantic schema tests — pure validation, no network.

Covers both runtime validation (the parser we use to score model output)
and the JSON Schema export (what we feed to vLLM / OpenAI structured output).
"""

import pytest
from pydantic import ValidationError

from service.schema import INVOICE_JSON_SCHEMA, Invoice, LineItem

FULL_VALID = {
    "vendor_name": "ACME CORP",
    "invoice_number": "INV-001",
    "invoice_date": "2024-01-15",
    "total_amount": 99.99,
    "currency": "USD",
    "line_items": [
        {"description": "Widget", "quantity": 2, "unit_price": 49.99, "amount": 99.98},
    ],
}


class TestInvoiceValidation:
    def test_full_valid_record_parses(self) -> None:
        inv = Invoice.model_validate(FULL_VALID)
        assert inv.vendor_name == "ACME CORP"
        assert inv.total_amount == 99.99
        assert len(inv.line_items) == 1
        assert inv.line_items[0].description == "Widget"

    def test_all_optionals_null_parses(self) -> None:
        inv = Invoice.model_validate(
            {
                "vendor_name": None,
                "invoice_number": None,
                "invoice_date": None,
                "total_amount": None,
                "currency": None,
                "line_items": [],
            }
        )
        assert inv.vendor_name is None
        assert inv.line_items == []

    def test_empty_line_items_is_valid(self) -> None:
        body = dict(FULL_VALID, line_items=[])
        inv = Invoice.model_validate(body)
        assert inv.line_items == []

    def test_missing_field_rejected(self) -> None:
        # Strict-mode contract: every key must be present (even when value is null).
        bad = {k: v for k, v in FULL_VALID.items() if k != "currency"}
        with pytest.raises(ValidationError) as ex:
            Invoice.model_validate(bad)
        assert "currency" in str(ex.value)

    def test_extra_field_rejected(self) -> None:
        bad = dict(FULL_VALID, weird_extra_key="oops")
        with pytest.raises(ValidationError) as ex:
            Invoice.model_validate(bad)
        assert "weird_extra_key" in str(ex.value)

    def test_wrong_type_for_total_rejected(self) -> None:
        bad = dict(FULL_VALID, total_amount="ninety-nine")
        with pytest.raises(ValidationError):
            Invoice.model_validate(bad)

    def test_line_item_with_extra_field_rejected(self) -> None:
        bad = dict(
            FULL_VALID,
            line_items=[
                {"description": "x", "quantity": 1, "unit_price": 1.0, "amount": 1.0, "tax": 0.1}
            ],
        )
        with pytest.raises(ValidationError):
            Invoice.model_validate(bad)

    def test_int_coerces_to_float_for_amounts(self) -> None:
        # Pragmatic: model may emit integers for whole amounts; accept them.
        inv = Invoice.model_validate(
            {
                **FULL_VALID,
                "total_amount": 100,
                "line_items": [
                    {"description": "x", "quantity": 1, "unit_price": 50, "amount": 100}
                ],
            }
        )
        assert inv.total_amount == 100.0
        assert isinstance(inv.line_items[0].amount, float)


class TestLineItemValidation:
    def test_full_valid(self) -> None:
        li = LineItem.model_validate(
            {"description": "Widget", "quantity": 1, "unit_price": 9.99, "amount": 9.99}
        )
        assert li.description == "Widget"

    def test_missing_description_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LineItem.model_validate({"quantity": 1, "unit_price": 1.0, "amount": 1.0})


class TestJsonSchemaExport:
    """The exported JSON Schema is what gets fed to vLLM / OpenAI. If Pydantic
    changes its emission format we want to know immediately."""

    def test_root_is_object(self) -> None:
        assert INVOICE_JSON_SCHEMA["type"] == "object"

    def test_all_invoice_fields_required(self) -> None:
        # OpenAI strict mode requires every property to appear in `required`.
        required = set(INVOICE_JSON_SCHEMA.get("required", []))
        assert required == {
            "vendor_name",
            "invoice_number",
            "invoice_date",
            "total_amount",
            "currency",
            "line_items",
        }

    def test_root_forbids_extra_properties(self) -> None:
        # OpenAI strict mode also requires additionalProperties: false everywhere.
        assert INVOICE_JSON_SCHEMA["additionalProperties"] is False

    def test_line_item_schema_in_defs(self) -> None:
        defs = INVOICE_JSON_SCHEMA.get("$defs", {})
        assert "LineItem" in defs
        li_schema = defs["LineItem"]
        assert li_schema["additionalProperties"] is False
        assert set(li_schema["required"]) == {"description", "quantity", "unit_price", "amount"}

    def test_optional_fields_allow_null(self) -> None:
        # Pydantic emits `anyOf: [{type: ...}, {type: "null"}]` for `T | None`.
        vendor_schema = INVOICE_JSON_SCHEMA["properties"]["vendor_name"]
        # Either anyOf with null, or type list including null
        if "anyOf" in vendor_schema:
            types = [b.get("type") for b in vendor_schema["anyOf"]]
            assert "null" in types
        else:
            assert "null" in vendor_schema.get("type", [])

    def test_line_items_is_array_of_line_items(self) -> None:
        items_schema = INVOICE_JSON_SCHEMA["properties"]["line_items"]
        assert items_schema["type"] == "array"
        # The items schema $refs LineItem from $defs.
        assert "$ref" in items_schema["items"]
        assert items_schema["items"]["$ref"].endswith("/LineItem")
