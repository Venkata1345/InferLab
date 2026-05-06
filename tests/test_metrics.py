"""Unit tests for eval/metrics.py — synthetic prediction/ground-truth pairs.

These tests gate every downstream eval. Every comparator, the schema-validity
check, the hallucination grounding rules, evaluate_record, and aggregate are
each exercised with positive + negative + null cases.
"""

import pytest

from eval.metrics import (
    CURRENCY_TOKENS,
    SCALAR_FIELDS,
    AggregateMetrics,
    aggregate,
    compare_currency,
    compare_invoice_date,
    compare_invoice_number,
    compare_total_amount,
    compare_vendor_name,
    evaluate_record,
    field_match,
    is_grounded,
    is_schema_valid,
)

OCR_SAMPLE = (
    "99 SPEED MART S/B (519537-X)\n"
    "INVOICE NO: 18396/102/T0362\n"
    "DATE: 13-05-18\n"
    "TOTAL: RM 22.90\n"
    "ITEM A 1 5.00 5.00\n"
    "ITEM B 2 8.95 17.90\n"
)


# ---------- per-field comparators ----------


class TestCompareVendorName:
    def test_exact(self) -> None:
        assert compare_vendor_name("ACME CORP", "ACME CORP") is True

    def test_case_insensitive(self) -> None:
        assert compare_vendor_name("acme corp", "ACME CORP") is True

    def test_whitespace_collapse(self) -> None:
        assert compare_vendor_name("ACME   CORP", "ACME CORP") is True

    def test_strips_trailing_reg_code(self) -> None:
        # The exact issue we saw with GPT-4o-mini in step 3.
        assert compare_vendor_name("99 SPEED MART S/B (519537-X)", "99 SPEED MART S/B") is True
        assert (
            compare_vendor_name("TF VALUE-MART SDN BHD (482123-U)", "TF VALUE-MART SDN BHD") is True
        )

    def test_does_not_strip_mid_string_parens(self) -> None:
        # Only TRAILING reg codes should be stripped.
        assert (
            compare_vendor_name(
                "BOOK TA .K (TAMAN DAYA) SDN BHD",
                "BOOK TA .K (TAMAN DAYA) SDN BHD",
            )
            is True
        )

    def test_mismatch(self) -> None:
        assert compare_vendor_name("ACME CORP", "WIDGETS LLC") is False

    def test_both_null(self) -> None:
        assert compare_vendor_name(None, None) is True

    def test_one_null(self) -> None:
        assert compare_vendor_name(None, "ACME") is False
        assert compare_vendor_name("ACME", None) is False


class TestCompareInvoiceNumber:
    def test_exact(self) -> None:
        assert compare_invoice_number("INV-001", "INV-001") is True

    def test_whitespace_ignored(self) -> None:
        assert (
            compare_invoice_number("INV 001", "INV-001") is False
        )  # different non-whitespace chars
        assert compare_invoice_number("INV001 ", " INV001") is True

    def test_case_insensitive(self) -> None:
        assert compare_invoice_number("inv-001", "INV-001") is True

    def test_mismatch(self) -> None:
        assert compare_invoice_number("INV-001", "INV-002") is False


class TestCompareInvoiceDate:
    def test_iso_exact(self) -> None:
        assert compare_invoice_date("2018-12-25", "2018-12-25") is True

    def test_different_formats_same_date(self) -> None:
        assert compare_invoice_date("25/12/2018", "2018-12-25") is True
        assert compare_invoice_date("25-12-2018", "2018-12-25") is True
        assert compare_invoice_date("25-Dec-2018", "2018-12-25") is True

    def test_with_time_component(self) -> None:
        # Model may emit ISO with time; ground truth is date-only.
        assert compare_invoice_date("2018-12-25T00:00:00", "2018-12-25") is True

    def test_different_dates(self) -> None:
        assert compare_invoice_date("2018-12-25", "2018-12-26") is False

    def test_unparseable(self) -> None:
        assert compare_invoice_date("not a date", "2018-12-25") is False

    def test_both_null(self) -> None:
        assert compare_invoice_date(None, None) is True

    def test_one_null(self) -> None:
        assert compare_invoice_date(None, "2018-12-25") is False


class TestCompareTotalAmount:
    def test_exact(self) -> None:
        assert compare_total_amount(9.0, 9.0) is True

    def test_within_one_percent(self) -> None:
        # 1% of 100 is 1; 100.99 vs 100 is within tolerance, 101.01 is not.
        assert compare_total_amount(100.99, 100.0) is True
        assert compare_total_amount(101.0, 100.0) is True
        assert compare_total_amount(101.01, 100.0) is False

    def test_int_vs_float(self) -> None:
        assert compare_total_amount(9, 9.0) is True

    def test_zero_vs_zero(self) -> None:
        assert compare_total_amount(0.0, 0.0) is True

    def test_zero_vs_nonzero(self) -> None:
        assert compare_total_amount(0.0, 1.0) is False

    def test_string_value_handled(self) -> None:
        # Defensive: shouldn't happen with our schema but if it slips through.
        assert compare_total_amount("9.0", 9.0) is True

    def test_unparseable_string(self) -> None:
        assert compare_total_amount("nope", 9.0) is False

    def test_both_null(self) -> None:
        assert compare_total_amount(None, None) is True

    def test_one_null(self) -> None:
        assert compare_total_amount(None, 9.0) is False

    def test_custom_tolerance(self) -> None:
        # 5% tolerance — 105 vs 100 should now match.
        assert compare_total_amount(105.0, 100.0, tol=0.05) is True
        assert compare_total_amount(106.0, 100.0, tol=0.05) is False


class TestCompareCurrency:
    def test_exact(self) -> None:
        assert compare_currency("USD", "USD") is True

    def test_case_insensitive(self) -> None:
        assert compare_currency("usd", "USD") is True

    def test_whitespace_stripped(self) -> None:
        assert compare_currency(" USD ", "USD") is True

    def test_mismatch(self) -> None:
        assert compare_currency("USD", "MYR") is False


class TestFieldMatchDispatch:
    def test_known_field(self) -> None:
        assert field_match("vendor_name", "ACME", "ACME") is True

    def test_unknown_field(self) -> None:
        # Defensive — unknown fields don't crash, just return False.
        assert field_match("not_a_real_field", "x", "x") is False


# ---------- schema validity ----------

VALID_INVOICE = {
    "vendor_name": "ACME",
    "invoice_number": "INV-001",
    "invoice_date": "2024-01-01",
    "total_amount": 9.99,
    "currency": "USD",
    "line_items": [],
}


class TestIsSchemaValid:
    def test_valid_dict(self) -> None:
        assert is_schema_valid(VALID_INVOICE) is True

    def test_valid_json_string(self) -> None:
        import json

        assert is_schema_valid(json.dumps(VALID_INVOICE)) is True

    def test_missing_required_field(self) -> None:
        bad = {k: v for k, v in VALID_INVOICE.items() if k != "currency"}
        assert is_schema_valid(bad) is False

    def test_extra_field(self) -> None:
        bad = dict(VALID_INVOICE, weird_key="x")
        assert is_schema_valid(bad) is False

    def test_invalid_json_string(self) -> None:
        assert is_schema_valid("not json {{{") is False

    def test_none(self) -> None:
        assert is_schema_valid(None) is False


# ---------- hallucination grounding ----------


class TestIsGroundedVendorName:
    def test_exact_in_text(self) -> None:
        assert is_grounded("vendor_name", "99 SPEED MART S/B", OCR_SAMPLE) is True

    def test_with_reg_code_appended_still_grounded(self) -> None:
        # GPT-4o-mini's typical addition: registration code in parens.
        assert is_grounded("vendor_name", "99 SPEED MART S/B (519537-X)", OCR_SAMPLE) is True

    def test_invented_vendor_not_grounded(self) -> None:
        assert is_grounded("vendor_name", "FAKE VENDOR LLC", OCR_SAMPLE) is False

    def test_null_is_grounded(self) -> None:
        assert is_grounded("vendor_name", None, OCR_SAMPLE) is True


class TestIsGroundedInvoiceNumber:
    def test_exact(self) -> None:
        assert is_grounded("invoice_number", "18396/102/T0362", OCR_SAMPLE) is True

    def test_whitespace_in_value_ok(self) -> None:
        assert is_grounded("invoice_number", "18396 / 102 / T0362", OCR_SAMPLE) is True

    def test_invented(self) -> None:
        assert is_grounded("invoice_number", "INV-9999", OCR_SAMPLE) is False


class TestIsGroundedInvoiceDate:
    def test_iso_form_grounded_via_dd_mm_yy(self) -> None:
        # OCR has "13-05-18"; predicted date "2018-05-13" must be grounded.
        assert is_grounded("invoice_date", "2018-05-13", OCR_SAMPLE) is True

    def test_wrong_date_not_grounded(self) -> None:
        assert is_grounded("invoice_date", "2018-05-14", OCR_SAMPLE) is False


class TestIsGroundedTotalAmount:
    def test_two_decimal_form(self) -> None:
        assert is_grounded("total_amount", 22.90, OCR_SAMPLE) is True

    def test_thousands_separator_in_text(self) -> None:
        text = "Subtotal: 1,234.56\nTotal: 1,234.56"
        assert is_grounded("total_amount", 1234.56, text) is True

    def test_not_in_text(self) -> None:
        assert is_grounded("total_amount", 99999.99, OCR_SAMPLE) is False

    def test_whole_number_form(self) -> None:
        text = "Total: 100"
        assert is_grounded("total_amount", 100.0, text) is True


class TestIsGroundedCurrency:
    def test_iso_code_in_text(self) -> None:
        text = "Currency: USD\nTotal: 9.99"
        assert is_grounded("currency", "USD", text) is True

    def test_symbol_in_text(self) -> None:
        # MYR is grounded by the "RM" symbol — the step-3 finding.
        assert is_grounded("currency", "MYR", OCR_SAMPLE) is True

    def test_dollar_grounds_usd(self) -> None:
        assert is_grounded("currency", "USD", "Total: $9.99") is True

    def test_invented_currency(self) -> None:
        assert is_grounded("currency", "EUR", OCR_SAMPLE) is False

    def test_currency_token_map_covers_common_codes(self) -> None:
        # Sanity check the token map didn't accidentally lose entries.
        for code in ("USD", "MYR", "EUR", "GBP", "SGD"):
            assert code in CURRENCY_TOKENS


class TestIsGroundedNullAndUnknown:
    def test_null_value_always_grounded(self) -> None:
        for f in SCALAR_FIELDS:
            assert is_grounded(f, None, OCR_SAMPLE) is True

    def test_unknown_field_vacuously_grounded(self) -> None:
        # line_items isn't in scope for hallucination v1.
        assert is_grounded("line_items", [{"description": "made up"}], OCR_SAMPLE) is True


# ---------- evaluate_record ----------

PRED_PERFECT = {
    "vendor_name": "99 SPEED MART S/B",
    "invoice_number": "18396/102/T0362",
    "invoice_date": "2018-05-13",
    "total_amount": 22.90,
    "currency": "MYR",
    "line_items": [],
}

EXPECTED_SROIE_STYLE = {
    "vendor_name": "99 SPEED MART S/B",
    "invoice_date": "2018-05-13",
    "total_amount": 22.90,
}


class TestEvaluateRecord:
    def test_perfect_prediction(self) -> None:
        r = evaluate_record(
            invoice_id="000",
            prediction=PRED_PERFECT,
            expected=EXPECTED_SROIE_STYLE,
            ocr_text=OCR_SAMPLE,
        )
        assert r.invoice_id == "000"
        assert r.schema_valid is True
        assert r.fully_correct is True
        assert set(r.fields_evaluated) == {"vendor_name", "invoice_date", "total_amount"}
        assert all(r.field_matches.values())
        assert r.hallucinated_fields == []

    def test_one_field_wrong(self) -> None:
        pred = dict(PRED_PERFECT, total_amount=99.0)  # wrong total
        r = evaluate_record(
            invoice_id="001", prediction=pred, expected=EXPECTED_SROIE_STYLE, ocr_text=OCR_SAMPLE
        )
        assert r.fully_correct is False
        assert r.field_matches["total_amount"] is False
        assert r.field_matches["vendor_name"] is True

    def test_ungrounded_total_flagged_as_hallucination(self) -> None:
        # Use a value that definitely doesn't appear in OCR_SAMPLE (no "12345").
        pred = dict(PRED_PERFECT, total_amount=12345.67)
        r = evaluate_record(
            invoice_id="001b", prediction=pred, expected=EXPECTED_SROIE_STYLE, ocr_text=OCR_SAMPLE
        )
        assert "total_amount" in r.hallucinated_fields

    def test_predicting_extra_field_outside_gt_not_scored(self) -> None:
        # Currency is not in expected_json (sparse GT), so it shouldn't be in field_matches.
        r = evaluate_record(
            invoice_id="002",
            prediction=PRED_PERFECT,
            expected=EXPECTED_SROIE_STYLE,
            ocr_text=OCR_SAMPLE,
        )
        assert "currency" not in r.field_matches
        # But currency IS counted as a non-null prediction, and must be grounded.
        assert "currency" in r.fields_predicted
        assert "currency" not in r.hallucinated_fields  # MYR ↔ RM in text

    def test_invented_currency_flagged(self) -> None:
        pred = dict(PRED_PERFECT, currency="EUR")  # not in OCR text
        r = evaluate_record(
            invoice_id="003",
            prediction=pred,
            expected=EXPECTED_SROIE_STYLE,
            ocr_text=OCR_SAMPLE,
        )
        assert "currency" in r.hallucinated_fields
        # Doesn't affect fully_correct because currency isn't in GT.
        assert r.fully_correct is True

    def test_null_predictions_not_in_fields_predicted(self) -> None:
        pred = dict(PRED_PERFECT, invoice_number=None, currency=None)
        r = evaluate_record(
            invoice_id="004",
            prediction=pred,
            expected=EXPECTED_SROIE_STYLE,
            ocr_text=OCR_SAMPLE,
        )
        assert "invoice_number" not in r.fields_predicted
        assert "currency" not in r.fields_predicted
        # Null on a field in GT would be wrong, but invoice_number/currency aren't in GT.
        assert r.fully_correct is True

    def test_null_for_required_gt_field_is_wrong(self) -> None:
        pred = dict(PRED_PERFECT, total_amount=None)
        r = evaluate_record(
            invoice_id="005",
            prediction=pred,
            expected=EXPECTED_SROIE_STYLE,
            ocr_text=OCR_SAMPLE,
        )
        assert r.field_matches["total_amount"] is False
        assert r.fully_correct is False

    def test_parse_failure_raw_invalid(self) -> None:
        r = evaluate_record(
            invoice_id="006",
            prediction=None,
            expected=EXPECTED_SROIE_STYLE,
            ocr_text=OCR_SAMPLE,
            raw_output="not json",
        )
        assert r.schema_valid is False
        assert r.fully_correct is False
        assert all(v is False for v in r.field_matches.values())
        assert r.fields_predicted == []

    def test_parse_failure_but_raw_valid_json(self) -> None:
        # Edge: prediction=None passed in but raw is actually valid JSON.
        # schema_valid must reflect the raw, not be hardcoded False.
        import json

        r = evaluate_record(
            invoice_id="007",
            prediction=None,
            expected=EXPECTED_SROIE_STYLE,
            ocr_text=OCR_SAMPLE,
            raw_output=json.dumps(PRED_PERFECT),
        )
        assert r.schema_valid is True
        assert r.fully_correct is False  # We treated it as "parse failure" upstream

    def test_line_items_in_gt_skipped_silently(self) -> None:
        # line_items in expected_json should not be scored (no comparator for it).
        expected = dict(EXPECTED_SROIE_STYLE, line_items=[{"description": "x"}])
        r = evaluate_record(
            invoice_id="008",
            prediction=PRED_PERFECT,
            expected=expected,
            ocr_text=OCR_SAMPLE,
        )
        assert "line_items" not in r.fields_evaluated


# ---------- aggregate ----------


class TestAggregate:
    def test_empty(self) -> None:
        am = aggregate([])
        assert am.n_records == 0
        assert am.schema_validity == 0.0

    def test_all_perfect(self) -> None:
        results = [
            evaluate_record(
                invoice_id=str(i),
                prediction=PRED_PERFECT,
                expected=EXPECTED_SROIE_STYLE,
                ocr_text=OCR_SAMPLE,
            )
            for i in range(10)
        ]
        am = aggregate(results)
        assert am.n_records == 10
        assert am.schema_validity == 1.0
        assert am.record_accuracy == 1.0
        assert am.field_accuracy_macro == 1.0
        assert am.field_accuracy_micro == 1.0
        assert am.hallucination_rate == 0.0
        for stats in am.field_accuracy.values():
            assert stats.correct == stats.total == 10

    def test_half_record_wrong_one_field(self) -> None:
        # 5 perfect, 5 with wrong total
        good = [
            evaluate_record(
                invoice_id=f"g{i}",
                prediction=PRED_PERFECT,
                expected=EXPECTED_SROIE_STYLE,
                ocr_text=OCR_SAMPLE,
            )
            for i in range(5)
        ]
        bad_pred = dict(PRED_PERFECT, total_amount=999.0)
        bad = [
            evaluate_record(
                invoice_id=f"b{i}",
                prediction=bad_pred,
                expected=EXPECTED_SROIE_STYLE,
                ocr_text=OCR_SAMPLE,
            )
            for i in range(5)
        ]
        am = aggregate(good + bad)
        assert am.n_records == 10
        assert am.record_accuracy == 0.5  # 5 of 10 records had every field right
        # Per-field: vendor_name + invoice_date are 10/10, total_amount is 5/10
        assert am.field_accuracy["vendor_name"].accuracy == 1.0
        assert am.field_accuracy["invoice_date"].accuracy == 1.0
        assert am.field_accuracy["total_amount"].accuracy == 0.5
        # Macro = (1 + 1 + 0.5) / 3
        assert am.field_accuracy_macro == pytest.approx((1.0 + 1.0 + 0.5) / 3)
        # Micro = (10 + 10 + 5) / (10 + 10 + 10)
        assert am.field_accuracy_micro == pytest.approx(25 / 30)

    def test_hallucination_rate(self) -> None:
        # Every prediction has currency=EUR (ungrounded). 5 scalar predictions per record.
        bad_pred = dict(PRED_PERFECT, currency="EUR")
        results = [
            evaluate_record(
                invoice_id=str(i),
                prediction=bad_pred,
                expected=EXPECTED_SROIE_STYLE,
                ocr_text=OCR_SAMPLE,
            )
            for i in range(10)
        ]
        am = aggregate(results)
        # 1 hallucination per record (currency), 5 predictions per record (all 5 scalars non-null)
        assert am.n_predictions == 50
        assert am.n_hallucinations == 10
        assert am.hallucination_rate == pytest.approx(10 / 50)

    def test_aggregate_returns_dataclass(self) -> None:
        am = aggregate([])
        assert isinstance(am, AggregateMetrics)
