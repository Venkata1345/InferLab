"""Unit tests for data/build_dataset.py — synthetic fixtures, no network/filesystem.

Tests the parsing + mapping primitives directly. The download / split / verify
flow is exercised end-to-end by `python -m data.build_dataset`.
"""

from pathlib import Path

import pytest

from data.build_dataset import (
    map_to_schema,
    normalize_date,
    normalize_total,
    parse_box_csv,
    verify_jsonl,
)


class TestNormalizeTotal:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("9.00", 9.0),
            ("1,234.56", 1234.56),
            ("RM 9.00", 9.0),
            ("$10.50", 10.5),
            ("  42 ", 42.0),
            ("0.00", 0.0),
            ("-1.50", -1.5),  # rare, but possible (refund)
        ],
    )
    def test_parses(self, raw: str, expected: float) -> None:
        assert normalize_total(raw) == expected

    @pytest.mark.parametrize("raw", ["", "  ", None, "abc", ".", "-", "RM"])
    def test_rejects(self, raw: str | None) -> None:
        assert normalize_total(raw) is None


class TestNormalizeDate:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("25/12/2018", "2018-12-25"),
            ("25-12-2018", "2018-12-25"),
            ("12-Aug-2017", "2017-08-12"),
            ("01/01/2020", "2020-01-01"),
            ("25/12/2018 8:13:39 PM", "2018-12-25"),  # fuzzy parse strips trailing time
        ],
    )
    def test_parses_to_iso(self, raw: str, expected: str) -> None:
        assert normalize_date(raw) == expected

    @pytest.mark.parametrize("raw", ["", "  ", None, "not a date", "99/99/9999"])
    def test_rejects(self, raw: str | None) -> None:
        assert normalize_date(raw) is None


class TestMapToSchema:
    def test_full_record(self) -> None:
        raw = {
            "company": "BOOK TA .K (TAMAN DAYA) SDN BHD",
            "date": "25/12/2018",
            "address": "NO.53, JOHOR BAHRU, JOHOR.",
            "total": "9.00",
        }
        assert map_to_schema(raw) == {
            "vendor_name": "BOOK TA .K (TAMAN DAYA) SDN BHD",
            "invoice_date": "2018-12-25",
            "total_amount": 9.0,
        }

    def test_address_is_dropped(self) -> None:
        # Address isn't in our target schema — must not appear in mapped output.
        out = map_to_schema({"company": "X", "address": "Somewhere"})
        assert "address" not in out

    def test_unparseable_total_is_omitted(self) -> None:
        out = map_to_schema({"company": "X", "total": "???"})
        assert "total_amount" not in out
        assert out["vendor_name"] == "X"

    def test_unparseable_date_is_omitted(self) -> None:
        out = map_to_schema({"company": "X", "date": "garbage"})
        assert "invoice_date" not in out

    def test_empty_input_yields_empty_dict(self) -> None:
        assert map_to_schema({}) == {}

    def test_no_invented_fields(self) -> None:
        # Spec: never fake fields SROIE doesn't have.
        out = map_to_schema({"company": "X", "date": "01/01/2020", "total": "1.00"})
        assert "invoice_number" not in out
        assert "currency" not in out
        assert "line_items" not in out


class TestParseBoxCsv:
    def test_strips_bbox_keeps_text_in_order(self, tmp_path: Path) -> None:
        csv = tmp_path / "0.csv"
        csv.write_text(
            "72,25,326,25,326,64,72,64,TAN WOON YANN\n"
            "50,82,440,82,440,121,50,121,BOOK SDN BND\n",
            encoding="utf-8",
        )
        assert parse_box_csv(csv) == "TAN WOON YANN\nBOOK SDN BND"

    def test_text_with_commas_preserved(self, tmp_path: Path) -> None:
        # 8 bbox commas, then text containing further commas — must keep them.
        csv = tmp_path / "0.csv"
        csv.write_text(
            "1,2,3,4,5,6,7,8,NO.53, JALAN SAGU 18, TAMAN DAYA\n",
            encoding="utf-8",
        )
        assert parse_box_csv(csv) == "NO.53, JALAN SAGU 18, TAMAN DAYA"

    def test_skips_blank_and_malformed_lines(self, tmp_path: Path) -> None:
        csv = tmp_path / "0.csv"
        csv.write_text(
            "1,2,3,4,5,6,7,8,GOOD\n"
            "\n"
            "too,few,fields\n"
            "1,2,3,4,5,6,7,8,ALSO GOOD\n",
            encoding="utf-8",
        )
        assert parse_box_csv(csv) == "GOOD\nALSO GOOD"


class TestVerifyJsonl:
    def test_well_formed_passes(self, tmp_path: Path) -> None:
        path = tmp_path / "eval.jsonl"
        path.write_text(
            '{"invoice_id":"000","input_text":"abc","expected_json":{"vendor_name":"X","invoice_date":"2020-01-01","total_amount":9.0}}\n'
            '{"invoice_id":"001","input_text":"def","expected_json":{"vendor_name":"Y"}}\n',
            encoding="utf-8",
        )
        n_ok, problems = verify_jsonl(path)
        assert n_ok == 2
        assert problems == []

    def test_missing_key_flagged(self, tmp_path: Path) -> None:
        path = tmp_path / "eval.jsonl"
        path.write_text('{"invoice_id":"000","input_text":"abc"}\n', encoding="utf-8")
        n_ok, problems = verify_jsonl(path)
        assert n_ok == 0
        assert len(problems) == 1
        assert "expected_json" in problems[0]

    def test_bad_date_flagged(self, tmp_path: Path) -> None:
        path = tmp_path / "eval.jsonl"
        path.write_text(
            '{"invoice_id":"000","input_text":"abc","expected_json":{"invoice_date":"2020/01/01"}}\n',
            encoding="utf-8",
        )
        n_ok, problems = verify_jsonl(path)
        assert n_ok == 0
        assert any("invoice_date" in p for p in problems)

    def test_non_numeric_total_flagged(self, tmp_path: Path) -> None:
        path = tmp_path / "eval.jsonl"
        path.write_text(
            '{"invoice_id":"000","input_text":"abc","expected_json":{"total_amount":"9.0"}}\n',
            encoding="utf-8",
        )
        n_ok, problems = verify_jsonl(path)
        assert n_ok == 0
        assert any("total_amount" in p for p in problems)

    def test_empty_expected_flagged(self, tmp_path: Path) -> None:
        path = tmp_path / "eval.jsonl"
        path.write_text(
            '{"invoice_id":"000","input_text":"abc","expected_json":{}}\n',
            encoding="utf-8",
        )
        n_ok, problems = verify_jsonl(path)
        assert n_ok == 0
        assert any("empty" in p for p in problems)
