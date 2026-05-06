"""Eval runner tests using a synthetic Predictor — no network."""

import json
from pathlib import Path

import pytest

from baselines.base import PredictionResult
from eval.runner import latency_stats, run_eval


class FakePredictor:
    """Returns a fixed prediction for every call. Used for runner-shape tests."""

    def __init__(self, prediction: dict, latency_ms: float = 100.0,
                 input_tokens: int = 200, output_tokens: int = 50, name: str = "fake"):
        self.name = name
        self._pred = prediction
        self._lat = latency_ms
        self._in_tok = input_tokens
        self._out_tok = output_tokens

    def extract(self, invoice_id: str, ocr_text: str) -> PredictionResult:
        return PredictionResult(
            invoice_id=invoice_id,
            prediction=dict(self._pred),
            raw_output=json.dumps(self._pred),
            latency_ms=self._lat,
            input_tokens=self._in_tok,
            output_tokens=self._out_tok,
            error=None,
        )


@pytest.fixture
def synth_eval(tmp_path: Path) -> Path:
    """Three records: 2 with all-correct ground truth, 1 with a different total."""
    records = [
        {
            "invoice_id": "001",
            "input_text": "ACME CORP\nINV-1\n2024-01-01\nTotal: USD 9.99",
            "expected_json": {"vendor_name": "ACME CORP", "invoice_date": "2024-01-01", "total_amount": 9.99},
        },
        {
            "invoice_id": "002",
            "input_text": "ACME CORP\nINV-1\n2024-01-01\nTotal: USD 9.99",
            "expected_json": {"vendor_name": "ACME CORP", "invoice_date": "2024-01-01", "total_amount": 9.99},
        },
        {
            "invoice_id": "003",
            "input_text": "ACME CORP\nINV-1\n2024-01-01\nTotal: USD 9.99",
            "expected_json": {"vendor_name": "ACME CORP", "invoice_date": "2024-01-01", "total_amount": 50.0},
        },
    ]
    path = tmp_path / "eval.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


PERFECT_PRED = {
    "vendor_name": "ACME CORP",
    "invoice_number": "INV-1",
    "invoice_date": "2024-01-01",
    "total_amount": 9.99,
    "currency": "USD",
    "line_items": [],
}


class TestRunEval:
    def test_writes_results_json(self, synth_eval: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "results"
        predictor = FakePredictor(PERFECT_PRED, name="fake-predictor")
        out_path = run_eval(predictor, eval_path=synth_eval, results_dir=out_dir)
        assert out_path == out_dir / "fake-predictor.json"
        assert out_path.exists()

    def test_results_shape(self, synth_eval: Path, tmp_path: Path) -> None:
        predictor = FakePredictor(PERFECT_PRED, name="fake-predictor")
        out_path = run_eval(predictor, eval_path=synth_eval, results_dir=tmp_path / "r")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["predictor"] == "fake-predictor"
        assert data["n_records"] == 3
        assert data["n_errors"] == 0
        assert "metrics" in data
        assert "latency_ms" in data
        assert "cost" in data  # token totals always present (pricing only when known)
        assert "records" in data
        assert len(data["records"]) == 3

    def test_metrics_match_expected(self, synth_eval: Path, tmp_path: Path) -> None:
        # Predictor returns total=9.99 always. 2 of 3 records have GT total=9.99 → 2/3 correct.
        predictor = FakePredictor(PERFECT_PRED, name="fake-predictor")
        out_path = run_eval(predictor, eval_path=synth_eval, results_dir=tmp_path / "r")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        m = data["metrics"]
        assert m["n_records"] == 3
        assert m["schema_validity"] == 1.0
        # Record 003 has wrong total → 2/3 fully correct.
        assert m["record_accuracy"] == pytest.approx(2 / 3)
        # Per-field: vendor + date are 3/3, total is 2/3.
        assert m["field_accuracy"]["total_amount"]["correct"] == 2
        assert m["field_accuracy"]["total_amount"]["total"] == 3

    def test_limit_truncates_eval(self, synth_eval: Path, tmp_path: Path) -> None:
        predictor = FakePredictor(PERFECT_PRED, name="fake-predictor")
        out_path = run_eval(
            predictor, eval_path=synth_eval, results_dir=tmp_path / "r", limit=1
        )
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["n_records"] == 1

    def test_missing_eval_file_raises(self, tmp_path: Path) -> None:
        predictor = FakePredictor(PERFECT_PRED, name="fake")
        with pytest.raises(FileNotFoundError):
            run_eval(predictor, eval_path=tmp_path / "nope.jsonl", results_dir=tmp_path)

    def test_unknown_predictor_no_pricing(self, synth_eval: Path, tmp_path: Path) -> None:
        # `fake-predictor` isn't in the pricing table — token totals should still be
        # there, but no pricing/cost fields.
        predictor = FakePredictor(PERFECT_PRED, name="fake-predictor")
        out_path = run_eval(predictor, eval_path=synth_eval, results_dir=tmp_path / "r")
        cost = json.loads(out_path.read_text(encoding="utf-8"))["cost"]
        assert "input_tokens_total" in cost
        assert "pricing" not in cost
        assert "cost_per_1k_invoices_usd" not in cost

    def test_known_predictor_includes_cost(self, synth_eval: Path, tmp_path: Path) -> None:
        # Use a name that matches PRICING_BY_PREDICTOR.
        predictor = FakePredictor(
            PERFECT_PRED, input_tokens=1000, output_tokens=100, name="openai-gpt-4o-mini"
        )
        out_path = run_eval(predictor, eval_path=synth_eval, results_dir=tmp_path / "r")
        cost = json.loads(out_path.read_text(encoding="utf-8"))["cost"]
        assert "pricing" in cost
        assert cost["pricing"]["as_of"] == "2026-05"
        assert cost["cost_per_1k_invoices_usd"] > 0


class TestLatencyStats:
    def test_empty(self) -> None:
        assert latency_stats([]) == {"n": 0}

    def test_single_value(self) -> None:
        s = latency_stats([100.0])
        assert s["n"] == 1
        assert s["min"] == s["max"] == s["p50"] == s["p99"] == 100.0

    def test_percentiles(self) -> None:
        # 100 evenly-spaced values [1, 2, ..., 100]; nearest-rank percentiles.
        s = latency_stats([float(i) for i in range(1, 101)])
        assert s["n"] == 100
        assert s["min"] == 1.0
        assert s["max"] == 100.0
        assert s["p50"] == 50.0
        assert s["p95"] == 95.0
        assert s["p99"] == 99.0
        assert s["mean"] == pytest.approx(50.5)
