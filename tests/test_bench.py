"""Bench/load tests using a synthetic Predictor — no network."""

import json
import time
from pathlib import Path

import pytest

from baselines.base import PredictionResult
from bench.load import (
    BenchResult,
    latency_stats,
    load_invoices,
    run_bench,
    run_sweep,
    sample_workload,
    write_result,
)


class SleepyPredictor:
    """Sleeps `latency_s` per call, then returns a fixed dummy success."""

    def __init__(self, *, latency_s: float = 0.05, name: str = "sleepy", error_every: int | None = None):
        self.name = name
        self.latency_s = latency_s
        self.error_every = error_every
        self._calls = 0

    def extract(self, invoice_id: str, ocr_text: str) -> PredictionResult:
        self._calls += 1
        time.sleep(self.latency_s)
        if self.error_every and self._calls % self.error_every == 0:
            return PredictionResult(
                invoice_id=invoice_id, prediction=None, raw_output=None,
                latency_ms=self.latency_s * 1000, input_tokens=None, output_tokens=None,
                error="synthetic error",
            )
        return PredictionResult(
            invoice_id=invoice_id,
            prediction={"vendor_name": "X"},
            raw_output='{"vendor_name": "X"}',
            latency_ms=self.latency_s * 1000,
            input_tokens=100, output_tokens=20, error=None,
        )


@pytest.fixture
def synth_eval(tmp_path: Path) -> Path:
    rows = [
        {"invoice_id": f"{i:03d}", "input_text": f"text {i}", "expected_json": {"vendor_name": "X"}}
        for i in range(10)
    ]
    path = tmp_path / "eval.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


# ---------- helpers ----------

class TestLatencyStats:
    def test_empty(self) -> None:
        assert latency_stats([]) == {"n": 0}

    def test_percentiles(self) -> None:
        s = latency_stats([float(i) for i in range(1, 101)])
        assert s["p50"] == 50.0 and s["p95"] == 95.0 and s["p99"] == 99.0


class TestSampleWorkload:
    def test_deterministic_with_seed(self) -> None:
        invs = [{"id": i} for i in range(10)]
        a = sample_workload(invs, 20, seed=42)
        b = sample_workload(invs, 20, seed=42)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        invs = [{"id": i} for i in range(10)]
        a = sample_workload(invs, 20, seed=42)
        b = sample_workload(invs, 20, seed=43)
        assert a != b  # not deterministic-equal across seeds (overwhelming probability)

    def test_n_samples(self) -> None:
        invs = [{"id": i} for i in range(5)]
        # Sample 20 from 5 invoices — proves replacement is happening.
        s = sample_workload(invs, 20, seed=42)
        assert len(s) == 20


# ---------- load_invoices ----------

class TestLoadInvoices:
    def test_loads_jsonl(self, synth_eval: Path) -> None:
        rows = load_invoices(synth_eval)
        assert len(rows) == 10
        assert rows[0]["invoice_id"] == "000"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_invoices(tmp_path / "nope.jsonl")


# ---------- run_bench ----------

class TestRunBench:
    def test_concurrency_speeds_things_up(self, synth_eval: Path) -> None:
        # 8 sequential calls @ 0.05s = 0.4s; at concurrency 4 should be ~0.1-0.2s.
        invs = load_invoices(synth_eval)
        p = SleepyPredictor(latency_s=0.05)
        seq = run_bench(p, invs, concurrency=1, n_requests=8, seed=42)
        par = run_bench(p, invs, concurrency=4, n_requests=8, seed=42)
        # Concurrency=4 should complete meaningfully faster.
        assert par.wall_clock_s < seq.wall_clock_s * 0.7

    def test_throughput_calculation(self, synth_eval: Path) -> None:
        invs = load_invoices(synth_eval)
        p = SleepyPredictor(latency_s=0.05)
        r = run_bench(p, invs, concurrency=4, n_requests=8, seed=42)
        # 8 successful in r.wall_clock_s seconds
        assert r.throughput_rps == pytest.approx(r.n_success / r.wall_clock_s)
        assert r.n_success == 8

    def test_error_counting(self, synth_eval: Path) -> None:
        invs = load_invoices(synth_eval)
        # Every 3rd call fails. concurrency=1 → call ordering is deterministic.
        p = SleepyPredictor(latency_s=0.01, error_every=3)
        r = run_bench(p, invs, concurrency=1, n_requests=9, seed=42)
        assert r.n_errors == 3      # calls 3, 6, 9
        assert r.n_success == 6
        assert r.n_errors + r.n_success == r.n_requests
        assert "synthetic error" in r.metadata["first_5_errors"]

    def test_invalid_concurrency(self, synth_eval: Path) -> None:
        invs = load_invoices(synth_eval)
        p = SleepyPredictor()
        with pytest.raises(ValueError):
            run_bench(p, invs, concurrency=0, n_requests=5)

    def test_invalid_n_requests(self, synth_eval: Path) -> None:
        invs = load_invoices(synth_eval)
        p = SleepyPredictor()
        with pytest.raises(ValueError):
            run_bench(p, invs, concurrency=1, n_requests=0)

    def test_latency_stats_in_result(self, synth_eval: Path) -> None:
        invs = load_invoices(synth_eval)
        p = SleepyPredictor(latency_s=0.05)
        r = run_bench(p, invs, concurrency=2, n_requests=4, seed=42)
        assert "p50" in r.latency_ms and r.latency_ms["p50"] >= 50.0


# ---------- write_result + sweep ----------

class TestWriteResult:
    def test_filename_format(self, tmp_path: Path) -> None:
        r = BenchResult(
            predictor="vllm-Qwen_Qwen2.5-3B-Instruct",
            concurrency=16,
            n_requests=100, n_success=100, n_errors=0,
            wall_clock_s=10.0, throughput_rps=10.0,
        )
        path = write_result(r, tmp_path)
        assert path.name == "vllm-Qwen_Qwen2.5-3B-Instruct_c16.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["concurrency"] == 16
        assert data["throughput_rps"] == 10.0


class TestRunSweep:
    def test_writes_one_file_per_level(self, synth_eval: Path, tmp_path: Path) -> None:
        p = SleepyPredictor(latency_s=0.01)
        paths = run_sweep(
            p, concurrencies=[1, 2, 4],
            eval_path=synth_eval, results_dir=tmp_path,
            n_requests=4, seed=42,
        )
        assert len(paths) == 3
        names = sorted(p.name for p in paths)
        assert names == ["sleepy_c01.json", "sleepy_c02.json", "sleepy_c04.json"]
