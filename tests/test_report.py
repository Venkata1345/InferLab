"""report.build tests — synthetic eval/bench JSONs in tmpdir, no network."""

import json
from pathlib import Path

import pytest

from report.build import (
    PredictorRow,
    build_comparison,
    build_row,
    load_bench_results,
    load_eval_results,
    render_json,
    render_markdown,
)


def _eval_doc(name: str, *, schema=1.0, macro=0.95, record=0.90,
              p50=2000.0, p99=5000.0, cost_per_1k=0.20) -> dict:
    return {
        "predictor": name,
        "n_records": 100,
        "n_errors": 0,
        "metrics": {
            "n_records": 100,
            "schema_validity": schema,
            "field_accuracy_macro": macro,
            "field_accuracy_micro": macro,
            "record_accuracy": record,
            "field_accuracy": {"vendor_name": {"correct": 95, "total": 100, "accuracy": 0.95}},
            "hallucination_rate": 0.05,
            "n_predictions": 500,
            "n_hallucinations": 25,
        },
        "latency_ms": {"n": 100, "p50": p50, "p95": p50 * 2, "p99": p99, "min": p50 * 0.5,
                       "max": p99, "mean": p50 * 1.2},
        "cost": {
            "input_tokens_total": 100000,
            "output_tokens_total": 20000,
            "input_tokens_per_record_mean": 1000,
            "output_tokens_per_record_mean": 200,
            "pricing": {"input_per_m": 0.15, "output_per_m": 0.60, "as_of": "2026-05",
                        "source": "test"},
            "cost_per_1k_invoices_usd": cost_per_1k,
            "total_cost_usd": cost_per_1k / 10,
        },
    }


def _bench_doc(name: str, c: int, throughput: float) -> dict:
    return {
        "predictor": name,
        "concurrency": c,
        "n_requests": 100,
        "n_success": 100,
        "n_errors": 0,
        "wall_clock_s": 100 / throughput,
        "throughput_rps": throughput,
        "latency_ms": {"n": 100, "min": 1000.0, "p50": 1500.0, "p95": 3000.0,
                       "p99": 4000.0, "max": 4500.0, "mean": 1700.0},
        "metadata": {"timestamp_utc": "2026-05-05T00:00:00Z", "seed": 42},
    }


@pytest.fixture
def synth_dirs(tmp_path: Path) -> tuple[Path, Path]:
    eval_dir = tmp_path / "eval"
    bench_dir = tmp_path / "bench"
    eval_dir.mkdir()
    bench_dir.mkdir()
    # OpenAI: eval only (API has no bench)
    (eval_dir / "openai-gpt-4o-mini.json").write_text(
        json.dumps(_eval_doc("openai-gpt-4o-mini", record=0.90, cost_per_1k=0.225))
    )
    # vLLM: eval + full bench sweep
    (eval_dir / "vllm-Qwen.json").write_text(
        json.dumps(_eval_doc("vllm-Qwen", record=0.66, p50=1200.0, p99=4100.0))
    )
    for c, thru in [(1, 0.68), (4, 2.49), (8, 4.66), (16, 7.95), (32, 11.50)]:
        (bench_dir / f"vllm-Qwen_c{c:02d}.json").write_text(
            json.dumps(_bench_doc("vllm-Qwen", c, thru))
        )
    return eval_dir, bench_dir


# ---------- loaders ----------

class TestLoaders:
    def test_load_eval(self, synth_dirs: tuple[Path, Path]) -> None:
        eval_dir, _ = synth_dirs
        results = load_eval_results(eval_dir)
        assert set(results) == {"openai-gpt-4o-mini", "vllm-Qwen"}

    def test_load_bench_groups_by_predictor(self, synth_dirs: tuple[Path, Path]) -> None:
        _, bench_dir = synth_dirs
        results = load_bench_results(bench_dir)
        assert set(results) == {"vllm-Qwen"}
        assert sorted(results["vllm-Qwen"].keys()) == [1, 4, 8, 16, 32]

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert load_eval_results(tmp_path / "nope") == {}
        assert load_bench_results(tmp_path / "nope") == {}

    def test_skip_invalid_json(self, tmp_path: Path) -> None:
        d = tmp_path / "eval"
        d.mkdir()
        (d / "broken.json").write_text("not json{{{")
        (d / "good.json").write_text(json.dumps(_eval_doc("good")))
        results = load_eval_results(d)
        assert set(results) == {"good"}


# ---------- build_row ----------

class TestBuildRow:
    def test_api_predictor_uses_eval_cost(self, synth_dirs: tuple[Path, Path]) -> None:
        eval_dir, _ = synth_dirs
        eval_results = load_eval_results(eval_dir)
        row = build_row(
            "openai-gpt-4o-mini", eval_results["openai-gpt-4o-mini"], None,
            gpu_dollar_per_hr=0.5, target_c=16,
        )
        assert row.cost_per_1k_usd == pytest.approx(0.225)
        assert "token cost" in row.cost_basis

    def test_vllm_cost_from_throughput(self, synth_dirs: tuple[Path, Path]) -> None:
        eval_dir, bench_dir = synth_dirs
        eval_results = load_eval_results(eval_dir)
        bench_results = load_bench_results(bench_dir)
        row = build_row(
            "vllm-Qwen", eval_results["vllm-Qwen"], bench_results["vllm-Qwen"],
            gpu_dollar_per_hr=0.5, target_c=16,
        )
        # $/1k = (0.5/3600) / 7.95 * 1000
        expected = (0.5 / 3600.0) / 7.95 * 1000.0
        assert row.cost_per_1k_usd == pytest.approx(expected)
        assert row.throughput_rps_at_target == 7.95
        assert row.target_concurrency == 16

    def test_bench_falls_back_to_closest_concurrency(self, synth_dirs: tuple[Path, Path]) -> None:
        _, bench_dir = synth_dirs
        bench_results = load_bench_results(bench_dir)
        # Ask for c=20 — no exact match, should pick c=16 (closer than c=32)
        row = build_row("vllm-Qwen", None, bench_results["vllm-Qwen"],
                        gpu_dollar_per_hr=0.5, target_c=20)
        assert row.target_concurrency == 16

    def test_no_eval_no_bench_returns_empty_row(self) -> None:
        row = build_row("phantom", None, None, gpu_dollar_per_hr=0.5, target_c=16)
        assert row.name == "phantom"
        assert row.schema_validity is None
        assert row.throughput_rps_at_target is None
        assert row.cost_per_1k_usd is None


# ---------- build_comparison ----------

class TestBuildComparison:
    def test_orders_vllm_first(self, synth_dirs: tuple[Path, Path]) -> None:
        eval_dir, bench_dir = synth_dirs
        rows = build_comparison(eval_dir=eval_dir, bench_dir=bench_dir)
        names = [r.name for r in rows]
        assert names == ["vllm-Qwen", "openai-gpt-4o-mini"]

    def test_includes_predictors_with_only_bench(self, synth_dirs: tuple[Path, Path]) -> None:
        eval_dir, bench_dir = synth_dirs
        # Add a predictor that only has bench data
        (bench_dir / "orphan_c16.json").write_text(json.dumps(_bench_doc("orphan", 16, 5.0)))
        rows = build_comparison(eval_dir=eval_dir, bench_dir=bench_dir)
        names = [r.name for r in rows]
        assert "orphan" in names


# ---------- renderers ----------

class TestRenderMarkdown:
    def test_table_has_header_separator_rows(self, synth_dirs: tuple[Path, Path]) -> None:
        eval_dir, bench_dir = synth_dirs
        rows = build_comparison(eval_dir=eval_dir, bench_dir=bench_dir)
        md = render_markdown(rows)
        lines = md.splitlines()
        assert lines[0].startswith("| Predictor |")
        assert lines[1].startswith("|---")
        assert len(lines) == 2 + len(rows)

    def test_em_dash_for_missing(self) -> None:
        rows = [PredictorRow(name="empty")]
        md = render_markdown(rows)
        cells = md.splitlines()[2].split(" | ")
        # Throughput renders "n/a" when missing; the other 5 metric cells render "—".
        assert "n/a" in cells, "throughput should render as n/a when no bench data"
        # Strip trailing " |" from the last cell for clean counting.
        normalized = [c.rstrip(" |") for c in cells]
        assert normalized.count("—") >= 5

    def test_throughput_includes_concurrency_label(self, synth_dirs: tuple[Path, Path]) -> None:
        eval_dir, bench_dir = synth_dirs
        rows = build_comparison(eval_dir=eval_dir, bench_dir=bench_dir)
        md = render_markdown(rows)
        # Should look like "7.95 req/s @ c=16" somewhere in the table.
        assert "req/s @ c=16" in md


class TestRenderJson:
    def test_round_trippable(self, synth_dirs: tuple[Path, Path]) -> None:
        eval_dir, bench_dir = synth_dirs
        rows = build_comparison(eval_dir=eval_dir, bench_dir=bench_dir)
        out = render_json(rows)
        loaded = json.loads(out)
        assert isinstance(loaded, list)
        assert loaded[0]["name"] == "vllm-Qwen"
