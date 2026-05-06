"""Stitch eval/results/*.json + bench/results/*.json into the comparison table.

Output: report/output/comparison.{md,json}. The Markdown drops directly into
README; the JSON is the canonical structured form for downstream tools.

Cost handling:
  - API predictors: $/1k taken from the eval result's `cost.cost_per_1k_invoices_usd`
    (token usage * dated pricing in eval/cost.py).
  - Self-hosted vLLM: $/1k = GPU $/hr / (throughput * 3600) * 1000 — needs a
    GPU price. Default is $0.50/hr (L4 on Colab Pro); override via --gpu-dollar-per-hr.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EVAL_RESULTS_DIR = ROOT / "eval" / "results"
BENCH_RESULTS_DIR = ROOT / "bench" / "results"
OUTPUT_DIR = ROOT / "report" / "output"

DEFAULT_GPU_DOLLAR_PER_HR = 0.50  # L4 on Colab Pro (rough; override per environment)
TARGET_CONCURRENCY = 16  # The headline "Throughput @16" column


@dataclass
class PredictorRow:
    """One row of the comparison table. All metrics are optional (Nones rendered as '—')."""

    name: str
    schema_validity: float | None = None
    field_accuracy_macro: float | None = None
    record_accuracy: float | None = None
    p50_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    throughput_rps_at_target: float | None = None
    target_concurrency: int = TARGET_CONCURRENCY
    cost_per_1k_usd: float | None = None
    cost_basis: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# ---------- loaders ----------


def load_eval_results(eval_dir: Path) -> dict[str, dict[str, Any]]:
    """Read every *.json under eval_dir → {predictor.name: full_eval_dict}."""
    out: dict[str, dict[str, Any]] = {}
    if not eval_dir.is_dir():
        return out
    for path in sorted(eval_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        name = data.get("predictor")
        if name:
            out[name] = data
    return out


def load_bench_results(bench_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Read every *.json under bench_dir → {predictor.name: {concurrency: bench_dict}}."""
    out: dict[str, dict[int, dict[str, Any]]] = {}
    if not bench_dir.is_dir():
        return out
    for path in sorted(bench_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        name = data.get("predictor")
        c = data.get("concurrency")
        if name and isinstance(c, int):
            out.setdefault(name, {})[c] = data
    return out


# ---------- row builder ----------


def build_row(
    name: str,
    eval_data: dict[str, Any] | None,
    bench_data: dict[int, dict[str, Any]] | None,
    *,
    gpu_dollar_per_hr: float,
    target_c: int,
) -> PredictorRow:
    row = PredictorRow(name=name)

    if eval_data:
        m = eval_data.get("metrics", {})
        row.schema_validity = m.get("schema_validity")
        row.field_accuracy_macro = m.get("field_accuracy_macro")
        row.record_accuracy = m.get("record_accuracy")
        lat = eval_data.get("latency_ms", {})
        if lat.get("n", 0) > 0:
            row.p50_latency_ms = lat.get("p50")
            row.p99_latency_ms = lat.get("p99")
        row.details["field_accuracy"] = m.get("field_accuracy", {})
        row.details["hallucination_rate"] = m.get("hallucination_rate")
        row.details["n_eval_records"] = m.get("n_records")

    if bench_data:
        # Prefer exact target_c; otherwise the closest available level.
        chosen_c = (
            target_c
            if target_c in bench_data
            else min(bench_data.keys(), key=lambda c: abs(c - target_c))
        )
        row.target_concurrency = chosen_c
        row.throughput_rps_at_target = bench_data[chosen_c].get("throughput_rps")
        row.details["bench_concurrencies"] = sorted(bench_data.keys())

    # Cost: self-hosted vs API
    if name.startswith("vllm-"):
        if row.throughput_rps_at_target and row.throughput_rps_at_target > 0:
            row.cost_per_1k_usd = gpu_dollar_per_hr / 3600.0 / row.throughput_rps_at_target * 1000.0
            row.cost_basis = (
                f"GPU ${gpu_dollar_per_hr:.2f}/hr / "
                f"{row.throughput_rps_at_target:.2f} req/s @ c={row.target_concurrency}"
            )
    else:
        if eval_data:
            cost = eval_data.get("cost", {})
            row.cost_per_1k_usd = cost.get("cost_per_1k_invoices_usd")
            pricing = cost.get("pricing", {})
            if pricing:
                row.cost_basis = f"token cost @ {pricing.get('as_of', 'unknown')}"

    return row


def build_comparison(
    *,
    eval_dir: Path = EVAL_RESULTS_DIR,
    bench_dir: Path = BENCH_RESULTS_DIR,
    gpu_dollar_per_hr: float = DEFAULT_GPU_DOLLAR_PER_HR,
    target_c: int = TARGET_CONCURRENCY,
) -> list[PredictorRow]:
    eval_results = load_eval_results(eval_dir)
    bench_results = load_bench_results(bench_dir)
    predictors = set(eval_results) | set(bench_results)
    # Order: vLLM first (the headline), then alphabetical.
    ordered = sorted(predictors, key=lambda n: (not n.startswith("vllm-"), n))
    return [
        build_row(
            n,
            eval_results.get(n),
            bench_results.get(n),
            gpu_dollar_per_hr=gpu_dollar_per_hr,
            target_c=target_c,
        )
        for n in ordered
    ]


# ---------- renderers ----------


def _fmt_pct(x: float | None) -> str:
    return f"{x:.1%}" if x is not None else "—"


def _fmt_sec(ms: float | None) -> str:
    return f"{ms / 1000:.2f} s" if ms is not None else "—"


def _fmt_rps(r: PredictorRow) -> str:
    if r.throughput_rps_at_target is None:
        return "n/a"
    return f"{r.throughput_rps_at_target:.2f} req/s @ c={r.target_concurrency}"


def _fmt_cost(x: float | None) -> str:
    if x is None:
        return "—"
    return f"${x:.3f}"


def render_markdown(rows: list[PredictorRow]) -> str:
    headers = [
        "Predictor",
        "Schema Valid",
        "Field Acc",
        "Record Acc",
        "p50 lat",
        "p99 lat",
        "Throughput",
        "$/1k",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        cells = [
            f"`{r.name}`",
            _fmt_pct(r.schema_validity),
            _fmt_pct(r.field_accuracy_macro),
            _fmt_pct(r.record_accuracy),
            _fmt_sec(r.p50_latency_ms),
            _fmt_sec(r.p99_latency_ms),
            _fmt_rps(r),
            _fmt_cost(r.cost_per_1k_usd),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_json(rows: list[PredictorRow]) -> str:
    return json.dumps([asdict(r) for r in rows], indent=2)


# ---------- CLI ----------


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, default=EVAL_RESULTS_DIR)
    ap.add_argument("--bench-dir", type=Path, default=BENCH_RESULTS_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument(
        "--gpu-dollar-per-hr",
        type=float,
        default=DEFAULT_GPU_DOLLAR_PER_HR,
        help="GPU price for self-hosted $/1k calculation (default: L4 @ $0.50/hr)",
    )
    ap.add_argument("--target-concurrency", type=int, default=TARGET_CONCURRENCY)
    args = ap.parse_args()

    rows = build_comparison(
        eval_dir=args.eval_dir,
        bench_dir=args.bench_dir,
        gpu_dollar_per_hr=args.gpu_dollar_per_hr,
        target_c=args.target_concurrency,
    )

    md = render_markdown(rows)
    print(md)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    md_path = args.output_dir / "comparison.md"
    json_path = args.output_dir / "comparison.json"
    md_path.write_text(md + "\n", encoding="utf-8")
    json_path.write_text(render_json(rows), encoding="utf-8")
    print(f"\nWrote {md_path}", file=sys.stderr)
    print(f"Wrote {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
