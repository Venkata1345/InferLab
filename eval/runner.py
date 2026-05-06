"""Run a Predictor against eval.jsonl: predictions, latencies, metrics, JSON output.

Sequential — one request at a time. This keeps per-request latency clean
(no contention between concurrent requests) and avoids hitting frontier-API
rate limits on a one-shot eval. Concurrency lives in bench/load.py instead.

Output: eval/results/<predictor.name>.json with per-record records + aggregates.
"""

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from baselines.base import Predictor
from eval.cost import lookup as lookup_pricing
from eval.metrics import AggregateMetrics, aggregate, evaluate_record

DEFAULT_EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "eval.jsonl"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Min/max/mean + p50/p95/p99 over the per-call latencies. Uses nearest-rank percentile."""
    if not latencies_ms:
        return {"n": 0}
    s = sorted(latencies_ms)
    n = len(s)

    def pct(p: float) -> float:
        # Nearest-rank: ceil(p * n) - 1, clamped.
        idx = max(0, min(n - 1, int(p * n + 0.999999) - 1))
        return s[idx]

    return {
        "n": n,
        "min": s[0],
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": s[-1],
        "mean": sum(s) / n,
    }


def _metrics_to_dict(am: AggregateMetrics) -> dict[str, Any]:
    """asdict, plus inject the .accuracy property since asdict drops it."""
    d = asdict(am)
    for fname, stats in d.get("field_accuracy", {}).items():
        total = stats["total"]
        stats["accuracy"] = stats["correct"] / total if total else 0.0
    return d


def _load_eval(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m data.build_dataset` first."
        )
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def run_eval(
    predictor: Predictor,
    *,
    eval_path: Path = DEFAULT_EVAL_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    limit: int | None = None,
) -> Path:
    """Run `predictor` against eval set, write a results JSON, return its path."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    examples = _load_eval(eval_path, limit)

    record_results = []
    raw_records = []

    for ex in tqdm(examples, desc=predictor.name, file=sys.stderr):
        result = predictor.extract(ex["invoice_id"], ex["input_text"])
        rr = evaluate_record(
            invoice_id=ex["invoice_id"],
            prediction=result.prediction,
            expected=ex["expected_json"],
            ocr_text=ex["input_text"],
            raw_output=result.raw_output,
        )
        record_results.append(rr)
        raw_records.append(
            {
                "invoice_id": ex["invoice_id"],
                "prediction": result.prediction,
                "expected": ex["expected_json"],
                "raw_output": result.raw_output,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "error": result.error,
                "schema_valid": rr.schema_valid,
                "field_matches": rr.field_matches,
                "fields_predicted": rr.fields_predicted,
                "hallucinated_fields": rr.hallucinated_fields,
                "fully_correct": rr.fully_correct,
            }
        )

    am = aggregate(record_results)
    succ = [r for r in raw_records if r["error"] is None]
    lat_stats = latency_stats([r["latency_ms"] for r in succ])
    n_errors = len(raw_records) - len(succ)
    total_in = sum((r["input_tokens"] or 0) for r in raw_records)
    total_out = sum((r["output_tokens"] or 0) for r in raw_records)
    n = len(raw_records)

    pricing = lookup_pricing(predictor.name)
    cost_block: dict[str, Any] = {
        "input_tokens_total": total_in,
        "output_tokens_total": total_out,
        "input_tokens_per_record_mean": total_in / n if n else 0.0,
        "output_tokens_per_record_mean": total_out / n if n else 0.0,
    }
    if pricing is not None:
        cost_block["pricing"] = asdict(pricing)
        cost_block["cost_per_1k_invoices_usd"] = pricing.cost_per_1k_invoices(
            total_in, total_out, n
        )
        cost_block["total_cost_usd"] = pricing.cost_for(total_in, total_out)

    output = {
        "predictor": predictor.name,
        "n_records": n,
        "n_errors": n_errors,
        "metrics": _metrics_to_dict(am),
        "latency_ms": lat_stats,
        "cost": cost_block,
        "run_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "eval_path": str(eval_path),
            "limit": limit,
        },
        "records": raw_records,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{predictor.name}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    _print_summary(predictor.name, am, lat_stats, n_errors, n, cost_block)
    print(f"\nWrote {out_path}", file=sys.stderr)
    return out_path


def _print_summary(
    name: str,
    am: AggregateMetrics,
    lat: dict[str, float],
    n_errors: int,
    n: int,
    cost_block: dict[str, Any],
) -> None:
    print(f"\n=== {name} ===")
    print(f"  records:           {n} ({n_errors} errors)")
    print(f"  schema validity:   {am.schema_validity:.1%}")
    print(f"  record accuracy:   {am.record_accuracy:.1%}")
    print(f"  field acc (macro): {am.field_accuracy_macro:.1%}")
    for fname, stats in am.field_accuracy.items():
        print(f"    - {fname:<16} {stats.correct}/{stats.total} ({stats.accuracy:.1%})")
    print(f"  hallucination:     {am.n_hallucinations}/{am.n_predictions} ({am.hallucination_rate:.1%})")
    if lat.get("n", 0):
        print(
            f"  latency (ms):      p50={lat['p50']:.0f}  p95={lat['p95']:.0f}  "
            f"p99={lat['p99']:.0f}  mean={lat['mean']:.0f}"
        )
    if "cost_per_1k_invoices_usd" in cost_block:
        print(
            f"  $ per 1k invoices: ${cost_block['cost_per_1k_invoices_usd']:.3f}  "
            f"(total ${cost_block['total_cost_usd']:.3f}, "
            f"pricing as of {cost_block['pricing']['as_of']})"
        )
