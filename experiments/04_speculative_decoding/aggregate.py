"""Aggregate speculative decoding experiment.

Reads:
  bench/results/vllm-Qwen-3B-spec-K{2,4,8}_c01.json   - per-K bench at c=1
  bench/results/vllm-Qwen-3B-vanilla_c01.json         - vanilla baseline (from step 3)
  eval/results/vllm-Qwen-3B-spec-K4-quality.json      - 25-record quality verify
  experiments/04_speculative_decoding/metrics-spec-K{2,4,8}.txt - /metrics snapshots

Writes results.json + prints a summary table with per-K speedup vs vanilla
and acceptance rate (parsed from the metrics snapshots).
"""

import json
import re
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parent.parent
BENCH_DIR = ROOT / "bench" / "results"
EVAL_DIR = ROOT / "eval" / "results"

K_VALUES = [2, 4, 8]
VANILLA_BENCH = BENCH_DIR / "vllm-Qwen-3B-vanilla_c01.json"
QUALITY_EVAL = EVAL_DIR / "vllm-Qwen-3B-spec-K4-quality.json"

# Acceptance metric names — vLLM uses different ones across versions.
_ACCEPT_PATTERNS = [
    re.compile(r"^vllm:spec_decode_num_accepted_tokens(?:_total)?\s+([\d.\-eE]+)", re.MULTILINE),
    re.compile(r"^vllm:spec_decode_accepted_tokens(?:_total)?\s+([\d.\-eE]+)", re.MULTILINE),
]
_PROPOSED_PATTERNS = [
    re.compile(r"^vllm:spec_decode_num_draft_tokens(?:_total)?\s+([\d.\-eE]+)", re.MULTILINE),
    re.compile(r"^vllm:spec_decode_num_emitted_tokens(?:_total)?\s+([\d.\-eE]+)", re.MULTILINE),
    re.compile(r"^vllm:spec_decode_draft_tokens(?:_total)?\s+([\d.\-eE]+)", re.MULTILINE),
]
# Generic spec-decode metrics (capture anything starting with vllm:spec_decode for inspection)
_GENERIC_RE = re.compile(r"^(vllm:spec_decode\w*)\s+([\d.\-eE]+)", re.MULTILINE)


def first_match(text: str, patterns: list[re.Pattern]) -> float | None:
    for p in patterns:
        m = p.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {"raw_path": str(path), "exists": False}
    text = path.read_text(encoding="utf-8")
    accepted = first_match(text, _ACCEPT_PATTERNS)
    proposed = first_match(text, _PROPOSED_PATTERNS)
    accept_rate = (accepted / proposed) if (accepted and proposed and proposed > 0) else None
    all_spec = dict(_GENERIC_RE.findall(text))
    return {
        "exists": True,
        "raw_path": str(path),
        "accepted_tokens": accepted,
        "proposed_tokens": proposed,
        "acceptance_rate": accept_rate,
        "all_spec_metrics": {k: float(v) for k, v in all_spec.items()},
    }


def load_bench(name: str) -> dict | None:
    path = BENCH_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_eval(name: str) -> dict | None:
    path = EVAL_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect() -> dict:
    out: dict = {"K": {}}

    vanilla = load_bench("vllm-Qwen-3B-vanilla_c01.json")
    out["vanilla_c1"] = (
        {
            "throughput_rps": vanilla["throughput_rps"],
            "p50_ms": vanilla["latency_ms"]["p50"],
            "p99_ms": vanilla["latency_ms"]["p99"],
            "n_success": vanilla["n_success"],
        }
        if vanilla
        else None
    )

    for k in K_VALUES:
        bench = load_bench(f"vllm-Qwen-3B-spec-K{k}_c01.json")
        metrics = load_metrics(EXP_DIR / f"metrics-spec-K{k}.txt")
        bench_summary = None
        if bench:
            lat = bench.get("latency_ms", {})
            bench_summary = {
                "throughput_rps": bench.get("throughput_rps"),
                "p50_ms": lat.get("p50"),
                "p99_ms": lat.get("p99"),
                "n_success": bench.get("n_success"),
                "n_errors": bench.get("n_errors"),
            }
        out["K"][str(k)] = {"bench_c1": bench_summary, "metrics": metrics}

    quality = load_eval("vllm-Qwen-3B-spec-K4-quality.json")
    if quality:
        m = quality["metrics"]
        out["quality_verify_K4"] = {
            "n_records": m["n_records"],
            "schema_validity": m["schema_validity"],
            "field_accuracy_macro": m["field_accuracy_macro"],
            "record_accuracy": m["record_accuracy"],
        }

    return out


def _delta_pct(new: float | None, old: float | None) -> str:
    if new is None or old is None or old == 0:
        return "—"
    pct = (new - old) / old * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _fmt(v: float | None, spec: str, missing: str = "—") -> str:
    return format(v, spec) if v is not None else missing


def print_summary(data: dict) -> None:
    v = data.get("vanilla_c1")
    print("\n=== Speculative decoding @ c=1 ===\n")
    if v is None:
        print(
            "vanilla baseline missing — re-run experiment 03 to recreate "
            "bench/results/vllm-Qwen-3B-vanilla_c01.json",
            file=sys.stderr,
        )
        v_p50 = v_thru = None
    else:
        v_p50 = v["p50_ms"]
        v_thru = v["throughput_rps"]
        print(
            f"vanilla baseline: throughput={v_thru:.2f} r/s  "
            f"p50={v_p50:.0f} ms  p99={v['p99_ms']:.0f} ms\n"
        )

    print(f"{'K':<4} {'p50 ms':>9} {'p99 ms':>9} {'thru r/s':>10} {'p50 Δ':>10} {'thru Δ':>10} {'accept':>10}")
    print("-" * 70)
    for k in K_VALUES:
        entry = data["K"].get(str(k), {})
        b = entry.get("bench_c1") or {}
        m = entry.get("metrics") or {}
        p50 = b.get("p50_ms")
        p99 = b.get("p99_ms")
        thru = b.get("throughput_rps")
        ar = m.get("acceptance_rate")
        print(
            f"{k:<4} "
            f"{_fmt(p50, '.0f'):>9} "
            f"{_fmt(p99, '.0f'):>9} "
            f"{_fmt(thru, '.2f'):>10} "
            f"{_delta_pct(p50, v_p50):>10} "
            f"{_delta_pct(thru, v_thru):>10} "
            f"{_fmt(ar, '.1%'):>10}"
        )

    q = data.get("quality_verify_K4")
    if q:
        print(
            f"\nQuality verify (K=4, n={q['n_records']}): "
            f"schema_valid={q['schema_validity']:.1%}  "
            f"field_macro={q['field_accuracy_macro']:.1%}  "
            f"record_acc={q['record_accuracy']:.1%}"
        )
        print(
            "(Spec dec is mathematically exact — these should match vanilla "
            "eval on the same first-25 records.)"
        )


def main() -> int:
    data = collect()
    out_path = EXP_DIR / "results.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print_summary(data)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
