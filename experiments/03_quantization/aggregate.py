"""Aggregate quantization experiment: vanilla BF16 vs AWQ INT4.

Reads:
  eval/results/vllm-Qwen-3B-vanilla.json
  eval/results/vllm-Qwen-3B-AWQ.json
  bench/results/vllm-Qwen-3B-vanilla_c{01,16}.json
  bench/results/vllm-Qwen-3B-AWQ_c{01,16}.json

Writes ./results.json with the eval + bench deltas in one place. Prints a
human-readable comparison table.
"""

import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parent.parent
EVAL_DIR = ROOT / "eval" / "results"
BENCH_DIR = ROOT / "bench" / "results"

CONFIGS = ["vanilla", "AWQ"]
CONCURRENCIES = [1, 16]


def load_eval(tag: str) -> dict | None:
    path = EVAL_DIR / f"vllm-Qwen-3B-{tag}.json"
    if not path.exists():
        print(f"  missing: {path}", file=sys.stderr)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_bench(tag: str, c: int) -> dict | None:
    path = BENCH_DIR / f"vllm-Qwen-3B-{tag}_c{c:02d}.json"
    if not path.exists():
        print(f"  missing: {path}", file=sys.stderr)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect() -> dict:
    out: dict = {}
    for tag in CONFIGS:
        ev = load_eval(tag)
        eval_summary = {}
        if ev:
            m = ev["metrics"]
            eval_summary = {
                "schema_validity": m["schema_validity"],
                "field_accuracy_macro": m["field_accuracy_macro"],
                "record_accuracy": m["record_accuracy"],
                "field_accuracy": {
                    f: stats["correct"] / stats["total"] if stats["total"] else None
                    for f, stats in m.get("field_accuracy", {}).items()
                },
                "p50_ms": ev["latency_ms"].get("p50") if ev.get("latency_ms", {}).get("n", 0) else None,
                "p99_ms": ev["latency_ms"].get("p99") if ev.get("latency_ms", {}).get("n", 0) else None,
            }
        bench_summary: dict = {}
        for c in CONCURRENCIES:
            bd = load_bench(tag, c)
            if bd:
                bench_summary[f"c{c}"] = {
                    "throughput_rps": bd["throughput_rps"],
                    "p50_ms": bd["latency_ms"]["p50"],
                    "p99_ms": bd["latency_ms"]["p99"],
                    "n_errors": bd["n_errors"],
                }
        out[tag] = {"eval": eval_summary, "bench": bench_summary}
    return out


def _delta_pct(new: float | None, old: float | None) -> str:
    if new is None or old is None or old == 0:
        return "—"
    pct = (new - old) / old * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _delta_pp(new: float | None, old: float | None) -> str:
    """Percentage-point delta for accuracy-style values already in [0, 1]."""
    if new is None or old is None:
        return "—"
    return f"{(new - old) * 100:+.1f}pp"


def print_summary(data: dict) -> None:
    v = data.get("vanilla", {}).get("eval", {})
    a = data.get("AWQ", {}).get("eval", {})
    vb = data.get("vanilla", {}).get("bench", {})
    ab = data.get("AWQ", {}).get("bench", {})

    print("\n=== Quantization comparison (BF16 vanilla -> AWQ INT4) ===\n")

    print(f"{'metric':<22} {'vanilla':>12} {'AWQ':>12} {'delta':>12}")
    print("-" * 60)

    rows = [
        ("schema_validity",      v.get("schema_validity"),      a.get("schema_validity"),      _delta_pp),
        ("field_acc_macro",      v.get("field_accuracy_macro"), a.get("field_accuracy_macro"), _delta_pp),
        ("record_accuracy",      v.get("record_accuracy"),      a.get("record_accuracy"),      _delta_pp),
        ("eval p50 latency ms",  v.get("p50_ms"),               a.get("p50_ms"),               _delta_pct),
        ("eval p99 latency ms",  v.get("p99_ms"),               a.get("p99_ms"),               _delta_pct),
    ]
    for label, vv, aa, deltafn in rows:
        if vv is None and aa is None:
            continue
        v_str = f"{vv:.4f}" if isinstance(vv, float) else "—" if vv is None else str(vv)
        a_str = f"{aa:.4f}" if isinstance(aa, float) else "—" if aa is None else str(aa)
        print(f"{label:<22} {v_str:>12} {a_str:>12} {deltafn(aa, vv):>12}")

    def _fmt(v: float | None, fmt: str) -> str:
        return format(v, fmt) if v is not None else "—"

    print("\nBench:")
    for c in CONCURRENCIES:
        v_t = vb.get(f"c{c}", {}).get("throughput_rps")
        a_t = ab.get(f"c{c}", {}).get("throughput_rps")
        v_p50 = vb.get(f"c{c}", {}).get("p50_ms")
        a_p50 = ab.get(f"c{c}", {}).get("p50_ms")
        if v_t is None and a_t is None:
            continue
        print(
            f"  c={c}  throughput  vanilla={_fmt(v_t, '.2f')} r/s  "
            f"AWQ={_fmt(a_t, '.2f')} r/s  delta={_delta_pct(a_t, v_t)}"
        )
        print(
            f"        p50 latency vanilla={_fmt(v_p50, '.0f')} ms  "
            f"AWQ={_fmt(a_p50, '.0f')} ms  delta={_delta_pct(a_p50, v_p50)}"
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
