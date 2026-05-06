"""Read results.json from run.py and produce two plots + a printed summary.

Plot 1: Peak KV cache usage % vs avg input tokens
Plot 2: Estimated max concurrency (before cache OOM) vs avg input tokens

The "max concurrency" estimate assumes cache usage scales linearly with
concurrent in-flight requests at a given input length:
    max_c_estimate = current_c * (target_cache_pct / observed_cache_pct)
where target_cache_pct = 0.95 (the practical OOM threshold).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results.json"

OOM_TARGET_CACHE_PCT = 0.95  # vLLM crashes well before 100% — 95% is a safer ceiling


def load() -> dict:
    if not RESULTS.exists():
        print(f"{RESULTS} not found — run experiments/02_kv_cache/run.py first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def estimate_max_concurrency(observed_cache: float, current_c: int) -> float | None:
    if observed_cache is None or observed_cache <= 0:
        return None
    return current_c * (OOM_TARGET_CACHE_PCT / observed_cache)


def plot_cache_vs_tokens(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [r["avg_input_tokens"] for r in rows]
    ys = [r["peak_cache_usage_perc"] * 100 if r["peak_cache_usage_perc"] is not None else 0 for r in rows]
    ax.plot(xs, ys, marker="o", linewidth=2, markersize=10, color="C0")
    for r in rows:
        if r["peak_cache_usage_perc"] is not None:
            ax.annotate(
                f"{r['peak_cache_usage_perc']:.1%}",
                (r["avg_input_tokens"], r["peak_cache_usage_perc"] * 100),
                textcoords="offset points", xytext=(8, 6),
            )
    ax.set_xlabel("Avg input tokens per request")
    ax.set_ylabel(f"Peak KV cache usage at c={rows[0]['concurrency']} (%)")
    ax.set_title("KV cache pressure vs input length")
    ax.grid(True, alpha=0.3)
    out = EXP_DIR / "kv_cache_usage_vs_input_length.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    return out


def plot_max_concurrency(rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [r["avg_input_tokens"] for r in rows]
    ys = [
        estimate_max_concurrency(r["peak_cache_usage_perc"], r["concurrency"]) or 0
        for r in rows
    ]
    ax.plot(xs, ys, marker="o", linewidth=2, markersize=10, color="C1")
    for x, y in zip(xs, ys):
        ax.annotate(f"~{y:.0f}", (x, y), textcoords="offset points", xytext=(8, 6))
    ax.set_xlabel("Avg input tokens per request")
    ax.set_ylabel("Estimated max concurrency before cache OOM")
    ax.set_title(
        f"Concurrent capacity vs input length\n"
        f"(extrapolated assuming linear cache scaling, OOM threshold {OOM_TARGET_CACHE_PCT:.0%})"
    )
    ax.grid(True, alpha=0.3)
    out = EXP_DIR / "max_concurrency_vs_input_length.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    return out


def print_summary(rows: list[dict]) -> None:
    print("\n=== KV cache instrumentation summary ===\n")
    print(
        f"{'target_chars':<13} {'avg_in_tok':<12} {'peak_cache':<12} {'est_max_c':<10} {'ok':<8}"
    )
    for r in rows:
        cache = r["peak_cache_usage_perc"]
        cache_str = f"{cache:.1%}" if cache is not None else "—"
        max_c = estimate_max_concurrency(cache, r["concurrency"])
        max_c_str = f"~{max_c:.0f}" if max_c is not None else "—"
        ok_str = f"{r['n_success']}/{r['n_requests']}"
        print(
            f"{r['target_chars']:<13} {r['avg_input_tokens']:<12.0f} "
            f"{cache_str:<12} {max_c_str:<10} {ok_str:<8}"
        )


def main() -> int:
    data = load()
    rows = data["runs"]
    if not rows:
        print("No runs in results.json", file=sys.stderr)
        return 1
    p1 = plot_cache_vs_tokens(rows)
    p2 = plot_max_concurrency(rows)
    print_summary(rows)
    print(f"\nWrote {p1}")
    print(f"Wrote {p2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
