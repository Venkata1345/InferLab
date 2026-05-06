"""Pareto frontier: quality (record accuracy) vs cost ($/1k invoices).

Reads eval/results/*.json for accuracy + token counts and bench/results/*.json
for vLLM throughput; combines with eval/cost.py pricing to derive per-config
$/1k. Cost for self-hosted is GPU $/hr / throughput (configurable via
--gpu-dollar-per-hr), matching what report/build.py already does.

Output: report/output/figures/pareto_quality_vs_cost.png
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval" / "results"
BENCH_DIR = ROOT / "bench" / "results"
OUTPUT_DIR = ROOT / "report" / "output" / "figures"

DEFAULT_GPU_DOLLAR_PER_HR = 1.50  # ~A100 spot or L4 mid-tier cloud

# Each entry maps a "display name" → (eval_filename, bench_filename, bench_concurrency,
# is_vllm_self_hosted). bench_filename can be None if we don't have throughput data.
CONFIGS = [
    ("vLLM-vanilla",        "vllm-Qwen-3B-vanilla.json",          "vllm-Qwen-3B-vanilla_c16.json",   16, True),
    ("vLLM-AWQ",            "vllm-Qwen-3B-AWQ.json",              "vllm-Qwen-3B-AWQ_c16.json",       16, True),
    ("vLLM-spec-K4",        "vllm-Qwen-3B-spec-K4-quality.json",  "vllm-Qwen-3B-spec-K4_c01.json",    1, True),
    ("gpt-4o-mini",         "openai-gpt-4o-mini.json",            "openai-gpt-4o-mini_c04.json",      4, False),
    ("gemini-flash-lite",   "gemini-gemini-2.5-flash-lite.json",  "gemini-gemini-2.5-flash-lite_c04.json", 4, False),
]


@dataclass
class Point:
    name: str
    record_acc: float | None
    throughput_rps: float | None
    throughput_c: int | None
    cost_per_1k_usd: float | None
    is_self_hosted: bool
    is_dominated: bool = False
    is_partial_data: bool = False  # True if we substituted from a non-target concurrency


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _record_acc(eval_data: dict | None) -> float | None:
    if not eval_data:
        return None
    return eval_data.get("metrics", {}).get("record_accuracy")


def _throughput(bench_data: dict | None) -> float | None:
    if not bench_data:
        return None
    return bench_data.get("throughput_rps")


def _cost_per_1k(eval_data: dict | None, throughput: float | None,
                 is_self_hosted: bool, gpu_dollar_per_hr: float) -> float | None:
    """API: token cost from eval cost block. vLLM: GPU $/hr / throughput."""
    if is_self_hosted:
        if throughput is None or throughput <= 0:
            return None
        return (gpu_dollar_per_hr / 3600.0 / throughput) * 1000.0
    if not eval_data:
        return None
    return eval_data.get("cost", {}).get("cost_per_1k_invoices_usd")


def collect(gpu_dollar_per_hr: float) -> list[Point]:
    points: list[Point] = []
    for name, eval_fn, bench_fn, c, is_self_hosted in CONFIGS:
        ev = _load_json(EVAL_DIR / eval_fn)
        be = _load_json(BENCH_DIR / bench_fn) if bench_fn else None
        thru = _throughput(be)
        acc = _record_acc(ev)
        cost = _cost_per_1k(ev, thru, is_self_hosted, gpu_dollar_per_hr)
        points.append(
            Point(
                name=name,
                record_acc=acc,
                throughput_rps=thru,
                throughput_c=c,
                cost_per_1k_usd=cost,
                is_self_hosted=is_self_hosted,
                is_partial_data=(c != 16 and is_self_hosted),
            )
        )
    return points


def mark_pareto(points: list[Point]) -> None:
    """A point is Pareto-optimal if no other point has BOTH higher accuracy AND lower cost.

    We mutate in place: set is_dominated=True for non-frontier points.
    """
    for p in points:
        if p.record_acc is None or p.cost_per_1k_usd is None:
            p.is_dominated = True
            continue
        for q in points:
            if q is p or q.record_acc is None or q.cost_per_1k_usd is None:
                continue
            # q dominates p if q has >= accuracy AND <= cost AND strictly better on at least one
            better_or_equal = (q.record_acc >= p.record_acc and q.cost_per_1k_usd <= p.cost_per_1k_usd)
            strictly_better = (q.record_acc > p.record_acc or q.cost_per_1k_usd < p.cost_per_1k_usd)
            if better_or_equal and strictly_better:
                p.is_dominated = True
                break


def plot_pareto(points: list[Point], out_path: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    # Frontier first (so it's behind dots), then dominated
    frontier_x = sorted(
        [(p.cost_per_1k_usd, p.record_acc) for p in points if not p.is_dominated and p.cost_per_1k_usd is not None]
    )
    if len(frontier_x) >= 2:
        ax.plot(
            [c for c, _ in frontier_x],
            [a for _, a in frontier_x],
            "--", color="green", alpha=0.4, linewidth=2, label="Pareto frontier",
        )

    for p in points:
        if p.cost_per_1k_usd is None or p.record_acc is None:
            continue
        if p.is_dominated:
            color, edge = "lightcoral", "darkred"
        else:
            color, edge = "limegreen", "darkgreen"
        ax.scatter(
            p.cost_per_1k_usd, p.record_acc * 100,
            s=180, marker="o", c=color, edgecolors=edge, linewidths=2, zorder=3,
        )
        # Annotate
        annotation = p.name
        if p.is_partial_data:
            annotation += f"\n(c={p.throughput_c})"
        ax.annotate(
            annotation,
            (p.cost_per_1k_usd, p.record_acc * 100),
            textcoords="offset points", xytext=(10, 6),
            fontsize=9,
        )

    ax.set_xlabel("Cost per 1k invoices (USD)")
    ax.set_ylabel("Record-level accuracy (%)")
    ax.set_title(
        "Pareto frontier — record accuracy vs $/1k invoices\n"
        "Green dashed = frontier; red = dominated"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.savefig(out_path, dpi=120, bbox_inches="tight")


def print_summary(points: list[Point]) -> None:
    print("\n=== Pareto frontier (quality vs cost) ===\n")
    print(f"{'config':<22} {'record_acc':>11} {'$/1k':>10} {'thru r/s':>10} {'@ c':>5} {'status':>14}")
    print("-" * 75)
    # Sort by cost ascending for readability
    sorted_points = sorted(
        points,
        key=lambda p: (p.cost_per_1k_usd if p.cost_per_1k_usd is not None else float("inf")),
    )
    for p in sorted_points:
        acc = f"{p.record_acc:.1%}" if p.record_acc is not None else "—"
        cost = f"${p.cost_per_1k_usd:.3f}" if p.cost_per_1k_usd is not None else "—"
        thru = f"{p.throughput_rps:.2f}" if p.throughput_rps is not None else "—"
        c_str = str(p.throughput_c) if p.throughput_c is not None else "—"
        status = "frontier" if not p.is_dominated else "dominated"
        print(f"{p.name:<22} {acc:>11} {cost:>10} {thru:>10} {c_str:>5} {status:>14}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--gpu-dollar-per-hr",
        type=float,
        default=DEFAULT_GPU_DOLLAR_PER_HR,
        help="GPU price for vLLM $/1k calculation (default 1.50).",
    )
    args = ap.parse_args()

    points = collect(gpu_dollar_per_hr=args.gpu_dollar_per_hr)
    mark_pareto(points)
    print_summary(points)

    out_path = OUTPUT_DIR / "pareto_quality_vs_cost.png"
    plot_pareto(points, out_path)
    print(f"\nWrote {out_path}")

    # Save raw points to JSON for downstream consumption
    out_json = OUTPUT_DIR.parent / "pareto.json"
    out_json.write_text(
        json.dumps(
            {"gpu_dollar_per_hr": args.gpu_dollar_per_hr,
             "points": [vars(p) for p in points]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
