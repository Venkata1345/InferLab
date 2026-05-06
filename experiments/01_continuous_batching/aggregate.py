"""Aggregate continuous-batching experiment results.

Reads bench JSONs from ./results/, organizes by (config, concurrency),
writes results.json, and produces throughput_vs_concurrency.png.

No project imports — only stdlib + matplotlib. Safe to run from anywhere.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"

CONFIGS = ["serial", "vanilla"]
CONFIG_LABELS = {
    "serial": "Serial (max_num_seqs=1)",
    "vanilla": "Continuous batching (max_num_seqs=256)",
}

# Filename: vllm-Qwen-3B-{config}_c{NN}.json
FILENAME_RE = re.compile(r"vllm-Qwen-3B-(?P<cfg>\w+)_c(?P<c>\d+)\.json")


def load_results() -> dict[str, list[dict]]:
    """Returns {config: [{concurrency, throughput_rps, p50_ms, p99_ms, n_errors}, ...]}."""
    by_cfg: dict[str, list[dict]] = {cfg: [] for cfg in CONFIGS}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        m = FILENAME_RE.match(path.name)
        if not m:
            continue
        cfg = m.group("cfg")
        if cfg not in by_cfg:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        by_cfg[cfg].append(
            {
                "concurrency": int(m.group("c")),
                "throughput_rps": data["throughput_rps"],
                "p50_ms": data["latency_ms"]["p50"],
                "p99_ms": data["latency_ms"]["p99"],
                "n_success": data["n_success"],
                "n_errors": data["n_errors"],
                "wall_clock_s": data["wall_clock_s"],
            }
        )
    for cfg in by_cfg:
        by_cfg[cfg].sort(key=lambda r: r["concurrency"])
    return by_cfg


def speedup_at(by_cfg: dict[str, list[dict]], concurrency: int) -> float | None:
    serial = next(
        (r["throughput_rps"] for r in by_cfg.get("serial", []) if r["concurrency"] == concurrency),
        None,
    )
    vanilla = next(
        (r["throughput_rps"] for r in by_cfg.get("vanilla", []) if r["concurrency"] == concurrency),
        None,
    )
    if serial and vanilla and serial > 0:
        return vanilla / serial
    return None


def write_results_json(by_cfg: dict[str, list[dict]]) -> Path:
    speedups = {f"c{c}": speedup_at(by_cfg, c) for c in (1, 4, 8, 16, 32)}
    out = {
        "experiment": "01_continuous_batching",
        "configs": by_cfg,
        "speedup_vanilla_over_serial": speedups,
    }
    path = EXP_DIR / "results.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def plot_throughput(by_cfg: dict[str, list[dict]]) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for cfg in CONFIGS:
        rows = by_cfg.get(cfg, [])
        if not rows:
            continue
        xs = [r["concurrency"] for r in rows]
        ys = [r["throughput_rps"] for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, markersize=8, label=CONFIG_LABELS.get(cfg, cfg))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16, 32])
    ax.set_xticklabels(["1", "4", "8", "16", "32"])
    ax.set_xlabel("Concurrency (closed-loop, in-flight requests)")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Continuous batching impact: throughput vs concurrency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = EXP_DIR / "throughput_vs_concurrency.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def print_summary(by_cfg: dict[str, list[dict]]) -> None:
    print("\n=== Summary ===\n")
    print(f"{'c':<6}{'serial r/s':<14}{'vanilla r/s':<16}{'speedup':<10}")
    for c in (1, 4, 8, 16, 32):
        s = next(
            (r["throughput_rps"] for r in by_cfg.get("serial", []) if r["concurrency"] == c),
            None,
        )
        v = next(
            (r["throughput_rps"] for r in by_cfg.get("vanilla", []) if r["concurrency"] == c),
            None,
        )
        ratio = (v / s) if (s and v and s > 0) else None
        s_str = f"{s:.2f}" if s is not None else "—"
        v_str = f"{v:.2f}" if v is not None else "—"
        r_str = f"{ratio:.2f}x" if ratio is not None else "—"
        print(f"{c:<6}{s_str:<14}{v_str:<16}{r_str:<10}")


def main() -> int:
    if not RESULTS_DIR.is_dir() or not any(RESULTS_DIR.glob("*.json")):
        print(
            f"No bench results in {RESULTS_DIR}. Run experiments/01_continuous_batching/run.sh first.",
            file=sys.stderr,
        )
        return 1
    by_cfg = load_results()
    json_path = write_results_json(by_cfg)
    png_path = plot_throughput(by_cfg)
    print_summary(by_cfg)
    print(f"\nWrote {json_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
