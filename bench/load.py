"""Closed-loop concurrency-sweep load generator.

ThreadPoolExecutor wraps the existing sync Predictor protocol — each worker
holds one in-flight request, so `concurrency=N` means up to N requests in
flight at once. Throughput is measured wall-clock; latency is per-request
(unaffected by concurrency).

Output: bench/results/<predictor>_c<NN>.json — one per (predictor, concurrency)
combination. Re-running one level doesn't clobber the others.
"""

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from baselines.base import Predictor

DEFAULT_EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "eval.jsonl"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class BenchResult:
    predictor: str
    concurrency: int
    n_requests: int
    n_success: int
    n_errors: int
    wall_clock_s: float
    throughput_rps: float                          # successful requests / wall_clock_s
    latency_ms: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    """Min/max/mean + p50/p95/p99 (nearest-rank). Mirrors eval/runner.py:latency_stats."""
    if not latencies_ms:
        return {"n": 0}
    s = sorted(latencies_ms)
    n = len(s)

    def pct(p: float) -> float:
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


def load_invoices(eval_path: Path) -> list[dict[str, Any]]:
    if not eval_path.exists():
        raise FileNotFoundError(f"{eval_path} not found. Run `python -m data.build_dataset`.")
    rows: list[dict[str, Any]] = []
    with eval_path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def sample_workload(
    invoices: list[dict[str, Any]], n: int, *, seed: int
) -> list[dict[str, Any]]:
    """Sample n invoices WITH replacement (deterministic by seed). Replacement is
    intentional — at high concurrency we may want to exceed the eval set size."""
    rng = random.Random(seed)
    return [rng.choice(invoices) for _ in range(n)]


def run_bench(
    predictor: Predictor,
    invoices: list[dict[str, Any]],
    *,
    concurrency: int,
    n_requests: int,
    seed: int = 42,
) -> BenchResult:
    """One concurrency level. Closed-loop: ThreadPoolExecutor caps in-flight reqs."""
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if n_requests < 1:
        raise ValueError(f"n_requests must be >= 1, got {n_requests}")

    workload = sample_workload(invoices, n_requests, seed=seed)
    latencies: list[float] = []
    errors: list[str] = []

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [
            ex.submit(predictor.extract, w["invoice_id"], w["input_text"])
            for w in workload
        ]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{predictor.name} c={concurrency}",
            file=sys.stderr,
        ):
            r = fut.result()
            if r.error is not None:
                errors.append(r.error)
            else:
                latencies.append(r.latency_ms)
    wall_clock_s = time.perf_counter() - t_start

    n_success = len(latencies)
    throughput = n_success / wall_clock_s if wall_clock_s > 0 else 0.0

    return BenchResult(
        predictor=predictor.name,
        concurrency=concurrency,
        n_requests=n_requests,
        n_success=n_success,
        n_errors=len(errors),
        wall_clock_s=wall_clock_s,
        throughput_rps=throughput,
        latency_ms=latency_stats(latencies),
        metadata={
            "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "seed": seed,
            "first_5_errors": errors[:5],
        },
    )


def write_result(result: BenchResult, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    safe_predictor = result.predictor.replace("/", "_")
    out_path = results_dir / f"{safe_predictor}_c{result.concurrency:02d}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    return out_path


def run_sweep(
    predictor: Predictor,
    *,
    concurrencies: list[int],
    eval_path: Path = DEFAULT_EVAL_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    n_requests: int = 100,
    seed: int = 42,
) -> list[Path]:
    """Run the bench at each concurrency level, save one JSON per level."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    invoices = load_invoices(eval_path)
    paths: list[Path] = []
    for c in concurrencies:
        print(f"\n>> bench {predictor.name} c={c}, n={n_requests}", file=sys.stderr)
        result = run_bench(
            predictor, invoices, concurrency=c, n_requests=n_requests, seed=seed
        )
        path = write_result(result, results_dir)
        paths.append(path)
        _print_summary(result)
        print(f"   wrote {path}", file=sys.stderr)
    return paths


def _print_summary(r: BenchResult) -> None:
    lat = r.latency_ms
    print(
        f"  c={r.concurrency:>2}  reqs={r.n_requests}  ok={r.n_success}  "
        f"err={r.n_errors}  wall={r.wall_clock_s:.1f}s  thru={r.throughput_rps:.2f} req/s"
    )
    if lat.get("n", 0):
        print(
            f"        lat ms: p50={lat['p50']:.0f}  p95={lat['p95']:.0f}  "
            f"p99={lat['p99']:.0f}  mean={lat['mean']:.0f}"
        )
