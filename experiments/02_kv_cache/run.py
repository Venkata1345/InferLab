"""Step 2: KV cache instrumentation.

Sends batches of requests at controlled input lengths to a running vLLM
server (assumed up at http://localhost:8000), scrapes /metrics in parallel,
records peak KV-cache usage per length. No vLLM restarts — same server
serves all three input lengths.

Run this AFTER starting vLLM with the vanilla config:
    bash service/configs/vanilla.sh > vllm.log 2>&1 &
    sleep 60
    python experiments/02_kv_cache/run.py
"""

# Make project modules importable (this dir name starts with a digit, so we can't
# treat experiments/02_kv_cache as a package — manually add the project root).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from openai import OpenAI

from service.prompt import build_messages
from service.schema import INVOICE_JSON_SCHEMA

# Local-dir import (same dir as this script — works without package init).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_inputs import base_invoices, make_input  # noqa: E402

EXP_DIR = Path(__file__).resolve().parent
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_METRICS_URL = "http://localhost:8000/metrics"
MODEL = "Qwen/Qwen2.5-3B-Instruct"

INPUT_LENGTHS_CHARS = [512, 1024, 2048]
N_REQUESTS_PER_LENGTH = 50
CONCURRENCY = 16

# vLLM exposes both names depending on version; we accept either.
CACHE_METRIC_NAMES = (
    "vllm:gpu_cache_usage_perc",
    "vllm:kv_cache_usage_perc",
)
WANTED_METRICS = set(CACHE_METRIC_NAMES) | {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
}

_METRIC_LINE = re.compile(r"^(?P<name>[a-z_:]+)(?:\{[^}]*\})?\s+(?P<value>[\d.\-eE]+)")


def parse_metrics(prom_text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in prom_text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m or m.group("name") not in WANTED_METRICS:
            continue
        try:
            # Take the LAST value if the metric appears multiple times (different labels).
            out[m.group("name")] = float(m.group("value"))
        except ValueError:
            pass
    return out


def scrape_metrics_loop(samples: list[dict], stop: threading.Event, interval: float = 0.25) -> None:
    with httpx.Client(timeout=5.0) as c:
        while not stop.is_set():
            try:
                r = c.get(VLLM_METRICS_URL)
                if r.status_code == 200:
                    parsed = parse_metrics(r.text)
                    parsed["t"] = time.perf_counter()
                    samples.append(parsed)
            except httpx.HTTPError:
                pass  # transient OK
            time.sleep(interval)


def send_one(client: OpenAI, text: str) -> dict:
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=build_messages(text),
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "Invoice", "schema": INVOICE_JSON_SCHEMA},
            },
        )
        return {
            "latency_s": time.perf_counter() - t0,
            "input_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "output_tokens": resp.usage.completion_tokens if resp.usage else None,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "latency_s": time.perf_counter() - t0,
            "input_tokens": None,
            "output_tokens": None,
            "error": f"{type(e).__name__}: {e}",
        }


def _peak_cache(samples: list[dict]) -> float | None:
    vals = []
    for s in samples:
        for name in CACHE_METRIC_NAMES:
            if name in s:
                vals.append(s[name])
                break
    return max(vals) if vals else None


def run_at_length(
    client: OpenAI,
    target_chars: int,
    base_texts: list[str],
) -> dict:
    inputs = [make_input(target_chars, base_texts, idx=i) for i in range(N_REQUESTS_PER_LENGTH)]
    samples: list[dict] = []
    stop = threading.Event()
    scraper = threading.Thread(
        target=scrape_metrics_loop, args=(samples, stop), daemon=True
    )
    scraper.start()

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        results = list(ex.map(lambda t: send_one(client, t), inputs))
    wall_clock = time.perf_counter() - t_start

    stop.set()
    scraper.join(timeout=2)

    successes = [r for r in results if r["error"] is None]
    avg_input_tokens = (
        sum(r["input_tokens"] for r in successes) / len(successes) if successes else 0.0
    )
    avg_output_tokens = (
        sum(r["output_tokens"] for r in successes) / len(successes) if successes else 0.0
    )
    peak_cache = _peak_cache(samples)
    n_metric_samples = len(samples)
    first_error = next((r["error"] for r in results if r["error"] is not None), None)

    return {
        "target_chars": target_chars,
        "n_requests": N_REQUESTS_PER_LENGTH,
        "concurrency": CONCURRENCY,
        "wall_clock_s": wall_clock,
        "n_success": len(successes),
        "n_errors": len(results) - len(successes),
        "first_error": first_error,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "peak_cache_usage_perc": peak_cache,
        "n_metric_samples": n_metric_samples,
        # Keep last 30 samples (~7.5s of data) for inspection without bloating the file
        "metrics_tail": samples[-30:],
    }


def main() -> int:
    base_texts = base_invoices(limit=5)
    if not base_texts:
        print(
            "No base invoices — run `python -m data.build_dataset` first.",
            file=sys.stderr,
        )
        return 1

    # Confirm vLLM is reachable before starting any benches.
    try:
        with httpx.Client(timeout=5.0) as c:
            r = c.get(f"{VLLM_BASE_URL}/models")
            r.raise_for_status()
    except httpx.HTTPError as e:
        print(f"vLLM not reachable at {VLLM_BASE_URL} ({e}). Start it first.", file=sys.stderr)
        return 2

    client = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL, max_retries=0, timeout=120.0)

    runs = []
    for length in INPUT_LENGTHS_CHARS:
        print(f"\n>> input_length={length} chars  n={N_REQUESTS_PER_LENGTH}  c={CONCURRENCY}")
        run = run_at_length(client, length, base_texts)
        cache_str = (
            f"{run['peak_cache_usage_perc']:.1%}"
            if run["peak_cache_usage_perc"] is not None
            else "(no metrics)"
        )
        print(
            f"   wall={run['wall_clock_s']:.1f}s  ok={run['n_success']}/{run['n_requests']}  "
            f"avg_in_tokens={run['avg_input_tokens']:.0f}  peak_cache={cache_str}  "
            f"metric_samples={run['n_metric_samples']}"
        )
        if run["first_error"]:
            print(f"   first error: {run['first_error']}")
        runs.append(run)

    out = {
        "experiment": "02_kv_cache",
        "model": MODEL,
        "concurrency": CONCURRENCY,
        "n_requests_per_length": N_REQUESTS_PER_LENGTH,
        "input_lengths_chars": INPUT_LENGTHS_CHARS,
        "runs": runs,
    }
    out_path = EXP_DIR / "results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
