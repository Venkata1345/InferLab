"""Async load generator: concurrency sweep, p50/p95/p99 latency, throughput, error rate.

TODO step 8: asyncio + httpx. Configurable concurrency (1, 4, 8, 16, 32) and
input-length distribution (short vs long invoices — matters for KV cache).
Outputs bench/results/<predictor>_<concurrency>.json.
"""
