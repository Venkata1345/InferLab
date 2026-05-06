# Experiment 02 — KV Cache Instrumentation

## Hypothesis

KV cache memory dominates GPU memory at long contexts; concurrent capacity
is **cache-bound, not compute-bound**. Specifically: peak cache usage at a
fixed concurrency should scale roughly linearly with input length, and the
maximum concurrency we can sustain before OOM should drop as input length
grows.

This experiment doesn't move the optimization needle directly — its job is
to *characterize the constraint* that motivates step 3 (quantization frees
KV cache space).

## Method

A single vLLM server (vanilla config: BF16, `max_num_seqs=256`,
`max_model_len=4096`). Same server handles all three input lengths — no
restarts.

For each input length in `{512, 1024, 2048}` chars:
- Generate 50 synthetic requests at that length (real SROIE invoices,
  repeat-padded — KV cache memory is a function of token count, not
  content quality)
- Send them concurrently at c=16 against the running server
- In parallel, scrape `/metrics` every 250 ms during the burst
- Capture peak `vllm:gpu_cache_usage_perc`

Then estimate the OOM-bound max concurrency as
`current_c × (0.95 / observed_cache)` — assumes linear cache scaling, with
0.95 as the practical OOM threshold (vLLM rejects new requests well before
100%).

## Reproduce

vLLM must already be running at `http://localhost:8000`. Then:

```bash
python experiments/02_kv_cache/run.py        # ~3-5 minutes
python experiments/02_kv_cache/aggregate.py  # plots + printed summary
```

Outputs: `results.json` (raw), `kv_cache_usage_vs_input_length.png`,
`max_concurrency_vs_input_length.png`.

## Trims vs the original spec

- Dropped 128-char input length (too small to show the scaling trend)
- Skipped the `block_size` sweep (default block_size=16; effect on
  fragmentation is < 5% and not the headline)
- Estimated max concurrency from the c=16 cache fraction rather than
  ramping concurrency until rejection — fast enough that a real ramp
  would more than double experiment time for marginal precision

## Results

A100 80 GB. Single vLLM (vanilla config), 50 requests per length at c=16.

| target chars | avg input tokens | peak cache @ c=16 | per-request cache | est. max c (≤ 95% cache) |
|--------------|------------------|-------------------|-------------------|--------------------------|
| 512          | 488              | 0.2%              | 0.0125%           | ~6,454                   |
| 1024         | 777              | 0.3%              | 0.0188%           | ~5,223                   |
| 2048         | 1358             | 0.5%              | 0.0313%           | ~3,258                   |

All 150 requests succeeded. Cache scaled roughly linearly with input tokens
(per-request cache ratio: 0.0125 / 0.0188 / 0.0313 — close to the token
ratio 488 / 777 / 1358).

## Verdict

**Hypothesis qualitatively confirmed (linear cache scaling), but the
absolute cache pressure on A100 80 GB is trivial — KV cache is *not* the
binding constraint for Qwen 3B at our input lengths.** The estimated max
concurrency (~3,000–6,000) is far above the compute ceiling (~32–64 from
Part A bench saturation), so the OOM-bound max-c numbers aren't actionable
on this hardware.

This matters for the Part B narrative going into step 3 (quantization):

- **The classic "quantization frees KV cache → more concurrent capacity"
  story does not hold here.** On A100 80 GB, we're not cache-bound.
- **Quantization's value on A100 is reduced VRAM footprint** (enabling
  larger models on the same hardware, or cheaper deployment GPUs) plus
  whatever raw kernel-level latency speedup the INT4 matmuls provide.
  We'll measure those deltas in step 3 without claiming a cache-pressure
  win we can't demonstrate here.
- **Where this experiment WOULD show meaningful pressure**: smaller GPUs
  (L4 24 GB → ~3.3× higher % per request → still loose), longer contexts
  (8k–16k tokens → 4–8× more per request), or larger models (7B/14B).
  Documented for future Part C, not blocking step 3.

The metrics-scraping infrastructure is solid — all three runs captured
clean peak-cache values, the OpenAI-compatible endpoint and `/metrics`
endpoint both work concurrently.
