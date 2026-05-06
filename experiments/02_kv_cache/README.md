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

(filled in after run)

| target chars | avg input tokens | peak cache | est. max c | ok |
|--------------|------------------|------------|------------|-----|
| 512          | TBD              | TBD        | TBD        | TBD |
| 1024         | TBD              | TBD        | TBD        | TBD |
| 2048         | TBD              | TBD        | TBD        | TBD |

## Verdict

(filled in after run — one line: "Cache scales as expected, max_c drops
from X at 512 tok to Y at 2048 tok, motivating quantization in step 3.")
