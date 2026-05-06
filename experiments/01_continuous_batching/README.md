# Experiment 01 — Continuous Batching Impact

## Hypothesis

vLLM's continuous batching (default `max_num_seqs=256`) provides a substantial
throughput advantage over serial execution (`max_num_seqs=1`, simulating the
pre-continuous-batching era). At concurrency=1 the two should be identical
(only one request in flight either way); the gap should grow with concurrency.

Expected: throughput ratio at c=16 of **3-5×**.

## Method

Two vLLM configurations, identical in every other respect:

| Config  | Flag                | Behavior                              |
|---------|---------------------|---------------------------------------|
| serial  | `--max-num-seqs 1`  | One request processed at a time       |
| vanilla | `--max-num-seqs 256`| vLLM's default — continuous batching  |

For each config, run the same load benchmark from Part A:
- Model: `Qwen/Qwen2.5-3B-Instruct` (BF16)
- 100 requests per concurrency level
- Concurrency sweep: 1, 4, 8, 16, 32
- Same eval-set sampling (seed=42, with replacement)

Same model, same hardware, same input distribution. The only knob varied is `max_num_seqs`.

## Reproduce

One command (orchestrates both vLLM restarts + both bench runs + aggregation):

```bash
bash experiments/01_continuous_batching/run.sh
```

Estimated time on A100: ~25 minutes total.

## Results

Filled in after run. Headline plot: `throughput_vs_concurrency.png`. Raw
per-(config, concurrency) bench JSONs live in `results/`. Aggregated summary
in `results.json`.

| Concurrency | serial throughput | vanilla throughput | speedup |
|-------------|-------------------|--------------------|---------|
| 1           | TBD               | TBD                | ~1.0×   |
| 4           | TBD               | TBD                | TBD     |
| 8           | TBD               | TBD                | TBD     |
| 16          | TBD               | TBD                | TBD     |
| 32          | TBD               | TBD                | TBD     |

## Verdict

(filled in after run — one line: "Helped / Hurt / No effect, by X%")
