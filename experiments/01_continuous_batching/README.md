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

A100 80GB, Qwen 2.5 3B-Instruct (BF16), 100 requests per concurrency level, seed=42.
Headline plot: `throughput_vs_concurrency.png`. Raw per-(config, concurrency) bench
JSONs in `results/`. Aggregated summary in `results.json`.

| Concurrency | serial r/s | vanilla r/s | speedup    |
|-------------|-----------:|------------:|-----------:|
| 1           | 0.66       | 0.66        | 1.01×      |
| 4           | 0.67       | 2.58        | 3.84×      |
| 8           | 0.67       | 4.72        | 7.01×      |
| 16          | 0.67       | **7.95**    | **11.82×** |
| 32          | 0.67       | 11.39       | 16.94×     |

Three observations:

1. **Serial throughput is flat** at 0.66–0.67 r/s across the entire concurrency
   sweep. That's the contract of `max_num_seqs=1` — additional concurrent
   clients just queue at the scheduler; the GPU still processes one request
   at a time. Confirms the experiment setup is doing what it claims to.

2. **Speedup at c=16 (11.8×) significantly exceeds the original 3–5× prediction.**
   The hypothesis was calibrated for an L4-class GPU. On A100, Qwen 3B is small
   enough that batched matmuls easily saturate FP16 compute — adding requests
   is almost free until the GPU's ceiling. Lesson: optimization payoffs depend
   heavily on hardware/model fit; revisit predictions for downstream experiments.

3. **Diminishing returns at c=32**: 1.43× throughput for 2× concurrency (vs the
   ~1.7× near-linear scaling at c=16). This matches the c=16→c=32 plateau Part A
   measured (53% efficiency vs 73%). At c=32 we're entering compute saturation,
   not scheduler-limited.

## Verdict

**Helped — by 12× throughput at concurrency 16 (16.94× at c=32), with no quality cost.**
Continuous batching is the single largest free win in the optimization stack.
Every subsequent experiment in Part B measures throughput improvements *on top
of* this baseline, not against the serial floor.
