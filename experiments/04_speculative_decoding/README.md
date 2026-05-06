# Experiment 04 — Speculative Decoding

## Hypothesis

Speculative decoding uses a small **draft** model (here, Qwen 2.5 0.5B) to
propose K candidate tokens per step; a single forward pass of the **target**
model (Qwen 2.5 3B) verifies them. Accepted tokens emit immediately;
rejected ones cause a fallback to single-token target generation.

The output is **mathematically identical** to running the target alone —
spec dec doesn't change accuracy, only the path to get there.

Expected outcome on A100 (revised after step 1):
- **Latency at c=1**: meaningful win. The target's compute headroom that
  showed up as "underutilized at low concurrency" in step 1 is exactly
  what spec dec converts into accepted draft tokens. Predict 1.3–2× speedup
  depending on K.
- **Latency at c=16+**: smaller or zero win — at high concurrency the
  target is already compute-saturated and there are no spare cycles for
  speculative work.
- **Acceptance rate**: structured-output tasks (JSON schema) tend to be
  highly predictable token sequences (`,`, `:`, `"`, etc.) — expect
  acceptance rate 50%+, possibly higher.
- **K curve**: throughput rises with K up to a knee, then drops as
  proposed-but-rejected tokens become wasted compute.

## Method

Three K values, c=1 only, 25 requests per K. Same target model, same
prompt + schema-constrained decoding as Part A. K=4 also runs a 25-record
quality eval (sanity check that spec dec output equals vanilla output).

| K | bench n | quality verify |
|---|---------|----------------|
| 2 | 25      | (skipped)      |
| 4 | 25      | yes (25 records) |
| 8 | 25      | (skipped)      |

Acceptance rate parsed from `vllm:spec_decode_*` metrics scraped after
each bench.

## Reproduce

```bash
bash experiments/04_speculative_decoding/run.sh
```

Estimated A100 time: ~10 min (3 vLLM restarts × ~90s each + 3 short
benches + 1 quality eval).

## Results

(filled in after run)

| K | p50 (ms) | p99 (ms) | throughput | p50 vs vanilla | accept rate |
|---|---------:|---------:|-----------:|---------------:|------------:|
| 2 | TBD      | TBD      | TBD        | TBD            | TBD         |
| 4 | TBD      | TBD      | TBD        | TBD            | TBD         |
| 8 | TBD      | TBD      | TBD        | TBD            | TBD         |

Quality verify (K=4, 25-record subset): TBD (should match vanilla exactly).

## Verdict

(filled in after run — best K, % speedup, where the curve plateaus.)
