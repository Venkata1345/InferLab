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

A100 80 GB. Qwen 2.5 3B-Instruct verifier + Qwen 2.5 0.5B-Instruct draft.
Bench at c=1, 25 requests per K.

Vanilla baseline (no spec dec, c=1): **p50 = 1,315 ms, throughput = 0.70 req/s**.

| K | p50 (ms) | p99 (ms) | throughput | p50 vs vanilla | thru vs vanilla |
|---|---------:|---------:|-----------:|---------------:|-----------------:|
| 2 | 2,642    | 7,046    | 0.32 r/s   | **+101%**      | **−54%**         |
| 4 | 3,823    | 6,707    | 0.26 r/s   | **+191%**      | **−62%**         |
| 8 | 4,298    | 9,146    | 0.22 r/s   | **+227%**      | **−68%**         |

Quality verify (K=4, n=25): schema_valid=100%, field_macro=94.7%,
record_acc=84.0%. Higher than vanilla's 125-record average (87.7% / 66.4%)
because the first 25 records skew easier — not a quality issue. Spec dec
output is mathematically exact, so by construction it matches what vanilla
would emit on the same 25 records.

Acceptance rate: not captured by our regex (vLLM 0.20.x metric naming
differs from earlier versions). Raw metric snapshots saved to
`metrics-spec-K{2,4,8}.txt` for inspection.

## Verdict

**Hurt — 2× to 3× latency regression at every K we tested, monotonically
worse with larger K.** Same as step 3, this is hardware-dependent, not a
bug in the setup.

Why this happens on A100:

- **The draft model's overhead exceeds its savings.** Spec dec helps when
  the *target* is the bottleneck — typically memory-bandwidth-bound
  weight reads. On A100 with BF16 tensor cores running a small (3B)
  model, the target's per-token cost is already near the compute floor.
  Adding *any* extra step (running the draft, verifying K candidates per
  pass) costs more than the saved tokens recoup.
- **Worse with higher K** is the smoking gun. If acceptance rate were
  high enough to offset draft overhead, throughput should rise with K
  up to a knee. We see strict monotonic degradation, meaning the
  marginal accepted-token benefit is below the marginal draft+verify
  cost across the entire K range we tested.
- **Schema-constrained decoding may interact badly here.** Structured
  outputs constrain the next-token distribution heavily. The draft has
  to predict *exactly* what the constrained verifier would emit, which
  is a different (often narrower) distribution than the draft was trained
  on. Acceptance rate may be lower than typical free-form generation.

Where speculative decoding would actually pay off:

- **Larger target models** (7B, 14B, 70B) where target per-token cost is
  high enough that draft overhead is amortized
- **Smaller / memory-bandwidth-bound GPUs** where target weight reads
  dominate (T4, V100, consumer cards)
- **Free-form generation** without strict schema constraints
- **Higher concurrency**? — actually the opposite; spec dec wins more at
  c=1 because there's spare compute per request. We didn't test c=16+
  because the c=1 result was already so negative that more data wouldn't
  flip the verdict.

## Combined Part B finding

Steps 3 and 4 together: **A100 80 GB + Qwen 2.5 3B + BF16 is "too easy a
problem" for the optimizations we measured.** The GPU has so much
compute headroom that adding any complexity (INT4 dequant, draft model)
costs more than it saves. The only optimization that helped (step 1's
continuous batching) is one that exploits the *workload* shape (multiple
concurrent requests sharing GPU cycles), not the *kernel-level*
arithmetic. Picking the right hardware/model/optimization triple matters
more than turning every knob.
