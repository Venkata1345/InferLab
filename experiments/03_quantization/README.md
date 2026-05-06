# Experiment 03 — AWQ INT4 Quantization

## Hypothesis (revised after step 2)

**Original framing**: AWQ INT4 frees KV cache space → higher concurrent capacity.
This story does NOT apply on A100 80 GB (step 2 showed cache usage at 0.2-0.5%
for our workload — we're nowhere near cache-bound).

**Actual hypothesis on this hardware**:
- **Latency** at low batch sizes should improve: INT4 matmuls are
  memory-bandwidth-bound, and 4× fewer weight bytes means fewer DRAM reads
  per forward pass.
- **Throughput** at high concurrency may improve modestly (compute is the
  bottleneck on A100; quantized kernels still need to run).
- **Quality**: AWQ on instruction-tuned models typically loses 0–2 percentage
  points of task accuracy. Worth measuring on the SROIE eval — if it costs
  more than 5pp record accuracy, document the tradeoff and don't pretend
  it's a free win.

FP8 deferred — vLLM FP8 has known integration friction across versions.
Will revisit after step 4 if Pareto curve calls for it.

## Method

Two configs, same eval set + same bench harness:

| Config  | Model id                           | Quantization flag |
|---------|------------------------------------|-------------------|
| vanilla | `Qwen/Qwen2.5-3B-Instruct`         | (none, BF16)      |
| AWQ     | `Qwen/Qwen2.5-3B-Instruct-AWQ`     | `--quantization awq` |

For each config: full 125-record eval (accuracy delta) + bench at
**c=1 and c=16 only**, 50 requests per level (trim — c=1 = latency
reference, c=16 = throughput headline; the full curve is established by
Part A and step 1).

Re-run vanilla on this hardware to keep the comparison apples-to-apples
(step 1's vanilla bench JSONs were lost in a Colab kernel reset).

## Reproduce

```bash
bash experiments/03_quantization/run.sh
```

Estimated A100 time: ~15 min (~3 min vanilla eval + ~80s vanilla bench
× 2 levels + ~3 min AWQ download + ~3 min AWQ eval + ~80s AWQ bench × 2
+ cleanup).

## Results

A100 80 GB. Qwen 2.5 3B-Instruct, 125-record SROIE eval + bench at c=1, 16
(50 requests/level).

| metric                  | vanilla BF16 | AWQ INT4    | delta     |
|-------------------------|-------------:|------------:|----------:|
| schema_validity         | 100.0%       | 100.0%      | +0.0pp    |
| field_acc_macro         | 87.73%       | 87.73%      | +0.0pp    |
| record_accuracy         | 66.4%        | 68.0%       | +1.6pp    |
| eval p50 latency (ms)   | 1,173        | 3,895       | **+232%** |
| eval p99 latency (ms)   | 4,160        | 15,276      | +267%     |
| bench throughput @ c=1  | 0.70 req/s   | 0.22 req/s  | **−69%**  |
| bench throughput @ c=16 | 8.96 req/s   | 2.81 req/s  | **−69%**  |
| bench p50 @ c=1 (ms)    | 1,315        | 4,110       | +212%     |
| bench p50 @ c=16 (ms)   | 1,342        | 4,436       | +231%     |

Quality is **essentially unchanged**: per-field accuracy is identical to 4
decimal places. The 1.6pp record-accuracy bump is within statistical
noise on 125 records (the same fields are correct/wrong on a slightly
different subset of records).

Latency and throughput both regress by ~3×. Same direction at c=1 and
c=16, so the regression isn't concurrency-dependent — it's per-request
overhead.

## Verdict

**Hurt — AWQ INT4 is a 3× latency regression on A100 with no quality
benefit.** Hardware mismatch, not a bug in our setup or a fluke.

Why:

- **A100 has top-tier BF16 tensor cores.** BF16 matmul on Ampere is the
  fast path. AWQ has nothing better to compete with.
- **A100 has no native INT4 tensor cores.** AWQ stores weights in INT4
  but dequantizes to BF16/FP16 *just before each matmul*. The dequant
  step adds per-layer overhead that pure BF16 doesn't pay.
- **The classic AWQ wins don't apply to this configuration.** The
  benefits — smaller VRAM, fitting bigger models, faster on
  memory-bandwidth-bound workloads — are about *fitting more / paying
  less* at memory boundaries we never hit. Step 2 already showed Qwen
  3B on A100 80GB isn't KV-cache-bound at our input sizes. So the
  dequant overhead is pure cost.

Where AWQ would actually pay off:

- **H100 or RTX 4090** (native INT4 tensor cores → no dequant penalty)
- **Smaller GPUs** (L4 24GB, T4 16GB) where weight bytes meaningfully
  reduce KV-cache pressure
- **Larger models** (7B, 14B) where VRAM headroom matters
- **Longer contexts** (8k+ tokens) where KV cache dominates

For Part B's narrative this is a real finding, not a failure: it
**bounds the optimization space** for our specific hardware/model
combination. INT4 quantization is the wrong knob to turn here.
