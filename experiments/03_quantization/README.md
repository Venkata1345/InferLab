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

(filled in after run)

| metric                | vanilla BF16 | AWQ INT4 | delta |
|-----------------------|-------------:|---------:|------:|
| schema_validity       | TBD          | TBD      | TBD   |
| field_acc_macro       | TBD          | TBD      | TBD   |
| record_accuracy       | TBD          | TBD      | TBD   |
| eval p50 latency      | TBD          | TBD      | TBD   |
| eval p99 latency      | TBD          | TBD      | TBD   |
| bench throughput @c=1 | TBD          | TBD      | TBD   |
| bench throughput @c=16| TBD          | TBD      | TBD   |
| bench p50 @c=1        | TBD          | TBD      | TBD   |

## Verdict

(filled in after run — one line: "Helped / Hurt / Wash, by X% on metric Y;
quality cost: Zpp.")
