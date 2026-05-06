| Predictor | Schema Valid | Field Acc | Record Acc | p50 lat | p99 lat | Throughput | $/1k |
|---|---|---|---|---|---|---|---|
| `vllm-Qwen_Qwen2.5-3B-Instruct` | 100.0% | 87.7% | 66.4% | 1.15 s | 4.15 s | 7.95 req/s @ c=16 | $0.052 |
| `gemini-gemini-2.5-flash-lite` | 100.0% | 95.7% | 87.2% | 1.53 s | 8.70 s | 2.62 req/s @ c=4 | $0.163 |
| `openai-gpt-4o-mini` | 100.0% | 96.8% | 90.4% | 2.36 s | 33.83 s | 1.21 req/s @ c=4 | $0.225 |
