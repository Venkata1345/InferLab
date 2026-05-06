"""Generate invoice-like text at a target char count.

KV cache memory scales with token count, not with content quality. For this
experiment we just need predictable input lengths — repeating real SROIE
invoices is fine.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_PATH = ROOT / "data" / "processed" / "eval.jsonl"


def base_invoices(limit: int = 5) -> list[str]:
    """Pull a few real SROIE invoices from the eval set as base content."""
    out: list[str] = []
    if not EVAL_PATH.exists():
        return out
    with EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["input_text"])
            if len(out) >= limit:
                break
    return out


def make_input(target_chars: int, base_texts: list[str], idx: int = 0) -> str:
    """Concat repeating base invoices until target_chars, then truncate."""
    if not base_texts:
        raise ValueError("base_texts is empty — run `python -m data.build_dataset` first")
    sep = "\n\n--- next invoice ---\n\n"
    parts = [base_texts[idx % len(base_texts)]]
    while len(sep.join(parts)) < target_chars:
        parts.append(base_texts[(idx + len(parts)) % len(base_texts)])
    return sep.join(parts)[:target_chars]
