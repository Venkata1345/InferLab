"""inferlab CLI: extract | bench | eval | compare.

Step 5 wires up `eval <predictor>`. extract / bench / compare get filled in
across steps 7/8/9.
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv


def _build_predictor(name: str, model: str | None):
    if name == "openai":
        from baselines.openai_caller import DEFAULT_MODEL, OpenAIPredictor

        return OpenAIPredictor(model=model or DEFAULT_MODEL)
    if name == "gemini":
        from baselines.gemini_caller import DEFAULT_MODEL, GeminiPredictor

        return GeminiPredictor(model=model or DEFAULT_MODEL)
    if name == "vllm":
        from baselines.vllm_caller import DEFAULT_MODEL, VLLMPredictor

        return VLLMPredictor(model=model or DEFAULT_MODEL)
    raise SystemExit(f"unknown predictor: {name}")


def cmd_eval(args: argparse.Namespace) -> int:
    from eval.runner import DEFAULT_EVAL_PATH, DEFAULT_RESULTS_DIR, run_eval

    predictor = _build_predictor(args.predictor, args.model)
    eval_path = Path(args.eval_path) if args.eval_path else DEFAULT_EVAL_PATH
    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    run_eval(predictor, eval_path=eval_path, results_dir=results_dir, limit=args.limit)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    raise SystemExit("`inferlab extract` not yet implemented (step 7)")


def cmd_bench(args: argparse.Namespace) -> int:
    raise SystemExit("`inferlab bench` not yet implemented (step 8)")


def cmd_compare(args: argparse.Namespace) -> int:
    raise SystemExit("`inferlab compare` not yet implemented (step 9)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inferlab", description="Invoice-extraction inference lab.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="Run eval set against a predictor")
    pe.add_argument("predictor", choices=["openai", "gemini", "vllm"])
    pe.add_argument("--model", default=None, help="Override the predictor's default model name")
    pe.add_argument("--limit", type=int, default=None, help="Run only the first N records")
    pe.add_argument("--eval-path", default=None, help="Override path to eval.jsonl")
    pe.add_argument("--results-dir", default=None, help="Override eval/results/ output dir")
    pe.set_defaults(func=cmd_eval)

    pb = sub.add_parser("bench", help="Run load benchmark (step 8)")
    pb.set_defaults(func=cmd_bench)

    px = sub.add_parser("extract", help="Extract one invoice via vLLM endpoint (step 7)")
    px.set_defaults(func=cmd_extract)

    pc = sub.add_parser("compare", help="Build the comparison table (step 9)")
    pc.set_defaults(func=cmd_compare)

    return p


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()  # pick up OPENAI_API_KEY / GEMINI_API_KEY from .env
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
