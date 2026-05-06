"""CLI argparse smoke tests — confirms the parser builds and routes correctly.

Doesn't exercise the full subcommand bodies (those need network / GPU).
Instead: parser construction, --help, and that each subcommand's `func` is
wired to the right callable.
"""

import pytest

from cli.main import (
    build_parser,
    cmd_bench,
    cmd_compare,
    cmd_eval,
    cmd_extract,
)


class TestBuildParser:
    def test_builds_without_error(self) -> None:
        parser = build_parser()
        assert parser is not None

    def test_eval_subcommand_routes_to_cmd_eval(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["eval", "openai"])
        assert args.func is cmd_eval
        assert args.predictor == "openai"

    def test_bench_subcommand_routes_to_cmd_bench(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["bench", "vllm", "--concurrency", "1,4,8"])
        assert args.func is cmd_bench
        assert args.predictor == "vllm"
        assert args.concurrency == "1,4,8"

    def test_compare_subcommand_routes_to_cmd_compare(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["compare"])
        assert args.func is cmd_compare

    def test_extract_subcommand_routes_to_cmd_extract(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["extract"])
        assert args.func is cmd_extract

    def test_unknown_subcommand_exits(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["does-not-exist"])

    def test_eval_invalid_predictor_exits(self) -> None:
        # Predictor argument is choices-constrained.
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["eval", "unknown-predictor"])

    def test_eval_limit_is_int(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["eval", "openai", "--limit", "5"])
        assert args.limit == 5

    def test_bench_default_concurrency(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["bench", "vllm"])
        assert args.concurrency == "1,4,8,16,32"
        assert args.n_requests == 100

    def test_bench_predictor_name_override(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["bench", "vllm", "--predictor-name", "vllm-Qwen-3B-serial"])
        assert args.predictor_name == "vllm-Qwen-3B-serial"

    def test_bench_predictor_name_default_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["bench", "vllm"])
        assert args.predictor_name is None

    def test_compare_gpu_dollar_default(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["compare"])
        assert args.gpu_dollar_per_hr == 0.50
        assert args.target_concurrency == 16


class TestSubcommandStubs:
    """Subcommands that aren't fully implemented should fail with a clear message."""

    def test_extract_says_not_implemented(self) -> None:
        with pytest.raises(SystemExit, match="not yet implemented"):
            cmd_extract(None)
