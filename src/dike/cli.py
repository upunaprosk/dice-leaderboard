from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from dike.config import (
    EvaluationConfig,
    ModelConfig,
    load_recipe,
)
from dike.results import save_results
from dike.runner import run


def main(
    argv: Sequence[str] | None = None,
) -> int:

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval":
        return _run_eval(args)

    if args.command == "leaderboard":
        return _run_leaderboard(args)

    parser.print_help()
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dike",
        description="DICE model evaluation",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    _add_eval_parser(
        subparsers
    )

    _add_leaderboard_parser(
        subparsers
    )

    return parser


def _add_eval_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate a model",
    )

    eval_parser.add_argument(
        "--model",
        required=True,
        help=(
            "HF model repository or local model path"
        ),
    )

    eval_parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Optional tokenizer repository/path."
            "Defaults to --model."
        ),
    )

    eval_parser.add_argument(
        "--mode",
        choices=[
            "base",
            "instruct",
        ],
        required=True,
        help=(
            "Evaluation mode"
        ),
    )

    eval_parser.add_argument(
        "--recipe",
        default=None,
        help=(
            "Recipe name or YAML path. "
            "Defaults to the selected mode, e.g.'base' -> recipes/base.yaml."
        ),
    )

    eval_parser.add_argument(
        "--recipes-dir",
        default="recipes",
        help=(
            "Directory containing named evaluation recipes" # TODO: add tests for ./
        ),
    )

    eval_parser.add_argument(
        "--backend",
        choices=[
            "transformers",
            "gptq",
        ],
        default="transformers",
        help=(
            "Backend used for evaluations. "
            "Default: transformers"
        ),
    )

    eval_parser.add_argument(
        "--quantization",
        choices=[
            "int4",
            "int8",
        ],
        default=None,
        help=(
            "BitsAndBytes quantization for  the Transformers backend"
        ),
    )

    eval_parser.add_argument(
        "--gptq-backend",
        default="auto",
        help=(
            "GPTQModel execution backend. "
            "Used only with --backend gptq"
        ),
    )

    eval_parser.add_argument(
        "--device-map",
        default="auto",
        help=(
            "Transformers/GPTQ device map"
        ),
    )
    eval_parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allow HF repos to execute custom model/tokenizer code" # TODO: add tests for gemma
        ),
    )

    eval_parser.add_argument(
        "--use-fast-tokenizer",
        action="store_true",
        help=(
            "Use the fast tokenizer implementation"
        ),
    )

    eval_parser.add_argument(
        "--revision",
        default=None,
        help=(
            "Optional model revision, branch, tag, or commit"
        ),
    )

    eval_parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help=(
            "Optional tokenizer revision"
        ),
    )

    eval_parser.add_argument(
        "--lm-eval-backend",
        choices=[
            "hf",
            "vllm",
        ],
        default=None,
        help=(
            "Override the lm-eval backend specified by the recipe "
        ),
    )

    eval_parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Override the recipe output directory"
        ),
    )

    eval_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override the recipe-level seed"
        ),
    )

    eval_parser.add_argument(
        "--output-file",
        default=None,
        help=(
            "Optional result filename"
            "(.json extension is added automatically)"
        ),
    )


def _add_leaderboard_parser(
    subparsers: argparse._SubParsersAction,
) -> None:
    leaderboard_parser = subparsers.add_parser(
        "leaderboard",
        help=(
            "Export result files to leaderboard CSV"
        ),
    )

    leaderboard_parser.add_argument(
        "results",
        nargs="+",
        help=(
            "Result JSON files"
        ),
    )

    leaderboard_parser.add_argument(
        "--mode",
        choices=[
            "base",
            "instruct",
        ],
        default=None,
        help=(
            "Leaderboard type"
        ),
    )

    leaderboard_parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output CSV file."
        ),
    )


def _run_eval(
    args: argparse.Namespace,
) -> int:
    recipe_name = (
        args.recipe
        if args.recipe is not None
        else args.mode
    )

    evaluation_config = load_recipe(
        recipe=recipe_name,
        recipes_dir=args.recipes_dir,
    )

    _apply_recipe_overrides(
        config=evaluation_config,
        args=args,
    )

    model_config = ModelConfig(
        model=args.model,
        tokenizer=args.tokenizer,
        mode=args.mode,
        backend=args.backend,
        quantization=args.quantization,
        trust_remote_code=args.trust_remote_code,
        use_fast_tokenizer=args.use_fast_tokenizer,
        revision=args.revision,
        tokenizer_revision=args.tokenizer_revision,
        device_map=args.device_map,
        gptq_backend=args.gptq_backend,
    )

    _print_run_summary(
        model_config=model_config,
        evaluation_config=evaluation_config,
        recipe=recipe_name,
    )

    results = run(
        model_config=model_config,
        evaluation_config=evaluation_config,
    )

    output_path = save_results(
        results=results,
        output_dir=evaluation_config.output_dir,
        filename=args.output_file,
    )

    print()
    print(
        f"Results saved to: {output_path}"
    )

    return 0


def _run_leaderboard(
    args: argparse.Namespace,
) -> int:
    """
    Export result files to leaderboard CSV rows
    """

    from dike.leaderboard import (
        export_leaderboard,
    )

    output_path = export_leaderboard(
        inputs=args.results,
        output=args.output,
        mode=args.mode,
    )

    print(
        f"Leaderboard saved to: {output_path}"
    )

    return 0


def _apply_recipe_overrides(
    config: EvaluationConfig,
    args: argparse.Namespace,
) -> None:
    """
    Override a loaded recipe with cli arguments
    """

    if args.output_dir is not None:
        config.output_dir = (
            args.output_dir
        )

    if args.seed is not None:
        config.seed = (
            args.seed
        )

    if args.lm_eval_backend is not None:
        for lm_eval_run in config.lm_eval:
            lm_eval_run.backend = (
                args.lm_eval_backend
            )


def _print_run_summary(
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
    recipe: str | Path,
) -> None:
    """
    Print a summary before evaluation
    """

    print(
        "=" * 60
    )

    print(
        "DICE evaluation"
    )

    print(
        "=" * 60
    )

    print(
        f"Model: {model_config.model}"
    )

    print(
        f"Tokenizer: {model_config.tokenizer}"
    )

    print(
        f"Mode: {model_config.mode}"
    )

    print(
        f"Model backend: {model_config.backend}"
    )

    if model_config.quantization is not None:
        print(
            f"Quantization:   " # TODO: check for llm compressor
            f"{model_config.quantization}"
        )

    if model_config.backend == "gptq":
        print(
            f"GPTQ backend:   "# TODO: fix format \s printed 
            f"{model_config.gptq_backend}"
        )

    print(
        f"Recipe:         {recipe}" # TODO: fix format \s printed
    )

    if evaluation_config.evaluations:
        print(
            "Evaluations: "
            + ", ".join(
                evaluation_config.evaluations
            )
        )

    if evaluation_config.lm_eval:
        for index, lm_run in enumerate(
            evaluation_config.lm_eval,
            start=1,
        ):
            print(
                f"lm-eval #{index}: "
                f"{lm_run.backend} / "
                + ", ".join(
                    lm_run.tasks
                )
            )

    print(
        f"Output:"
        f"{evaluation_config.output_dir}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

