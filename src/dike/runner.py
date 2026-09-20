from __future__ import annotations

from typing import Any, Callable

from dike.config import EvaluationConfig, ModelConfig
from dike.models import load_model
from dike.results import (
    add_evaluation_result,
    add_lm_eval_result,
    create_results,
)


Evaluator = Callable[
    [Any, Any, dict[str, Any]],
    dict[str, Any],
]


def _evaluations() -> dict[str, Evaluator]:
    """
    Return the evaluation registry
    """

    from dike.evals.holistic_bias import (
        evaluate as holistic_bias,
    )
    from dike.evals.moral_stories import (
        evaluate as moral_stories,
    )
    from dike.evals.perplexity import (
        evaluate as perplexity,
    )
    from dike.evals.sofa import (
        evaluate as sofa,
    )
    from dike.evals.stereoset import (
        evaluate as stereoset,
    )

    return {
        "perplexity": perplexity,
        "holistic_bias": holistic_bias,
        "sofa": sofa,
        "stereoset": stereoset,
        "moral_stories": moral_stories,
    }


def run(
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    """
    Execute one evaluation recipe.
    Order:
    1. Validate model/recipe
    2. Run lm-evaluation-harness
    3. Load the DICE model and tokenizer
    4. Run all evaluations
    5. Return the aggregated results

    lm-eval runs are executed beforeto avoid keeping 2 large models in GPU memory
    """

    evaluation_config.validate_model(
        model_config
    )

    results = create_results(
        model_config=model_config,
        evaluation_config=evaluation_config,
    )

    if evaluation_config.has_lm_eval:
        _run_lm_eval(
            results=results,
            model_config=model_config,
            evaluation_config=evaluation_config,
        )

    if evaluation_config.has_evaluations:
        _run_evaluations(
            results=results,
            model_config=model_config,
            evaluation_config=evaluation_config,
        )

    return results


def _run_lm_eval(
    results: dict[str, Any],
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
) -> None:
    """
    Execute lm-evaluation-harness runs
    """

    from dike.evals.lm_eval import (
        evaluate as run_lm_eval,
    )

    total_runs = len(
        evaluation_config.lm_eval
    )

    for index, run_config in enumerate(
        evaluation_config.lm_eval,
        start=1,
    ):
        task_names = ", ".join(
            run_config.tasks
        )

        print(
            f"[lm-eval {index}/{total_runs}] "
            f"Running: {task_names}"
        )

        result = run_lm_eval(
            model_config=model_config,
            config=run_config,
            seed=evaluation_config.seed,
        )

        add_lm_eval_result(
            results=results,
            tasks=run_config.tasks,
            value=result,
        )

        print(
            f"[lm-eval {index}/{total_runs}] "
            f"Finished: {task_names}"
        )


def _run_evaluations(
    results: dict[str, Any],
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
) -> None:
    """
    Load the model and execute all DICE(not lm-eval) evaluations
    """

    available = _evaluations()

    _validate_evaluations(
        requested=evaluation_config.evaluations,
        available=available,
    )

    print(
        "Loading model for DICE evaluations: "
        f"{model_config.model}"
    )

    model, tokenizer = load_model(
        model_config
    )

    total = len(
        evaluation_config.evaluations
    )

    for index, name in enumerate(
        evaluation_config.evaluations,
        start=1,
    ):
        evaluator = available[name]

        options = _evaluation_options(
            name=name,
            model_config=model_config,
            evaluation_config=evaluation_config,
        )

        print(
            f"[evaluation {index}/{total}] "
            f"Running: {name}"
        )

        result = evaluator(
            model=model,
            tokenizer=tokenizer,
            config=options,
        )

        add_evaluation_result(
            results=results,
            name=name,
            value=result,
        )

        print(
            f"[evaluation {index}/{total}] "
            f"Finished: {name}"
        )


def _evaluation_options(
    name: str,
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    """
    Build the configuration passed to evaluator
    """

    options = dict(
        evaluation_config.options_for(
            name
        )
    )

    options.setdefault(
        "seed",
        evaluation_config.seed,
    )

    options.setdefault(
        "model_name",
        model_config.model,
    )

    return options


def _validate_evaluations(
    requested: list[str],
    available: dict[str, Evaluator],
) -> None:
    """
    Validate evaluation names before loading the model
    """

    unknown = [
        name
        for name in requested
        if name not in available
    ]

    if not unknown:
        return

    unknown_names = ", ".join(
        unknown
    )

    available_names = ", ".join(
        sorted(available)
    )

    raise ValueError(
        "Unknown DICE evaluation(s): "
        f"{unknown_names}. "
        f"Available evaluations: {available_names}."
    )

