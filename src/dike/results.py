from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dike.config import EvaluationConfig, ModelConfig
from dike.model_info import inspect_model


def create_results(
    model_config: ModelConfig,
    evaluation_config: EvaluationConfig,
) -> dict[str, Any]:
    """
    Create the result config for one run
    """

    model_info = inspect_model(
        model_config
    )

    return {
        "model": {
            "name": model_config.model,
            "display_name": model_info.get(
                "display_name"
            ),
            "mode": model_config.mode,
            "base_model": model_info.get(
                "base_model"
            ),
            "tokenizer": model_config.tokenizer,
            "backend": model_config.backend,
            "quantization": model_config.quantization,
            "revision": model_config.revision,
            "tokenizer_revision": (
                model_config.tokenizer_revision
            ),
            "dtype": model_info.get(
                "dtype"
            ),
            "source": model_info.get(
                "source"
            ),
            "compression": model_info.get(
                "compression"
            ),
            "warnings": model_info.get(
                "warnings",
                [],
            ),
        },

        "run": {
            "recipe_mode": evaluation_config.mode,
            "seed": evaluation_config.seed,
            "evaluations": list(
                evaluation_config.evaluations
            ),
            "lm_eval_runs": [
                {
                    "backend": run.backend,
                    "tasks": list(
                        run.tasks
                    ),
                    "batch_size": run.batch_size,
                    "device": run.device,
                    "limit": run.limit,
                    "apply_chat_template": (
                        run.apply_chat_template
                    ),
                    "fewshot_as_multiturn": (
                        run.fewshot_as_multiturn
                    ),
                    "num_fewshot": (
                        run.num_fewshot
                    ),
                }
                for run in evaluation_config.lm_eval
            ],
        },

        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluations": {},
        "lm_eval": {},
    }


def add_evaluation_result(
    results: dict[str, Any],
    name: str,
    value: Any,
) -> None:
    """
    Add the output of a single run:

    add_evaluation_result(
        results, "perplexity",
        {
            "score": 12.4,
        },)
    """

    results["evaluations"][
        name
    ] = value


def add_lm_eval_result(
    results: dict[str, Any],
    tasks: list[str],
    value: Any,
) -> None:
    """
    Add the output of a single lm-evaluation-harness run
    """

    base_key = "+".join(
        tasks
    )

    key = base_key
    suffix = 2

    while key in results["lm_eval"]:
        key = (
            f"{base_key}#{suffix}"
        )
        suffix += 1

    results["lm_eval"][
        key
    ] = value


def save_results(
    results: dict[str, Any],
    output_dir: str | Path,
    filename: str | None = None,
) -> Path:
    """
    Save complete results in JSON.
    """

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if filename is None:
        filename = _default_filename(
            results
        )

    if not filename.endswith(
        ".json"
    ):
        filename = (
            f"{filename}.json"
        )

    path = (
        output_dir
        / filename
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )

    return path


def _default_filename(
    results: dict[str, Any],
) -> str:
    """
    Generate a result filename
    """

    model = str(
        results["model"]["name"]
    )

    mode = str(
        results["model"]["mode"]
    )

    safe_model_name = _sanitize_filename(
        model
    )

    return (
        f"{safe_model_name}_"
        f"{mode}_results.json"
    )


def _sanitize_filename(
    value: str,
) -> str:
    """
    Replace characters in result filenames
    """

    replacements = {
        "/": "_",
        "\\": "_",
        ".": "_",
        " ": "_",
        ":": "_",
    }

    for old, new in replacements.items():
        value = value.replace(
            old,
            new,
        )

    return value


def _json_default(
    value: Any,
) -> Any:
    """
    Convert non-JSON values
    """

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        set,
    ):
        return sorted(
            value
        )

    if hasattr(
        value,
        "item",
    ):
        return value.item()

    if hasattr(
        value,
        "tolist",
    ):
        return value.tolist()

    raise TypeError(
        f"Object of type "
        f"{type(value).__name__} "
        "is not JSON serializable."
    )

