from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


ModelMode = Literal[
    "base",
    "instruct",
]

ModelBackend = Literal[
    "transformers",
    "gptq",
]

LMEvalBackend = Literal[
    "hf",
    "vllm",
]

Quantization = Literal[
    "int4",
    "int8",
]


@dataclass
class ModelConfig:
    """
    Configuration describing the model used by DICE evaluations.

    Notes
    -----
    - ``tokenizer`` defaults to ``model``.
    - ``transformers`` supports standard, INT4, and INT8 loading.
    - ``gptq`` represents an already GPTQ-quantized checkpoint.
    - vLLM is configured separately for lm-evaluation-harness runs.
    """

    model: str
    tokenizer: str | None = None

    mode: ModelMode = "base"

    backend: ModelBackend = "transformers"
    quantization: Quantization | None = None

    trust_remote_code: bool = False

    # Historical DICE CLI used action="store_true", so the default
    # behavior was the slow tokenizer.
    use_fast_tokenizer: bool = False

    revision: str | None = None
    tokenizer_revision: str | None = None

    device_map: str = "auto"

    # Used only when backend == "gptq".
    gptq_backend: str = "auto"

    def __post_init__(self) -> None:
        self.model = self.model.strip()

        if not self.model:
            raise ValueError(
                "Model name/path cannot be empty."
            )

        if self.tokenizer is None:
            self.tokenizer = self.model
        else:
            self.tokenizer = self.tokenizer.strip()

        if not self.tokenizer:
            raise ValueError(
                "Tokenizer name/path cannot be empty."
            )

        if self.mode not in {
            "base",
            "instruct",
        }:
            raise ValueError(
                f"Unsupported model mode: {self.mode}. "
                "Expected 'base' or 'instruct'."
            )

        if self.backend not in {
            "transformers",
            "gptq",
        }:
            raise ValueError(
                f"Unsupported model backend: "
                f"{self.backend}."
            )

        if self.quantization not in {
            None,
            "int4",
            "int8",
        }:
            raise ValueError(
                f"Unsupported quantization: "
                f"{self.quantization}."
            )

        if (
            self.backend == "gptq"
            and self.quantization is not None
        ):
            raise ValueError(
                "Do not specify int4/int8 quantization "
                "for a GPTQ checkpoint. "
                "GPTQ checkpoints are already quantized."
            )

        if (
            self.backend == "gptq"
            and not self.gptq_backend
        ):
            raise ValueError(
                "`gptq_backend` cannot be empty when "
                "backend='gptq'."
            )


@dataclass
class LMEvalRunConfig:
    """
    Configuration for one lm-evaluation-harness invocation.

    A recipe may contain multiple lm-eval runs because different task
    groups may require different parameters.

    The lm-eval backend is independent from the model backend used by
    DICE evaluations.
    """

    tasks: list[str]

    backend: LMEvalBackend = "hf"

    batch_size: int | str = 1
    device: str = "cuda:0"

    limit: int | float | None = None

    apply_chat_template: bool = False

    fewshot_as_multiturn: bool = False
    num_fewshot: int | None = None

    extra: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError(
                "An lm-eval run must contain at least "
                "one task."
            )

        self.tasks = [
            str(task)
            for task in self.tasks
        ]

        if self.backend not in {
            "hf",
            "vllm",
        }:
            raise ValueError(
                f"Unsupported lm-eval backend: "
                f"{self.backend}."
            )

        if isinstance(
            self.batch_size,
            int,
        ):
            if self.batch_size <= 0:
                raise ValueError(
                    "`batch_size` must be greater than 0."
                )

        if self.num_fewshot is not None:
            if self.num_fewshot < 0:
                raise ValueError(
                    "`num_fewshot` cannot be negative."
                )


@dataclass
class EvaluationConfig:
    """
    Configuration describing a complete DICE evaluation recipe.

    There are two groups:

    1. DICE evaluations implemented in this repository:
       - perplexity
       - Holistic Bias
       - SOFA
       - StereoSet
       - Moral Stories

    2. lm-evaluation-harness runs:
       - BBQ
       - HellaSwag
       - CrowS-Pairs
       - RealToxicityPrompts
       - HarmBench
       - etc.

    ``options`` contains configuration for DICE evaluations.
    """

    evaluations: list[str] = field(
        default_factory=list
    )

    lm_eval: list[LMEvalRunConfig] = field(
        default_factory=list
    )

    seed: int = 42
    output_dir: str = "results"

    # Optional recipe restriction.
    mode: ModelMode | None = None

    # Evaluation-specific options.
    options: dict[
        str,
        dict[str, Any],
    ] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not self.evaluations
            and not self.lm_eval
        ):
            raise ValueError(
                "A recipe must contain at least one "
                "evaluation or lm-eval run."
            )

        if self.mode not in {
            None,
            "base",
            "instruct",
        }:
            raise ValueError(
                f"Unsupported recipe mode: "
                f"{self.mode}."
            )

        self.evaluations = [
            str(name)
            for name in self.evaluations
        ]

    def options_for(
        self,
        evaluation: str,
    ) -> dict[str, Any]:
        """
        Return configuration for one DICE evaluation.

        Example
        -------
        config.options_for("perplexity")
        """

        return self.options.get(
            evaluation,
            {},
        )

    def validate_model(
        self,
        model_config: ModelConfig,
    ) -> None:
        """
        Ensure a mode-specific recipe is used with the expected model.
        """

        if (
            self.mode is not None
            and self.mode != model_config.mode
        ):
            raise ValueError(
                f"This recipe is for '{self.mode}' models, "
                f"but the selected model mode is "
                f"'{model_config.mode}'."
            )

    @property
    def has_evaluations(self) -> bool:
        """Return True when DICE evaluations are requested."""

        return bool(
            self.evaluations
        )

    @property
    def has_lm_eval(self) -> bool:
        """Return True when lm-eval runs are requested."""

        return bool(
            self.lm_eval
        )


def load_recipe(
    recipe: str | Path,
    recipes_dir: str | Path = "recipes",
) -> EvaluationConfig:
    """
    Load a DICE evaluation recipe.

    Parameters
    ----------
    recipe:
        Either a recipe name:

            "base"
            "instruct"

        or an explicit YAML path:

            "recipes/base.yaml"
            "my_experiment.yaml"

    recipes_dir:
        Directory containing named recipes.
    """

    path = _resolve_recipe_path(
        recipe=recipe,
        recipes_dir=recipes_dir,
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(
            file
        ) or {}

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError(
            f"Recipe '{path}' must contain "
            "a YAML mapping."
        )

    evaluations = raw.get(
        "evaluations",
        [],
    ) or []

    if not isinstance(
        evaluations,
        list,
    ):
        raise ValueError(
            f"Recipe '{path}': "
            "'evaluations' must be a list."
        )

    evaluations = [
        str(name)
        for name in evaluations
    ]

    options = {
        name: raw.get(
            name,
            {},
        ) or {}
        for name in evaluations
    }

    for name, value in options.items():
        if not isinstance(
            value,
            dict,
        ):
            raise ValueError(
                f"Configuration for evaluation "
                f"'{name}' must be a YAML mapping."
            )

    lm_eval = _parse_lm_eval_runs(
        raw.get(
            "lm_eval",
            [],
        ),
        recipe_path=path,
    )

    return EvaluationConfig(
        evaluations=evaluations,
        lm_eval=lm_eval,
        seed=int(
            raw.get(
                "seed",
                42,
            )
        ),
        output_dir=str(
            raw.get(
                "output_dir",
                "results",
            )
        ),
        mode=raw.get(
            "mode"
        ),
        options=options,
    )


def _parse_lm_eval_runs(
    raw_runs: Any,
    recipe_path: Path,
) -> list[LMEvalRunConfig]:
    """
    Parse the ``lm_eval`` section of a recipe.
    """

    if raw_runs is None:
        return []

    if not isinstance(
        raw_runs,
        list,
    ):
        raise ValueError(
            f"Recipe '{recipe_path}': "
            "'lm_eval' must be a list."
        )

    runs: list[
        LMEvalRunConfig
    ] = []

    known_fields = {
        "tasks",
        "backend",
        "batch_size",
        "device",
        "limit",
        "apply_chat_template",
        "fewshot_as_multiturn",
        "num_fewshot",
    }

    for index, raw_run in enumerate(
        raw_runs,
        start=1,
    ):
        if not isinstance(
            raw_run,
            dict,
        ):
            raise ValueError(
                f"Recipe '{recipe_path}': "
                f"lm_eval entry #{index} must be "
                "a YAML mapping."
            )

        tasks = raw_run.get(
            "tasks"
        )

        if (
            not isinstance(
                tasks,
                list,
            )
            or not tasks
        ):
            raise ValueError(
                f"Recipe '{recipe_path}': "
                f"lm_eval entry #{index} must define "
                "a non-empty 'tasks' list."
            )

        backend = raw_run.get(
            "backend",
            "hf",
        )

        if backend not in {
            "hf",
            "vllm",
        }:
            raise ValueError(
                f"Recipe '{recipe_path}': "
                f"lm_eval entry #{index} has unsupported "
                f"backend '{backend}'. "
                "Expected 'hf' or 'vllm'."
            )

        extra = {
            key: value
            for key, value in raw_run.items()
            if key not in known_fields
        }

        runs.append(
            LMEvalRunConfig(
                tasks=[
                    str(task)
                    for task in tasks
                ],
                backend=backend,
                batch_size=raw_run.get(
                    "batch_size",
                    1,
                ),
                device=str(
                    raw_run.get(
                        "device",
                        "cuda:0",
                    )
                ),
                limit=raw_run.get(
                    "limit"
                ),
                apply_chat_template=bool(
                    raw_run.get(
                        "apply_chat_template",
                        False,
                    )
                ),
                fewshot_as_multiturn=bool(
                    raw_run.get(
                        "fewshot_as_multiturn",
                        False,
                    )
                ),
                num_fewshot=raw_run.get(
                    "num_fewshot"
                ),
                extra=extra,
            )
        )

    return runs


def _resolve_recipe_path(
    recipe: str | Path,
    recipes_dir: str | Path,
) -> Path:
    """
    Resolve a recipe name or explicit path to an existing YAML file.
    """

    recipe = Path(
        recipe
    )

    if recipe.exists():
        return recipe

    recipes_dir = Path(
        recipes_dir
    )

    if recipe.suffix == "":
        candidate = (
            recipes_dir
            / f"{recipe.name}.yaml"
        )
    else:
        candidate = (
            recipes_dir
            / recipe.name
        )

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Recipe not found: '{recipe}'. "
        f"Also checked '{candidate}'."
    )

