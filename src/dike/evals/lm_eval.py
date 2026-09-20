from __future__ import annotations

from typing import Any

from dike.config import LMEvalRunConfig, ModelConfig


def evaluate(
    model_config: ModelConfig,
    config: LMEvalRunConfig,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run lm-evaluation-harness evaluation
    """

    try:
        import lm_eval

    except ImportError as exc:
        raise ImportError(
            "lm-evaluation-harness is required for lm-eval tasks. "
        ) from exc

    backend, model_args = _build_model(
        model_config=model_config,
        run_config=config,
    )

    kwargs: dict[str, Any] = {
        "model": backend,
        "model_args": model_args,
        "tasks": config.tasks,
        "batch_size": config.batch_size,
        "device": config.device,
        "limit": config.limit,
        "apply_chat_template": config.apply_chat_template,
        "fewshot_as_multiturn": config.fewshot_as_multiturn,
        "num_fewshot": config.num_fewshot,
        "random_seed": seed,
        "numpy_random_seed": seed,
        "torch_random_seed": seed,
        "fewshot_random_seed": seed,
    }
    kwargs.update(
        config.extra
    )
    kwargs = {
        key: value
        for key, value in kwargs.items()
        if value is not None
    }

    result = lm_eval.simple_evaluate(
        **kwargs
    )

    if result is None:
        raise RuntimeError(
            "lm-evaluation-harness returned no result."
        )

    return _prepare_results(
        result=result,
        backend=backend,
        model_args=model_args,
        config=config,
    )


def _build_model(
    model_config: ModelConfig,
    run_config: LMEvalRunConfig,
) -> tuple[str, dict[str, Any]]:
    """
    Map configuration into an lm-eval backend (vllm argument from cli)
    """

    if run_config.backend == "hf":
        return (
            "hf",
            _build_hf_args(
                model_config
            ),
        )

    if run_config.backend == "vllm":
        return (
            "vllm",
            _build_vllm_args(
                model_config
            ),
        )

    raise ValueError(
        f"Unsupported lm-eval backend: "
        f"{run_config.backend}"
    )


def _build_hf_args(
    config: ModelConfig,
) -> dict[str, Any]:
    """
    Build lm-eval HF model arguments
    """

    args: dict[str, Any] = {
        "pretrained": config.model,
        "trust_remote_code": config.trust_remote_code,
    }

    if (
        config.tokenizer
        and config.tokenizer != config.model
    ):
        args["tokenizer"] = (
            config.tokenizer
        )

    if config.revision is not None:
        args["revision"] = (
            config.revision
        )

    if config.tokenizer_revision is not None:
        args["tokenizer_revision"] = (
            config.tokenizer_revision
        )

    if config.backend == "gptq":
        args["gptqmodel"] = True

    elif config.quantization == "int4":
        args["load_in_4bit"] = True

    elif config.quantization == "int8":
        args["load_in_8bit"] = True

    return args


def _build_vllm_args(
    config: ModelConfig,
) -> dict[str, Any]:
    """
    Build lm-eval vLLM model arguments
    """

    args: dict[str, Any] = {
        "pretrained": config.model,
        "trust_remote_code": config.trust_remote_code,
    }

    if (
        config.tokenizer
        and config.tokenizer != config.model
    ):
        args["tokenizer"] = (
            config.tokenizer
        )

    if config.revision is not None:
        args["revision"] = (
            config.revision
        )

    if config.tokenizer_revision is not None:
        args["tokenizer_revision"] = (
            config.tokenizer_revision
        )

    return args


def _prepare_results(
    result: dict[str, Any],
    backend: str,
    model_args: dict[str, Any],
    config: LMEvalRunConfig,
) -> dict[str, Any]:
    """
    Return aggregated lm-eval output
    """

    return {
        "backend": backend,
        "model_args": model_args,
        "tasks": list(
            config.tasks
        ),
        "batch_size": config.batch_size,
        "device": config.device,
        "limit": config.limit,
        "apply_chat_template": (
            config.apply_chat_template
        ),
        "fewshot_as_multiturn": (
            config.fewshot_as_multiturn
        ),
        "num_fewshot": (
            config.num_fewshot
        ),

        "results": result.get(
            "results",
            {},
        ),
        "groups": result.get(
            "groups",
            {},
        ),

        "versions": result.get(
            "versions",
            {},
        ),

        "n-shot": result.get(
            "n-shot",
            {},
        ),

        "higher_is_better": result.get(
            "higher_is_better",
            {},
        ),

        "n-samples": result.get(
            "n-samples",
            {},
        ),
    }

