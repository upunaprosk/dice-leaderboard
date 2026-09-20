from __future__ import annotations

from typing import Any

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from dike.config import ModelConfig


def load_model(
    config: ModelConfig,
) -> tuple[Any, Any]:
    """
    Load a model and tokenizer.
    Supported backends:
    - transformers
    - gptq

    (vLLM is supported only through lm-eval)

    Padding is changed further, since:
    - likelihood/scoring tasks use right padding
    - generation tasks use left padding
    """

    if config.backend == "vllm":
        raise ValueError(
            "backend='vllm' is supported only for lm-eval-harness "
        )

    tokenizer = load_tokenizer(
        config
    )

    if config.backend == "transformers":
        model = _load_transformers_model(
            config
        )

    elif config.backend == "gptq":
        model = _load_gptq_model(
            config
        )

    else:
        # Normally prevented by ModelConfig validation.
        raise ValueError(
            f"Unsupported model backend: "
            f"{config.backend}"
        )

    if hasattr(model, "eval"):
        model.eval()

    return model, tokenizer


def load_tokenizer(
    config: ModelConfig,
) -> Any:
    """
    Load the tokenizer for a model
    """

    tokenizer_revision = _tokenizer_revision(
        config
    )

    return AutoTokenizer.from_pretrained(
        config.tokenizer,
        revision=tokenizer_revision,
        use_fast=config.use_fast_tokenizer,
        trust_remote_code=config.trust_remote_code,
    )


def _tokenizer_revision(
    config: ModelConfig,
) -> str | None:
    """
    The tokenizer revision.
    """

    if config.tokenizer_revision is not None:
        return config.tokenizer_revision

    if config.tokenizer == config.model:
        return config.revision

    return None


def _load_transformers_model(
    config: ModelConfig,
) -> Any:
    """
    Load a causal language model through HF.
    Supported:
    - standard model loading
    - bitsandbytes INT4
    - bitsandbytes INT8
    """

    kwargs: dict[str, Any] = {
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
    }

    if config.revision is not None:
        kwargs["revision"] = (
            config.revision
        )

    if config.quantization == "int4":
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(
                load_in_4bit=True,
            )
        )

    elif config.quantization == "int8":
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(
                load_in_8bit=True,
            )
        )

    else:
        kwargs["torch_dtype"] = "auto"

    return AutoModelForCausalLM.from_pretrained(
        config.model,
        **kwargs,
    )


def _load_gptq_model(
    config: ModelConfig,
) -> Any:
    """
    Load a GPTQ-model with GPTQModel.
    """

    try:
        from gptqmodel import GPTQModel

    except ImportError as exc:
        raise ImportError(
            "GPTQModel is required when backend='gptq'. "
            "Install the GPTQ dependencies"
        ) from exc

    kwargs: dict[str, Any] = {
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
        "backend": config.gptq_backend,
    }

    if config.revision is not None:
        kwargs["revision"] = (
            config.revision
        )

    return GPTQModel.load(
        config.model,
        **kwargs,
    )