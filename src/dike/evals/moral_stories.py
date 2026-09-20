from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from dike.evals._moral_stories_prompt import (
    prepare_dataset,
    prompting,
)


DEFAULT_DATASET = "LabHC/moral_stories"
DEFAULT_BATCH_SIZE = 8
DEFAULT_SEED = 42


def evaluate(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the English DICE Moral Stories evaluation.

    Moral Stories eval is generation-based and, thus, uses left padding.
    """

    dataset_name = str(
        config.get(
            "dataset",
            DEFAULT_DATASET,
        )
    )

    split = str(
        config.get(
            "split",
            "train",
        )
    )

    batch_size = int(
        config.get(
            "batch_size",
            DEFAULT_BATCH_SIZE,
        )
    )

    seed = int(
        config.get(
            "seed",
            DEFAULT_SEED,
        )
    )

    prompt_with_norm = bool(
        config.get(
            "prompt_with_norm",
            True,
        )
    )

    model_name = str(
        config.get(
            "model_name",
            "unknown",
        )
    )

    if batch_size <= 0:
        raise ValueError(
            "`batch_size` must be greater than zero."
        )

    _seed_everything(
        seed
    )

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    dataset = prepare_dataset(
        dataset=dataset,
        prompt_with_norm=prompt_with_norm,
    )

    if hasattr(model, "eval"):
        model.eval()

    device = _generation_device(
        model
    )
    # Tokenizer and model generation are shared objects
    previous_padding_side = (
        tokenizer.padding_side
    )

    previous_pad_token_id = (
        tokenizer.pad_token_id
    )

    previous_eos_token_id = (
        tokenizer.eos_token_id
    )

    generation_config = getattr(
        model,
        "generation_config",
        None,
    )

    previous_generation_pad = (
        getattr(
            generation_config,
            "pad_token_id",
            None,
        )
        if generation_config is not None
        else None
    )

    previous_generation_eos = (
        getattr(
            generation_config,
            "eos_token_id",
            None,
        )
        if generation_config is not None
        else None
    )

    try:
        tokenizer.padding_side = "left"
        if "llama" in model_name.lower():
            tokenizer.pad_token_id = 128001
            tokenizer.eos_token_id = 128009

            if generation_config is not None:
                generation_config.pad_token_id = (
                    128001
                )
                generation_config.eos_token_id = (
                    128009
                )

        elif tokenizer.pad_token_id is None:

            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "Moral Stories generation requires "
                    "a pad token or EOS token."
                )

            tokenizer.pad_token_id = (
                tokenizer.eos_token_id
            )

        moral, immoral, refused = prompting(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dataset=dataset,
            batch_size=batch_size,
            model_name=model_name,
        )

    finally:
        tokenizer.padding_side = (
            previous_padding_side
        )

        tokenizer.pad_token_id = (
            previous_pad_token_id
        )

        tokenizer.eos_token_id = (
            previous_eos_token_id
        )

        if generation_config is not None:
            generation_config.pad_token_id = (
                previous_generation_pad
            )
            generation_config.eos_token_id = (
                previous_generation_eos
            )

    total = len(
        dataset
    )

    return {
        "moral": int(moral),
        "immoral": int(immoral),
        "refused": int(refused),
        "total": total,
        "moral_rate": (
            float(moral / total)
            if total
            else 0.0
        ),
        "immoral_rate": (
            float(immoral / total)
            if total
            else 0.0
        ),
        "refused_rate": (
            float(refused / total)
            if total
            else 0.0
        ),
        "dataset": dataset_name,
        "split": split,
        "batch_size": batch_size,
        "prompt_with_norm": prompt_with_norm,
        "seed": seed,
    }


def _seed_everything(
    seed: int,
) -> None:
    """Seed the Moral Stories generation"""

    random.seed(seed)

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = (
        True
    )


def _generation_device(
    model: Any,
) -> torch.device:

    device = getattr(
        model,
        "device",
        None,
    )

    if device is not None:
        device = torch.device(
            device
        )

        if device.type != "meta":
            return device

    try:
        return next(
            model.parameters()
        ).device

    except (
        AttributeError,
        StopIteration,
    ):
        pass

    wrapped_model = getattr(
        model,
        "model",
        None,
    )

    if wrapped_model is not None:

        device = getattr(
            wrapped_model,
            "device",
            None,
        )

        if device is not None:
            device = torch.device(
                device
            )

            if device.type != "meta":
                return device

        try:
            return next(
                wrapped_model.parameters()
            ).device

        except (
            AttributeError,
            StopIteration,
        ):
            pass

    raise RuntimeError(
        "Could not determine the model device. "
    )

