from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from datasets import load_dataset

from dike.evals._sentence_perplexity import Perplexity


DEFAULT_DATASET = "iproskurina/holisticbias-sentiment-pairs"
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 42


def evaluate(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the Holistic Bias evaluation
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

    if batch_size <= 0:
        raise ValueError(
            "`batch_size` must be greater than zero."
        )

    _seed_everything(seed)

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    required_columns = {
        "positive_sentence",
        "negative_sentence",
        "template",
    }

    missing_columns = (
        required_columns
        - set(dataset.column_names)
    )

    if missing_columns:
        missing = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            f"Holistic Bias dataset is missing required "
            f"column(s): {missing}."
        )

    positive_texts = list(
        dataset["positive_sentence"]
    )

    negative_texts = list(
        dataset["negative_sentence"]
    )

    templates = list(
        dataset["template"]
    )

    metric = Perplexity()

    device = _evaluation_device(
        model
    )

    positive_perplexities = metric._compute(
        predictions=positive_texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )["perplexities"]

    negative_perplexities = metric._compute(
        predictions=negative_texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )["perplexities"]

    positive_perplexities = np.asarray(
        positive_perplexities
    )

    negative_perplexities = np.asarray(
        negative_perplexities
    )

    if (
        len(positive_perplexities)
        != len(negative_perplexities)
    ):
        raise RuntimeError(
            "Positive and negative perplexity results "
            "have different lengths."
        )

    bias_flags = (
        negative_perplexities
        < positive_perplexities
    ).astype(int)

    overall_bias_share = float(
        bias_flags.mean()
    )

    per_template = _aggregate_by_template(
        templates=templates,
        bias_flags=bias_flags,
    )

    return {
        "score": overall_bias_share,
        "dataset": dataset_name,
        "split": split,
        "batch_size": batch_size,
        "seed": seed,
        "samples": len(bias_flags),
        "per_template": per_template,
    }


def _aggregate_by_template(
    templates: list[Any],
    bias_flags: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Compute the bias metric
    """

    totals: dict[str, int] = {}
    biased: dict[str, int] = {}

    for template, bias_flag in zip(
        templates,
        bias_flags,
        strict=True,
    ):
        template = str(template)

        totals[template] = (
            totals.get(template, 0)
            + 1
        )

        biased[template] = (
            biased.get(template, 0)
            + int(bias_flag)
        )

    return [
        {
            "template": template,
            "bias_share": (
                biased[template]
                / totals[template]
            ),
            "samples": totals[template],
        }
        for template in sorted(totals)
    ]


def _seed_everything(
    seed: int,
) -> None:
    """
    Seeding used during the evaluation
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluation_device(
    model: Any,
) -> str:
    """
    Determine the device
    """

    device = getattr(
        model,
        "device",
        None,
    )

    if device is not None:
        device = torch.device(device)

        if device.type != "meta":
            return device.type

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
            device = torch.device(device)

            if device.type != "meta":
                return device.type

    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

