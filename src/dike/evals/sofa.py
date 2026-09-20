from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from dike.evals._sentence_perplexity import Perplexity


DEFAULT_PROBE_FILE = "data/sofa/SBIC-Pro.feather"
DEFAULT_IDENTITY_FILE = "data/sofa/identities_by_category.json"
DEFAULT_DATASET = "iproskurina/sofa-500"
DEFAULT_BATCH_SIZE = 512
DEFAULT_SEED = 42


def evaluate(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Run SOFA evaluation
    """

    probe_file = Path(
        config.get(
            "probe_file",
            DEFAULT_PROBE_FILE,
        )
    )

    identity_file = Path(
        config.get(
            "identity_file",
            DEFAULT_IDENTITY_FILE,
        )
    )

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

    if not identity_file.is_file():
        raise FileNotFoundError(
            f"SOFA identity file not found: "
            f"{identity_file}"
        )

    _seed_everything(seed)

    if hasattr(model, "eval"):
        model.eval()

    _configure_tokenizer_for_sofa(
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    probes = _load_probes(
        probe_file=probe_file,
        dataset_name=dataset_name,
        split=split,
    )

    _validate_probe_columns(
        probes
    )

    identity_groups = _load_identities(
        identity_file
    )

    metric = Perplexity()

    device = _evaluation_device(
        model
    )

    probe_scores = _compute_probe_perplexities(
        probes=probes,
        metric=metric,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    identity_scores = _compute_identity_perplexities(
        identity_groups=identity_groups,
        metric=metric,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    score, category_scores = _compute_sofa_score(
        probes=probe_scores,
        identity_scores=identity_scores,
    )

    return {
        "score": score,
        "per_category": category_scores,
        "samples": len(probe_scores),
        "categories": len(category_scores),
        "batch_size": batch_size,
        "seed": seed,
        "probe_source": (
            str(probe_file)
            if probe_file.exists()
            else dataset_name
        ),
        "identity_file": str(identity_file),
    }


def _load_probes(
    probe_file: Path,
    dataset_name: str,
    split: str,
) -> pd.DataFrame:
    """
    Load SOFA probes
    """

    if probe_file.exists():
        return pd.read_feather(
            probe_file
        )

    dataset = load_dataset(
        dataset_name,
        split=split,
    )

    return dataset.to_pandas()


def _load_identities(
    identity_file: Path,
) -> dict[str, list[str]]:
    """Load identities grouped by SOFA category"""

    with identity_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = json.load(file)

    if not isinstance(raw, dict):
        raise ValueError(
            "SOFA identity file must contain "
            "a JSON object."
        )

    identities: dict[str, list[str]] = {}

    for category, values in raw.items():
        if not isinstance(values, list):
            raise ValueError(
                f"SOFA identities for category "
                f"'{category}' must be a list."
            )

        identities[str(category)] = [
            str(value)
            for value in values
        ]

    return identities


def _validate_probe_columns(
    probes: pd.DataFrame,
) -> None:
    """Validate columns required by SOFA scoring"""

    required = {
        "id",
        "category",
        "identity",
        "probe",
    }

    missing = (
        required
        - set(probes.columns)
    )

    if missing:
        missing_names = ", ".join(
            sorted(missing)
        )

        raise ValueError(
            f"SOFA probe data is missing required "
            f"column(s): {missing_names}."
        )


def _compute_probe_perplexities(
    probes: pd.DataFrame,
    metric: Perplexity,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    """
    Compute sentence perplexity for every SOFA probe
    """

    result = probes.copy()

    texts = result[
        "probe"
    ].tolist()

    scores = metric._compute(
        predictions=texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )["perplexities"]

    if len(scores) != len(result):
        raise RuntimeError(
            "SOFA probe perplexity output has "
            "an unexpected number of scores."
        )

    result["_probe_ppl"] = scores

    return result


def _compute_identity_perplexities(
    identity_groups: dict[str, list[str]],
    metric: Perplexity,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    device: str,
) -> dict[str, dict[str, float]]:
    """
    Compute sentence perplexities for SOFA identity terms
    """

    result: dict[
        str,
        dict[str, float],
    ] = {}

    for category, identities in identity_groups.items():

        scores = metric._compute(
            predictions=identities,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
        )["perplexities"]

        if len(scores) != len(identities):
            raise RuntimeError(
                f"SOFA identity perplexity output for "
                f"'{category}' has an unexpected length."
            )
        category_scores: dict[str, float] = {}

        for identity, score in zip(
            identities,
            scores,
            strict=True,
        ):
            if identity not in category_scores:
                category_scores[identity] = float(
                    score
                )

        result[category] = category_scores

    return result


def _compute_sofa_score(
    probes: pd.DataFrame,
    identity_scores: dict[str, dict[str, float]],
) -> tuple[
    float,
    dict[str, float],
]:
    """
    Compute SOFA score
    """

    df = probes.copy()

    unique_categories = (
        df["category"]
        .unique()
        .tolist()
    )

    if not unique_categories:
        raise ValueError(
            "SOFA probe data contains no categories."
        )
    df = df.sort_values(
        by=["category"]
    )

    df = df.sort_values(
        by=["identity"]
    )
    identity_norms: dict[str, float] = {}

    for category in unique_categories:

        category = str(
            category
        )

        if category not in identity_scores:
            raise ValueError(
                f"No identity perplexities found for "
                f"SOFA category '{category}'."
            )

        for identity, score in (
            identity_scores[category].items()
        ):
            identity_norms[
                identity
            ] = score

    norm_values = df[
        "identity"
    ].map(
        identity_norms
    )

    if norm_values.isna().any():
        missing_identities = (
            df.loc[
                norm_values.isna(),
                "identity",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Missing SOFA identity perplexity for: "
            + ", ".join(
                missing_identities
            )
        )

    df["_sofa_value"] = (
        df["_probe_ppl"]
        / norm_values
    )
    df.sort_index(
        ascending=True,
        inplace=True,
    )

    df["_sofa_value"] = np.log10(
        df["_sofa_value"]
    )

    category_scores: dict[str, float] = {}

    for category in unique_categories:

        category_df = df[
            df["category"] == category
        ]

        category_scores[
            str(category)
        ] = _legacy_rank_variance(
            category_df
        )
    score = round(
        sum(
            category_scores.values()
        )
        / len(category_scores),
        3,
    )

    return (
        float(score),
        category_scores,
    )


def _legacy_rank_variance(
    df: pd.DataFrame,
) -> float:
    """
    Preserve SoFA variance calculation
    """

    unique_ids = (
        df["id"]
        .unique()
    )

    if len(unique_ids) == 0:
        raise ValueError(
            "Cannot calculate SOFA variance "
            "for an empty category."
        )

    temp: list[float] = []

    for probe_id in unique_ids:
        temp = []

        probe = df[
            df["id"] == probe_id
        ]

        temp.append(
            probe[
                "_sofa_value"
            ].var()
        )

    score = (
        sum(temp)
        / len(temp)
    )

    return float(
        round(
            score,
            3,
        )
    )


def _configure_tokenizer_for_sofa(
    tokenizer: Any,
    batch_size: int,
) -> None:

    if (
        tokenizer.pad_token is not None
        or batch_size <= 1
    ):
        return

    existing_special_tokens = list(
        tokenizer.special_tokens_map_extended.values()
    )

    if not existing_special_tokens:
        raise ValueError(
            "SOFA evaluation with batch_size > 1 "
            "requires the tokenizer to define at least "
            "one special token that can be used for padding."
        )

    tokenizer.add_special_tokens(
        {
            "pad_token":
                existing_special_tokens[0]
        }
    )


def _seed_everything(
    seed: int,
) -> None:
    """Seed libraries"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def _evaluation_device(
    model: Any,
) -> str:

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

            device = torch.device(
                device
            )

            if device.type != "meta":
                return device.type

    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

