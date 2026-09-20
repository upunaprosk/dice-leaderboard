from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import hf_hub_download

from dike.config import ModelConfig


_METADATA_FILES = (
    "config.json",
    "quantize_config.json",
    "quant_config.json",
    "recipe.yaml",
    "recipe.yml",
    "README.md",
)


def inspect_model(
    config: ModelConfig,
) -> dict[str, Any]:
    """
    Inspect model metadata without downloading model weights.

    Supported:
    - dense base models
    - dense instruction-tuned models
    - GPTQ / GPTQModel
    - AWQ
    - BitsAndBytes
    - compressed-tensors
    - llm-compressor recipes
    - SparseGPT / Wanda / magnitude pruning
    - SmoothQuant
    - combinations such as SparseGPT + GPTQ

    Runtime BitsAndBytes quantization configured through ModelConfig is
    also recorded even when the source repository itself is dense.
    """

    files, warnings = _load_metadata_files(
        model=config.model,
        revision=config.revision,
    )

    model_config = _read_json(
        files.get("config.json")
    )

    quantize_config = _read_json(
        files.get("quantize_config.json")
        or files.get("quant_config.json")
    )

    recipe = _read_yaml(
        files.get("recipe.yaml")
        or files.get("recipe.yml")
    )

    card = _read_model_card_metadata(
        files.get("README.md")
    )

    dtype = _detect_dtype(
        model_config
    )

    quantization_config = _merge_quantization_configs(
        model_config=model_config,
        quantize_config=quantize_config,
    )

    compression = _detect_compression(
        model_config=config,
        quantization_config=quantization_config,
        recipe=recipe,
        dtype=dtype,
    )

    display_name = _display_name(
        config.model
    )

    source_type = (
        "local"
        if Path(config.model).is_dir()
        else "huggingface"
    )

    source_url = None

    if source_type == "huggingface":
        source_url = (
            f"https://huggingface.co/{config.model}"
        )

    return {
        "name": config.model,
        "display_name": display_name,
        "mode": config.mode,
        "base_model": _base_model(card),
        "dtype": dtype,
        "source": {
            "type": source_type,
            "url": source_url,
            "revision": config.revision,
        },
        "compression": compression,
        "warnings": warnings,
    }


def _load_metadata_files(
    model: str,
    revision: str | None,
) -> tuple[dict[str, Path], list[str]]:
    """
    Resolve small metadata files from either a local model directory
    or a Hugging Face Hub repository.

    Model weights are never downloaded.
    """

    model_path = Path(model)

    if model_path.is_dir():
        files: dict[str, Path] = {}

        for filename in _METADATA_FILES:
            path = model_path / filename

            if path.is_file():
                files[filename] = path

        return files, []

    files = {}
    warnings: list[str] = []

    for filename in _METADATA_FILES:
        try:
            downloaded = hf_hub_download(
                repo_id=model,
                filename=filename,
                revision=revision,
            )

        except Exception:
            # Missing recipe.yaml does not prevent evaluation of a valid model (default is used)
            continue

        files[filename] = Path(
            downloaded
        )

    if "config.json" not in files:
        warnings.append(
            "Could not inspect config.json; model recipe detection "
            "may be incomplete."
        )

    return files, warnings


def _read_json(
    path: Path | None,
) -> dict[str, Any]:
    if path is None:
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            value = json.load(
                file
            )

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return {}

    if isinstance(
        value,
        dict,
    ):
        return value

    return {}


def _read_yaml(
    path: Path | None,
) -> dict[str, Any]:
    if path is None:
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            value = yaml.safe_load(
                file
            )

    except (
        OSError,
        yaml.YAMLError,
    ):
        return {}

    if isinstance(
        value,
        dict,
    ):
        return value

    return {}


def _read_model_card_metadata(
    path: Path | None,
) -> dict[str, Any]:
    """
    Read YAML front matter from model card.
    Only the metadata header is parsed; the model-card body is ignored
    """

    if path is None:
        return {}

    try:
        text = path.read_text(
            encoding="utf-8"
        )

    except OSError:
        return {}

    if not text.startswith("---"):
        return {}

    parts = text.split(
        "---",
        2,
    )

    if len(parts) < 3:
        return {}

    try:
        metadata = yaml.safe_load(
            parts[1]
        )

    except yaml.YAMLError:
        return {}

    if isinstance(
        metadata,
        dict,
    ):
        return metadata

    return {}


def _base_model(
    card: dict[str, Any],
) -> str | list[str] | None:
    value = card.get(
        "base_model"
    )

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        list,
    ):
        return [
            str(item)
            for item in value
        ]

    return None


def _display_name(
    model: str,
) -> str:
    path = Path(
        model
    )

    if path.is_dir():
        return path.name

    return model.rstrip(
        "/"
    ).split("/")[-1]


def _detect_dtype(
    model_config: dict[str, Any],
) -> str | None:
    value = (
        model_config.get("dtype")
        or model_config.get("torch_dtype")
    )

    if value is None:
        return None

    return _normalize_dtype(
        str(value)
    )


def _normalize_dtype(
    dtype: str,
) -> str:
    value = (
        dtype
        .lower()
        .replace("torch.", "")
        .replace("_", "")
    )

    aliases = {
        "bfloat16": "BF16",
        "bf16": "BF16",
        "float16": "FP16",
        "fp16": "FP16",
        "half": "FP16",
        "float32": "FP32",
        "fp32": "FP32",
        "float64": "FP64",
        "fp64": "FP64",
        "float8e4m3fn": "FP8",
        "float8e5m2": "FP8",
    }

    return aliases.get(
        value,
        dtype.upper(),
    )


def _merge_quantization_configs(
    model_config: dict[str, Any],
    quantize_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge Transformers quantization_config with a standalone GPTQ-style quantization configuration when both exist
    """

    embedded = model_config.get(
        "quantization_config"
    )

    if not isinstance(
        embedded,
        dict,
    ):
        embedded = {}

    merged = dict(
        embedded
    )

    for key, value in quantize_config.items():
        if value is not None:
            merged.setdefault(
                key,
                value,
            )

    return merged


def _detect_compression(
    model_config: ModelConfig,
    quantization_config: dict[str, Any],
    recipe: dict[str, Any],
    dtype: str | None,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []

    detected_from: list[str] = []

    if recipe:
        recipe_steps = _steps_from_recipe(
            recipe
        )

        if recipe_steps:
            steps.extend(
                recipe_steps
            )

            detected_from.append(
                "recipe.yaml"
            )

    if quantization_config:
        _merge_checkpoint_compression(
            steps=steps,
            quantization_config=quantization_config,
        )

        detected_from.append(
            "quantization_config"
        )

    if model_config.quantization is not None:
        _merge_runtime_quantization(
            steps=steps,
            quantization=model_config.quantization,
        )

        detected_from.append(
            "runtime"
        )

    format_name = _compression_format(
        quantization_config
    )

    framework = _compression_framework(
        quantization_config=quantization_config,
        recipe=recipe,
        model_config=model_config,
    )

    steps = _deduplicate_steps(
        steps
    )

    compressed = bool(
        steps
    )

    if compressed:
        label = _format_recipe_label(
            steps=steps,
            format_name=format_name,
        )

    else:
        label = (
            f"Dense {dtype}"
            if dtype
            else "Dense"
        )

    return {
        "compressed": compressed,
        "steps": steps,
        "format": format_name,
        "framework": framework,
        "label": label,
        "detected_from": detected_from,
    }


def _steps_from_recipe(
    recipe: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract known compression modifiers from a recipe.yaml
    """

    modifiers = _find_modifiers(
        recipe
    )

    steps: list[dict[str, Any]] = []

    generic_quantization: list[
        dict[str, Any]
    ] = []

    for name, settings in modifiers:
        modifier = name.lower()

        if "sparsegpt" in modifier:
            steps.append(
                _pruning_step(
                    algorithm="sparsegpt",
                    settings=settings,
                )
            )

        elif "wanda" in modifier:
            steps.append(
                _pruning_step(
                    algorithm="wanda",
                    settings=settings,
                )
            )

        elif "magnitudepruning" in modifier:
            steps.append(
                _pruning_step(
                    algorithm="magnitude",
                    settings=settings,
                )
            )

        elif "constantpruning" in modifier:
            steps.append(
                _pruning_step(
                    algorithm="pruning",
                    settings=settings,
                )
            )

        elif "smoothquant" in modifier:
            step: dict[str, Any] = {
                "type": "quantization_preparation",
                "algorithm": "smoothquant",
            }

            strength = (
                settings.get(
                    "smoothing_strength"
                )
                or settings.get(
                    "smoothing_alpha"
                )
            )

            if strength is not None:
                step["strength"] = strength

            steps.append(
                step
            )

        elif "gptq" in modifier:
            step = _quantization_step_from_settings(
                algorithm="gptq",
                settings=settings,
            )

            steps.append(
                step
            )

        elif "awq" in modifier:
            step = _quantization_step_from_settings(
                algorithm="awq",
                settings=settings,
            )

            steps.append(
                step
            )

        elif (
            modifier == "quantizationmodifier"
            or "quantizationmodifier" in modifier
        ):
            generic_quantization.append(
                _quantization_step_from_settings(
                    algorithm="quantization",
                    settings=settings,
                )
            )

    # QuantizationModifier contains an actual bit-width
    for generic in generic_quantization:
        specific = _last_specific_quantization(
            steps
        )

        if specific is not None:
            _enrich_step(
                specific,
                generic,
            )
        else:
            steps.append(
                generic
            )

    return steps


def _find_modifiers(
    value: Any,
) -> list[tuple[str, dict[str, Any]]]:
    found: list[
        tuple[str, dict[str, Any]]
    ] = []

    if isinstance(
        value,
        dict,
    ):
        for key, child in value.items():
            key_text = str(
                key
            )

            if (
                "modifier" in key_text.lower()
                and isinstance(
                    child,
                    dict,
                )
            ):
                found.append(
                    (
                        key_text,
                        child,
                    )
                )

            found.extend(
                _find_modifiers(
                    child
                )
            )

    elif isinstance(
        value,
        list,
    ):
        for child in value:
            found.extend(
                _find_modifiers(
                    child
                )
            )

    return found


def _pruning_step(
    algorithm: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "type": "pruning",
        "algorithm": algorithm,
    }

    sparsity = _first_value(
        settings,
        (
            "sparsity",
            "final_sparsity",
            "target_sparsity",
        ),
    )

    structure = _first_value(
        settings,
        (
            "mask_structure",
            "sparsity_structure",
            "structure",
        ),
    )

    if sparsity is not None:
        step["sparsity"] = sparsity

    if structure is not None:
        step["structure"] = structure

    return step


def _quantization_step_from_settings(
    algorithm: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "type": "quantization",
        "algorithm": algorithm,
    }

    scheme = settings.get(
        "scheme"
    )

    if isinstance(
        scheme,
        str,
    ):
        _apply_scheme(
            step,
            scheme,
        )

    config_groups = settings.get(
        "config_groups"
    )

    if isinstance(
        config_groups,
        dict,
    ):
        _apply_config_groups(
            step,
            config_groups,
        )

    group_size = _first_value(
        settings,
        (
            "group_size",
            "block_size",
        ),
    )

    if group_size is not None:
        step.setdefault(
            "group_size",
            group_size,
        )

    return step


def _merge_checkpoint_compression(
    steps: list[dict[str, Any]],
    quantization_config: dict[str, Any],
) -> None:
    method = str(
        quantization_config.get(
            "quant_method",
            "",
        )
    ).lower()

    if method in {
        "gptq",
        "awq",
    }:
        step = _find_quantization_step(
            steps,
            method,
        )

        if step is None:
            step = {
                "type": "quantization",
                "algorithm": method,
            }

            steps.append(
                step
            )

        bits = quantization_config.get(
            "bits"
        )

        group_size = quantization_config.get(
            "group_size"
        )

        if bits is not None:
            step.setdefault(
                "weight_bits",
                bits,
            )

        if group_size is not None:
            step.setdefault(
                "group_size",
                group_size,
            )

        if "sym" in quantization_config:
            step.setdefault(
                "symmetric",
                quantization_config["sym"],
            )

        if "zero_point" in quantization_config:
            step.setdefault(
                "zero_point",
                quantization_config[
                    "zero_point"
                ],
            )

        if "desc_act" in quantization_config:
            step.setdefault(
                "desc_act",
                quantization_config[
                    "desc_act"
                ],
            )

        return

    if method in {
        "bitsandbytes",
        "bnb",
    }:
        _merge_bitsandbytes_config(
            steps,
            quantization_config,
        )

        return

    if method in {
        "compressed-tensors",
        "compressed_tensors",
    }:
        _merge_compressed_tensors_config(
            steps,
            quantization_config,
        )

        return

    if method:
        step = {
            "type": "quantization",
            "algorithm": method,
        }

        bits = quantization_config.get(
            "bits"
        )

        if bits is not None:
            step["weight_bits"] = bits

        steps.append(
            step
        )


def _merge_bitsandbytes_config(
    steps: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    step = _find_quantization_step(
        steps,
        "bitsandbytes",
    )

    if step is None:
        step = {
            "type": "quantization",
            "algorithm": "bitsandbytes",
        }

        steps.append(
            step
        )

    if config.get(
        "load_in_4bit"
    ):
        step["weight_bits"] = 4

        quant_type = config.get(
            "bnb_4bit_quant_type"
        )

        if quant_type:
            step["quant_type"] = str(
                quant_type
            ).upper()

        if config.get(
            "bnb_4bit_use_double_quant"
        ):
            step["double_quant"] = True

    elif config.get(
        "load_in_8bit"
    ):
        step["weight_bits"] = 8
        step["quant_type"] = "INT8"


def _merge_runtime_quantization(
    steps: list[dict[str, Any]],
    quantization: str,
) -> None:
    step = _find_quantization_step(
        steps,
        "bitsandbytes",
    )

    if step is None:
        step = {
            "type": "quantization",
            "algorithm": "bitsandbytes",
        }

        steps.append(
            step
        )

    if quantization == "int4":
        step["weight_bits"] = 4
        step.setdefault(
            "quant_type",
            "INT4",
        )

    elif quantization == "int8":
        step["weight_bits"] = 8
        step["quant_type"] = "INT8"


def _merge_compressed_tensors_config(
    steps: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    config_groups = config.get(
        "config_groups"
    )

    if isinstance(
        config_groups,
        dict,
    ):
        specific = _last_specific_quantization(
            steps
        )

        if specific is None:
            specific = {
                "type": "quantization",
                "algorithm": "compressed-tensors",
            }

            steps.append(
                specific
            )

        _apply_config_groups(
            specific,
            config_groups,
        )

    sparsity_config = config.get(
        "sparsity_config"
    )

    if isinstance(
        sparsity_config,
        dict,
    ):
        if not any(
            step.get("type") == "pruning"
            for step in steps
        ):
            step: dict[str, Any] = {
                "type": "pruning",
                "algorithm": "sparsity",
            }

            sparsity = _first_value(
                sparsity_config,
                (
                    "global_sparsity",
                    "sparsity",
                ),
            )

            structure = _first_value(
                sparsity_config,
                (
                    "sparsity_structure",
                    "structure",
                ),
            )

            if sparsity is not None:
                step["sparsity"] = sparsity

            if structure is not None:
                step["structure"] = structure

            steps.insert(
                0,
                step,
            )


def _apply_config_groups(
    step: dict[str, Any],
    config_groups: dict[str, Any],
) -> None:
    """
    Extract a quantization scheme
    """

    for group in config_groups.values():
        if not isinstance(
            group,
            dict,
        ):
            continue

        weights = group.get(
            "weights"
        )

        activations = group.get(
            "input_activations"
        )

        if isinstance(
            weights,
            dict,
        ):
            bits = weights.get(
                "num_bits"
            )

            if bits is not None:
                step.setdefault(
                    "weight_bits",
                    bits,
                )

            group_size = weights.get(
                "group_size"
            )

            if group_size is not None:
                step.setdefault(
                    "group_size",
                    group_size,
                )

            strategy = weights.get(
                "strategy"
            )

            if strategy is not None:
                step.setdefault(
                    "strategy",
                    strategy,
                )

            weight_type = weights.get(
                "type"
            )

            if weight_type is not None:
                step.setdefault(
                    "weight_type",
                    weight_type,
                )

            symmetric = weights.get(
                "symmetric"
            )

            if symmetric is not None:
                step.setdefault(
                    "symmetric",
                    symmetric,
                )

        if isinstance(
            activations,
            dict,
        ):
            bits = activations.get(
                "num_bits"
            )

            if bits is not None:
                step.setdefault(
                    "activation_bits",
                    bits,
                )

            activation_type = activations.get(
                "type"
            )

            if activation_type is not None:
                step.setdefault(
                    "activation_type",
                    activation_type,
                )

        if (
            "weight_bits" in step
            or "activation_bits" in step
        ):
            return


def _apply_scheme(
    step: dict[str, Any],
    scheme: str,
) -> None:
    """
    Parse schemes such as W4A16 and W8A8
    """

    value = scheme.upper()

    if not value.startswith(
        "W"
    ):
        return

    try:
        weights, activations = value.split(
            "A",
            1,
        )

        step["weight_bits"] = int(
            weights[1:]
        )

        step["activation_bits"] = int(
            activations
        )

    except (
        ValueError,
        IndexError,
    ):
        return


def _compression_format(
    config: dict[str, Any],
) -> str | None:
    value = (
        config.get("format")
        or config.get("checkpoint_format")
    )

    if value is not None:
        return str(
            value
        )

    method = config.get(
        "quant_method"
    )

    if method is not None:
        return str(
            method
        )

    return None


def _compression_framework(
    quantization_config: dict[str, Any],
    recipe: dict[str, Any],
    model_config: ModelConfig,
) -> str | None:
    if recipe:
        return "llm-compressor"

    meta = quantization_config.get(
        "meta"
    )

    if isinstance(
        meta,
        dict,
    ):
        quantizer = meta.get(
            "quantizer"
        )

        if isinstance(
            quantizer,
            list,
        ):
            for value in quantizer:
                if "gptqmodel" in str(
                    value
                ).lower():
                    return "gptqmodel"

        elif (
            quantizer
            and "gptqmodel"
            in str(quantizer).lower()
        ):
            return "gptqmodel"

    method = str(
        quantization_config.get(
            "quant_method",
            "",
        )
    ).lower()

    if method in {
        "bitsandbytes",
        "bnb",
    }:
        return "bitsandbytes"

    if model_config.backend == "gptq":
        return "gptqmodel"

    if model_config.quantization is not None:
        return "bitsandbytes"

    return None


def _format_recipe_label(
    steps: list[dict[str, Any]],
    format_name: str | None,
) -> str:
    parts: list[str] = []

    for step in steps:
        step_type = step.get(
            "type"
        )

        algorithm = str(
            step.get(
                "algorithm",
                "",
            )
        ).lower()

        if step_type == "pruning":
            name = {
                "sparsegpt": "SparseGPT",
                "wanda": "Wanda",
                "magnitude": "Magnitude",
                "pruning": "Pruned",
                "sparsity": "Sparse",
            }.get(
                algorithm,
                algorithm.title(),
            )

            values = [
                name
            ]

            sparsity = step.get(
                "sparsity"
            )

            if isinstance(
                sparsity,
                (int, float),
            ):
                value = (
                    sparsity * 100
                    if sparsity <= 1
                    else sparsity
                )

                values.append(
                    f"{value:g}%"
                )

            structure = step.get(
                "structure"
            )

            if structure:
                values.append(
                    str(structure)
                )

            parts.append(
                " ".join(values)
            )

        elif (
            step_type
            == "quantization_preparation"
        ):
            if algorithm == "smoothquant":
                parts.append(
                    "SmoothQuant"
                )
            else:
                parts.append(
                    algorithm.title()
                )

        elif step_type == "quantization":
            if algorithm == "gptq":
                name = "GPTQ"

            elif algorithm == "awq":
                name = "AWQ"

            elif algorithm == "bitsandbytes":
                name = "BitsAndBytes"

            elif algorithm == "compressed-tensors":
                name = "Compressed-Tensors"

            elif algorithm == "quantization":
                name = "Quantized"

            else:
                name = algorithm.upper()

            values = [
                name
            ]

            weight_bits = step.get(
                "weight_bits"
            )

            activation_bits = step.get(
                "activation_bits"
            )

            quant_type = step.get(
                "quant_type"
            )

            if algorithm == "bitsandbytes":
                if quant_type:
                    values.append(
                        str(quant_type)
                    )

                elif weight_bits:
                    values.append(
                        f"{weight_bits}-bit"
                    )

            elif weight_bits is not None:
                if activation_bits is None:
                    activation_bits = 16

                values.append(
                    f"W{weight_bits}A"
                    f"{activation_bits}"
                )

            group_size = step.get(
                "group_size"
            )

            if group_size not in {
                None,
                -1,
            }:
                values.append(
                    f"G{group_size}"
                )

            if step.get(
                "double_quant"
            ):
                values.append(
                    "double-quant"
                )

            parts.append(
                " ".join(values)
            )

    if parts:
        return " + ".join(
            parts
        )

    if format_name:
        return str(
            format_name
        )

    return "Compressed"


def _find_quantization_step(
    steps: list[dict[str, Any]],
    algorithm: str,
) -> dict[str, Any] | None:
    for step in steps:
        if (
            step.get("type")
            == "quantization"
            and step.get("algorithm")
            == algorithm
        ):
            return step

    return None


def _last_specific_quantization(
    steps: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for step in reversed(
        steps
    ):
        if (
            step.get("type")
            == "quantization"
            and step.get("algorithm")
            not in {
                None,
                "quantization",
                "compressed-tensors",
            }
        ):
            return step

    return None


def _enrich_step(
    target: dict[str, Any],
    source: dict[str, Any],
) -> None:
    for key, value in source.items():
        if key in {
            "type",
            "algorithm",
        }:
            continue

        target.setdefault(
            key,
            value,
        )


def _deduplicate_steps(
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[
        dict[str, Any]
    ] = []

    seen: set[
        tuple[Any, ...]
    ] = set()

    for step in steps:
        identity = (
            step.get("type"),
            step.get("algorithm"),
            step.get("weight_bits"),
            step.get("activation_bits"),
            step.get("sparsity"),
            step.get("structure"),
        )

        if identity in seen:
            continue

        seen.add(
            identity
        )

        result.append(
            step
        )

    return result


def _first_value(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
) -> Any:
    for key in keys:
        value = mapping.get(
            key
        )

        if value is not None:
            return value

    return None

