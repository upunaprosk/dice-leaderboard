from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Literal


LeaderboardMode = Literal[
    "base",
    "instruct",
]


BASE_HEADER = [
    "Model",
    "Model Recipe",
    "PPL",
    "BBQ (Acc)",
    "CrowS-Pairs",
    "Holistic Bias",
    "SOFA",
    "StereoSet",
    "Link",
]

INSTRUCT_HEADER = [
    "Model",
    "Model Recipe",
    "PPL",
    "ETHICS",
    "Moral Stories",
    "Moral Stories (Refusal)",
    "RealToxicityPrompts",
    "HarmBench",
    "Link",
]


def export_leaderboard(
    inputs: Iterable[str | Path],
    output: str | Path,
    mode: LeaderboardMode | None = None,
) -> Path:
    """
    Export DICE result JSON files to leaderboard CSV
    `inputs` contains individual JSON files or directories.
    Directories are searched recursively for ``*_results.json``.
    """

    paths = _collect_result_files(
        inputs
    )

    if not paths:
        raise ValueError(
            "No result files found"
        )

    results = [
        _load_result(path)
        for path in paths
    ]

    if mode is not None:
        results = [
            result
            for result in results
            if _mode(result) == mode
        ]

        if not results:
            raise ValueError(
                f"No '{mode}' result files found"
            )

    else:
        modes = {
            _mode(result)
            for result in results
        }

        if len(modes) != 1:
            names = ", ".join(
                sorted(modes)
            )

            raise ValueError(
                "Result files contain multiple modes "
                f"({names}). Specify --mode base or "
                "--mode instruct"
            )

        mode = next(
            iter(modes)
        )

    rows = [
        _build_row(
            result=result,
            mode=mode,
        )
        for result in results
    ]

    output = Path(
        output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    header = (
        BASE_HEADER
        if mode == "base"
        else INSTRUCT_HEADER
    )

    file_exists = (
            output.exists()
            and output.stat().st_size > 0
    )

    if file_exists:
        with output.open(
                "r",
                newline="",
                encoding="utf-8",
        ) as file:
            reader = csv.reader(
                file
            )

            existing_header = next(
                reader,
                None,
            )

        if existing_header != header:
            raise ValueError(
                f"Leaderboard header does not match existing file: {output}"
            )

    with output.open(
            "a",
            newline="",
            encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        if not file_exists:
            writer.writerow(
                header
            )

        writer.writerows(
            rows
        )

    return output


def _collect_result_files(
    inputs: Iterable[str | Path],
) -> list[Path]:
    paths: list[Path] = []

    for value in inputs:
        path = Path(
            value
        )

        if path.is_file():
            paths.append(
                path
            )

        elif path.is_dir():
            paths.extend(
                path.rglob(
                    "*_results.json"
                )
            )

        else:
            raise FileNotFoundError(
                f"Result path not found: {path}"
            )

    return sorted(
        set(paths)
    )


def _load_result(
    path: Path,
) -> dict[str, Any]:
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
    ) as exc:
        raise ValueError(
            f"Could not read result file: {path}"
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            f"DICE result file must contain a JSON object: {path}"
        )

    if not isinstance(
        value.get("model"),
        dict,
    ):
        raise ValueError(
            f"Missing model metadata in: {path}"
        )

    return value


def _build_row(
    result: dict[str, Any],
    mode: LeaderboardMode,
) -> list[str]:
    if mode == "base":
        return _base_row(
            result
        )

    return _instruct_row(
        result
    )


def _base_row(
    result: dict[str, Any],
) -> list[str]:
    return [
        _model_name(result),
        _model_recipe(result),

        _fmt_number(
            _evaluation_metric(
                result,
                "perplexity",
                (
                    "score",
                    "perplexity",
                ),
            )
        ),

        _fmt_percent(
            _lm_eval_metric(
                result,
                task="bbq",
                metrics=(
                    "acc",
                ),
                allow_subtasks=True,
            )
        ),

        _fmt_percent(
            _lm_eval_metric(
                result,
                task="crows_pairs_english",
                metrics=(
                    "pct_stereotype",
                ),
                allow_subtasks=True,
            )
        ),

        _fmt_percent(
            _evaluation_metric(
                result,
                "holistic_bias",
                (
                    "score",
                    "bias_share",
                ),
            )
        ),

        _fmt_number(
            _evaluation_metric(
                result,
                "sofa",
                (
                    "score",
                    "sofa_score",
                ),
            ),
            digits=3,
        ),

        _fmt_number(
            _evaluation_metric(
                result,
                "stereoset",
                (
                    "icat_score",
                    "score",
                ),
            )
        ),

        _model_link(result),
    ]

def _instruct_row(
    result: dict[str, Any],
) -> list[str]:
    return [
        _model_name(result),
        _model_recipe(result),

        _fmt_number(
            _evaluation_metric(
                result,
                "perplexity",
                (
                    "score",
                    "perplexity",
                ),
            )
        ),

        _fmt_percent(
            _lm_eval_metric(
                result,
                task="ethics_cm",
                metrics=(
                    "acc",
                    "score",
                ),
            )
        ),

        _fmt_percent(
            _moral_stories_rate(
                result,
                kind="moral",
            )
        ),

        _fmt_percent(
            _moral_stories_rate(
                result,
                kind="refused",
            )
        ),

        _fmt_percent(
            _lm_eval_metric(
                result,
                task="realtoxicityprompts",
                metrics=(
                    "score",
                ),
            )
        ),

        _fmt_percent(
            _lm_eval_metric(
                result,
                task="harmbench",
                metrics=(
                    "score",
                    "acc",
                ),
            )
        ),

        _model_link(result),
    ]

def _mode(
    result: dict[str, Any],
) -> LeaderboardMode:
    mode = result.get(
        "model",
        {},
    ).get(
        "mode"
    )

    if mode not in {
        "base",
        "instruct",
    }:
        raise ValueError(
            f"Unsupported or missing model mode: {mode}"
        )

    return mode


def _model_name(
    result: dict[str, Any],
) -> str:
    model = result.get(
        "model",
        {},
    )

    display_name = model.get(
        "display_name"
    )

    if display_name:
        return str(display_name)

    name = model.get(
        "name"
    )

    if not name:
        return ""

    return str(name).rstrip("/").split("/")[-1]

def _model_recipe(
    result: dict[str, Any],
) -> str:
    compression = result.get(
        "model",
        {},
    ).get(
        "compression",
        {},
    )

    if not isinstance(
        compression,
        dict,
    ):
        return ""

    label = compression.get(
        "label"
    )

    return str(
        label or ""
    )


def _model_link(
    result: dict[str, Any],
) -> str:
    source = result.get(
        "model",
        {},
    ).get(
        "source",
        {},
    )

    if not isinstance(
        source,
        dict,
    ):
        return ""

    url = source.get(
        "url"
    )

    return str(
        url or ""
    )


def _evaluation_metric(
    result: dict[str, Any],
    evaluation: str,
    keys: tuple[str, ...],
) -> float | None:
    evaluations = result.get(
        "evaluations",
        {},
    )

    if not isinstance(
        evaluations,
        dict,
    ):
        return None

    value = evaluations.get(
        evaluation
    )

    if isinstance(
        value,
        (int, float),
    ):
        return float(
            value
        )

    if not isinstance(
        value,
        dict,
    ):
        return None

    for key in keys:
        metric = value.get(
            key
        )

        if isinstance(
            metric,
            (int, float),
        ):
            return float(
                metric
            )

    return None


def _lm_eval_metric(
    result: dict[str, Any],
    task: str,
    metrics: tuple[str, ...],
    allow_subtasks: bool = False,
) -> float | None:
    """
    Extract 1 lm-evaluation-harness metric.

    Exact task/group results are preferred.
    """
    # TODO: test for CP_english child tasks
    runs = result.get(
        "lm_eval",
        {},
    )

    if not isinstance(
        runs,
        dict,
    ):
        return None

    for run in runs.values():
        if not isinstance(
            run,
            dict,
        ):
            continue

        for section_name in (
            "groups",
            "results",
        ):
            section = run.get(
                section_name,
                {},
            )

            if not isinstance(
                section,
                dict,
            ):
                continue

            task_result = section.get(
                task
            )

            if isinstance(
                task_result,
                dict,
            ):
                metric = _metric_from_task(
                    task_result,
                    metrics,
                )

                if metric is not None:
                    return metric

    if allow_subtasks:
        return _aggregate_subtasks(
            runs=runs,
            prefix=f"{task}_",
            metrics=metrics,
        )

    return None


def _aggregate_subtasks(
    runs: dict[str, Any],
    prefix: str,
    metrics: tuple[str, ...],
) -> float | None:
    """
    Weighted fallback for an lm-eval group.
    """

    values: list[
        tuple[float, float | None]
    ] = []

    for run in runs.values():
        if not isinstance(
            run,
            dict,
        ):
            continue

        task_results = run.get(
            "results",
            {},
        )

        sample_counts = run.get(
            "n-samples",
            {},
        )

        if not isinstance(
            task_results,
            dict,
        ):
            continue

        for task_name, task_result in task_results.items():
            if not str(
                task_name
            ).startswith(
                prefix
            ):
                continue

            if not isinstance(
                task_result,
                dict,
            ):
                continue

            metric = _metric_from_task(
                task_result,
                metrics,
            )

            if metric is None:
                continue

            weight = None

            if isinstance(
                sample_counts,
                dict,
            ):
                sample = sample_counts.get(
                    task_name
                )

                if isinstance(
                    sample,
                    dict,
                ):
                    count = (
                        sample.get("effective")
                        or sample.get("original")
                    )

                    if isinstance(
                        count,
                        (int, float),
                    ):
                        weight = float(
                            count
                        )

                elif isinstance(
                    sample,
                    (int, float),
                ):
                    weight = float(
                        sample
                    )

            values.append(
                (
                    metric,
                    weight,
                )
            )

    if not values:
        return None

    if all(
        weight is not None
        for _, weight in values
    ):
        total_weight = sum(
            weight
            for _, weight in values
            if weight is not None
        )

        if total_weight:
            return sum(
                value * weight
                for value, weight in values
                if weight is not None
            ) / total_weight

    return sum(
        value
        for value, _ in values
    ) / len(
        values
    )


def _metric_from_task(
    task_result: dict[str, Any],
    metrics: tuple[str, ...],
) -> float | None:
    for metric in metrics:
        candidates = (
            f"{metric},none",
            metric,
        )

        for key in candidates:
            value = task_result.get(
                key
            )

            if isinstance(
                value,
                (int, float),
            ):
                return float(
                    value
                )

        prefix = f"{metric},"

        for key, value in task_result.items():
            if (
                str(key).startswith(
                    prefix
                )
                and not str(key).endswith(
                    "_stderr"
                )
                and isinstance(
                    value,
                    (int, float),
                )
            ):
                return float(
                    value
                )

    return None


def _moral_stories_rate(
    result: dict[str, Any],
    kind: Literal[
        "moral",
        "refused",
    ],
) -> float | None:
    evaluations = result.get(
        "evaluations",
        {},
    )

    if not isinstance(
        evaluations,
        dict,
    ):
        return None

    value = evaluations.get(
        "moral_stories"
    )

    if not isinstance(
        value,
        dict,
    ):
        return None

    english = value.get(
        "en"
    )

    if isinstance(
        english,
        dict,
    ):
        value = english

    rate_keys = {
        "moral": (
            "moral_rate",
            "moral_share",
        ),
        "refused": (
            "refusal_rate",
            "refused_rate",
            "refusal_share",
        ),
    }

    for key in rate_keys[
        kind
    ]:
        metric = value.get(
            key
        )

        if isinstance(
            metric,
            (int, float),
        ):
            return float(
                metric
            )

    count = value.get(
        kind
    )

    total = (
        value.get("total")
        or value.get("samples")
        or value.get("num_samples")
    )

    if (
        isinstance(
            count,
            (int, float),
        )
        and isinstance(
            total,
            (int, float),
        )
        and total > 0
    ):
        return float(
            count
        ) / float(
            total
        )

    return None


def _fmt_percent(
    value: float | None,
) -> str:
    if value is None:
        return ""

    return f"{value * 100:.2f}"


def _fmt_number(
    value: float | None,
    digits: int = 2,
) -> str:
    if value is None:
        return ""

    return f"{value:.{digits}f}"

