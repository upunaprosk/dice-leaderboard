import csv
import json

from dike.leaderboard import (
    export_leaderboard,
)


def test_export_base_leaderboard(
    tmp_path,
):
    result = {
        "model": {
            "name": "org/model",
            "display_name": "model",
            "mode": "base",
            "compression": {
                "label": "GPTQ W4A16 G128",
            },
            "source": {
                "type": "huggingface",
                "url": "https://huggingface.co/org/model",
            },
        },
        "evaluations": {
            "perplexity": {
                "score": 12.345,
            },
            "holistic_bias": {
                "score": 0.25,
            },
            "stereoset": {
                "icat_score": 71.2,
            },
            "sofa": {
                "score": 0.123,
            },
        },
        "lm_eval": {
            "base": {
                "groups": {
                    "crows_pairs_english": {
                        "pct_stereotype,none": 0.60,
                    },
                },
                "results": {},
                "n-samples": {},
            },
        },
    }

    source = (
        tmp_path
        / "model_base_results.json"
    )

    source.write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    output = (
        tmp_path
        / "leaderboard.csv"
    )

    export_leaderboard(
        inputs=[source],
        output=output,
        mode="base",
    )

    with output.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.reader(file)
        )

    assert rows[0] == [
        "Model",
        "Model Recipe",
        "PPL",
        "CP",
        "Holistic Bias",
        "StereoSet",
        "SOFA",
        "Link",
    ]

    assert rows[1][0] == "org/model"
    assert rows[1][1] == "GPTQ W4A16 G128"
    assert rows[1][2] == "12.35"
    assert rows[1][4] == "25.00"
    assert rows[1][5] == "71.20"

def test_export_instruct_leaderboard(
    tmp_path,
):
    result = {
        "model": {
            "name": "org/model-instruct",
            "mode": "instruct",
            "compression": {
                "label": "Dense BF16",
            },
            "source": {
                "type": "huggingface",
                "url": (
                    "https://huggingface.co/"
                    "org/model-instruct"
                ),
            },
        },
        "evaluations": {
            "perplexity": {
                "score": 8.0,
            },
            "moral_stories": {
                "moral": 80,
                "immoral": 15,
                "refused": 5,
                "total": 100,
            },
        },
        "lm_eval": {
            "rtp": {
                "results": {
                    "realtoxicityprompts": {
                        "score,none": 0.12,
                    },
                },
            },
            "ethics": {
                "results": {
                    "ethics_cm": {
                        "acc,none": 0.71,
                    },
                    "harmbench": {
                        "score,none": 0.18,
                    },
                },
            },
        },
    }

    source = (
        tmp_path
        / "model_instruct_results.json"
    )

    source.write_text(
        json.dumps(result),
        encoding="utf-8",
    )

    output = (
        tmp_path
        / "instruct.csv"
    )

    export_leaderboard(
        inputs=[source],
        output=output,
        mode="instruct",
    )

    with output.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.reader(file)
        )

    row = rows[1]

    assert row[0] == "org/model-instruct"
    assert row[1] == "Dense BF16"
    assert row[2] == "8.00"
    assert row[3] == "71.00"
    assert row[4] == "80.00"
    assert row[5] == "5.00"
    assert row[6] == "12.00"
    assert row[7] == "18.00"