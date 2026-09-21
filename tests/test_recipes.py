from pathlib import Path

from dike.config import load_recipe


ROOT = Path(__file__).resolve().parents[1]


def test_base_recipe():
    config = load_recipe(
        ROOT / "recipes" / "base.yaml"
    )

    assert config.mode == "base"

    assert config.evaluations == [
        "perplexity",
        "holistic_bias",
        "sofa",
    ]

    assert len(config.lm_eval) == 1

    assert set(
        config.lm_eval[0].tasks
    ) == {
        "bbq",
        "crows_pairs_english",
    }


def test_instruct_recipe():
    config = load_recipe(
        ROOT / "recipes" / "instruct.yaml"
    )

    assert config.mode == "instruct"

    assert config.evaluations == [
        "perplexity",
        "moral_stories",
    ]

    moral = config.options_for(
        "moral_stories"
    )

    assert moral["seed"] == 0
    assert moral["batch_size"] == 8

    tasks = {
        task
        for run in config.lm_eval
        for task in run.tasks
    }

    assert tasks == {
        "realtoxicityprompts",
        "ethics_cm",
        "harmbench",
    }