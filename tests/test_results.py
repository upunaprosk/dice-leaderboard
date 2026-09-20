from dike.config import (
    EvaluationConfig,
    ModelConfig,
)
from dike.results import (
    add_evaluation_result,
    add_lm_eval_result,
    create_results,
    save_results,
)


def test_create_results(
    monkeypatch,
):
    def fake_inspect_model(config):
        return {
            "display_name": "test-model",
            "base_model": None,
            "dtype": "BF16",
            "source": {
                "type": "huggingface",
                "url": "https://huggingface.co/org/test-model",
                "revision": None,
            },
            "compression": {
                "compressed": False,
                "steps": [],
                "format": None,
                "framework": None,
                "label": "Dense BF16",
                "detected_from": [],
            },
            "warnings": [],
        }

    monkeypatch.setattr(
        "dike.results.inspect_model",
        fake_inspect_model,
    )

    model = ModelConfig(
        model="org/test-model",
        mode="base",
    )

    evaluation = EvaluationConfig(
        evaluations=[
            "perplexity",
        ],
        mode="base",
    )

    results = create_results(
        model_config=model,
        evaluation_config=evaluation,
    )

    assert results["model"]["name"] == "org/test-model"
    assert (
        results["model"]["compression"]["label"]
        == "Dense BF16"
    )

    assert results["evaluations"] == {}
    assert results["lm_eval"] == {}


def test_add_evaluation_result():
    results = {
        "evaluations": {},
    }

    add_evaluation_result(
        results,
        "perplexity",
        {
            "score": 12.5,
        },
    )

    assert (
        results["evaluations"]["perplexity"]["score"]
        == 12.5
    )


def test_lm_eval_result_does_not_overwrite():
    results = {
        "lm_eval": {},
    }

    add_lm_eval_result(
        results,
        ["hellaswag"],
        {"score": 1},
    )

    add_lm_eval_result(
        results,
        ["hellaswag"],
        {"score": 2},
    )

    assert "hellaswag" in results["lm_eval"]
    assert "hellaswag#2" in results["lm_eval"]


def test_save_results(tmp_path):
    results = {
        "model": {
            "name": "org/model",
            "mode": "base",
        },
        "evaluations": {},
        "lm_eval": {},
    }

    path = save_results(
        results=results,
        output_dir=tmp_path,
    )

    assert path.exists()
    assert path.name == "org_model_base_results.json"