import pytest

from dike.config import (
    EvaluationConfig,
    LMEvalRunConfig,
    ModelConfig,
)


def test_model_config_defaults_tokenizer_to_model():
    config = ModelConfig(
        model="org/model",
    )

    assert config.tokenizer == "org/model"
    assert config.backend == "transformers"
    assert config.use_fast_tokenizer is False


def test_gptq_cannot_use_bitsandbytes_quantization():
    with pytest.raises(
        ValueError,
        match="already quantized",
    ):
        ModelConfig(
            model="org/model",
            backend="gptq",
            quantization="int4",
        )


def test_lm_eval_backend_is_independent():
    model = ModelConfig(
        model="org/model",
        backend="transformers",
    )

    run = LMEvalRunConfig(
        tasks=["hellaswag"],
        backend="vllm",
    )

    assert model.backend == "transformers"
    assert run.backend == "vllm"


def test_evaluation_config_requires_work():
    with pytest.raises(ValueError):
        EvaluationConfig()


def test_recipe_mode_must_match_model():
    evaluation = EvaluationConfig(
        evaluations=["perplexity"],
        mode="base",
    )

    model = ModelConfig(
        model="org/model",
        mode="instruct",
    )

    with pytest.raises(
        ValueError,
        match="recipe is for 'base'",
    ):
        evaluation.validate_model(
            model
        )