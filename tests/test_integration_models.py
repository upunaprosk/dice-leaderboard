import pytest
import torch

from dike.config import ModelConfig
from dike.models import load_model


pytestmark = pytest.mark.integration


def test_load_small_base_model():
    config = ModelConfig(
        model="facebook/opt-125m",
        mode="base",
        backend="transformers",
        device_map="cpu",
        use_fast_tokenizer=True,
    )

    model, tokenizer = load_model(
        config
    )

    assert model is not None
    assert tokenizer is not None

    inputs = tokenizer(
        "The capital of France is",
        return_tensors="pt",
    )

    with torch.no_grad():
        output = model(
            **inputs
        )

    assert output.logits.shape[0] == 1
    assert output.logits.shape[1] > 0


def test_load_small_instruct_model():
    config = ModelConfig(
        model=(
            "HuggingFaceTB/"
            "SmolLM2-135M-Instruct"
        ),
        mode="instruct",
        backend="transformers",
        device_map="cpu",
        use_fast_tokenizer=True,
    )

    model, tokenizer = load_model(
        config
    )

    assert model is not None
    assert tokenizer is not None

    messages = [
        {
            "role": "user",
            "content": "Say hello.",
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert isinstance(
        prompt,
        str,
    )

    assert prompt