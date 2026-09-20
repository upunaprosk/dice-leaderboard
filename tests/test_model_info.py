import json

from dike.config import ModelConfig
from dike.model_info import inspect_model


def _write_json(
    path,
    value,
):
    path.write_text(
        json.dumps(value),
        encoding="utf-8",
    )


def test_dense_base_model(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "torch_dtype": "bfloat16",
        },
    )

    config = ModelConfig(
        model=str(tmp_path),
        mode="base",
    )

    info = inspect_model(
        config
    )

    assert info["mode"] == "base"
    assert info["compression"]["compressed"] is False
    assert info["compression"]["label"] == "Dense BF16"


def test_dense_instruct_model(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "torch_dtype": "float16",
        },
    )

    config = ModelConfig(
        model=str(tmp_path),
        mode="instruct",
    )

    info = inspect_model(
        config
    )

    assert info["mode"] == "instruct"
    assert info["compression"]["compressed"] is False
    assert info["compression"]["label"] == "Dense FP16"


def test_gptq_model(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "torch_dtype": "float16",
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "desc_act": True,
                "checkpoint_format": "gptq",
                "meta": {
                    "quantizer": [
                        "gptqmodel:2.0.0",
                    ],
                },
            },
        },
    )

    config = ModelConfig(
        model=str(tmp_path),
        backend="gptq",
    )

    info = inspect_model(
        config
    )

    compression = info[
        "compression"
    ]

    assert compression["compressed"] is True
    assert compression["framework"] == "gptqmodel"
    assert compression["format"] == "gptq"
    assert compression["label"] == "GPTQ W4A16 G128"

    step = compression["steps"][0]

    assert step["algorithm"] == "gptq"
    assert step["weight_bits"] == 4
    assert step["group_size"] == 128


def test_runtime_bitsandbytes_int4(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "torch_dtype": "float16",
        },
    )

    config = ModelConfig(
        model=str(tmp_path),
        quantization="int4",
    )

    info = inspect_model(
        config
    )

    compression = info[
        "compression"
    ]

    assert compression["compressed"] is True
    assert compression["framework"] == "bitsandbytes"
    assert compression["label"] == "BitsAndBytes INT4"


def test_sparsegpt_plus_gptq_recipe(tmp_path):
    _write_json(
        tmp_path / "config.json",
        {
            "torch_dtype": "float16",
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "compressed-tensors",
            },
        },
    )

    (
        tmp_path
        / "recipe.yaml"
    ).write_text(
        """
stage:
  SparseGPTModifier:
    sparsity: 0.5
    mask_structure: "2:4"

  GPTQModifier:
    scheme: W4A16
    block_size: 128
""",
        encoding="utf-8",
    )

    config = ModelConfig(
        model=str(tmp_path),
    )

    info = inspect_model(
        config
    )

    label = info[
        "compression"
    ][
        "label"
    ]

    assert "SparseGPT 50% 2:4" in label
    assert "GPTQ W4A16 G128" in label