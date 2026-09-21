<div align="center">

<p align="center">
  <img src="./logo.png" alt="Open Diké" width="140">
</p>

# Open Diké Leaderboard

### Bias, Fairness, Ethics, and Safety in Compressed LLMs

Evaluation framework for the [Open Diké Leaderboard](https://huggingface.co/spaces/LabHC/dike-leaderboard), developed as part of the [Diké Project](https://www.anr-dike.fr/).

🤗 **[View the Open Diké Leaderboard on Hugging Face](https://huggingface.co/spaces/LabHC/dike-leaderboard)**

[![Tests](https://github.com/upunaprosk/dice-leaderboard/actions/workflows/tests.yml/badge.svg)](https://github.com/upunaprosk/dice-leaderboard/actions/workflows/tests.yml)
[![Release](https://img.shields.io/github/v/release/upunaprosk/dice-leaderboard)](https://github.com/upunaprosk/dice-leaderboard/releases)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/upunaprosk/dice-leaderboard)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/spaces/LabHC/dike-leaderboard)

</div>


---

## Overview

The Open Diké Leaderboard is designed to evaluate how model compression affects bias, fairness, ethical alignment, and safety in large language model outputs.

The leaderboard is organized into two evaluation tracks, presented as separate subpages:

* **Base models**
* **Instruction-tuned models**

Dense and compressed models are supported, including GPTQ, AWQ, BitsAndBytes INT4/INT8, and sparse models.

Model and compression metadata are detected automatically when available.

---

## Evaluation

### Base models metrics

Defined in `recipes/base.yaml`.

| Metric            | Description                                                           |
| ----------------- | --------------------------------------------------------------------- |
| **PPL**           | WikiText-2 perplexity (**lower is better**)                           |
| **BBQ**           | Bias Benchmark for Question Answering accuracy (**higher is better**) |
| **CrowS-Pairs**   | Stereotype preference (**closer to 50% is better**)                   |
| **Holistic Bias** | Negative-sentiment preference rate (**closer to 50% is better**)            |
| **SOFA**          | Social fairness bias score (**lower is better**)                      |

### Instruction-tuned models metrics

Defined in `recipes/instruct.yaml`.

| Metric                      | Description                                          |
| --------------------------- | ---------------------------------------------------- |
| **PPL**                     | WikiText-2 perplexity (**lower is better**)          |
| **ETHICS**                  | Commonsense morality accuracy (**higher is better**) |
| **Moral Stories**           | Morally aligned response rate (**higher is better**) |
| **Moral Stories (Refusal)** | Refusal rate (**lower is better**)                   |
| **RealToxicityPrompts**     | Toxicity score (**lower is better**)                 |
| **HarmBench**               | Harmful-response rate (**lower is better**)          |

---

## Installation

```
git clone https://github.com/upunaprosk/dice-leaderboard.git
cd dice-leaderboard

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Set a Hugging Face token before evaluation created at https://huggingface.co/settings/tokens:

```
export HF_TOKEN=hf_your_token_here
```

---

## Running Evaluations

### Base model

```
dike eval \
  --model facebook/opt-125m \
  --mode base
```

### Instruction-tuned model

```
dike eval \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --mode instruct
```

The corresponding recipe is selected automatically from:

```
recipes/base.yaml
recipes/instruct.yaml
```

### Evaluation arguments

The `dike eval` arguments:
| Argument                        | Values / default         | Description                                                                                                                                                                                                                               |
| ------------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model MODEL`                 | Required                 | HF model or path to a local checkpoint. The path may refer to a dense, quantized, sparse, or otherwise compressed model.                                                                                         |
| `--mode {base,instruct}`        | `base`, `instruct`       | Evaluation mode. The `base` mode evaluates pretrained language models, while `instruct` evaluates instruction-tuned models.                                                                                                              |
| `--recipe PATH`                 | Optional                 | Path to a custom evaluation recipe.                                                                                                          |
| `--backend {transformers,gptq}` | Default: `transformers`  | Backend used to load the model. `transformers` is used for standard HF checkpoints, including models compressed with LLM Compressor or stored using `compressed-tensors`. `gptq` is used for checkpoints requiring GPTQModel.   |
| `--quantization {int4,int8}`    | Optional                 | Applies BitsAndBytes 4-bit or 8-bit quantization at model loading time. This option is intended for dense checkpoints and should not be used when the model is already quantized.                                                         |
| `--gptq-backend BACKEND`        | Default: `auto`          | Execution backend used by GPTQModel when `--backend gptq` is selected.                                                                           |
| `--lm-eval-backend {hf,vllm}`   | Default: `hf`            | Backend used for evaluations with `lm-evaluation-harness`. Use `vllm` for sparse compressed models with `llm-compressor`.                                                                                       |
| `--tokenizer TOKENIZER`         | Default: model tokenizer | HF tokenizer or path to a local tokenizer. This option is required only when the tokenizer differs from the evaluated model path.                                                                              |
| `--revision REVISION`           | Optional                 | Model revision to load: a branch, tag, or commit identifier.                                                                                                                                  |
| `--tokenizer-revision REVISION` | Optional                 | Tokenizer revision to load when it differs from the model revision.                                                                                                                                                                       |
| `--device-map DEVICE_MAP`       | Default: `auto`          | Device configuration used when loading the model.                                                                                                                                                                               |
| `--trust-remote-code`           | Disabled by default      | Enables model- or tokenizer-specific code provided by the corresponding HF repository.                                                                                                                                          |
| `--use-fast-tokenizer`          | Disabled by default      | Uses the fast tokenizer implementation when available.                                                                                                                                                                                    |
| `--output-dir PATH`             | Default: `results`       | Directory in which evaluation results are stored.                                                                                                                                                                                         |
| `--output-file PATH`            | Optional                 | Explicit path for the generated JSON result file.                                                                                                                                                                                         |
| `--seed INTEGER`                | Recipe default           | Overrides the random seed defined in the evaluation recipe.                                                                                                                                                                               |
| Sparse / LLM Compressor models  | `--backend transformers` | Sparse checkpoints, including models produced with SparseGPT, Wanda, magnitude pruning, structured sparsity, or LLM Compressor, are evaluated directly from the compressed checkpoint. No additional sparsification argument is required. |
| Quantized and sparse models     | `--backend transformers` | Models combining sparsity and quantization, for example SparseGPT with GPTQ or other `compressed-tensors` recipes, are loaded as already-compressed checkpoints. Runtime `--quantization` should not be applied to such models.           |


---

## Compressed Models

### GPTQ

Install GPTQModel support:

```
pip install -e ".[dev,gptq]"
```

Evaluate a GPTQ-quantized model:

```
dike eval \
  --model iproskurina/opt-125m-GPTQ-4bit-g128 \
  --mode base \
  --backend gptq
```

### BitsAndBytes

Install BitsAndBytes:

```
pip install -e ".[dev,bnb]"
```

Runtime INT4:

```
dike eval \
  --model HuggingFaceTB/SmolLM2-1.7B \
  --mode base \
  --quantization int4
```

Runtime INT8:

```
dike eval \
  --model HuggingFaceTB/SmolLM2-1.7B \
  --mode base \
  --quantization int8
```

### AWQ and sparse models

AWQ, SparseGPT, Wanda, and other checkpoints supported directly by Transformers can normally be evaluated without a special backend:

```
dike eval \
  --model MODEL_NAME \
  --mode base
```

Diké records available compression metadata in the result JSON.

---

## Results

Results are saved as JSON:

```
results/facebook_opt-125m_base_results.json
```

They contain evaluation scores together with model, compression, and run metadata.

Example model-recipe labels:

```
Dense BF16
GPTQ W4A16 G128
AWQ W4A16 G128
BitsAndBytes INT4
SparseGPT 50% + GPTQ W4A16 G128
```

---

## Generate Leaderboard scores

Base models:

```
dike leaderboard \
  results/ \
  --mode base \
  --output leaderboard_base.csv
```

Instruction-tuned models:

```
dike leaderboard \
  results/ \
  --mode instruct \
  --output leaderboard_instruct.csv
```

---

## Testing

Unit tests:

```
python -m pytest -m "not integration"
```

Integration tests with models:

```
python -m pytest -m integration
```

---

## Citation

If you use the Open Diké Leaderboard in your work, please cite the software.

**BibTeX**

```
@software{proskurina_open_dike_leaderboard,
  author    = {Proskurina, Irina},
  title     = {Open Diké Leaderboard},
  version   = {0.1.0},
  url       = {https://github.com/upunaprosk/dice-leaderboard}
}
```

**Plain**

```
Proskurina, I. Open Diké Leaderboard, version 0.1.0.
https://github.com/upunaprosk/dice-leaderboard
```

Citation metadata is also available in [`CITATION.cff`](CITATION.cff).


---

## License

See [LICENSE](LICENSE).

