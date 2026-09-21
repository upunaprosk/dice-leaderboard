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

The leaderboard is organized into two evaluation tracks:

* **Base models**
* **Instruction-tuned models**

Dense and compressed models are supported, including GPTQ, AWQ, BitsAndBytes INT4/INT8, sparse models, and hybrid compression recipes.

Model and compression metadata are detected automatically when available.

---

## Contribute a model

The Open Diké Leaderboard accepts community model evaluations.

To contribute a dense or compressed model:

1. Choose a Hugging Face model or compression recipe that you want to evaluate.
2. Run the official `base` or `instruct` evaluation without modifying the evaluation protocol.
3. Keep the generated result JSON.
4. Submit the result through a GitHub pull request.

Leaderboard scores should come directly from the generated evaluation result and should not be edited manually.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full submission procedure.

---

## Evaluation

### Base models metrics

Defined in `recipes/base.yaml`.

| Metric            | Description                                                           |
| ----------------- | --------------------------------------------------------------------- |
| **PPL**           | WikiText-2 perplexity (**lower is better**)                           |
| **BBQ**           | Bias Benchmark for Question Answering accuracy (**higher is better**) |
| **CrowS-Pairs**   | Stereotype preference (**closer to 50% is better**)                   |
| **Holistic Bias** | Negative-sentiment preference rate (**closer to 50% is better**)      |
| **SOFA**          | Social fairness bias score (**lower is better**)                      |
| **StereoSet**     | ICAT score combining language-model ability and stereotype preference |

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

<details>
<summary><b>Using uv</b></summary>

<br>

```bash
git clone https://github.com/upunaprosk/dice-leaderboard.git
cd dice-leaderboard

uv sync --locked
```

</details>

<details>
<summary><b>Using pip</b></summary>

<br>

```bash
git clone https://github.com/upunaprosk/dice-leaderboard.git
cd dice-leaderboard

python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

</details>

Set a Hugging Face token before evaluation. Tokens can be created at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Linux/macOS:

```bash
export HF_TOKEN=hf_your_token_here
```

Windows PowerShell:

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

---

## Running Evaluations

<details>
<summary><b>Base model</b></summary>

<br>

```bash
dike eval \
  --model facebook/opt-125m \
  --mode base
```

The corresponding evaluation recipe is:

```text
recipes/base.yaml
```

</details>

<details>
<summary><b>Instruction-tuned model</b></summary>

<br>

```bash
dike eval \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --mode instruct
```

The corresponding evaluation recipe is:

```text
recipes/instruct.yaml
```

</details>

<details>
<summary><b>GPTQ model</b></summary>

<br>

Install GPTQModel support:

```bash
pip install -e ".[dev,gptq]"
```

Evaluate:

```bash
dike eval \
  --model iproskurina/opt-125m-GPTQ-4bit-g128 \
  --mode base \
  --backend gptq
```

</details>

<details>
<summary><b>BitsAndBytes INT4 / INT8</b></summary>

<br>

Install BitsAndBytes:

```bash
pip install -e ".[dev,bnb]"
```

Runtime INT4:

```bash
dike eval \
  --model HuggingFaceTB/SmolLM2-1.7B \
  --mode base \
  --quantization int4
```

Runtime INT8:

```bash
dike eval \
  --model HuggingFaceTB/SmolLM2-1.7B \
  --mode base \
  --quantization int8
```

Runtime quantization is intended for dense checkpoints and should not be applied to a checkpoint that is already quantized.

</details>

<details>
<summary><b>AWQ, sparse, and compressed-tensors models</b></summary>

<br>

Already-compressed checkpoints supported by Transformers can normally be evaluated directly:

```bash
dike eval \
  --model MODEL_NAME \
  --mode base \
  --backend transformers
```

This includes supported AWQ, SparseGPT, Wanda, structured or unstructured sparse, and `compressed-tensors` checkpoints.

If the tokenizer is stored separately:

```bash
dike eval \
  --model MODEL_NAME \
  --tokenizer TOKENIZER_NAME \
  --mode base
```

If model and tokenizer revisions differ:

```bash
dike eval \
  --model MODEL_NAME \
  --revision MODEL_REVISION \
  --tokenizer TOKENIZER_NAME \
  --tokenizer-revision TOKENIZER_REVISION \
  --mode base
```

No additional sparsification argument is required when evaluating an already-compressed checkpoint.

</details>

---

## Evaluation arguments

<details>
<summary><b>Show dike eval arguments</b></summary>

<br>

| Argument                        | Values / default         | Description                                                |
| ------------------------------- | ------------------------ | ---------------------------------------------------------- |
| `--model MODEL`                 | Required                 | Hugging Face model or path to a local checkpoint.          |
| `--mode {base,instruct}`        | `base`, `instruct`       | Evaluation mode.                                           |
| `--recipe PATH`                 | Optional                 | Path to a custom evaluation recipe.                        |
| `--backend {transformers,gptq}` | Default: `transformers`  | Model-loading backend.                                     |
| `--quantization {int4,int8}`    | Optional                 | Applies BitsAndBytes runtime 4-bit or 8-bit quantization.  |
| `--gptq-backend BACKEND`        | Default: `auto`          | GPTQModel execution backend.                               |
| `--lm-eval-backend {hf,vllm}`   | Default: `hf`            | Execution backend for `lm-evaluation-harness` evaluations. |
| `--tokenizer TOKENIZER`         | Default: model tokenizer | Hugging Face tokenizer or local tokenizer path.            |
| `--revision REVISION`           | Optional                 | Model branch, tag, or commit identifier.                   |
| `--tokenizer-revision REVISION` | Optional                 | Tokenizer revision when different from the model revision. |
| `--device-map DEVICE_MAP`       | Default: `auto`          | Device map used during model loading.                      |
| `--trust-remote-code`           | Disabled by default      | Enables repository-specific model or tokenizer code.       |
| `--use-fast-tokenizer`          | Disabled by default      | Uses the fast tokenizer implementation when available.     |
| `--output-dir PATH`             | Default: `results`       | Directory for evaluation result files.                     |
| `--output-file PATH`            | Optional                 | Explicit output JSON path.                                 |
| `--seed INTEGER`                | Recipe default           | Overrides the random seed from the evaluation recipe.      |

</details>

---

## Results

Results are stored as JSON files containing evaluation scores together with model, compression, and run metadata.

Example:

```text
results/facebook_opt-125m_base_results.json
```

<details>
<summary><b>Model Recipe examples</b></summary>

<br>

```text
Dense BF16
GPTQ W4A16 G128
AWQ W4A16 G128
BitsAndBytes INT4
SparseGPT 50%
Wanda 2:4 + GPTQ W4A16 G128
```

</details>

---

## Generate Leaderboard scores

<details>
<summary><b>Generate leaderboard CSV files</b></summary>

<br>

Base models:

```bash
dike leaderboard \
  results/ \
  --mode base \
  --output leaderboard_base.csv
```

Instruction-tuned models:

```bash
dike leaderboard \
  results/ \
  --mode instruct \
  --output leaderboard_instruct.csv
```

</details>

---

## Testing

<details>
<summary><b>Run tests and build checks</b></summary>

<br>

Validate the lockfile:

```bash
uv lock --check
```

Install the locked environment:

```bash
uv sync --locked
```

Unit tests:

```bash
uv run python -m pytest -m "not integration"
```

Integration tests:

```bash
uv run python -m pytest -m integration
```

Build the package:

```bash
uv build
```

</details>

---

## Citation

If you use the Open Diké Leaderboard in your work, please cite the software.

**BibTeX**

```bibtex
@software{proskurina_open_dike_leaderboard,
  author  = {Proskurina, Irina},
  title   = {Open Diké Leaderboard},
  version = {0.1.0},
  url     = {https://github.com/upunaprosk/dice-leaderboard}
}
```

**Plain**

```text
Proskurina, I. Open Diké Leaderboard, version 0.1.0.
https://github.com/upunaprosk/dice-leaderboard
```

Citation metadata is also available in [`CITATION.cff`](CITATION.cff).

---

## License

See [LICENSE](LICENSE).
