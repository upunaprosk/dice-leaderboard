
# Contributing to the Open Diké Leaderboard

Thank you for your interest in contributing to the Open Diké Leaderboard!
Open Diké is an open leaderboard for evaluating bias, fairness, ethics, and safety in dense and compressed language models.
We welcome community evaluations of new models, compression methods, and model configurations.

**Accepted evaluation submissions are added to the public [Open Diké Leaderboard](https://huggingface.co/spaces/LabHC/dike-leaderboard), making the results openly accessible to the research community.**

All leaderboard submissions must follow the official Open Diké evaluation protocol to ensure reproducibility and comparability across models.

## What can be contributed?

You can submit evaluations of:

- New pretrained/base language models.
- New instruction-tuned or chat models.
- Compressed versions of models already on the leaderboard.
- Alternative compression recipes for existing models.
- Different quantization or sparsity configurations.

Supported configurations include dense models, GPTQ, AWQ, BitsAndBytes runtime quantization, sparse models, and hybrid compression recipes.
We also welcome bug reports, documentation improvements, and contributions to the evaluation framework.

## 1. Fork, clone, and install

Fork the [Open Diké repository](https://github.com/upunaprosk/dice-leaderboard) on GitHub.

Clone your fork:

```
git clone https://github.com/YOUR_USERNAME/dice-leaderboard.git
cd dice-leaderboard
```

Install the project using the locked environment:

```
uv sync --locked
```

Set a Hugging Face token before evaluation.

Linux/macOS:

```
export HF_TOKEN=hf_your_token_here
```

Windows PowerShell:

```
$env:HF_TOKEN="hf_your_token_here"
```

Keep your token private. Do not include it in result files, commits, or pull requests.

## 2. Choose the evaluation mode

Open Diké provides two evaluation modes.

### Base models

For pretrained/base language models, use:

```
--mode base
```

### Instruction-tuned models

For instruction-tuned or chat models, use:

```
--mode instruct
```

The official evaluation protocols are defined in:

```
recipes/base.yaml
recipes/instruct.yaml
```

**Do not modify the official recipes when producing a leaderboard submission.**

Custom recipes may be used for independent experiments, but results generated with modified protocols are not considered standard leaderboard submissions.

## 3. Run the evaluation

Use `uv run` to execute the evaluation CLI within the project environment.

Replace `MODEL_NAME` with the Hugging Face repository identifier of the model you want to evaluate.

### Base model

```
uv run dike eval \
  --model MODEL_NAME \
  --mode base
```

### Instruction-tuned model

```
uv run dike eval \
  --model MODEL_NAME \
  --mode instruct
```

### GPTQ model

```
uv run dike eval \
  --model MODEL_NAME \
  --mode base \
  --backend gptq
```

### BitsAndBytes INT4

```
uv run dike eval \
  --model MODEL_NAME \
  --mode base \
  --quantization int4
```

### Sparse or Transformers-compatible checkpoint

For an already-compressed sparse or Transformers-compatible checkpoint:

```
uv run dike eval \
  --model MODEL_NAME \
  --mode base \
  --backend transformers
```

### Separate tokenizer

If the model requires a separate tokenizer:

```
uv run dike eval \
  --model MODEL_NAME \
  --tokenizer TOKENIZER_NAME \
  --mode base
```

### Specific model and tokenizer revisions

For reproducibility, model and tokenizer revisions can be specified explicitly:

```
uv run dike eval \
  --model MODEL_NAME \
  --revision MODEL_REVISION \
  --tokenizer TOKENIZER_NAME \
  --tokenizer-revision TOKENIZER_REVISION \
  --mode base
```

Select the appropriate evaluation mode and loading configuration for your model.

For additional options, refer to the project README.

## 4. Check the generated results

Evaluation results are written to:

```
results/
```

The generated JSON contains benchmark scores and available metadata, including:

- Model repository.
- Evaluation mode.
- Tokenizer repository.
- Model and tokenizer revisions.
- Loading backend.
- Quantization configuration.
- Compression recipe.
- Evaluation configuration.
- Benchmark results.

Before submitting, verify that:

- The evaluation completed successfully.
- All expected benchmark metrics are present.
- The correct evaluation mode was used.
- The official evaluation recipe was not modified.
- Model and compression metadata are accurate.
- The generated result file has not been manually altered.

Do not manually modify benchmark scores or evaluation metadata.

## 5. Submit your evaluation

Leaderboard results are submitted through GitHub Pull Requests.

### Create a branch

From your local fork, create a dedicated branch:

```
git checkout -b add-model-name
```

### Add the generated result

Add the generated JSON file under the repository's `results/` directory, following the existing submission structure.

Stage only the relevant result file:

```
git add results/YOUR_RESULT_FILE.json
```

Commit your evaluation:

```
git commit -m "Add MODEL_NAME evaluation"
```

Push the branch to your fork:

```
git push origin add-model-name
```

### Open a Pull Request

Open a Pull Request from your fork to the upstream Open Diké repository.

The PR should contain the generated result JSON and sufficient information to identify and reproduce the evaluated configuration.

Please include:

- Hugging Face model repository.
- Evaluation mode (`base` or `instruct`).
- Compression method, if applicable.
- Model revision, if applicable.
- Tokenizer repository, if different from the model repository.
- Tokenizer revision, if applicable.
- Hardware used for evaluation.
- Generated Open Diké result JSON.

For compressed models, include the relevant compression configuration described below.

**Once the submission has been reviewed and accepted, its results will be incorporated into the public [Open Diké Leaderboard](https://huggingface.co/spaces/LabHC/dike-leaderboard).**

Accepted results become publicly available for comparison with other evaluated models.

## Evaluation requirements

To maintain consistency and comparability, leaderboard submissions must follow the official Open Diké evaluation protocol.

Please ensure that:

1. The appropriate official evaluation mode is used.
2. The official benchmark configuration remains unchanged.
3. Results are generated using the Open Diké evaluation framework.
4. The evaluated model configuration is clearly identifiable.
5. The submitted JSON is the original output of the evaluation.

Results obtained with custom or modified evaluation recipes should not be submitted as standard leaderboard results.

If a model requires an unsupported backend, custom model code, or a non-standard loading configuration, please open an issue before submitting the evaluation.

## Compressed models

For compressed checkpoints, provide sufficient information to identify the exact evaluated configuration.

Where applicable, include:

```
model repository
model revision
tokenizer repository
tokenizer revision
compression method
weight precision
activation precision
group size
sparsity level
sparsity pattern
runtime quantization configuration
```

Examples of supported model configurations include:

```
Dense BF16
GPTQ W4A16 G128
AWQ W4A16 G128
BitsAndBytes INT4
SparseGPT 50%
Wanda 2:4 + GPTQ W4A16 G128
```

Different compression configurations of the same underlying model may be submitted separately.

Please ensure that each submission clearly identifies the corresponding checkpoint and compression recipe.

## Questions and unsupported models

If your model requires:

- An unsupported loading backend.
- An unusual tokenizer configuration.
- Custom model code.
- A non-standard compression recipe.
- Another unsupported loading procedure.

Please [open an issue](https://github.com/upunaprosk/dice-leaderboard/issues) before submitting the result.
Include the model repository, relevant configuration, and a description of the error.

---

Thank you for contributing to Open Diké!
Community submissions help expand the coverage of the leaderboard and support reproducible research into the effects of model compression on language model behavior.
**[Explore the Open Diké Leaderboard](https://huggingface.co/spaces/LabHC/dike-leaderboard)**
