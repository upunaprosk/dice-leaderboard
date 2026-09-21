# Contributing to the Open Diké Leaderboard

The Open Diké Leaderboard accepts community evaluations of dense and compressed language models.

Contributions should use the official Open Diké evaluation protocol so that results remain comparable across models.

## What can be contributed

You can submit:

* a new base language model;
* a new instruction-tuned language model;
* a compressed version of a model already on the leaderboard;
* a different compression recipe for an existing model;
* a different quantization or sparsity configuration.

Supported model recipes include dense models, GPTQ, AWQ, BitsAndBytes runtime quantization, sparse models, and hybrid compression recipes.

## 1. Clone and install

```
git clone https://github.com/upunaprosk/dice-leaderboard.git
cd dice-leaderboard
```

Using the locked environment:

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

## 2. Choose the evaluation track

Use:

```
--mode base
```

for pretrained/base language models.

Use:

```
--mode instruct
```

for instruction-tuned or chat models.

The official protocols are defined in:

```
recipes/base.yaml
recipes/instruct.yaml
```

Do not modify these recipes when producing a standard leaderboard submission.

## 3. Run the evaluation

Base example:

```
dike eval \
  --model MODEL_NAME \
  --mode base
```

Instruction-tuned example:

```
dike eval \
  --model MODEL_NAME \
  --mode instruct
```

GPTQ example:

```
dike eval \
  --model MODEL_NAME \
  --mode base \
  --backend gptq
```

BitsAndBytes INT4 example:

```
dike eval \
  --model MODEL_NAME \
  --mode base \
  --quantization int4
```

For an already-compressed sparse or Transformers-compatible checkpoint:

```
dike eval \
  --model MODEL_NAME \
  --mode base \
  --backend transformers
```

If a separate tokenizer is required:

```
dike eval \
  --model MODEL_NAME \
  --tokenizer TOKENIZER_NAME \
  --mode base
```

If the model uses a specific revision:

```
dike eval \
  --model MODEL_NAME \
  --revision MODEL_REVISION \
  --tokenizer TOKENIZER_NAME \
  --tokenizer-revision TOKENIZER_REVISION \
  --mode base
```

## 4. Check the result

Evaluation results are written to:

```
results/
```

A result JSON contains the benchmark scores and available metadata about:

* model repository;
* model mode;
* tokenizer;
* model revision;
* tokenizer revision;
* loading backend;
* quantization;
* compression recipe;
* evaluation configuration;
* benchmark results.

Before submitting, verify that the evaluation completed successfully and that the expected metrics are present.

## 5. Submit the model

Create a branch for the evaluation:

```
git checkout -b add-model-name
```

Add the generated result file according to the repository submission structure.

Commit the evaluation:

```
git add .
git commit -m "Add MODEL_NAME evaluation"
git push origin add-model-name
```

Then open a pull request against `main`.

The pull request should include:

* Hugging Face model repository;
* `base` or `instruct` track;
* compression method, if applicable;
* model revision, if applicable;
* tokenizer repository, if different from the model repository;
* tokenizer revision, if applicable;
* hardware used for the evaluation;
* generated Open Diké result JSON.

## Evaluation requirements

Leaderboard submissions must use the official Open Diké evaluation protocol.
If a model cannot be evaluated with the standard protocol because it requires a special loading configuration, open an issue before submitting the result.

## Compressed models

For compressed checkpoints, provide enough information to identify the evaluated model configuration.
Where applicable, this includes:

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

Examples of model recipes:
```
Dense BF16
GPTQ W4A16 G128
AWQ W4A16 G128
BitsAndBytes INT4
SparseGPT 50%
Wanda 2:4 + GPTQ W4A16 G128
```

## Questions and unsupported models

If a model requires an unsupported backend, unusual tokenizer configuration, custom model code, or another non-standard loading procedure, open an issue:

https://github.com/upunaprosk/dice-leaderboard/issues
