# Dice-Leaderboard
Code for evaluating LLMs in the Diké (Dice) leaderboard

Leaderboard contains the following metrics:
1) LLM performance on zero-shot benchmarks and ppl on wikitext-2 data.
2) Fairness: monolingual and multilingual
3) Ethics
4) Toxicity
5) Calibration
6) Efficiency metrics for quantized LLMs

## Perplexity Evaluation

We evaluate perplexity on WikiText-2. 
To evaluate the perplexity of a `model` that has been quantized, run the following command:

```bash
ppl_eval.py --model "$model" --backend auto --is_gptqmodel
```
Other supported arguments: `is_vllm_quantized` (model quantized with vllm), `is_int4` (bitsandbytes quantization), `is_int8` (bitsandbytes quantization).

## LM-Eval-Harness

For general LLM performance evaluation and fairness evaluation (`bbq, simple_cooccurrence_bias,winogender,crows_pairs_english`), we use the EleutherAI lm-evaluation-harness:
```bash
# Clone lm-evaluation-harness and install dependencies
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e ".[gptqmodel]"

# Define model arguments
model_args="pretrained=$repo_id,gptqmodel=True"

# Run lm_eval command for evaluation
lm_eval --model hf --model_args $model_args --device cuda:0 \
  --tasks bbq,arc_easy,openbookqa,xstorycloze_en,hellaswag,piqa,arc_challenge,simple_cooccurrence_bias,winogender,crows_pairs_english
  --batch_size 8 --write_out --log_samples --output_path ./results/ --trust_remote_code --seed 42
```

## Fairness Evaluation

`python sofa.py --model_name "$model" --is_gptqmodel`
`python holistic_bias.py --model_name "$model" --is_gptqmodel`


```angular2html
from unqover import *
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16,  device_map='auto')

tokenizer.pad_token = tokenizer.eos_token

tag_split = model_name.split("/")
tag = tag_split[-1].replace(".", "").replace("-", "_")
unqover_evaluate(
    tag=tag,
    component=tag+"_baseline",
    model=model,
    tokenizer=tokenizer,
    few_shot=True,
    persistent_dir="./results",
    device=device,
    baseline=True,
    verbose=False
)

```

Other supported arguments: `is_vllm_quantized` (model quantized with vllm), `is_int4` (bitsandbytes quantization), `is_int8` (bitsandbytes quantization).
