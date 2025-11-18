# Dice-Leaderboard

Code for evaluating LLMs in the Diké (Dice) leaderboard:  
https://huggingface.co/spaces/iproskurina/dike-leaderboard

This repository provides tools to evaluate your model and format the results for submission. There are two types of models supported:

- **Instruct models**
- **Non-instruct (base) models**

## Installation

```bash
git clone https://github.com/upunaprosk/dice-leaderboard.git
cd dice-leaderboard
pip install -r requirements.txt
pip install datasets colorama logbar
```

Additional dependencies depending on model type:

- **GPTQ models**
  ```bash
  pip install gptqmodel
  ```

- **Models compressed with vLLM or stored in vLLM format**
  ```bash
  pip install vllm llm-compressor
  ```

- **Models quantized with bitsandbytes**
  ```bash
  pip install bitsandbytes
  ```

## Evaluating Non-Instruct Models

```bash
git clone https://github.com/upunaprosk/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
cd ..

chmod +x run_eval.sh

# Example GPTQ model evaluation
bash run_eval.sh iproskurina/opt-350m-int4-tb --gptq

python base_model_evaluation.py    --model iproskurina/Llama-3.1-8B-gptqmodel-4bit    --is_gptqmodel    --use_fast_tokenizer    --trust_remote_code
```

## Evaluating Instruct Models

```bash
git clone https://github.com/upunaprosk/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
cd ..

chmod +x run_eval_instruct.sh

# Example GPTQ instruct model evaluation
bash run_eval_instruct.sh model --gptq

python instruct_model_evaluation.py    --model model_name    --is_gptqmodel    --use_fast_tokenizer    --trust_remote_code
```
