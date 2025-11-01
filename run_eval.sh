#!/bin/bash

model="$1"

if [[ -z "$model" ]]; then
  echo "Model parameter is required!"
  exit 1
fi

# Check if the model is quantized
quantized=false
if [[ "$model" == *"int4"* ]]; then
  quantized=true
fi

# Run ppl-eval.py
if $quantized; then
  python ppl-eval.py --model "$model" --backend auto --is_quantized
else
  python ppl-eval.py --model "$model" --backend auto
fi

cd EMNLP2023_llm_compression_and_social_bias || { echo "Directory not found!"; exit 1; }

python -m pip install -e .

model_dir="${model//\//_}"
model_filename="results/stereoset/stereoset_m-AutoModelForCausalLM_c-${model_dir}_s-42.json"

python -c 'import random; import torch; random.seed(42); torch.manual_seed(42)'

if $quantized; then
  python experiments/stereoset.py --model "AutoModelForCausalLM" --model_name "$model" --seed 42 --batch_size 16 --file_name "test.json" --is_quantized
else
  python experiments/stereoset.py --model "AutoModelForCausalLM" --model_name "$model" --seed 42 --batch_size 16 --file_name "test.json"
fi

# Run stereoset_evaluation.py
python experiments/stereoset_evaluation.py --predictions_file "$model_filename" --file_name "test.json"

cd ..

# LM-Eval
cd lm-evaluation-harness || { echo "LM Evaluation Harness directory not found!"; exit 1; }

pip install -e ".[gptqmodel]"

repo_id="$model"
echo "Evaluating Model: $repo_id"
echo "========================================"

echo "Running LM-Eval Evaluation..."

if $quantized; then
  lm_eval --model hf --model_args pretrained=$repo_id,gptqmodel=True --device cuda:0 \
  --tasks bbq,truthfulqa,arc_easy,openbookqa,xstorycloze_en,hellaswag,piqa,arc_challenge,simple_cooccurrence_bias,winogender,crows_pairs_english \
  --batch_size 8 --write_out --log_samples --output_path ./results/ --trust_remote_code --seed 42
else
  lm_eval --model hf --model_args pretrained=$repo_id --device cuda:0 \
  --tasks bbq,truthfulqa,arc_easy,xstorycloze_en,openbookqa,hellaswag,piqa,arc_challenge,simple_cooccurrence_bias,winogender,crows_pairs_english \
  --batch_size 8 --write_out --log_samples --output_path ./results/ --trust_remote_code --seed 42
fi

cd ..

if $quantized; then
  python sofa.py --model_name "$model" --gptqmodel
else
  python sofa.py --model_name "$model"
fi

cd ..

echo "========================================"
echo "Completed Evaluations for $repo_id"

echo "All Evaluations Completed!"
