#!/bin/bash

set -e

model="$1"

if [[ -z "$model" ]]; then
  echo "Model parameter is required!"
  exit 1
fi

is_gptqmodel=false
is_vllm_quantized=false
is_int4=false
is_int8=false

for arg in "$@"; do
  case "$arg" in
    --gptq) is_gptqmodel=true ;;
    --vllm) is_vllm_quantized=true ;;
    --int4) is_int4=true ;;
    --int8) is_int8=true ;;
  esac
done

cd lm-evaluation-harness || { echo "Missing lm-evaluation-harness directory!"; exit 1; }
pip install -e .

repo_id="$model"

if $is_vllm_quantized; then

  lm_eval \
    --model vllm \
    --model_args pretrained="$repo_id" \
    --tasks realtoxicityprompts \
    --limit 1000 \
    --apply_chat_template \
    --device cuda:0 \
    --batch_size 16 \
    --output_path ./results-eval \
    --seed 42

  lm_eval \
    --model vllm \
    --model_args pretrained="$repo_id" \
    --tasks ethics_cm,harmbench \
    --apply_chat_template \
    --device cuda:0 \
    --batch_size 16 \
    --output_path ./results-eval \
    --seed 42

else
  model_args="pretrained=$repo_id"

  if $is_gptqmodel; then
    model_args="$model_args,gptqmodel=True"
  fi
  if $is_int4; then
    model_args="$model_args,load_in_4bit=True"
  fi
  if $is_int8; then
    model_args="$model_args,load_in_8bit=True"
  fi

  lm_eval \
    --model hf \
    --model_args "$model_args" \
    --tasks realtoxicityprompts \
    --limit 1000 \
    --apply_chat_template \
    --device cuda:0 \
    --batch_size 16 \
    --output_path ./results-eval \
    --seed 42

  lm_eval \
    --model hf \
    --model_args "$model_args" \
    --tasks ethics_cm,harmbench \
    --apply_chat_template \
    --device cuda:0 \
    --batch_size 16 \
    --output_path ./results-eval \
    --seed 42
fi

echo "Done."