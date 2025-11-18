#!/bin/bash
#
# Example usage:
#
#   ./run_all.sh my-org/my-model
#   ./run_all.sh my-org/my-model --gptq
#   ./run_all.sh my-org/my-model --int4
#   ./run_all.sh my-org/my-model --int8
#   ./run_all.sh my-org/my-model --vllm
#
# Combined examples:
#
#   # GPTQ quantized model
#   ./run_all.sh my-org/llama-3-8b-gptq --gptq
#
#   # 4-bit quantized model
#   ./run_all.sh my-org/llama-3-8b-4bit --int4
#
#   # 8-bit quantized model
#   ./run_all.sh my-org/llama-3-8b-8bit --int8
#
#   # vLLM backend ONLY in lm-eval-harness
#   ./run_all.sh my-org/llama-3-8b-awq --vllm
#
# All flags can be combined except `--vllm` (vLLM is used ONLY in LM-Eval):
#
#   ./run_all.sh my-org/model --gptq --int4   # valid
#   ./run_all.sh my-org/model --vllm --int4   # vLLM overrides everything else
#

set -e

# -----------------------------
# Required argument
# -----------------------------
model="$1"

if [[ -z "$model" ]]; then
  echo "❌ Model parameter is required!"
  echo "Usage: ./run_all.sh <model_name> [--gptq] [--vllm] [--int4] [--int8]"
  exit 1
fi

# -----------------------------
# Optional flags
# -----------------------------
is_gptqmodel=false
is_vllm_quantized=false
is_int4=false
is_int8=false

tasks_eval="bbq,hellaswag,simple_cooccurrence_bias,crows_pairs_english"

for arg in "$@"; do
  case "$arg" in
    --gptq) is_gptqmodel=true ;;
    --vllm) is_vllm_quantized=true ;;
    --int4) is_int4=true ;;
    --int8) is_int8=true ;;
  esac
done

echo "========================================"
echo " Model: $model"
echo " GPTQ:   $is_gptqmodel"
echo " vLLM:   $is_vllm_quantized"
echo " INT4:   $is_int4"
echo " INT8:   $is_int8"
echo " Tasks:  $tasks_eval"
echo "========================================"

# -----------------------------
# StereoSet
# -----------------------------
echo "Running StereoSet…"
python -m pip install -e .
python - <<EOF
import random, torch
random.seed(42)
torch.manual_seed(42)
EOF

if $is_gptqmodel; then
  python experiments/stereoset.py \
    --model AutoModelForCausalLM \
    --model_name_or_path "$model" \
    --seed 42 --batch_size 16 --file_name test.json \
    --is_gptqmodel --is_quantized

elif $is_int4; then
  python experiments/stereoset.py \
    --model AutoModelForCausalLM \
    --model_name_or_path "$model" \
    --seed 42 --batch_size 16 --file_name test.json \
    --is_int4 --is_quantized

elif $is_int8; then
  python experiments/stereoset.py \
    --model AutoModelForCausalLM \
    --model_name_or_path "$model" \
    --seed 42 --batch_size 16 --file_name test.json \
    --is_int8 --is_quantized

else
  python experiments/stereoset.py \
    --model AutoModelForCausalLM \
    --model_name_or_path "$model" \
    --seed 42 --batch_size 16 --file_name test.json
fi

python experiments/stereoset_evaluation.py --file_name test.json

# -----------------------------
# LLM-Evaluaton-Harness
# -----------------------------
echo "Running LM-Eval Harness…"

cd lm-evaluation-harness || { echo "Missing lm-evaluation-harness directory!"; exit 1; }
pip install -e .

repo_id="$model"

echo "Evaluating Model: $repo_id"
echo "========================================"

# -----------------------------
# vLLM path (only for lm-eval)
# -----------------------------
if $is_vllm_quantized; then
  echo "Using vLLM backend"
  lm_eval \
    --model vllm \
    --model_args pretrained="$repo_id" \
    --device cuda:0 \
    --tasks $tasks_eval \
    --batch_size 8 \
    --output_path ./results/ \
    --trust_remote_code \
    --seed 42

else
  # -----------------------------
  # HuggingFace backend
  # -----------------------------
  echo "Using HF backend"

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
    --device cuda:0 \
    --tasks $tasks_eval \
    --batch_size 8 \
    --output_path ./results/ \
    --trust_remote_code \
    --seed 42
fi

echo "========================================"
echo "Completed Evaluations for $repo_id"
echo "All Evaluations Completed!"
echo "========================================"