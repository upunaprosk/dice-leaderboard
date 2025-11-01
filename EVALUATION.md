# Evaluation Setup for Quantized Models

In our paper, we report the following evaluations for quantized models:

- Perplexity on WikiText-2
- Stereotype-bias scores
- Zero-shot accuracy on ARC Easy, PIQA, HellaSwag, and Cloze-En.

# Complete Evaluation Setup

To perform the complete evaluation, use a shell script `run_eval.sh`:

```bash
# Clone Bias-bench framework for stereotype-bias evaluation
git clone https://github.com/gsgoncalves/emnlp2023_llm_compression_and_social_bias.git

# Clone lm-evaluation-harness for zero-shot and group bias evaluation
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git

# Navigate to lm-evaluation-harness and make run_lm_eval.sh executable
cd lm-evaluation-harness
chmod +x run_lm_eval.sh

# Define variables for model and log file
model="$model"
log_file="evaluation.log"

# Run the evaluation script with model argument and redirect output to log file
./run_lm_eval.sh "$model" > "$log_file" 2>&1
```


## Perplexity Evaluation

We evaluate perplexity on WikiText-2. To evaluate the perplexity of a `model` that has been quantized, run the following command:

```bash
ppl-eval.py --model "$model" --backend auto --is_quantized
```

## Stereotype-bias Scores (Stereoset)

To evaluate stereotype-bias scores using the Bias-bench framework, run:

```bash
# Clone Bias-bench framework
git clone https://github.com/gsgoncalves/emnlp2023_llm_compression_and_social_bias.git
cd emnlp2023_llm_compression_and_social_bias || { echo "Directory not found!"; exit 1; }
# Install Bias-bench framework
python -m pip install -e .
# Define model directory and filename for results
model_dir="${model//\//_}"
model_filename="results/stereoset/stereoset_m-AutoModelForCausalLM_c-${model_dir}_s-42.json"
# Run Stereoset script for predictions
python experiments/stereoset.py --model "AutoModelForCausalLM" --model_name "$model" --seed 42 --batch_size 16 --file_name "test.json" --is_quantized
# Evaluate Stereoset predictions using evaluation script
python experiments/stereoset_evaluation.py --predictions_file "$model_filename" --file_name "test.json"
```

## Stereotype-bias Scores (SoFA)

To evaluate stereotype-bias scores using the SoFA metric as implemented in the source paper, run:

```bash
python sofa.py --model_name "$model" --gptqmodel
```

## Zero-shot Evaluation and Other Group Bias Evaluation

To evaluate zero-shot performance and other group bias metrics using the lm-evaluation-harness framework, execute the following commands in a single code cell:

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

