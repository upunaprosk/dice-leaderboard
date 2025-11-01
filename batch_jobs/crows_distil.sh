#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:05:00" ["albert-base-v2"]="00:05:00" ["roberta-base"]="00:05:00" ["gpt2"]="00:05:00"
#declare -A time=(["bert-base-uncased"]="00:10:00" ["albert-base-v2"]="00:10:00" ["roberta-base"]="00:10:00" ["gpt2"]="00:10:00")
declare -A time=(["distilbert-base-uncased"]="00:20:00" ["distilroberta-base"]="00:20:00")

for model in ${distil_masked_lm_models[@]}; do
    for bias_type in ${all_crows_bias_types[@]}; do
        experiment_id="crows_m-${model}_c-${model_to_distilmodel_name_or_path[${model}]}_t-${bias_type}"
        if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
            echo ${experiment_id}
            sbatch \
                --gres=gpu:nvidia_a100-pcie-40gb:1 \
                --time ${time[${model_to_distilmodel_name_or_path[${model}]}]} \
                -J ${experiment_id} \
                -o ${persistent_dir}/logs/%x.%j.out \
                -e ${persistent_dir}/logs/%x.%j.err \
                python_job.sh experiments/crows.py \
                    --model ${model} \
                    --model_name_or_path ${model_to_distilmodel_name_or_path[${model}]} \
                    --bias_type ${bias_type}
        fi
    done
done

