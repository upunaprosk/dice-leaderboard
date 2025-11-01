#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:10:00" ["albert-base-v2"]="00:10:00" ["roberta-base"]="00:10:00" ["gpt2"]="00:10:00"
declare -A time=(["bert-base-uncased"]="00:20:00" ["albert-base-v2"]="00:20:00" ["roberta-base"]="00:20:00" ["gpt2"]="00:20:00")


for model in ${masked_lm_base_models[@]}; do
    experiment_id="stereoset_m-${model}_c-${model_to_model_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/stereoset/${experiment_id}.json" ]; then
        echo ${experiment_id}
	echo ${persistent_dir}
        sbatch \
            --gres=gpu:nvidia_a100-pcie-40gb:1 \
            --time ${time[${model_to_model_name_or_path[${model}]}]} \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/%x.%j.out \
            -e ${persistent_dir}/logs/%x.%j.err \
            -N 1 -n 1 -c 16 --mem=16G \
            python_job.sh experiments/stereoset.py \
                --model ${model} \
                --model_name_or_path ${model_to_model_name_or_path[${model}]}
    fi
done

