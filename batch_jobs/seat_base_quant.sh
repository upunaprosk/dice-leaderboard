#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:15:00" ["albert-base-v2"]="00:10:00" ["roberta-base"]="00:10:00" ["gpt2"]="00:15:00"
#declare -A time=(["bert-base-uncased"]="00:30:00" ["albert-base-v2"]="00:30:00" ["roberta-base"]="00:30:00" ["gpt2"]="00:30:00")
declare -A time=(["bert-base-uncased"]="00:30:00" ["roberta-base"]="00:30:00")


for model in ${masked_lm_quant_models[@]}; do
    experiment_id="seat_m-${model}_c-${model_to_quantbasemodel_name_or_path[${model}]}"
    if [ ! -f "${persistent_dir}/results/seat/${experiment_id}.json" ]; then
        echo ${experiment_id}
        sbatch \
            --gres=shard:nvidia_a100-pcie-40gb:0 \
            --time ${time[${model_to_quantbasemodel_name_or_path[${model}]}]} \
            -J ${experiment_id} \
            -o ${persistent_dir}/logs/%x.%j.out \
            -e ${persistent_dir}/logs/%x.%j.err \
            -N 1 -n 1 -c 4 --mem=16G \
            python_job.sh experiments/seat.py \
                --tests ${seat_tests} \
                --model ${model} \
                --model_name_or_path ${model_to_quantbasemodel_name_or_path[${model}]}
                --quant_type "dynamic" \
                --quant_prec "int8"
    fi
done
