#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:05:00" ["albert-base-v2"]="00:05:00" ["roberta-base"]="00:05:00" ["gpt2"]="00:05:00"
#declare -A time=(["bert-base-uncased"]="00:10:00" ["albert-base-v2"]="00:10:00" ["roberta-base"]="00:10:00" ["gpt2"]="00:10:00")
declare -A time=(["bert-base-uncased"]="00:20:00" ["roberta-base"]="00:20:00")
#declare -A quant_type=dynamic
#declare -A quant_prec=fp16 int8

for model in ${masked_lm_quant_models[@]}; do
    for bias_type in ${all_crows_bias_types[@]}; do
        #for quant_type in ${quant_type[@]}; do
        #    for quant_prec in ${quant_prec[@]}; do
        for quant_type in "dynamic"; do
            for quant_prec in "int8"; do
                experiment_id="crows_m-${model}_c-${model_to_quantbasemodel_name_or_path[${model}]}_t-${bias_type}_qt-${quant_type}_qp-${quant_prec}"
                if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
                    echo ${experiment_id}
                    sbatch \
                        --gres=gpu:nvidia_a100-pcie-40gb:0 \
                        -J ${experiment_id} \
                        -o ${persistent_dir}/logs/%x.%j.out \
                        -e ${persistent_dir}/logs/%x.%j.err \
                        -N 1 -n 1 -c 16 --mem=32G \
                        --time ${time[${model_to_quantbasemodel_name_or_path[${model}]}]} \
                        python_job.sh experiments/crows.py \
                            --model ${model} \
                            --model_name_or_path ${model_to_quantbasemodel_name_or_path[${model}]} \
                            --bias_type ${bias_type} \
                            --quant_type ${quant_type} \
                            --quant_prec ${quant_prec}
                fi
            done
        done
    done
done
