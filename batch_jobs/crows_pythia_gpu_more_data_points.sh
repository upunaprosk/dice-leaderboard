#!/bin/bash

source "batch_jobs/_experiment_configuration.sh"

# Actual time:   ["bert-base-uncased"]="00:10:00" ["albert-base-v2"]="00:10:00" ["roberta-base"]="00:10:00" ["gpt2"]="00:10:00"
#declare -A time=(["bert-large-uncased"]="02:40:00" ["roberta-large"]="02:40:00")

MODEL="GPTNeoXForCausalLM"
for model_name in "pythia-6.9b-deduped" "pythia-2.8b-deduped" "pythia-1.4b-deduped" "pythia-1b-deduped" "pythia-410m-deduped" "pythia-160m-deduped" "pythia-70m-deduped"; do
#for model_name in "pythia-70m-deduped" "pythia-160m-deduped" "pythia-410m-deduped" "pythia-1b-deduped" "pythia-1.4b-deduped" "pythia-2.8b-deduped" "pythia-6.9b-deduped"; do
  #for revision in "step1000" "step15000" "step29000" "step43000" "step57000" "step72000" "step86000" "step100000" "step114000" "step128000" "step143000"; do
  for revision in "step1000" "step7000" "step14000" "step21000" "step29000" "step36000" "step43000" "step50000" "step57000" "step64000" "step72000" "step79000" "step86000" "step93000" "step100000" "step107000" "step114000" "step122000" "step129000" "step136000" "step143000"; do
    for bias_type in "gender" "religion" "race"; do
      experiment_id="crows_m-${MODEL}_c-${model_name//-/_}_rev-${revision}_t-${bias_type}"
      cache_dir="/home/ggoncalves/data/pythia_cache/${model_name}/${revision}"
      mkdir -p ${cache_dir}
      if [ ! -f "${persistent_dir}/results/crows/${experiment_id}.json" ]; then
        echo ${experiment_id}
        echo ${persistent_dir}
        sbatch \
          --gres=gpu:nvidia_a100-pcie-40gb:1 \
          --time 0 \
          -J ${experiment_id} \
          -o ${persistent_dir}/logs/%x.%j.out \
          -e ${persistent_dir}/logs/%x.%j.err \
          -N 1 -n 1 -c 32 --mem=64G \
            python_job.sh experiments/crows.py \
              --model ${MODEL} \
              --model_name_or_path "EleutherAI/${model_name}" \
              --bias_type ${bias_type} \
              --revision ${revision} \
              --cache_dir ${cache_dir}
      fi
    done
  done
done

