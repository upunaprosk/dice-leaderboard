import argparse
import os
from transformers import AutoTokenizer
import json
import sys
import numpy as np
import torch
from datasets import load_dataset
from logbar import LogBar
import logging
import pandas as pd

from ppl_eval import Perplexity as PPL_Perplexity
from holistic_bias import *
from sofa import *

logger = LogBar.shared()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate Model on Wiki2, Holistic Bias and SoFA.")

    parser.add_argument("--model", type=str, default="ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v2.5")
    parser.add_argument("--tasks", type=str,
                        default="wiki_2,bbq,crows_pairs_english,hellaswag,holistic_bias,stereoset,sofa")

    parser.add_argument("--is_gptqmodel", action="store_true")
    parser.add_argument("--is_vllm_quantized", action="store_true")
    parser.add_argument("--is_int4", action="store_true")
    parser.add_argument("--is_int8", action="store_true")

    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--backend",
                        choices=['auto', 'marlin', 'exllama_v1', 'exllama_v2',
                                 'triton', 'cuda', 'torch', 'ipex', 'bitblas'],
                        default='auto')

    parser.add_argument("--n_ctx", type=int, default=1024)
    parser.add_argument("--n_batch", type=int, default=1024)
    parser.add_argument("--dataset_path", type=str, default="wikitext")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text_column", type=str, default="text")

    parser.add_argument("--hb_dataset", type=str, default="iproskurina/holisticbias-sentiment-pairs")
    parser.add_argument("--hb_batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")

    parser.add_argument("--probe_file", type=str, default="data/sofa/SBIC-Pro.feather")
    parser.add_argument("--identity_file", type=str, default="data/sofa/identities_by_category.json")
    parser.add_argument("--sofa_batch", type=int, default=512)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.is_gptqmodel:
        from gptqmodel import BACKEND, GPTQModel
        model = GPTQModel.load(
            args.model,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            backend=BACKEND(args.backend.lower()),
        )
    elif args.is_vllm_quantized:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
    elif args.is_int4:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )
    elif args.is_int8:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=args.trust_remote_code,
        )

    ppl = PPL_Perplexity(
        model,
        tokenizer,
        args.dataset_path,
        args.dataset_name,
        args.split,
        args.text_column,
    )
    all_perplexity = ppl.calculate(args.n_ctx, args.n_batch)
    average_perplexity = float(np.mean(all_perplexity))

    evaluation_results = {
        "model": args.model,
        "perplexity": average_perplexity,
    }

    print("Average ppl:", average_perplexity)

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    ds = load_dataset(args.hb_dataset, split="train")
    df = ds.to_pandas()

    metric = Perplexity()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pos_ppl = metric._compute(df["positive_sentence"].tolist(), model, tokenizer,
                              batch_size=args.hb_batch, device=device)["perplexities"]
    neg_ppl = metric._compute(df["negative_sentence"].tolist(), model, tokenizer,
                              batch_size=args.hb_batch, device=device)["perplexities"]

    df["pos_ppl"] = pos_ppl
    df["neg_ppl"] = neg_ppl
    df["bias_flag"] = (np.array(neg_ppl) < np.array(pos_ppl)).astype(int)

    overall_bias_share = df["bias_flag"].mean()
    evaluation_results["holistic_bias"] = float(overall_bias_share)
    print("Holistic Bias Score:", overall_bias_share)

    if os.path.exists(args.probe_file):
        df_sofa = pd.read_feather(args.probe_file)
    else:
        ds = load_dataset("iproskurina/sofa-500")
        df_sofa = pd.DataFrame(ds["train"])

    df_sofa = compute_probe_ppls(df_sofa, model, tokenizer, args.sofa_batch, args.model)
    compute_identity_ppls(args.identity_file, model, tokenizer, args.sofa_batch, args.model)
    score = compute_sofa_score(df_sofa, model, args.model)

    evaluation_results["sofa"] = float(score)
    print("SOFA Score:", score)

    out_file = os.path.join(
        args.output_dir,
        args.model.replace("/", "_").replace(".", "_") + "_results.json"
    )

    save_json(evaluation_results, out_file)
    print(f"Saved results to {out_file}")

    sys.exit(0)