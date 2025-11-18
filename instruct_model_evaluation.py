#!/usr/bin/env python3
import argparse
import os
import json
import sys
import numpy as np
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from logbar import LogBar

from ppl_eval import Perplexity as PPL_Perplexity
from moral_stories_declarative_prompt import *

logger = LogBar.shared()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_main_model(args, tokenizer):
    if args.is_gptqmodel:
        from gptqmodel import BACKEND, GPTQModel
        return GPTQModel.load(
            args.model,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            backend=BACKEND(args.backend.lower()),
        )
    elif args.is_vllm_quantized:
        return AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", trust_remote_code=args.trust_remote_code
        )
    elif args.is_int4:
        from transformers import BitsAndBytesConfig
        return AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )
    elif args.is_int8:
        from transformers import BitsAndBytesConfig
        return AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=args.trust_remote_code,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--is_gptqmodel", action="store_true")
    parser.add_argument("--is_vllm_quantized", action="store_true")
    parser.add_argument("--is_int4", action="store_true")
    parser.add_argument("--is_int8", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--backend", choices=['auto','marlin','exllama_v1','exllama_v2','triton','cuda','torch','ipex','bitblas'], default="auto")
    parser.add_argument("--dataset_path", type=str, default="wikitext")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--n_ctx", type=int, default=1024)
    parser.add_argument("--n_batch", type=int, default=1024)
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--prompt_with_norm", choices=[True, False], default=True)
    parser.add_argument("--moral_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_main_model(args, tokenizer)
    evaluation_results = {"model": args.model}

    ppl = PPL_Perplexity(
        model,
        tokenizer,
        args.dataset_path,
        args.dataset_name,
        args.split,
        args.text_column,
    )
    all_perplexity = ppl.calculate(args.n_ctx, args.n_batch)
    evaluation_results["perplexity"] = float(np.mean(all_perplexity))

    seed_everything(args.seed)
    moral_scores = {}

    for lang, dataset_name in {"en": "LabHC/moral_stories", "fr": "LabHC/histoires_morales"}.items():
        dataset = load_data(dataset_name, args)
        model2, tokenizer2, device2 = load_model(args)
        moral, immoral, refused = prompting(model2, tokenizer2, device2, dataset, args)
        moral_scores[lang] = {"moral": moral, "immoral": immoral, "refused": refused}

    evaluation_results["moral_stories"] = moral_scores

    out_file = os.path.join(args.output_dir, args.model.replace("/", "_").replace(".", "_") + "_results.json")
    save_json(evaluation_results, out_file)
    print(out_file)
    sys.exit(0)
