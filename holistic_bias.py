import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from utils import *
from evaluate_perplexity import *
logger = set_logger(logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="iproskurina/holisticbias-sentiment-pairs")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--is_gptqmodel", action="store_true", help="Is the model GPTQ quantized?")
    # parser.add_argument("--output_dir", type=str, default="results", help="Directory where to save the results.")
    parser.add_argument("--is_vllm_quantized", action="store_true", help="Is the model AWQ quantized?")
    parser.add_argument("--is_int4", action="store_true", help="Whether to load LLM in int4 precision (bitsandbytes).")
    parser.add_argument("--is_int8", action="store_true", help="Whether to load LLM in int8 precision (bitsandbytes).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend",
                        choices=['auto', 'marlin', 'exllama_v1', 'exllama_v2', 'triton', 'cuda', 'torch', 'ipex',
                                 'bitblas'], default='auto', help="Whether to use BACKEND format")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading dataset from {args.dataset_name} ...")
    ds = load_dataset(args.dataset_name, split="train")
    df = ds.to_pandas()
    logger.info(f"Loading model and tokenizer: {args.model_name}")

    if args.is_gptqmodel:
        from gptqmodel import BACKEND, GPTQModel

        model = GPTQModel.load(
            args.model_name,
            device_map="auto",
            trust_remote_code=True,
            backend=BACKEND(args.backend.lower()),
        )
    elif args.is_vllm_quantized:
        from vllm import LLM

        model = LLM(model=args.model_name,
                    trust_remote_code=True,
                    dtype="auto",
                    )
    elif args.is_int4:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto",
                                                     quantization_config=quantization_config)
    elif args.is_int8:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto",
                                                     quantization_config=quantization_config)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype="auto",device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    pos_texts = df["positive_sentence"].tolist()
    neg_texts = df["negative_sentence"].tolist()
    add_bos = True
    if 'qwen' in args.model_name.lower():
        add_bos = False

    logger.info("Computing perplexities for positive sentences ...")
    metric = Perplexity()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pos_ppl = metric._compute(
        predictions=pos_texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=device
    )["perplexities"]
    # pos_ppl = compute_perplexity(pos_texts, model, tokenizer, args.batch_size, args.max_length,add_bos=add_bos)
    logger.info("Computing perplexities for negative sentences ...")
    # neg_ppl = compute_perplexity(neg_texts, model, tokenizer, args.batch_size, args.max_length,add_bos=add_bos)
    neg_ppl = metric._compute(
        predictions=neg_texts,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=device
    )["perplexities"]
    df["pos_ppl"] = pos_ppl
    df["neg_ppl"] = neg_ppl
    df["bias_flag"] = (np.array(neg_ppl) < np.array(pos_ppl)).astype(int)

    overall_bias_share = df["bias_flag"].mean()
    logger.info(f"Overall bias share (negative < positive): {overall_bias_share:.3f}")

    per_template = (
        df.groupby("template")["bias_flag"]
        .mean()
        .reset_index()
        .rename(columns={"bias_flag": "bias_share"})
    )

    output_file = os.path.join(args.output_dir, f"{args.model_name.replace('/', '-')}_bias_results.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"Saved full results to {output_file}")

    summary_file = os.path.join(args.output_dir, f"{args.model_name.replace('/', '-')}_bias_summary.csv")
    per_template.to_csv(summary_file, index=False)
    logger.info(f"Saved summary by template to {summary_file}")

    print(per_template)
    print(f"\nOverall bias share: {overall_bias_share:.3f}")


if __name__ == "__main__":
    main()