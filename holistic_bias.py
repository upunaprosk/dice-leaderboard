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

logger = set_logger(logging.INFO)


def tokenize_all(texts, tokenizer, max_length, add_bos=True):
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length - 1 if add_bos else max_length,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    if add_bos and tokenizer.bos_token_id is not None:
        # Only add BOS if it’s not already there
        if not torch.all(input_ids[:, 0] == tokenizer.bos_token_id):
            bos = tokenizer.bos_token_id
            bos_tokens = torch.full((input_ids.size(0), 1), bos, dtype=input_ids.dtype)
            input_ids = torch.cat([bos_tokens, input_ids[:, :-1]], dim=1)
            attention_mask = torch.cat(
                [torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype),
                attention_mask[:, :-1]], dim=1
            )
    # if add_bos:
    #     bos = tokenizer.bos_token_id
    #     bos_tokens = torch.full((input_ids.size(0), 1), bos)
    #     input_ids = torch.cat([bos_tokens, input_ids[:, :-1]], dim=1)
    #     attention_mask = torch.cat([torch.ones((attention_mask.size(0), 1)), attention_mask[:, :-1]], dim=1)
    return input_ids, attention_mask


def compute_perplexity(texts, model, tokenizer, batch_size=64, max_length=64, device=None, add_bos=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids, attention_mask = tokenize_all(texts, tokenizer, max_length, add_bos=add_bos)
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    loss_fct = CrossEntropyLoss(reduction="none")
    perplexities = []
    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader, desc="Computing perplexity"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size()) * shift_mask
            loss = loss.sum(1) / shift_mask.sum(1)
            batch_ppl = torch.exp(loss)
            perplexities.extend(batch_ppl.tolist())
    return [round(p, 3) for p in perplexities]


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
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    pos_texts = df["positive_sentence"].tolist()
    neg_texts = df["negative_sentence"].tolist()
    add_bos = True
    if 'qwen' in args.model_name.lower():
        add_bos = False

    logger.info("Computing perplexities for positive sentences ...")
    pos_ppl = compute_perplexity(pos_texts, model, tokenizer, args.batch_size, args.max_length,add_bos=add_bos)
    logger.info("Computing perplexities for negative sentences ...")
    neg_ppl = compute_perplexity(neg_texts, model, tokenizer, args.batch_size, args.max_length,add_bos=add_bos)

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
