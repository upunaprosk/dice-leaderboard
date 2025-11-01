import argparse
import os
import json

import transformers

from bias_bench.benchmark.crows import CrowSPairsRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id, _is_generative

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs CrowS-Pairs benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertForMaskedLM",
    #choices=[
    #    "DistilBertForMaskedLM",
    #    "BertForMaskedLM",
    #    "AlbertForMaskedLM",
    #    "RobertaForMaskedLM",
    #    "DistilRobertaForMaskedLM",
    #    "GPT2LMHeadModel",
    #    "IBertForMaskedLM"
    #],
    help="Model to evalute (e.g., BertForMaskedLM). Typically, these correspond to a HuggingFace "
    "class.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    #choices=["distilbert-base-uncased", "distilroberta-base", "bert-base-uncased", "albert-base-v2", "roberta-base",
    #         "gpt2", "kssteven/ibert-roberta-base"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    default=None,
    choices=["gender", "religion", "race", "sexual-orientation", "age", "nationality", "disability",
             "physical-appearance", "socioeconomic"],
    help="Determines which CrowS-Pairs dataset split to evaluate against.",
)

parser.add_argument(
    "--quant_prec",
    action="store",
    type=str,
    #default="fp16",
    default=None,
    help="Choose quantization precision (e.g., fp16, int8)."
)

parser.add_argument(
    "--quant_type",
    action="store",
    type=str,
    #default=DYNAMIC,
    default=None,
    help="Choose type of quantization (e.g., dynamic, static)."
)

parser.add_argument(
    "--revision",
    action="store",
    type=str,
    #default=DYNAMIC,
    default=None,
    help="Pythia checkpoint revision e.g. step3000"
)

parser.add_argument(
    "--cache_dir",
    action="store",
    type=str,
    #default=DYNAMIC,
    default=None,
    help="Pythia model cache directory e.g. /home/username/.cache/pythia"
)


if __name__ == "__main__":
    args = parser.parse_args()

    experiment_id = generate_experiment_id(
        name="crows",
        model=args.model,
        model_name_or_path=args.model_name_or_path,
        bias_type=args.bias_type,
        quant_type=args.quant_type, quant_prec=args.quant_prec,
        revision=args.revision,
    )

    print("Running CrowS-Pairs benchmark:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - bias_type: {args.bias_type}")
    print(f" - quant_prec: {args.quant_prec}")
    print(f" - quant_type: {args.quant_type}")
    print(f" - revision: {args.revision}")
    print(f" - cache_dir: {args.cache_dir}")

    # Load model and tokenizer.
    if args.quant_prec is not None and args.quant_type is not None:
        if args.revision is None:
            model = getattr(models, args.model)(args.model_name_or_path, args.quant_prec, args.quant_type)
        else:
            model = getattr(models, args.model)(args.model_name_or_path, args.revision, args.cache_dir, args.quant_prec,
                                                args.quant_type)
    elif args.revision is not None:
        model = getattr(models, args.model)(args.model_name_or_path, args.revision, args.cache_dir)
    else:
        model = getattr(models, args.model)(args.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    runner = CrowSPairsRunner(
        model=model,
        tokenizer=tokenizer,
        input_file=f"{args.persistent_dir}/data/crows/crows_pairs_anonymized.csv",
        bias_type=args.bias_type,
        is_generative=_is_generative(args.model),  # Affects model scoring.
    )
    results = runner()

    print(f"Metric: {results}")

    os.makedirs(f"{args.persistent_dir}/results/crows", exist_ok=True)
    with open(f"{args.persistent_dir}/results/crows/{experiment_id}.json", "w") as f:
        json.dump(results, f)
