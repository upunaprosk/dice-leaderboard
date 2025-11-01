import argparse
import json
import os
import sys

import transformers

from bias_bench.benchmark.seat import SEATRunner
from bias_bench.model import models
from bias_bench.util import generate_experiment_id
from bias_bench.util.compress_utils import DYNAMIC, QAT, PTQ


thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs SEAT benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--tests",
    action="store",
    nargs="*",
    help="List of SEAT tests to run. Test files should be in `data_dir` and have "
    "corresponding names with extension .jsonl.",
)
parser.add_argument(
    "--n_samples",
    action="store",
    type=int,
    default=100000,
    help="Number of permutation test samples used when estimating p-values "
    "(exact test is used if there are fewer than this many permutations).",
)
parser.add_argument(
    "--parametric",
    action="store_true",
    help="Use parametric test (normal assumption) to compute p-values.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="bert-base-uncased",
    #choices=["bert-base-uncased", "albert-base-v2", "roberta-base", "gpt2"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="BertModel",
    #choices=["BertModel", "AlbertModel", "RobertaModel", "GPT2Model"],
    help="Model to evalute (e.g., BertModel). Typically, these correspond to a HuggingFace "
    "class.",
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
    "--device",
    action="store",
    type=str,
    #default=DYNAMIC,
    default=None,
    help="Use cuda or cpu."
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
        name="seat", model=args.model, model_name_or_path=args.model_name_or_path,
        quant_type=args.quant_type, quant_prec=args.quant_prec, revision=args.revision
    )

    print("Running SEAT benchmark:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - tests: {args.tests}")
    print(f" - n_samples: {args.n_samples}")
    print(f" - parametric: {args.parametric}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - quant_prec: {args.quant_prec}")
    print(f" - quant_type: {args.quant_type}")
    print(f" - device: {args.device}")
    print(f" - revision: {args.revision}")
    print(f" - cache_dir: {args.cache_dir}")

    # Load model and tokenizer.
    if args.quant_prec is not None and args.quant_type is not None:
        if args.revision is None:
            model = getattr(models, args.model)(args.model_name_or_path, args.quant_prec, args.quant_type)
        else:
            model = getattr(models, args.model)(args.model_name_or_path, args.revision, args.cache_dir, args.quant_prec, args.quant_type)
    elif args.revision is not None:
        model = getattr(models, args.model)(args.model_name_or_path, args.revision, args.cache_dir)
    else:
        model = getattr(models, args.model)(args.model_name_or_path)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    runner = SEATRunner(
        experiment_id=experiment_id,
        tests=args.tests,
        data_dir=f"{args.persistent_dir}/data/seat",
        n_samples=args.n_samples,
        parametric=args.parametric,
        model=model,
        tokenizer=tokenizer,
        device=args.device,
    )
    results = runner()
    print(results)

    os.makedirs(f"{args.persistent_dir}/results/seat", exist_ok=True)
    with open(f"{args.persistent_dir}/results/seat/{experiment_id}.json", "w") as f:
        json.dump(results, f)
    sys.exit(0)
