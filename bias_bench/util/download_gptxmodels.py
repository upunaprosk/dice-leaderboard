import argparse
import os
import sys
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs Download models to cache.")

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
    "--revision",
    action="store",
    type=str,
    #default=DYNAMIC,
    default=None,
    required=True,
    help="Pythia checkpoint revision e.g. step3000"
)

parser.add_argument(
    "--cache_dir",
    action="store",
    type=str,
    #default=DYNAMIC,
    default=None,
    required=True,
    help="Pythia model cache directory e.g. /home/username/.cache/pythia"
)

if __name__ == "__main__":
    args = parser.parse_args()

    print("Download models:")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - revision: {args.revision}")
    print(f" - cache_dir: {args.cache_dir}")

    device = torch.device('cpu')

    model_name_or_path = args.model_name_or_path
    revision = args.revision
    cache_dir = args.cache_dir

    model = GPTNeoXForCausalLM.from_pretrained(
        model_name_or_path,
        revision=revision,
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        revision=revision,
        cache_dir=cache_dir,
    )

    inputs = tokenizer("Hello, I am", return_tensors="pt")
    tokens = model.generate(**inputs)
    tokenizer.decode(tokens[0])

    sys.exit(0)
