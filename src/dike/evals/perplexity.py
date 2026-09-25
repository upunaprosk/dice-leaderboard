"""DICE WikiText-2 token perplexity (sliding-window protocol v1).
"""

from __futurUpdae__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk

DEFAULT_DATASET = "Salesforce/wikitext"
DEFAULT_WIKITEXT_CONFIG = "wikitext-2-raw-v1"
DEFAULT_N_CTX = 1024
DEFAULT_STRIDE = 512


class Perplexity:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        dataset_path: str = DEFAULT_DATASET,
        dataset_name: str | None = None,
        split: str = "test",
        text_column: str = "text",
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._dataset_path = (
            DEFAULT_DATASET if dataset_path == "wikitext" else dataset_path
        )
        self._dataset_name = (
            DEFAULT_WIKITEXT_CONFIG
            if self._dataset_path == DEFAULT_DATASET and dataset_name is None
            else dataset_name
        )
        self._split = split
        self._text_column = text_column
        self.scored_tokens = 0
        self.total_nll = 0.0
        self._text = self._prepare_data()

    @property
    def dataset_name(self) -> str | None:
        return self._dataset_name

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

    def _load_dataset(self):
        path = Path(self._dataset_path)
        if path.exists() and path.is_dir():
            data = load_from_disk(str(path))
            if hasattr(data, "keys") and self._split in data:
                return data[self._split]
            return data
        if path.exists() and path.suffix == ".gz":
            if self._dataset_name is None:
                raise ValueError("dataset_name is required for a local .gz file")
            return load_dataset(
                self._dataset_name, data_files=str(path), split=self._split
            )
        return load_dataset(
            self._dataset_path, self._dataset_name, split=self._split
        )

    def _prepare_data(self) -> str:
        data = self._load_dataset()
        # Entire test corpus; no length filtering or 1024-sample cap.
        # Preserve blank rows as separators, as in the HF WikiText example.
        return "\n\n".join(
            "" if row[self._text_column] is None else str(row[self._text_column])
            for row in data
        )

    def _get_model_device(self) -> torch.device:
        embeddings = getattr(self._model, "get_input_embeddings", lambda: None)()
        if embeddings is not None:
            device = embeddings.weight.device
            if device.type != "meta":
                return device
        device = getattr(self._model, "device", None)
        if device is not None:
            device = torch.device(device)
            if device.type != "meta":
                return device
        for parameter in self._model.parameters():
            if parameter.device.type != "meta":
                return parameter.device
        raise RuntimeError("Unable to determine the model's input device")

    def calculate(
        self,
        n_ctx: int = DEFAULT_N_CTX,
        stride: int = DEFAULT_STRIDE,
    ) -> list[float]:
        if n_ctx < 2:
            raise ValueError("n_ctx must be at least 2")
        # With stride == n_ctx, the first token of every later window has
        # no preceding token in the window and would be silently skipped.
        if not 1 <= stride < n_ctx:
            raise ValueError("stride must satisfy 1 <= stride < n_ctx")

        original_max_length = getattr(self._tokenizer, "model_max_length", None)
        if original_max_length is not None:
            self._tokenizer.model_max_length = sys.maxsize
        try:
            tokens = self._tokenizer(
                self._text,
                add_special_tokens=False,
                truncation=False,
                return_tensors="pt",
            ).input_ids
        finally:
            if original_max_length is not None:
                self._tokenizer.model_max_length = original_max_length

        seq_len = tokens.size(1)
        if seq_len < 2:
            raise ValueError("Evaluation corpus must contain at least two tokens")

        self._model.eval()
        device = self._get_model_device()
        self.total_nll = 0.0
        self.scored_tokens = 0
        cumulative_perplexities: list[float] = []

        previous_end = 0
        end = min(n_ctx, seq_len)
        while True:
            begin = max(0, end - n_ctx)
            # Each target is scored once; token zero lacks a predecessor.
            target_start = max(previous_end, begin + 1)
            first_logit = target_start - begin - 1
            last_logit = end - begin - 1

            input_ids = tokens[:, begin:end].to(device)
            with torch.inference_mode():
                logits = self._model(
                    input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                ).logits
                prediction_logits = logits[0, first_logit:last_logit].float()
                target_ids = tokens[0, target_start:end].to(
                    prediction_logits.device
                )
                if prediction_logits.size(0) != target_ids.numel():
                    raise AssertionError("Next-token logits and targets misaligned")
                window_nll = F.cross_entropy(
                    prediction_logits, target_ids, reduction="sum"
                )

            self.total_nll += window_nll.item()
            self.scored_tokens += target_ids.numel()
            cumulative_perplexities.append(
                math.exp(self.total_nll / self.scored_tokens)
            )
            previous_end = end
            if end == seq_len:
                break
            end = min(end + stride, seq_len)

        if self.scored_tokens != seq_len - 1:
            raise AssertionError(
                f"Expected {seq_len - 1} targets; scored {self.scored_tokens}"
            )
        return cumulative_perplexities


def evaluate(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    # Old n_batch is a different parameter, NOT an alias for stride.
    if "n_batch" in config:
        raise ValueError(
            "This sliding-window protocol uses 'stride', not 'n_batch'. "
            "Update the official recipe before running."
        )

    evaluator = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path=config.get("dataset_path", DEFAULT_DATASET),
        dataset_name=config.get("dataset_name"),
        split=config.get("split", "test"),
        text_column=config.get("text_column", "text"),
    )
    n_ctx = int(config.get("n_ctx", DEFAULT_N_CTX))
    stride = int(config.get("stride", DEFAULT_STRIDE))
    perplexities = evaluator.calculate(n_ctx=n_ctx, stride=stride)
    return {
        "score": perplexities[-1],
        "dataset_path": evaluator.dataset_path,
        "dataset_name": evaluator.dataset_name,
        "split": config.get("split", "test"),
        "text_column": config.get("text_column", "text"),
        "n_ctx": n_ctx,
        "stride": stride,
        "contexts": len(perplexities),
        "scored_tokens": evaluator.scored_tokens,
        "protocol": "wikitext2-raw-sliding-v1",
    }
