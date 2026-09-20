from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset, load_from_disk


DEFAULT_DATASET = "wikitext"
DEFAULT_WIKITEXT_CONFIG = "wikitext-2-raw-v1"

DEFAULT_N_CTX = 1024
DEFAULT_N_BATCH = 1024


class Perplexity:
    """
    Perplexity evaluator.
    The leaderboard protocol uses:
        n_ctx = 1024
        n_batch = 1024
    """

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

        self._dataset_path = dataset_path
        self._dataset_name = dataset_name
        self._split = split
        self._text_column = text_column

        self._text = self._prepare_data()

    @property
    def dataset_name(self) -> str | None:
        """Return the resolved dataset configuration name."""
        return self._dataset_name

    def _prepare_data(self) -> str:
        """
        Load and prepare the evaluation corpus (wikitext-2)
        """

        if self._dataset_path == DEFAULT_DATASET:
            self._dataset_name = DEFAULT_WIKITEXT_CONFIG

        minimum_length = (
            512
            if self._dataset_path == DEFAULT_DATASET
            else 2048
        )

        data = self._load_dataset()

        texts: list[str] = []

        for sample in data:
            if self._text_column not in sample:
                raise KeyError(
                    f"Dataset does not contain column "
                    f"'{self._text_column}'."
                )

            text = sample[self._text_column]

            if text is None:
                continue

            if not isinstance(text, str):
                text = str(text)

            if len(text) < minimum_length:
                continue
            texts.append(
                " \n" if text == "" else text
            )

            if len(texts) >= 1024:
                break

        if not texts:
            raise ValueError(
                "No suitable text samples were found "
                "for perplexity evaluation."
            )

        return "".join(texts)

    def _load_dataset(self):
        """
        Load a HF dataset or a local dataset
        """

        path = Path(self._dataset_path)

        if path.exists():

            if path.is_dir():
                data = load_from_disk(
                    str(path)
                )

                if (
                    hasattr(data, "keys")
                    and self._split in data
                ):
                    return data[self._split]

                return data

            if str(path).endswith(".gz"):

                if self._dataset_name is None:
                    raise ValueError(
                        "`dataset_name` must be provided "
                        "when loading a local compressed dataset."
                    )

                return load_dataset(
                    self._dataset_name,
                    data_files=str(path),
                    split=self._split,
                )

        return load_dataset(
            self._dataset_path,
            self._dataset_name,
            split=self._split,
        )

    @staticmethod
    def softmax(
        logits: torch.Tensor,
    ) -> torch.Tensor:
        e_x = torch.exp(
            logits - torch.max(logits)
        )

        return e_x / torch.sum(
            e_x,
            dim=0,
        )

    def calculate(
        self,
        n_ctx: int = DEFAULT_N_CTX,
        n_batch: int = DEFAULT_N_BATCH,
    ) -> list[float]:
        """
        Calculate perplexity values
        """

        if n_ctx <= 0:
            raise ValueError(
                "`n_ctx` must be greater than zero."
            )

        if n_batch <= 0:
            raise ValueError(
                "`n_batch` must be greater than zero."
            )
        if n_batch != n_ctx:
            raise ValueError(
                "Historical DICE perplexity requires "
                "`n_batch == n_ctx`. "
                f"Received n_ctx={n_ctx}, n_batch={n_batch}."
            )

        original_max_length = getattr(
            self._tokenizer,
            "model_max_length",
            None,
        )
        self._tokenizer.model_max_length = sys.maxsize

        try:
            tokens = self._tokenizer(
                self._text,
                truncation=False,
                return_tensors="pt",
            ).input_ids

        finally:
            if original_max_length is not None:
                self._tokenizer.model_max_length = (
                    original_max_length
                )

        tokens = tokens.to(
            self._get_model_device()
        )

        number_of_contexts = (
            len(tokens[0]) // n_ctx
        )

        if number_of_contexts == 0:
            raise ValueError(
                "The prepared dataset contains fewer "
                f"than {n_ctx} tokens."
            )

        nll = 0.0
        count = 0

        all_perplexity: list[float] = []

        for i in range(number_of_contexts):

            nll, count = self._process_batch(
                i=i,
                n_ctx=n_ctx,
                n_batch=n_batch,
                tokens=tokens,
                nll=nll,
                count=count,
            )

            current_perplexity = float(
                np.exp(nll / count)
            )

            all_perplexity.append(
                current_perplexity
            )

        return all_perplexity

    def _process_batch(
        self,
        i: int,
        n_ctx: int,
        n_batch: int,
        tokens: torch.Tensor,
        nll: float,
        count: int,
    ) -> tuple[float, int]:
        """
        Process one context window
        """

        start = i * n_ctx
        end = start + n_ctx

        num_batches = (
            n_ctx + n_batch - 1
        ) // n_batch

        logits: list[torch.Tensor] = []

        for j in range(num_batches):

            batch_start = (
                start + j * n_batch
            )

            batch_size = min(
                end - batch_start,
                n_batch,
            )

            token_org = (
                tokens[0][batch_start].item()
            )

            if (
                j == 0
                and self._tokenizer.bos_token_id
                is not None
            ):
                tokens[0][batch_start] = (
                    self._tokenizer.bos_token_id
                )

            batch_logits = (
                self._compute_batch_logits(
                    tokens=tokens,
                    batch_start=batch_start,
                    batch_size=batch_size,
                )
            )
            tokens[0][batch_start] = token_org

            logits.append(
                batch_logits
            )
        score_start = min(
            512,
            n_ctx // 2,
        )

        for j in range(
            score_start,
            n_ctx - 1,
        ):

            tok_logits = logits[0][0][j]

            probability = self.softmax(
                tok_logits
            )[
                tokens[0][start + j + 1]
            ]

            probability = torch.where(
                probability > 0,
                probability,
                torch.tensor(
                    1e-8,
                    device=probability.device,
                ),
            )

            nll += -torch.log(
                probability
            ).item()

            count += 1

        return nll, count

    def _compute_batch_logits(
        self,
        tokens: torch.Tensor,
        batch_start: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute model logits without gradient tracking."""

        with torch.no_grad():

            outputs = self._model(
                tokens[
                    :,
                    batch_start:
                    batch_start + batch_size,
                ]
            )

        return outputs.logits.detach()

    def _get_model_device(
        self,
    ) -> torch.device:

        device = getattr(
            self._model,
            "device",
            None,
        )

        if device is not None:

            device = torch.device(
                device
            )

            if device.type != "meta":
                return device

        try:
            return next(
                self._model.parameters()
            ).device

        except (
            AttributeError,
            StopIteration,
        ):
            pass

        wrapped_model = getattr(
            self._model,
            "model",
            None,
        )

        if wrapped_model is not None:

            device = getattr(
                wrapped_model,
                "device",
                None,
            )

            if device is not None:

                device = torch.device(
                    device
                )

                if device.type != "meta":
                    return device

            try:
                return next(
                    wrapped_model.parameters()
                ).device

            except (
                AttributeError,
                StopIteration,
            ):
                pass

        raise RuntimeError(
            "Could not determine the model device."
        )


def evaluate(
    model: Any,
    tokenizer: Any,
    config: dict[str, Any],
) -> dict[str, Any]:

    dataset_path = config.get(
        "dataset_path",
        DEFAULT_DATASET,
    )

    dataset_name = config.get(
        "dataset_name"
    )

    split = config.get(
        "split",
        "test",
    )

    text_column = config.get(
        "text_column",
        "text",
    )

    n_ctx = int(
        config.get(
            "n_ctx",
            DEFAULT_N_CTX,
        )
    )

    n_batch = int(
        config.get(
            "n_batch",
            DEFAULT_N_BATCH,
        )
    )

    evaluator = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        split=split,
        text_column=text_column,
    )

    perplexities = evaluator.calculate(
        n_ctx=n_ctx,
        n_batch=n_batch,
    )

    average_perplexity = float(
        np.mean(perplexities)
    )

    return {
        "score": average_perplexity,
        "dataset_path": dataset_path,
        "dataset_name": evaluator.dataset_name,
        "split": split,
        "text_column": text_column,
        "n_ctx": n_ctx,
        "n_batch": n_batch,
        "contexts": len(perplexities),
    }