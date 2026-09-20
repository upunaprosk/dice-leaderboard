from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


class Perplexity:
    """
    Sentence-level perplexity evaluation used in (copied from evaluate.perpexity with added quantized support)
    - Holistic Bias
    - SOFA
    """

    def _compute(
        self,
        predictions: list[str],
        model: Any,
        tokenizer: Any,
        batch_size: int = 16,
        add_start_token: bool = True,
        device: str | None = None,
        max_length: int | None = None,
    ) -> dict[str, Any]:
        """
        Compute perplexity for each input
        """

        if batch_size <= 0:
            raise ValueError(
                "`batch_size` must be greater than zero."
            )

        if not predictions:
            raise ValueError(
                "`predictions` cannot be empty."
            )

        device = _resolve_device(
            device
        )

        previous_padding_side = (
            tokenizer.padding_side
        )

        previous_pad_token = (
            tokenizer.pad_token
        )

        try:
            # Sentence perplexity is a likelihood/scoring operation.
            tokenizer.padding_side = "right"
            if (
                tokenizer.pad_token is None
                and batch_size > 1
            ):
                existing_special_tokens = list(
                    tokenizer
                    .special_tokens_map_extended
                    .values()
                )

                if not existing_special_tokens:
                    raise ValueError(
                        "If batch_size > 1, the tokenizer must "
                        "define at least one special token that "
                        "can also be used for padding."
                    )

                tokenizer.add_special_tokens(
                    {
                        "pad_token":
                            existing_special_tokens[0]
                    }
                )

            if (
                add_start_token
                and max_length is not None
            ):
                if tokenizer.bos_token is None:
                    raise ValueError(
                        "A BOS token is required when "
                        "add_start_token=True and max_length "
                        "is specified."
                    )

                # Leave one position for the BOS token.
                max_tokenized_len = (
                    max_length - 1
                )

            else:
                max_tokenized_len = (
                    max_length
                )

            encodings = tokenizer(
                predictions,
                add_special_tokens=False,
                padding=True,
                truncation=(
                    max_tokenized_len
                    is not None
                ),
                max_length=max_tokenized_len,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(
                device
            )

            encoded_texts = (
                encodings["input_ids"]
            )

            attention_masks = (
                encodings["attention_mask"]
            )

            _validate_inputs(
                attention_masks=attention_masks,
                add_start_token=add_start_token,
            )

            perplexities = (
                _compute_perplexities(
                    model=model,
                    tokenizer=tokenizer,
                    encoded_texts=encoded_texts,
                    attention_masks=attention_masks,
                    batch_size=batch_size,
                    add_start_token=add_start_token,
                    device=device,
                )
            )

        finally:
            tokenizer.padding_side = (
                previous_padding_side
            )

            tokenizer.pad_token = (
                previous_pad_token
            )

        return {
            "perplexities": perplexities,
            "mean_perplexity": float(
                np.mean(
                    perplexities
                )
            ),
        }


def _compute_perplexities(
    model: Any,
    tokenizer: Any,
    encoded_texts: torch.Tensor,
    attention_masks: torch.Tensor,
    batch_size: int,
    add_start_token: bool,
    device: str,
) -> list[float]:
    """
    Sentence perplexity evaluation
    """

    perplexities: list[float] = []

    loss_function = CrossEntropyLoss(
        reduction="none"
    )

    for start_index in tqdm(
        range(
            0,
            len(encoded_texts),
            batch_size,
        ),
        desc="Sentence perplexity",
    ):
        end_index = min(
            start_index + batch_size,
            len(encoded_texts),
        )

        encoded_batch = encoded_texts[
            start_index:end_index
        ]

        attention_mask = attention_masks[
            start_index:end_index
        ]
        if (
            add_start_token
            and tokenizer.bos_token_id
            is not None
        ):
            bos_tokens = torch.tensor(
                [
                    [tokenizer.bos_token_id]
                ]
                * encoded_batch.size(0),
                device=device,
                dtype=encoded_batch.dtype,
            )

            encoded_batch = torch.cat(
                [
                    bos_tokens,
                    encoded_batch,
                ],
                dim=1,
            )

            bos_attention = torch.ones(
                bos_tokens.size(),
                device=device,
                dtype=attention_mask.dtype,
            )

            attention_mask = torch.cat(
                [
                    bos_attention,
                    attention_mask,
                ],
                dim=1,
            )

        labels = encoded_batch

        with torch.no_grad():
            logits = model(
                encoded_batch,
                attention_mask=attention_mask,
            ).logits

        shift_logits = (
            logits[
                ...,
                :-1,
                :,
            ]
            .contiguous()
            .float()
        )

        shift_labels = (
            labels[
                ...,
                1:,
            ]
            .contiguous()
        )

        shift_attention_mask = (
            attention_mask[
                ...,
                1:,
            ]
            .contiguous()
        )

        token_losses = loss_function(
            shift_logits.transpose(
                1,
                2,
            ),
            shift_labels,
        )

        perplexity_batch = torch.exp(
            (
                token_losses
                * shift_attention_mask
            ).sum(1)
            / shift_attention_mask.sum(1)
        )

        perplexities.extend(
            perplexity_batch.tolist()
        )

    return perplexities


def _validate_inputs(
    attention_masks: torch.Tensor,
    add_start_token: bool,
) -> None:
    """
    Minimum input-length check
    """

    token_counts = (
        attention_masks.sum(1)
    )

    if add_start_token:
        if not torch.all(
            token_counts >= 1
        ):
            raise ValueError(
                "Each input text must contain at least "
                "one token."
            )

    else:
        if not torch.all(
            token_counts >= 2
        ):
            raise ValueError(
                "When add_start_token=False, each input "
                "text must contain at least two tokens."
            )


def _resolve_device(
    device: str | None,
) -> str:
    """
    Resolve the device
    """

    if device is None:
        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    if device not in {
        "gpu",
        "cuda",
        "cpu",
    }:
        raise ValueError(
            "`device` must be one of "
            "'gpu', 'cuda', or 'cpu'."
        )

    if device == "gpu":
        return "cuda"

    return device
