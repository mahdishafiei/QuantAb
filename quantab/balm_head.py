"""BALM-paired with a sequence-level head (the lab's BALM-paper downstream classifier).

This loads ``brineylab/BALM-paired`` with a ``AutoModelForSequenceClassification`` head —
the same head architecture the lab fine-tunes on top of the LM — but wired to *our* curated
DMS binding data (heavy + light chains, continuous ``kd_nm`` affinity), not the paper's
native-vs-shuffled dataset.

Heavy and light chains are tokenized in BALM's native paired format
(``<s> H </s></s> L </s>``) via the two-sequence tokenizer call.

Default is regression (``num_labels=1``, ``problem_type="regression"``) because the label is a
continuous KD. The head is randomly initialized and must be fine-tuned on the DMS data.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BALM_NAME = "brineylab/BALM-paired"
MAX_LENGTH = 512  # BALM max_position_embeddings; paired H+L (~230 tokens) fits easily
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_balm_head(
    num_labels: int = 1,
    problem_type: str = "regression",
    device: str = DEVICE,
):
    """Load BALM-paired with a sequence-classification/regression head.

    Args:
        num_labels: 1 for regression on affinity; >=2 for classification.
        problem_type: "regression" or "single_label_classification".
        device: torch device string.

    Returns:
        (tokenizer, model) — the head is untrained and needs fine-tuning on the DMS data.
    """
    tokenizer = AutoTokenizer.from_pretrained(BALM_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        BALM_NAME,
        num_labels=num_labels,
        problem_type=problem_type,
    ).to(device)
    return tokenizer, model


def tokenize_pairs(
    tokenizer,
    heavy: list[str],
    light: list[str],
    max_length: int = MAX_LENGTH,
):
    """Tokenize (heavy, light) pairs in BALM's native paired format.

    Produces ``<s> H </s></s> L </s>`` per sample. Light chains that are ``None`` become "".
    """
    light = ["" if l is None else l for l in light]
    return tokenizer(
        list(heavy),
        light,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


@torch.no_grad()
def predict(model, tokenizer, heavy: list[str], light: list[str],
            batch_size: int = 32, device: str = DEVICE) -> np.ndarray:
    """Run the BALM head over (heavy, light) pairs and return scalar predictions.

    For an untrained head this is just a forward-pass smoke test; after fine-tuning it
    returns the predicted affinity per sample.
    """
    model.eval()
    out: list[np.ndarray] = []
    for i in range(0, len(heavy), batch_size):
        enc = tokenize_pairs(tokenizer, heavy[i:i + batch_size], light[i:i + batch_size]).to(device)
        logits = model(**enc).logits
        out.append(logits.squeeze(-1).cpu().float().numpy())
    return np.concatenate(out)
