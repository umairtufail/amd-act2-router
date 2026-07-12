"""Local inference for the binary tier0/tier3 router.

The model is loaded lazily from offline checkpoint artifacts and returns the
probability that the cheapest model can answer a prompt successfully.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from data.schema import CATEGORIES
from router.features import extract_features
from router.labels import CHEAP_OK_LABEL
from router.model import MultiTierRouter

CKPT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_PATH = CKPT_DIR / "binary_router.pt"
CONFIG_PATH = CKPT_DIR / "binary_router_config.json"
DEFAULT_TAU = 0.8


@lru_cache(maxsize=1)
def _load():
    """Load the binary classifier once; CPU inference is intentional."""
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)
    trained_categories = config.get("categories")
    if trained_categories is not None and trained_categories != CATEGORIES:
        raise RuntimeError(
            "Binary router category vocabulary differs from its checkpoint; "
            "run `python -m router.train_binary_router`."
        )
    label_names = config["label_names"]
    if CHEAP_OK_LABEL not in label_names:
        raise ValueError(
            f"{CONFIG_PATH} does not contain the {CHEAP_OK_LABEL!r} label"
        )

    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR / "tokenizer")
    encoder_config = AutoConfig.from_pretrained(CKPT_DIR / "encoder_config")
    model = MultiTierRouter(
        num_tiers=len(label_names), encoder_config=encoder_config
    )
    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return config, tokenizer, model


def checkpoint_available() -> bool:
    """Return whether every artifact required for offline inference is present."""
    return all(
        path.exists()
        for path in (
            CHECKPOINT_PATH,
            CONFIG_PATH,
            CKPT_DIR / "tokenizer",
            CKPT_DIR / "encoder_config",
        )
    )


def _prepare_batch(prompt: str, category: str | None, config: dict, tokenizer):
    feats = extract_features(prompt, category)
    enc = tokenizer(
        feats["text"],
        truncation=True,
        max_length=config["max_len"],
        padding="max_length",
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "numeric": torch.tensor([feats["numeric"]], dtype=torch.float32),
        "category_index": torch.tensor(
            [feats["category_index"]], dtype=torch.long
        ),
    }


def predict_cheap_ok_proba(prompt: str, category: str | None = None) -> float:
    """Return P(cheap_ok), the probability that tier0 will pass."""
    config, tokenizer, model = _load()
    batch = _prepare_batch(prompt, category, config, tokenizer)
    probabilities = model.predict_proba(**batch).squeeze(0)
    cheap_ok_index = config["label_names"].index(CHEAP_OK_LABEL)
    return float(probabilities[cheap_ok_index].item())


def predict_tier(
    prompt: str, category: str | None = None, tau: float = DEFAULT_TAU
) -> str:
    """Threshold P(cheap_ok) and return only ``tier0`` or ``tier3``."""
    probability = predict_cheap_ok_proba(prompt, category)
    return "tier0" if probability >= tau else "tier3"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict tier0 or tier3 with the local binary router."
    )
    parser.add_argument("prompt", help="the query text")
    parser.add_argument("--category", default=None, help="optional task category")
    parser.add_argument(
        "--tau", type=float, default=DEFAULT_TAU, help="cheap_ok threshold"
    )
    args = parser.parse_args()

    probability = predict_cheap_ok_proba(args.prompt, args.category)
    tier = "tier0" if probability >= args.tau else "tier3"
    print(f"P(cheap_ok): {probability:.4f}")
    print(f"Threshold: {args.tau:.3f}")
    print(f"Predicted tier: {tier}")


if __name__ == "__main__":
    main()
