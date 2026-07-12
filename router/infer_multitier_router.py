"""Local inference for the multi-tier router. Zero Fireworks tokens.

Loads the trained checkpoint once (lazy singleton) and predicts which tier
should answer a prompt. Everything loads from local files saved by training,
so this works offline inside the submission container.

CLI test:
  python -m router.infer_multitier_router "Explain transformers vs CNNs" --category summarization
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from data.schema import CATEGORIES
from router.features import extract_features
from router.model import MultiTierRouter

logger = logging.getLogger(__name__)

CKPT_DIR = Path(__file__).parent / "checkpoints"


@lru_cache(maxsize=1)
def _load():
    """Load config, tokenizer, and weights once. CPU is intentional:
    inference must run in a slim container with no GPU."""
    with open(CKPT_DIR / "router_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    trained_categories = config.get("categories")
    if trained_categories is not None and trained_categories != CATEGORIES:
        raise RuntimeError(
            "Multitier router category vocabulary differs from its checkpoint; "
            "run `python -m router.train_multitier_router`."
        )
    tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR / "tokenizer")
    encoder_config = AutoConfig.from_pretrained(CKPT_DIR / "encoder_config")
    model = MultiTierRouter(num_tiers=len(config["tier_names"]),
                            encoder_config=encoder_config)
    state = torch.load(CKPT_DIR / "multitier_router.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return config, tokenizer, model


def checkpoint_available() -> bool:
    return (CKPT_DIR / "multitier_router.pt").exists() and \
           (CKPT_DIR / "router_config.json").exists()


def predict_tier(prompt: str, category: str | None = None) -> str:
    """Return 'tier0'..'tier3' for a prompt. Pure local compute."""
    config, tokenizer, model = _load()
    feats = extract_features(prompt, category)
    enc = tokenizer(feats["text"], truncation=True, max_length=config["max_len"],
                    padding="max_length", return_tensors="pt")
    batch = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "numeric": torch.tensor([feats["numeric"]], dtype=torch.float32),
        "category_index": torch.tensor([feats["category_index"]], dtype=torch.long),
    }
    index = model.predict(**batch).item()
    return config["tier_names"][index]


def predict_tier_proba(prompt: str, category: str | None = None) -> dict[str, float]:
    """Tier -> probability, useful for the demo and debugging."""
    config, tokenizer, model = _load()
    feats = extract_features(prompt, category)
    enc = tokenizer(feats["text"], truncation=True, max_length=config["max_len"],
                    padding="max_length", return_tensors="pt")
    batch = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "numeric": torch.tensor([feats["numeric"]], dtype=torch.float32),
        "category_index": torch.tensor([feats["category_index"]], dtype=torch.long),
    }
    probs = model.predict_proba(**batch).squeeze(0).tolist()
    return dict(zip(config["tier_names"], probs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the routing tier for a prompt.")
    parser.add_argument("prompt", help="the query text")
    parser.add_argument("--category", default=None, help="optional task category")
    args = parser.parse_args()

    tier = predict_tier(args.prompt, args.category)
    probs = predict_tier_proba(args.prompt, args.category)
    print(f"Predicted tier: {tier}")
    print("Probabilities: " + ", ".join(f"{t}={p:.3f}" for t, p in probs.items()))


if __name__ == "__main__":
    main()
