"""Fine-tune the local router as a binary cheap-vs-strong classifier.

The classifier predicts ``cheap_ok`` (tier0) or ``needs_strong`` (tier3).
Only the training split is oversampled; validation remains representative of
the original labeled data.

Run with::

    python -m router.train_binary_router
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from data.schema import CATEGORIES
from router.features import extract_features
from router.labels import CHEAP_OK_LABEL, NEEDS_STRONG_LABEL, tier_label_to_binary
from router.model import DEFAULT_ENCODER, MultiTierRouter, pick_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "labeled_multitier.jsonl"
CKPT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_PATH = CKPT_DIR / "binary_router.pt"
CONFIG_PATH = CKPT_DIR / "binary_router_config.json"
MAX_LEN = 256
TAU_DEFAULT = 0.8
LABEL_NAMES = [CHEAP_OK_LABEL, NEEDS_STRONG_LABEL]


class RouterDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, label_to_index: dict[str, int]):
        self.records = records
        self.tokenizer = tokenizer
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        feats = extract_features(rec["prompt"], rec.get("category"))
        enc = self.tokenizer(
            feats["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        binary_label = tier_label_to_binary(rec["tier_label"])
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "numeric": torch.tensor(feats["numeric"], dtype=torch.float32),
            "category_index": torch.tensor(feats["category_index"], dtype=torch.long),
            "label": torch.tensor(self.label_to_index[binary_label], dtype=torch.long),
        }


def load_records() -> list[dict]:
    """Load labeled rows without rewriting ``none`` as a passing tier."""
    if not DATA_PATH.exists():
        raise SystemExit(
            f"{DATA_PATH} not found - run labeling first: "
            "python -m data.label_multitier"
        )
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def binary_label(record: dict) -> str:
    return tier_label_to_binary(record["tier_label"])


def stratified_split(
    records: list[dict], val_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split within each binary class, retaining rare negatives in training."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for rec in records:
        by_label.setdefault(binary_label(rec), []).append(rec)

    train: list[dict] = []
    val: list[dict] = []
    for group in by_label.values():
        group = group.copy()
        rng.shuffle(group)
        if len(group) > 1:
            n_val = max(1, int(len(group) * val_frac))
            n_val = min(n_val, len(group) - 1)
        else:
            n_val = 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def oversample_negatives(
    records: list[dict], neg_multiple: int, seed: int
) -> list[dict]:
    """Return a shuffled training set with negative rows repeated N times."""
    positives = [r for r in records if binary_label(r) == CHEAP_OK_LABEL]
    negatives = [r for r in records if binary_label(r) == NEEDS_STRONG_LABEL]
    sampled = positives + [r for r in negatives for _ in range(neg_multiple)]
    random.Random(seed).shuffle(sampled)
    return sampled


def evaluate(
    model: MultiTierRouter,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    correct = total = 0
    confusion = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=int)
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("label").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions = model(**batch).argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
            for target, prediction in zip(labels.tolist(), predictions.tolist()):
                confusion[target][prediction] += 1
    return correct / max(total, 1), confusion


def _load_tokenizer():
    local_path = CKPT_DIR / "tokenizer"
    if local_path.exists():
        logger.info("Reusing tokenizer from %s", local_path)
        return AutoTokenizer.from_pretrained(local_path)
    return AutoTokenizer.from_pretrained(DEFAULT_ENCODER)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the binary tier0/tier3 router.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--neg-multiple",
        type=int,
        default=8,
        help="effective copies of each needs_strong row in the training split",
    )
    args = parser.parse_args()

    if args.neg_multiple < 1:
        parser.error("--neg-multiple must be at least 1")
    if not 0.0 < args.val_frac < 1.0:
        parser.error("--val-frac must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    label_to_index = {name: index for index, name in enumerate(LABEL_NAMES)}
    records = load_records()
    distribution = Counter(binary_label(rec) for rec in records)
    logger.info(
        "Loaded %d records. Binary distribution: %s",
        len(records),
        dict(distribution),
    )

    train_records, val_records = stratified_split(records, args.val_frac, args.seed)
    train_distribution = Counter(binary_label(rec) for rec in train_records)
    sampled_train_records = oversample_negatives(
        train_records, args.neg_multiple, args.seed
    )
    sampled_distribution = Counter(binary_label(rec) for rec in sampled_train_records)
    logger.info(
        "Train: %d original / %d oversampled %s | Val: %d %s",
        len(train_records),
        len(sampled_train_records),
        dict(sampled_distribution),
        len(val_records),
        dict(Counter(binary_label(rec) for rec in val_records)),
    )

    tokenizer = _load_tokenizer()
    train_dataset = RouterDataset(sampled_train_records, tokenizer, label_to_index)
    val_dataset = RouterDataset(val_records, tokenizer, label_to_index)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    device = pick_device()
    logger.info("Training on device: %s", device)
    model = MultiTierRouter(num_tiers=len(LABEL_NAMES)).to(device)

    # Use the unmodified training split for weights so oversampling and class
    # weighting both counter the severe original imbalance.
    counts = [max(train_distribution.get(name, 0), 1) for name in LABEL_NAMES]
    weights = torch.tensor(
        [max(counts) / count for count in counts], dtype=torch.float32, device=device
    )
    logger.info(
        "Class weights: %s",
        {name: round(weight.item(), 2) for name, weight in zip(LABEL_NAMES, weights)},
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    best_confusion = np.zeros((len(LABEL_NAMES), len(LABEL_NAMES)), dtype=int)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("label").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = loss_fn(model(**batch), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_acc, confusion = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d | train loss %.4f | val acc %.3f",
            epoch,
            epoch_loss / max(len(train_loader), 1),
            val_acc,
        )
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_confusion = confusion.copy()
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    logger.info("Best val accuracy: %.3f", best_val_acc)
    logger.info(
        "Best confusion matrix (rows=true, cols=pred, order %s):\n%s",
        LABEL_NAMES,
        best_confusion,
    )

    tokenizer.save_pretrained(CKPT_DIR / "tokenizer")
    model.encoder.config.save_pretrained(CKPT_DIR / "encoder_config")
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "encoder_name": DEFAULT_ENCODER,
                "label_names": LABEL_NAMES,
                "label_to_index": label_to_index,
                "max_len": MAX_LEN,
                "best_val_acc": best_val_acc,
                "tau_default": TAU_DEFAULT,
                "neg_multiple": args.neg_multiple,
                "categories": CATEGORIES,
            },
            f,
            indent=2,
        )
    logger.info("Saved binary checkpoint and config to %s", CKPT_DIR)


if __name__ == "__main__":
    main()
