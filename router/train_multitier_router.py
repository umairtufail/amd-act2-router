"""Fine-tune the multi-tier router on the labeled dataset.

Reads data/labeled_multitier.jsonl, trains MultiTierRouter (DistilBERT +
numeric features, 4 output classes), and saves:
  - router/checkpoints/multitier_router.pt   (model weights)
  - router/checkpoints/router_config.json    (label map, encoder name)
  - router/checkpoints/tokenizer/            (tokenizer files, so the
    container can load everything offline — no HF download at boot)

Runs unchanged on CPU, Apple MPS, or AMD MI300X via ROCm (reported as cuda).
Small dataset + small model: expect well under a minute of training.

Human runs:  python -m router.train_multitier_router
"""

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

from config import get_tier_names
from data.schema import CATEGORIES
from router.features import extract_features
from router.model import DEFAULT_ENCODER, MultiTierRouter, pick_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "labeled_multitier.jsonl"
CKPT_DIR = Path(__file__).parent / "checkpoints"
MAX_LEN = 256


class RouterDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, label_to_index: dict):
        self.records = records
        self.tokenizer = tokenizer
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        feats = extract_features(rec["prompt"], rec.get("category"))
        enc = self.tokenizer(
            feats["text"], truncation=True, max_length=MAX_LEN,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "numeric": torch.tensor(feats["numeric"], dtype=torch.float32),
            "category_index": torch.tensor(feats["category_index"], dtype=torch.long),
            "label": torch.tensor(self.label_to_index[rec["tier_label"]], dtype=torch.long),
        }


def load_records() -> list[dict]:
    if not DATA_PATH.exists():
        raise SystemExit(
            f"{DATA_PATH} not found — run labeling first: python -m data.label_multitier"
        )
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    # Tasks where no tier passed can't teach the router a useful tier choice;
    # map them to the strongest tier (best remaining option at serve time).
    strongest = get_tier_names()[-1]
    for r in records:
        if r["tier_label"] == "none":
            r["tier_label"] = strongest
    return records


def stratified_split(records: list[dict], val_frac: float, seed: int):
    """Per-class shuffle & split so rare tiers appear in both sets."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for r in records:
        by_label.setdefault(r["tier_label"], []).append(r)
    train, val = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * val_frac)) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    return train, val


def evaluate(model, loader, device, num_classes: int):
    model.eval()
    correct = total = 0
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("label").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
            for t, p in zip(labels.tolist(), preds.tolist()):
                confusion[t][p] += 1
    return correct / max(total, 1), confusion


def _load_tokenizer():
    """Reuse packaged tokenizer files so retraining never needs a hub lookup."""
    local_path = CKPT_DIR / "tokenizer"
    if local_path.exists():
        logger.info("Reusing tokenizer from %s", local_path)
        return AutoTokenizer.from_pretrained(local_path)
    return AutoTokenizer.from_pretrained(DEFAULT_ENCODER)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multi-tier router.")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tier_names = get_tier_names()
    label_to_index = {name: i for i, name in enumerate(tier_names)}

    records = load_records()
    dist = Counter(r["tier_label"] for r in records)
    logger.info("Loaded %d records. Tier distribution: %s", len(records), dict(dist))

    train_recs, val_recs = stratified_split(records, args.val_frac, args.seed)
    logger.info("Train: %d | Val: %d", len(train_recs), len(val_recs))

    tokenizer = _load_tokenizer()
    train_ds = RouterDataset(train_recs, tokenizer, label_to_index)
    val_ds = RouterDataset(val_recs, tokenizer, label_to_index)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = pick_device()
    logger.info("Training on device: %s", device)
    model = MultiTierRouter(num_tiers=len(tier_names)).to(device)

    # Inverse-frequency class weights so rare tiers (usually tier2/3) aren't ignored.
    counts = [max(dist.get(name, 0), 1) for name in tier_names]
    weights = torch.tensor([max(counts) / c for c in counts], dtype=torch.float32).to(device)
    logger.info("Class weights: %s", {n: round(w.item(), 2) for n, w in zip(tier_names, weights)})
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("label").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = loss_fn(model(**batch), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_acc, confusion = evaluate(model, val_loader, device, len(tier_names))
        logger.info("Epoch %d | train loss %.4f | val acc %.3f",
                    epoch, epoch_loss / max(len(train_loader), 1), val_acc)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_DIR / "multitier_router.pt")

    logger.info("Best val accuracy: %.3f", best_val_acc)
    logger.info("Final confusion matrix (rows=true, cols=pred, order %s):\n%s",
                tier_names, confusion)

    # Save everything inference needs, fully offline-loadable: the trained
    # weights are in the .pt file, so inference only needs the encoder
    # architecture config + tokenizer files, never a HuggingFace download.
    tokenizer.save_pretrained(CKPT_DIR / "tokenizer")
    model.encoder.config.save_pretrained(CKPT_DIR / "encoder_config")
    with open(CKPT_DIR / "router_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "encoder_name": DEFAULT_ENCODER,
            "tier_names": tier_names,
            "label_to_index": label_to_index,
            "max_len": MAX_LEN,
            "best_val_acc": best_val_acc,
            "categories": CATEGORIES,
        }, f, indent=2)
    logger.info("Saved checkpoint + tokenizer + config to %s", CKPT_DIR)


if __name__ == "__main__":
    main()
