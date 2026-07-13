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
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from data.integrity import (
    assert_disjoint_prompt_groups,
    collapse_prompt_groups,
    dataset_hash,
    prompt_group_key,
)
from data.schema import CATEGORIES
from router.features import extract_features
from router.labels import CHEAP_OK_LABEL, NEEDS_STRONG_LABEL, tier_label_to_binary
from router.model import DEFAULT_ENCODER, MultiTierRouter, pick_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "labeled_multitier.jsonl"
CKPT_DIR = Path(__file__).parent / "checkpoints"
ACTIVE_CHECKPOINT_PATH = CKPT_DIR / "binary_router.pt"
ACTIVE_CONFIG_PATH = CKPT_DIR / "binary_router_config.json"
DEFAULT_CANDIDATE_DIR = CKPT_DIR / "candidates" / "binary_router"
MAX_LEN = 256
TAU_DEFAULT = 0.8
LABEL_NAMES = [CHEAP_OK_LABEL, NEEDS_STRONG_LABEL]
MIN_HARD_GROUPS_FOR_PROMOTION = 10


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
        target_label = binary_label(rec)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "numeric": torch.tensor(feats["numeric"], dtype=torch.float32),
            "category_index": torch.tensor(feats["category_index"], dtype=torch.long),
            "label": torch.tensor(self.label_to_index[target_label], dtype=torch.long),
        }


def load_records(data_paths: list[Path] | None = None) -> list[dict]:
    """Load train-only labeled rows without rewriting ``none`` labels."""
    paths = data_paths or [DATA_PATH]
    records: list[dict] = []
    seen_ids: set[str] = set()
    for path in paths:
        is_legacy_default = path.resolve() == DATA_PATH.resolve()
        if not path.exists():
            raise SystemExit(
                f"{path} not found - run labeling first: "
                "python -m data.label_multitier"
            )
        with path.open("r", encoding="utf-8") as handle:
            loaded = [json.loads(line) for line in handle if line.strip()]
        for record in loaded:
            split = record.get("dataset_split")
            if split == "stress_holdout":
                raise SystemExit(
                    f"refusing stress_holdout training row {record.get('id')!r} "
                    f"from {path}"
                )
            if split is None and not is_legacy_default:
                raise SystemExit(
                    f"custom training source {path} contains row "
                    f"{record.get('id')!r} without dataset_split='train'"
                )
            if split not in {None, "train"}:
                raise SystemExit(
                    f"unsupported dataset_split {split!r} in training row "
                    f"{record.get('id')!r}"
                )
            partial = (
                record.get("measurement_status") == "partial"
                or record.get("tier_label") == "unresolved"
            )
            if partial and not _confirmed_binary_hard_case(record):
                raise SystemExit(
                    f"refusing partially measured training row "
                    f"{record.get('id')!r} from {path}"
                )
            record_id = str(record.get("id", ""))
            if not record_id:
                raise SystemExit(f"training row in {path} is missing an id")
            if record_id in seen_ids:
                raise SystemExit(f"duplicate training row id {record_id!r}")
            seen_ids.add(record_id)
        records.extend(loaded)
    return records


def binary_label(record: dict) -> str:
    explicit = record.get("binary_target")
    if explicit in LABEL_NAMES:
        return str(explicit)
    return tier_label_to_binary(record["tier_label"])


def _confirmed_binary_hard_case(record: dict) -> bool:
    """Validate selector evidence before accepting a partial mined row."""
    if (
        record.get("dataset_split") != "train"
        or record.get("binary_target") != NEEDS_STRONG_LABEL
        or record.get("hard_case_confirmed") is not True
    ):
        return False
    tier0 = record.get("tier_results", {}).get("tier0")
    if not isinstance(tier0, dict) or tier0.get("passed") is not False:
        return False
    policy_hash = record.get("selection_request_policy_hash")
    if not isinstance(policy_hash, str) or tier0.get("request_policy_hash") != policy_hash:
        return False
    for tier in ("tier1", "tier2", "tier3"):
        result = record.get("tier_results", {}).get(tier)
        tokens = result.get("total_tokens") if isinstance(result, dict) else None
        if (
            isinstance(result, dict)
            and result.get("passed") is True
            and result.get("request_policy_hash") == policy_hash
            and isinstance(tokens, (int, float))
            and not isinstance(tokens, bool)
            and tokens > 0
            and str(result.get("finish_reason") or "").strip().lower() != "error"
        ):
            return True
    return False


def collapse_binary_records(records: list[dict]) -> list[dict]:
    """Return one conservative label per unique normalized prompt.

    Repeated model calls are observations, not independent prompts.  If any
    observed tier0 attempt failed, the group is ``needs_strong``; this avoids
    turning a contradictory duplicate into cheap-model evidence merely because
    another stochastic attempt happened to pass.
    """

    def resolve(group: list[dict]) -> dict:
        tier0_observations: list[bool] = []
        for record in group:
            tier0 = record.get("tier_results", {}).get("tier0")
            if isinstance(tier0, dict) and "passed" in tier0:
                tier0_observations.append(bool(tier0["passed"]))
            else:
                tier0_observations.append(record.get("tier_label") == "tier0")
        cheap_ok = bool(tier0_observations) and all(tier0_observations)
        return {"tier_label": "tier0" if cheap_ok else "tier3"}

    return collapse_prompt_groups(records, resolve)


def stratified_split(
    records: list[dict], val_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Split atomic prompt families within each conservative binary class."""
    if not all("_prompt_group_key" in record for record in records):
        records = collapse_binary_records(records)
    rng = random.Random(seed)
    atomic_units: dict[str, list[dict]] = {}
    for rec in records:
        family = rec.get("prompt_family_id")
        unit_key = (
            f"family:{str(family).strip().casefold()}"
            if isinstance(family, str) and family.strip()
            else f"prompt:{prompt_group_key(rec)}"
        )
        atomic_units.setdefault(unit_key, []).append(rec)

    by_label: dict[str, list[list[dict]]] = {}
    for unit in atomic_units.values():
        # A family containing any observed hard case is stratified as hard, but
        # every member stays on the same side of the boundary.
        unit_label = (
            NEEDS_STRONG_LABEL
            if any(binary_label(record) == NEEDS_STRONG_LABEL for record in unit)
            else CHEAP_OK_LABEL
        )
        by_label.setdefault(unit_label, []).append(unit)

    train: list[dict] = []
    val: list[dict] = []
    for units in by_label.values():
        units = units.copy()
        rng.shuffle(units)
        if len(units) > 1:
            n_val = max(1, int(len(units) * val_frac))
            n_val = min(n_val, len(units) - 1)
        else:
            n_val = 0
        val.extend(record for unit in units[:n_val] for record in unit)
        train.extend(record for unit in units[n_val:] for record in unit)

    rng.shuffle(train)
    rng.shuffle(val)
    assert_disjoint_prompt_groups(train, val)
    train_families = {
        str(record["prompt_family_id"]).strip().casefold()
        for record in train
        if record.get("prompt_family_id")
    }
    validation_families = {
        str(record["prompt_family_id"]).strip().casefold()
        for record in val
        if record.get("prompt_family_id")
    }
    overlap = train_families & validation_families
    if overlap:
        raise AssertionError(
            f"prompt-family leakage across train/validation ({len(overlap)} families)"
        )
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


def classification_metrics(confusion: np.ndarray) -> dict[str, float | int]:
    """Binary metrics with ``needs_strong`` treated as the hard/positive class."""
    if confusion.shape != (2, 2):
        raise ValueError("binary confusion matrix must be 2x2")
    true_cheap_pred_cheap, true_cheap_pred_hard = confusion[0].tolist()
    true_hard_pred_cheap, true_hard_pred_hard = confusion[1].tolist()
    tp = int(true_hard_pred_hard)
    fp = int(true_cheap_pred_hard)
    fn = int(true_hard_pred_cheap)
    tn = int(true_cheap_pred_cheap)
    total = tp + fp + fn + tn
    cheap_recall = tn / max(tn + fp, 1)
    hard_recall = tp / max(tp + fn, 1)
    return {
        "accuracy": (tp + tn) / max(total, 1),
        "hard_precision": tp / max(tp + fp, 1),
        "hard_recall": hard_recall,
        "hard_f1": (2 * tp) / max(2 * tp + fp + fn, 1),
        "cheap_recall": cheap_recall,
        "balanced_accuracy": (cheap_recall + hard_recall) / 2.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def promotion_recommended(
    candidate: dict[str, float | int],
    baseline: dict[str, float | int],
    *,
    hard_group_count: int,
) -> bool:
    """Conservative gate before an explicit active-checkpoint promotion.

    Overall accuracy cannot approve a model on this skewed dataset.  The
    candidate must see at least ten unique hard groups, identify at least one hard
    validation example, materially beat the constant router on balanced
    accuracy, retain near-baseline overall accuracy, and achieve a useful hard
    F1.  Promotion still requires the separate ``--promote`` flag.
    """
    return bool(
        hard_group_count >= MIN_HARD_GROUPS_FOR_PROMOTION
        and float(candidate["hard_recall"]) > 0.0
        and float(candidate["hard_f1"]) >= 0.5
        and float(candidate["balanced_accuracy"])
        >= float(baseline["balanced_accuracy"]) + 0.05
        and float(candidate["accuracy"]) >= float(baseline["accuracy"]) - 0.02
    )


def checkpoint_selection_key(
    candidate: dict[str, float | int],
    baseline: dict[str, float | int],
) -> tuple[float, float, float, float, float]:
    """Prefer useful hard recall while preserving near-baseline accuracy."""
    accuracy = float(candidate["accuracy"])
    return (
        float(accuracy >= float(baseline["accuracy"]) - 0.02),
        float(candidate["balanced_accuracy"]),
        float(candidate["hard_f1"]),
        float(candidate["hard_recall"]),
        accuracy,
    )


def _promote_candidate(candidate_dir: Path) -> None:
    """Promote one staged bundle with rollback on any replacement failure."""
    artifacts = (
        (candidate_dir / "binary_router.pt", ACTIVE_CHECKPOINT_PATH, False),
        (candidate_dir / "binary_router_config.json", ACTIVE_CONFIG_PATH, False),
        (candidate_dir / "tokenizer", CKPT_DIR / "tokenizer", True),
        (candidate_dir / "encoder_config", CKPT_DIR / "encoder_config", True),
    )
    for source, _, _ in artifacts:
        if not source.exists():
            raise FileNotFoundError(f"candidate artifact is missing: {source}")

    def remove_path(path: Path) -> None:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    staged: list[tuple[Path, Path, Path]] = []
    for source, target, is_directory in artifacts:
        promoting = target.with_name(target.name + ".promoting")
        backup = target.with_name(target.name + ".previous")
        remove_path(promoting)
        remove_path(backup)
        if is_directory:
            shutil.copytree(source, promoting)
        else:
            shutil.copy2(source, promoting)
        staged.append((target, promoting, backup))

    activated: list[tuple[Path, Path]] = []
    backed_up: list[tuple[Path, Path]] = []
    try:
        for target, _, backup in staged:
            if target.exists():
                target.replace(backup)
                backed_up.append((target, backup))
        for target, promoting, _ in staged:
            promoting.replace(target)
            activated.append((target, promoting))
    except Exception:
        for target, _ in reversed(activated):
            remove_path(target)
        for target, backup in reversed(backed_up):
            if backup.exists():
                backup.replace(target)
        raise
    finally:
        for _, promoting, backup in staged:
            remove_path(promoting)
            remove_path(backup)


def _validate_candidate_dir(candidate_dir: Path) -> Path:
    """Keep ungated writes away from every active artifact path."""
    resolved = candidate_dir.resolve()
    active_targets = {
        ACTIVE_CHECKPOINT_PATH.resolve(),
        ACTIVE_CONFIG_PATH.resolve(),
        (CKPT_DIR / "tokenizer").resolve(),
        (CKPT_DIR / "encoder_config").resolve(),
    }
    candidate_targets = {
        (resolved / "binary_router.pt").resolve(),
        (resolved / "binary_router_config.json").resolve(),
        (resolved / "tokenizer").resolve(),
        (resolved / "encoder_config").resolve(),
    }
    if resolved == CKPT_DIR.resolve() or active_targets & candidate_targets:
        raise ValueError(
            "candidate directory overlaps active checkpoint artifacts; choose "
            "a separate directory such as router/checkpoints/candidates/binary_router"
        )
    return resolved


def _logical_source_name(path: Path) -> str:
    """Record reproducible source names without leaking local absolute paths."""
    project_root = DATA_PATH.parent.parent.resolve()
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return path.name


def constant_cheap_confusion(records: list[dict]) -> np.ndarray:
    """Confusion matrix for the honest majority baseline: always use tier0."""
    confusion = np.zeros((2, 2), dtype=int)
    for record in records:
        target = 0 if binary_label(record) == CHEAP_OK_LABEL else 1
        confusion[target, 0] += 1
    return confusion


def _load_tokenizer():
    local_path = CKPT_DIR / "tokenizer"
    if local_path.exists():
        logger.info("Reusing tokenizer from %s", local_path)
        return AutoTokenizer.from_pretrained(local_path)
    return AutoTokenizer.from_pretrained(DEFAULT_ENCODER)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the binary tier0/tier3 router.")
    parser.add_argument(
        "--data-path",
        type=Path,
        action="append",
        help="train-only labeled JSONL; repeat to combine sources",
    )
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
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_DIR,
        help="candidate artifact directory (active checkpoint is untouched)",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="promote the candidate only if the hard-class safety gate passes",
    )
    args = parser.parse_args()

    if args.neg_multiple < 1:
        parser.error("--neg-multiple must be at least 1")
    if not 0.0 < args.val_frac < 1.0:
        parser.error("--val-frac must be between 0 and 1")
    try:
        candidate_dir = _validate_candidate_dir(args.candidate_dir)
    except ValueError as exc:
        parser.error(str(exc))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    label_to_index = {name: index for index, name in enumerate(LABEL_NAMES)}
    data_paths = [path.resolve() for path in (args.data_path or [DATA_PATH])]
    source_records = load_records(data_paths)
    records = collapse_binary_records(source_records)
    distribution = Counter(binary_label(rec) for rec in records)
    logger.info(
        "Loaded %d rows / %d unique prompt groups (%d duplicates removed). "
        "Binary distribution: %s",
        len(source_records),
        len(records),
        len(source_records) - len(records),
        dict(distribution),
    )
    train_records, val_records = stratified_split(records, args.val_frac, args.seed)
    train_distribution = Counter(binary_label(rec) for rec in train_records)
    sampled_train_records = oversample_negatives(
        train_records, args.neg_multiple, args.seed
    )
    sampled_distribution = Counter(binary_label(rec) for rec in sampled_train_records)
    baseline_confusion = constant_cheap_confusion(val_records)
    baseline_metrics = classification_metrics(baseline_confusion)
    logger.info(
        "Train: %d original / %d oversampled %s | Val: %d %s",
        len(train_records),
        len(sampled_train_records),
        dict(sampled_distribution),
        len(val_records),
        dict(Counter(binary_label(rec) for rec in val_records)),
    )
    logger.info(
        "Constant tier0 validation baseline: acc %.3f | hard precision %.3f | "
        "hard recall %.3f | confusion=%s",
        baseline_metrics["accuracy"],
        baseline_metrics["hard_precision"],
        baseline_metrics["hard_recall"],
        baseline_confusion.tolist(),
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

    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_checkpoint_path = candidate_dir / "binary_router.pt"
    candidate_config_path = candidate_dir / "binary_router_config.json"
    best_val_acc = -1.0
    best_selection_key = (-1.0, -1.0, -1.0, -1.0, -1.0)
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
        epoch_metrics = classification_metrics(confusion)
        selection_key = checkpoint_selection_key(epoch_metrics, baseline_metrics)
        logger.info(
            "Epoch %d | train loss %.4f | val acc %.3f | balanced %.3f | "
            "hard F1 %.3f",
            epoch,
            epoch_loss / max(len(train_loader), 1),
            val_acc,
            epoch_metrics["balanced_accuracy"],
            epoch_metrics["hard_f1"],
        )
        if selection_key >= best_selection_key:
            best_selection_key = selection_key
            best_val_acc = val_acc
            best_confusion = confusion.copy()
            torch.save(model.state_dict(), candidate_checkpoint_path)

    logger.info("Selected checkpoint val accuracy: %.3f", best_val_acc)
    best_metrics = classification_metrics(best_confusion)
    hard_group_count = len(
        {
            (
                f"family:{str(record['prompt_family_id']).strip().casefold()}"
                if record.get("prompt_family_id")
                else f"prompt:{prompt_group_key(record)}"
            )
            for record in records
            if binary_label(record) == NEEDS_STRONG_LABEL
        }
    )
    recommend_promotion = promotion_recommended(
        best_metrics,
        baseline_metrics,
        hard_group_count=hard_group_count,
    )
    logger.info(
        "Best confusion matrix (rows=true, cols=pred, order %s):\n%s",
        LABEL_NAMES,
        best_confusion,
    )
    logger.info(
        "Best hard-class precision %.3f | recall %.3f | F1 %.3f | "
        "constant baseline acc %.3f",
        best_metrics["hard_precision"],
        best_metrics["hard_recall"],
        best_metrics["hard_f1"],
        baseline_metrics["accuracy"],
    )
    logger.info(
        "Promotion gate: %s (unique hard groups=%d). Active checkpoint remains "
        "unchanged unless --promote is supplied.",
        "PASS" if recommend_promotion else "FAIL",
        hard_group_count,
    )

    tokenizer.save_pretrained(candidate_dir / "tokenizer")
    model.encoder.config.save_pretrained(candidate_dir / "encoder_config")
    with candidate_config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "encoder_name": DEFAULT_ENCODER,
                "label_names": LABEL_NAMES,
                "label_to_index": label_to_index,
                "max_len": MAX_LEN,
                "best_val_acc": best_val_acc,
                "checkpoint_selection_key": list(best_selection_key),
                "checkpoint_selection_objective": (
                    "accuracy_floor, balanced_accuracy, hard_f1, hard_recall, accuracy"
                ),
                "best_validation_metrics": best_metrics,
                "best_confusion": best_confusion.tolist(),
                "constant_tier0_validation_metrics": baseline_metrics,
                "constant_tier0_confusion": baseline_confusion.tolist(),
                "tau_default": TAU_DEFAULT,
                "neg_multiple": args.neg_multiple,
                "categories": CATEGORIES,
                "source_rows": len(source_records),
                "source_data_paths": [_logical_source_name(path) for path in data_paths],
                "source_data_files_sha256": {
                    _logical_source_name(path): dataset_hash(
                        [
                            json.loads(line)
                            for line in path.read_text(encoding="utf-8").splitlines()
                            if line.strip()
                        ]
                    )
                    for path in data_paths
                },
                "unique_prompt_groups": len(records),
                "duplicates_removed": len(source_records) - len(records),
                "source_dataset_sha256": dataset_hash(source_records),
                "training_dataset_sha256": dataset_hash(records),
                "training_prompt_groups_sha256": dataset_hash(
                    sorted(prompt_group_key(record) for record in train_records)
                ),
                "training_prompt_families_sha256": dataset_hash(
                    sorted(
                        {
                            str(record["prompt_family_id"])
                            for record in train_records
                            if record.get("prompt_family_id")
                        }
                    )
                ),
                "validation_prompt_groups_sha256": dataset_hash(
                    sorted(prompt_group_key(record) for record in val_records)
                ),
                "hard_group_count": hard_group_count,
                "promotion_recommended": recommend_promotion,
                "promotion_gate": {
                    "minimum_hard_groups": MIN_HARD_GROUPS_FOR_PROMOTION,
                    "requires_nonzero_hard_recall": True,
                    "minimum_hard_f1": 0.5,
                    "balanced_accuracy_margin_over_constant": 0.05,
                    "maximum_accuracy_drop_from_constant": 0.02,
                },
            },
            f,
            indent=2,
        )
    logger.info("Saved candidate checkpoint and config to %s", candidate_dir)
    if args.promote:
        if not recommend_promotion:
            raise SystemExit(
                "Candidate was NOT promoted: the hard-class promotion gate "
                "failed. Active checkpoint is unchanged."
            )
        _promote_candidate(candidate_dir)
        logger.info("Promoted gated candidate to active checkpoint paths in %s", CKPT_DIR)


if __name__ == "__main__":
    main()
