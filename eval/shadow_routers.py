"""Offline-only lightweight router baselines for honest experiments.

These models are intentionally small analysis tools, not submission runtime
components.  They never read credentials, call APIs, load transformer models,
or choose a Fireworks model.  Callers must supply explicit train and validation
splits with group identifiers; the public comparison helper refuses any group
overlap so duplicate prompts cannot leak across the split.

Only NumPy and the Python standard library are used.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


_CATEGORY_ALIASES = {
    "mathematical_reasoning": "math_reasoning",
    "sentiment_classification": "sentiment",
    "text_summarization": "summarization",
    "named_entity_recognition": "ner",
}
_TOKEN = re.compile(r"[^\W_]+(?:[-'\u2019][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True)
class ShadowSplit:
    """A caller-defined split with mandatory duplicate/group identities."""

    prompts: Sequence[str]
    categories: Sequence[str | None]
    labels: Sequence[int]
    groups: Sequence[str]

    def __post_init__(self) -> None:
        lengths = {
            len(self.prompts),
            len(self.categories),
            len(self.labels),
            len(self.groups),
        }
        if len(lengths) != 1:
            raise ValueError("prompts, categories, labels, and groups must have equal length")
        if len(self.prompts) == 0:
            raise ValueError("a shadow split must contain at least one record")
        if any(label not in (0, 1, False, True) for label in self.labels):
            raise ValueError("binary labels must be 0 (easy) or 1 (hard)")
        if any(not isinstance(group, str) or not group.strip() for group in self.groups):
            raise ValueError("every record requires a non-empty group identifier")


@dataclass(frozen=True)
class BinaryMetrics:
    """Binary metrics where label 1 is the hard/needs-strong class."""

    accuracy: float
    hard_precision: float
    hard_recall: float
    hard_f1: float
    true_easy_pred_easy: int
    true_easy_pred_hard: int
    true_hard_pred_easy: int
    true_hard_pred_hard: int

    @property
    def confusion(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return ``((true_easy), (true_hard))`` rows by predicted class."""
        return (
            (self.true_easy_pred_easy, self.true_easy_pred_hard),
            (self.true_hard_pred_easy, self.true_hard_pred_hard),
        )

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "hard_precision": self.hard_precision,
            "hard_recall": self.hard_recall,
            "hard_f1": self.hard_f1,
            "confusion": {
                "true_easy_pred_easy": self.true_easy_pred_easy,
                "true_easy_pred_hard": self.true_easy_pred_hard,
                "true_hard_pred_easy": self.true_hard_pred_easy,
                "true_hard_pred_hard": self.true_hard_pred_hard,
            },
        }


@dataclass(frozen=True)
class ShadowRun:
    """Predictions and metrics from one analysis-only router."""

    name: str
    predictions: tuple[int, ...]
    hard_probabilities: tuple[float, ...]
    metrics: BinaryMetrics

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "predictions": list(self.predictions),
            "hard_probabilities": list(self.hard_probabilities),
            "metrics": self.metrics.as_dict(),
        }


def _normalized_category(category: str | None) -> str:
    value = (category or "unknown").strip().lower() or "unknown"
    return _CATEGORY_ALIASES.get(value, value)


def _feature_strings(prompt: str, category: str | None) -> list[str]:
    """Build non-semantic token/category features with no learned vocabulary."""
    tokens = [token.lower() for token in _TOKEN.findall(prompt)]
    features = [f"category={_normalized_category(category)}"]
    features.extend(f"token={token}" for token in tokens)
    features.extend(
        f"bigram={left}\x1f{right}" for left, right in zip(tokens, tokens[1:])
    )
    features.append(f"length_bucket={min(len(tokens) // 8, 31)}")
    if any(character.isdigit() for character in prompt):
        features.append("shape=has_digit")
    if "```" in prompt:
        features.append("shape=has_code_fence")
    if "json" in prompt.lower():
        features.append("shape=mentions_json")
    return features


def _hash_feature(feature: str, n_features: int) -> tuple[int, float]:
    # A cryptographic digest, rather than Python's salted hash(), guarantees
    # identical feature indices and signs in every process and platform.
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    index = value % n_features
    sign = 1.0 if value & (1 << 63) else -1.0
    return index, sign


def hash_prompt_features(
    prompts: Sequence[str],
    categories: Sequence[str | None],
    *,
    n_features: int = 1024,
) -> np.ndarray:
    """Return deterministic, signed, row-normalized hashed feature vectors."""
    if len(prompts) != len(categories):
        raise ValueError("prompts and categories must have equal length")
    if n_features < 8:
        raise ValueError("n_features must be at least 8")

    matrix = np.zeros((len(prompts), n_features), dtype=np.float64)
    for row, (prompt, category) in enumerate(zip(prompts, categories)):
        if not isinstance(prompt, str):
            raise TypeError("every prompt must be a string")
        for feature in _feature_strings(prompt, category):
            column, sign = _hash_feature(feature, n_features)
            matrix[row, column] += sign
    if len(prompts):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1.0)
    return matrix


def validate_group_safe_split(train: ShadowSplit, validation: ShadowSplit) -> None:
    """Raise if any prompt/entity group appears on both sides of the split."""
    overlap = set(train.groups).intersection(validation.groups)
    if overlap:
        preview = sorted(overlap)[:5]
        raise ValueError(
            f"train/validation group leakage detected ({len(overlap)} group(s)): {preview}"
        )
    # Exact prompt overlap is leakage even if a caller accidentally assigned
    # different group IDs.  Group IDs remain necessary to catch duplicates or
    # paraphrase families that are not byte-identical.
    train_prompts = {prompt.strip() for prompt in train.prompts}
    prompt_overlap = train_prompts.intersection(prompt.strip() for prompt in validation.prompts)
    if prompt_overlap:
        raise ValueError(
            "train/validation exact-prompt leakage detected "
            f"({len(prompt_overlap)} prompt(s))"
        )


def binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> BinaryMetrics:
    """Compute accuracy and hard-class metrics without sklearn."""
    truth = np.asarray(labels)
    predicted = np.asarray(predictions)
    if truth.ndim != 1 or predicted.ndim != 1 or len(truth) != len(predicted):
        raise ValueError("labels and predictions must be equal-length one-dimensional arrays")
    if not len(truth):
        raise ValueError("metrics require at least one prediction")
    if not np.isin(truth, [0, 1]).all() or not np.isin(predicted, [0, 1]).all():
        raise ValueError("labels and predictions must contain only 0 or 1")
    truth = truth.astype(np.int8)
    predicted = predicted.astype(np.int8)

    tn = int(np.sum((truth == 0) & (predicted == 0)))
    fp = int(np.sum((truth == 0) & (predicted == 1)))
    fn = int(np.sum((truth == 1) & (predicted == 0)))
    tp = int(np.sum((truth == 1) & (predicted == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BinaryMetrics(
        accuracy=(tp + tn) / len(truth),
        hard_precision=precision,
        hard_recall=recall,
        hard_f1=f1,
        true_easy_pred_easy=tn,
        true_easy_pred_hard=fp,
        true_hard_pred_easy=fn,
        true_hard_pred_hard=tp,
    )


class _ShadowRouter(Protocol):
    def predict_proba(
        self, prompts: Sequence[str], categories: Sequence[str | None]
    ) -> np.ndarray: ...


@dataclass
class LogisticShadowRouter:
    """Class-balanced full-batch logistic regression over hashed features."""

    n_features: int = 1024
    learning_rate: float = 0.3
    epochs: int = 500
    l2: float = 1e-3
    weights: np.ndarray | None = None
    intercept: float = 0.0
    class_weights: tuple[float, float] | None = None

    def fit(
        self, prompts: Sequence[str], categories: Sequence[str | None], labels: Sequence[int]
    ) -> "LogisticShadowRouter":
        features = hash_prompt_features(prompts, categories, n_features=self.n_features)
        target = np.asarray(labels, dtype=np.float64)
        if len(features) != len(target) or target.ndim != 1:
            raise ValueError("features and labels must have matching lengths")
        if not np.isin(target, [0.0, 1.0]).all():
            raise ValueError("binary labels must contain only 0 or 1")
        counts = np.bincount(target.astype(np.int8), minlength=2)
        if np.any(counts == 0):
            raise ValueError("class-balanced logistic regression requires both classes")

        total = float(len(target))
        easy_weight = total / (2.0 * counts[0])
        hard_weight = total / (2.0 * counts[1])
        self.class_weights = (easy_weight, hard_weight)
        sample_weight = np.where(target == 1.0, hard_weight, easy_weight)
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.intercept = 0.0

        for _ in range(self.epochs):
            logits = np.clip(features @ self.weights + self.intercept, -35.0, 35.0)
            probability = 1.0 / (1.0 + np.exp(-logits))
            residual = (probability - target) * sample_weight
            gradient = (features.T @ residual) / total + self.l2 * self.weights
            intercept_gradient = float(np.sum(residual) / total)
            self.weights -= self.learning_rate * gradient
            self.intercept -= self.learning_rate * intercept_gradient
        return self

    def predict_proba(
        self, prompts: Sequence[str], categories: Sequence[str | None]
    ) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("fit the logistic shadow router before prediction")
        features = hash_prompt_features(prompts, categories, n_features=self.n_features)
        logits = np.clip(features @ self.weights + self.intercept, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass
class KNNShadowRouter:
    """Deterministic cosine k-NN baseline over the same hashed features."""

    n_features: int = 1024
    k: int = 3
    _features: np.ndarray | None = None
    _labels: np.ndarray | None = None

    def fit(
        self, prompts: Sequence[str], categories: Sequence[str | None], labels: Sequence[int]
    ) -> "KNNShadowRouter":
        if self.k < 1:
            raise ValueError("k must be at least 1")
        features = hash_prompt_features(prompts, categories, n_features=self.n_features)
        target = np.asarray(labels)
        if len(features) != len(target) or target.ndim != 1:
            raise ValueError("features and labels must have matching lengths")
        if not np.isin(target, [0, 1]).all():
            raise ValueError("binary labels must contain only 0 or 1")
        target = target.astype(np.int8)
        self._features = features
        self._labels = target
        return self

    def predict_proba(
        self, prompts: Sequence[str], categories: Sequence[str | None]
    ) -> np.ndarray:
        if self._features is None or self._labels is None:
            raise RuntimeError("fit the k-NN shadow router before prediction")
        queries = hash_prompt_features(prompts, categories, n_features=self.n_features)
        k = min(self.k, len(self._labels))
        probabilities = np.zeros(len(queries), dtype=np.float64)
        for row, query in enumerate(queries):
            similarities = self._features @ query
            # Stable sorting makes ties resolve by original training order.
            nearest = np.argsort(-similarities, kind="stable")[:k]
            # Shift cosine similarity into a non-negative weighting range.  A
            # small floor prevents zero-weight degeneracy on collision-heavy
            # or vocabulary-disjoint examples.
            weights = np.maximum(similarities[nearest] + 1.0, 1e-9)
            probabilities[row] = float(
                np.sum(weights * self._labels[nearest]) / np.sum(weights)
            )
        return probabilities


def evaluate_shadow_router(
    name: str,
    router: _ShadowRouter,
    validation: ShadowSplit,
    *,
    threshold: float = 0.5,
) -> ShadowRun:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    probabilities = router.predict_proba(validation.prompts, validation.categories)
    predictions = (probabilities >= threshold).astype(np.int8)
    return ShadowRun(
        name=name,
        predictions=tuple(int(value) for value in predictions),
        hard_probabilities=tuple(float(value) for value in probabilities),
        metrics=binary_metrics(validation.labels, predictions),
    )


def run_shadow_baselines(
    train: ShadowSplit,
    validation: ShadowSplit,
    *,
    n_features: int = 1024,
    threshold: float = 0.5,
    knn_k: int = 3,
    logistic_epochs: int = 500,
    logistic_learning_rate: float = 0.3,
    logistic_l2: float = 1e-3,
) -> dict[str, ShadowRun]:
    """Fit and evaluate both baselines after enforcing a group-safe split."""
    validate_group_safe_split(train, validation)
    logistic = LogisticShadowRouter(
        n_features=n_features,
        epochs=logistic_epochs,
        learning_rate=logistic_learning_rate,
        l2=logistic_l2,
    ).fit(train.prompts, train.categories, train.labels)
    knn = KNNShadowRouter(n_features=n_features, k=knn_k).fit(
        train.prompts, train.categories, train.labels
    )
    return {
        "logistic": evaluate_shadow_router(
            "logistic", logistic, validation, threshold=threshold
        ),
        "knn": evaluate_shadow_router("knn", knn, validation, threshold=threshold),
    }


__all__ = [
    "BinaryMetrics",
    "KNNShadowRouter",
    "LogisticShadowRouter",
    "ShadowRun",
    "ShadowSplit",
    "binary_metrics",
    "evaluate_shadow_router",
    "hash_prompt_features",
    "run_shadow_baselines",
    "validate_group_safe_split",
]
