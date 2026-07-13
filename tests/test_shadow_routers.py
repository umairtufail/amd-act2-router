"""Focused offline tests for lightweight analysis-only shadow routers."""

import numpy as np
import pytest

from eval.shadow_routers import (
    KNNShadowRouter,
    LogisticShadowRouter,
    ShadowSplit,
    binary_metrics,
    hash_prompt_features,
    run_shadow_baselines,
    validate_group_safe_split,
)


def _training_split() -> ShadowSplit:
    return ShadowSplit(
        prompts=[
            "Name the capital city",
            "Name the chemical element",
            "Classify this positive review",
            "Classify this neutral review",
            "Extract ambiguous nested entities from this complex sentence",
            "Extract ambiguous overlapping entities from this complex sentence",
        ],
        categories=[
            "factual_knowledge",
            "factual_knowledge",
            "sentiment",
            "sentiment",
            "ner",
            "ner",
        ],
        labels=[0, 0, 0, 0, 1, 1],
        groups=["train-e1", "train-e2", "train-e3", "train-e4", "train-h1", "train-h2"],
    )


def _validation_split() -> ShadowSplit:
    return ShadowSplit(
        prompts=[
            "Name the capital",
            "Classify this positive message",
            "Extract ambiguous entities from this complex passage",
        ],
        categories=["factual_knowledge", "sentiment", "ner"],
        labels=[0, 0, 1],
        groups=["valid-e1", "valid-e2", "valid-h1"],
    )


def test_feature_hashing_is_deterministic_and_category_aware():
    prompts = ["Extract Ada from Acme", "Extract Ada from Acme"]
    categories = ["ner", "factual_knowledge"]
    first = hash_prompt_features(prompts, categories, n_features=128)
    second = hash_prompt_features(prompts, categories, n_features=128)
    assert np.array_equal(first, second)
    assert not np.array_equal(first[0], first[1])
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)

    alias = hash_prompt_features(
        ["same"], ["named_entity_recognition"], n_features=128
    )
    canonical = hash_prompt_features(["same"], ["ner"], n_features=128)
    assert np.array_equal(alias, canonical)


def test_splits_require_explicit_groups_and_reject_leakage():
    with pytest.raises(ValueError, match="equal length"):
        ShadowSplit(["one"], ["ner"], [1], [])

    train = _training_split()
    validation = ShadowSplit(
        prompts=["new prompt"],
        categories=["ner"],
        labels=[1],
        groups=[train.groups[-1]],
    )
    with pytest.raises(ValueError, match="group leakage"):
        validate_group_safe_split(train, validation)
    with pytest.raises(ValueError, match="group leakage"):
        run_shadow_baselines(train, validation, n_features=128, logistic_epochs=10)

    exact_prompt = ShadowSplit(
        prompts=[train.prompts[0]],
        categories=["factual_knowledge"],
        labels=[0],
        groups=["a-different-group-id"],
    )
    with pytest.raises(ValueError, match="exact-prompt leakage"):
        validate_group_safe_split(train, exact_prompt)


def test_binary_metrics_report_hard_class_and_confusion():
    metrics = binary_metrics([0, 0, 1, 1], [0, 1, 0, 1])
    assert metrics.accuracy == 0.5
    assert metrics.hard_precision == 0.5
    assert metrics.hard_recall == 0.5
    assert metrics.hard_f1 == 0.5
    assert metrics.confusion == ((1, 1), (1, 1))
    assert metrics.as_dict()["confusion"]["true_hard_pred_hard"] == 1

    never_hard = binary_metrics([0, 1], [0, 0])
    assert never_hard.hard_precision == 0.0
    assert never_hard.hard_recall == 0.0
    assert never_hard.hard_f1 == 0.0
    with pytest.raises(ValueError, match="only 0 or 1"):
        binary_metrics([0, 0.5], [0, 1])


def test_class_balanced_logistic_router_learns_minority_pattern():
    train = _training_split()
    validation = _validation_split()
    router = LogisticShadowRouter(
        n_features=256, epochs=700, learning_rate=0.4, l2=1e-4
    ).fit(train.prompts, train.categories, train.labels)
    probabilities = router.predict_proba(validation.prompts, validation.categories)
    predictions = (probabilities >= 0.5).astype(int)
    assert router.class_weights == (0.75, 1.5)
    assert predictions.tolist() == validation.labels


def test_knn_router_is_deterministic():
    train = _training_split()
    validation = _validation_split()
    router = KNNShadowRouter(n_features=256, k=1).fit(
        train.prompts, train.categories, train.labels
    )
    first = router.predict_proba(validation.prompts, validation.categories)
    second = router.predict_proba(validation.prompts, validation.categories)
    assert np.array_equal(first, second)
    assert (first >= 0.5).astype(int).tolist() == validation.labels


def test_comparison_runner_returns_serializable_metrics_for_both_models():
    results = run_shadow_baselines(
        _training_split(),
        _validation_split(),
        n_features=256,
        knn_k=1,
        logistic_epochs=700,
        logistic_learning_rate=0.4,
        logistic_l2=1e-4,
    )
    assert set(results) == {"logistic", "knn"}
    for name, run in results.items():
        assert run.name == name
        assert len(run.predictions) == 3
        assert 0.0 <= run.metrics.accuracy <= 1.0
        assert "hard_f1" in run.as_dict()["metrics"]
