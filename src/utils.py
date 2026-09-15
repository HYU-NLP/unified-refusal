"""Shared numerical utilities for the representation analyses."""

import numpy as np


def unit_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return an L2-normalized vector, with a stable zero-vector fallback."""
    return vector / (np.linalg.norm(vector) + eps)


def pair_cosine_similarities(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return the cosine similarity for every matched pair of rows."""
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have the same shape, got {X.shape} and {Y.shape}")
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    return np.sum(X_norm * Y_norm, axis=1)


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a percentile bootstrap confidence interval for a sample mean."""
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def paired_bootstrap_gap(
    first: np.ndarray,
    second: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Estimate ``mean(first) - mean(second)`` with a paired bootstrap CI."""
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape or first.ndim != 1 or len(first) == 0:
        raise ValueError("first and second must be non-empty 1D arrays of equal shape")
    gap = first - second
    lo, hi = bootstrap_ci(gap, n_resamples=n_resamples, ci=ci, seed=seed)
    return float(gap.mean()), lo, hi


def kfold_probe_group(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Evaluate an L2-logistic probe with pair-preserving GroupKFold splits."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"need at least {n_splits} matched pairs for cross-validation; "
            f"received {len(unique_groups)}"
        )

    accs, aucs = [], []
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        classifier = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        classifier.fit(X[train_idx], y[train_idx])
        probabilities = classifier.predict_proba(X[test_idx])[:, 1]
        predictions = classifier.predict(X[test_idx])
        accs.append(float(accuracy_score(y[test_idx], predictions)))
        aucs.append(float(roc_auc_score(y[test_idx], probabilities)))

    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "null_acc": 0.5,
    }
