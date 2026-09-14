# utils.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


# ── Cosine similarity ──────────────────────────────────────────────────────────

def pair_cosine_similarities(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Per-pair cosine similarity between rows of X and Y.
    X, Y: [N, d] float32.  Returns [N] float32.
    """
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
    return np.sum(X_norm * Y_norm, axis=1)


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """Bootstrap CI for the mean. Returns (lo, hi)."""
    rng = np.random.RandomState(seed)
    n = len(values)
    means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_resamples)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


# ── Linear probe ───────────────────────────────────────────────────────────────


def kfold_probe_group(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """
    GroupKFold logistic regression probe.
    Keeps both samples of the same pair in the same fold.
    X: [2N, d], y: [2N] binary, groups: [2N] pair index repeated twice.
    """
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    accs, aucs = [], []
    for tr_idx, te_idx in gkf.split(X, y, groups=groups):
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
        clf.fit(X[tr_idx], y[tr_idx])
        accs.append(float(accuracy_score(y[te_idx], clf.predict(X[te_idx]))))
        aucs.append(float(roc_auc_score(y[te_idx], clf.predict_proba(X[te_idx])[:, 1])))
    return {
        "mean_acc": float(np.mean(accs)),
        "std_acc":  float(np.std(accs)),
        "mean_auc": float(np.mean(aucs)),
        "std_auc":  float(np.std(aucs)),
        "null_acc": 0.5,
    }
