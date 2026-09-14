"""
Analysis: Cross-Condition Probe Transfer
Train probe on K-refusal vs K-control, evaluate on S-refusal vs S-control (and vice versa).
Per-layer accuracy. High cross-transfer accuracy = same separating direction for K and S.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import MODELS, HIDDEN_STATES_DIR, ANALYSIS_DIR, SEED
from utils import kfold_probe_group


def _fit_probe(X_train, y_train, seed=SEED):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    clf.fit(X_tr, y_train)
    return clf, scaler


def _eval_probe(clf, scaler, X_test, y_test) -> float:
    return float(clf.score(scaler.transform(X_test), y_test))


def compute_cross_probe(data_path: Path) -> dict:
    """
    Per layer:
      - Train on K vs KC, evaluate on S vs SC  (K→S transfer)
      - Train on S vs SC, evaluate on K vs KC  (S→K transfer)
      - In-condition CV accuracy as baseline (5-fold)
    Returns dict with per-layer results.
    """
    data = np.load(data_path, allow_pickle=True)
    k  = data["knowledge_hidden"]    # [N, L, d]
    s  = data["safety_hidden"]       # [N, L, d]
    kc = data["k_control_hidden"]    # [N, L, d]
    sc = data["s_control_hidden"]    # [N, L, d]
    n_layers = k.shape[1]

    results = []
    for l in range(n_layers):
        k_l  = k[:, l, :]
        s_l  = s[:, l, :]
        kc_l = kc[:, l, :]
        sc_l = sc[:, l, :]

        N = len(k_l)
        X_k  = np.concatenate([k_l,  kc_l])
        y_k  = np.array([1] * N + [0] * N)
        X_s  = np.concatenate([s_l,  sc_l])
        y_s  = np.array([1] * N + [0] * N)

        # K→S transfer
        clf_k, scaler_k = _fit_probe(X_k, y_k)
        k2s = _eval_probe(clf_k, scaler_k, X_s, y_s)

        # S→K transfer
        clf_s, scaler_s = _fit_probe(X_s, y_s)
        s2k = _eval_probe(clf_s, scaler_s, X_k, y_k)

        # In-condition 5-fold CV (pair-level GroupKFold)
        groups = np.concatenate([np.arange(N), np.arange(N)])
        k_cv_result = kfold_probe_group(X_k, y_k, groups, n_splits=5, seed=SEED)
        s_cv_result = kfold_probe_group(X_s, y_s, groups, n_splits=5, seed=SEED)

        results.append({
            "layer":      l,
            "k2s":        k2s,
            "s2k":        s2k,
            "k_in_cv":    k_cv_result["mean_acc"],
            "s_in_cv":    s_cv_result["mean_acc"],
        })
        print(f"  layer {l:2d}: k→s={k2s:.3f}  s→k={s2k:.3f}  k_cv={k_cv_result['mean_acc']:.3f}  s_cv={s_cv_result['mean_acc']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    args = parser.parse_args()

    data_path = HIDDEN_STATES_DIR / f"{args.model}.npz"
    print(f"Computing cross-probe transfer for {args.model} ...")
    results = compute_cross_probe(data_path)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"{args.model}_cross_probe.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
