"""RQ2: asymmetric cross-type probe transfer.

At each layer, an L2-regularized logistic probe trained on KR vs. KC is
evaluated on SR vs. SC without retraining, and vice versa. Confidence intervals
for the directional transfer gap use paired resampling at the quadruple level.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import ANALYSIS_DIR, HIDDEN_STATES_DIR, MODELS, N_BOOTSTRAP, SEED
from utils import bootstrap_ci, kfold_probe_group, paired_bootstrap_gap, unit_vector


def _fit_probe(X_train: np.ndarray, y_train: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    classifier = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
    classifier.fit(X_scaled, y_train)
    return classifier, scaler


def _correctness_by_pair(
    classifier,
    scaler: StandardScaler,
    refusal: np.ndarray,
    control: np.ndarray,
) -> np.ndarray:
    """Return mean correctness across the two test items in each pair."""
    n_pairs = len(refusal)
    X_test = np.concatenate([refusal, control])
    y_test = np.array([1] * n_pairs + [0] * n_pairs)
    predictions = classifier.predict(scaler.transform(X_test))
    correct = (predictions == y_test).astype(np.float64)
    return (correct[:n_pairs] + correct[n_pairs:]) / 2.0


def compute_cross_probe(
    data_path: Path,
    n_bootstrap: int = N_BOOTSTRAP,
) -> list[dict]:
    """Compute within-type and cross-type RQ2 metrics at every layer."""
    with np.load(data_path, allow_pickle=False) as data:
        kr = data["knowledge_hidden"]
        sr = data["safety_hidden"]
        kc = data["k_control_hidden"]
        sc = data["s_control_hidden"]
    n_pairs, n_layers, _ = kr.shape

    results = []
    groups = np.concatenate([np.arange(n_pairs), np.arange(n_pairs)])
    labels = np.array([1] * n_pairs + [0] * n_pairs)

    for layer in range(n_layers):
        X_k = np.concatenate([kr[:, layer], kc[:, layer]])
        X_s = np.concatenate([sr[:, layer], sc[:, layer]])
        d_k = unit_vector((kr[:, layer] - kc[:, layer]).mean(axis=0))
        d_s = unit_vector((sr[:, layer] - sc[:, layer]).mean(axis=0))
        shift_alignment = float(np.dot(d_k, d_s))

        k_classifier, k_scaler = _fit_probe(X_k, labels)
        k2s_by_pair = _correctness_by_pair(
            k_classifier, k_scaler, sr[:, layer], sc[:, layer]
        )

        s_classifier, s_scaler = _fit_probe(X_s, labels)
        s2k_by_pair = _correctness_by_pair(
            s_classifier, s_scaler, kr[:, layer], kc[:, layer]
        )

        k2s = float(k2s_by_pair.mean())
        s2k = float(s2k_by_pair.mean())
        k2s_ci = bootstrap_ci(k2s_by_pair, n_resamples=n_bootstrap, seed=SEED)
        s2k_ci = bootstrap_ci(s2k_by_pair, n_resamples=n_bootstrap, seed=SEED)
        gap, gap_lo, gap_hi = paired_bootstrap_gap(
            s2k_by_pair,
            k2s_by_pair,
            n_resamples=n_bootstrap,
            seed=SEED,
        )

        k_cv = kfold_probe_group(X_k, labels, groups, n_splits=5, seed=SEED)
        s_cv = kfold_probe_group(X_s, labels, groups, n_splits=5, seed=SEED)

        results.append({
            "layer": layer,
            "relative_layer": layer / max(n_layers - 1, 1),
            "shift_alignment": shift_alignment,
            "k2s": k2s,
            "k2s_ci95": list(k2s_ci),
            "s2k": s2k,
            "s2k_ci95": list(s2k_ci),
            "gap_s2k_minus_k2s": gap,
            "gap_ci95": [gap_lo, gap_hi],
            "k_in_cv": k_cv["mean_acc"],
            "k_in_cv_std": k_cv["std_acc"],
            "s_in_cv": s_cv["mean_acc"],
            "s_in_cv_std": s_cv["std_acc"],
        })
        print(
            f"  layer {layer:2d}: KR->SR={k2s:.3f}  SR->KR={s2k:.3f}  "
            f"gap={gap:+.3f} [{gap_lo:+.3f}, {gap_hi:+.3f}]"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ2: asymmetric KR/SR cross-type probe transfer"
    )
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--fast", action="store_true", help="Use 50 bootstrap resamples")
    args = parser.parse_args()

    n_bootstrap = 50 if args.fast else args.n_bootstrap
    data_path = HIDDEN_STATES_DIR / f"{args.model}.npz"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found; run extraction first")

    print(f"Computing RQ2 cross-probe transfer for {args.model} ...")
    results = compute_cross_probe(data_path, n_bootstrap=n_bootstrap)
    with np.load(data_path, allow_pickle=False) as data:
        n_pairs = int(data["knowledge_hidden"].shape[0])
    peak_layer = int(np.argmax([row["shift_alignment"] for row in results]))
    output = {
        "model": args.model,
        "analysis": "asymmetric_cross_type_transfer",
        "n_pairs": n_pairs,
        "n_bootstrap": n_bootstrap,
        "peak_alignment_layer": peak_layer,
        "peak_layer_metrics": results[peak_layer],
        "layers": results,
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"{args.model}_cross_probe.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
