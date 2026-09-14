"""RQ1 template-control similarity analysis.

This script compares KR/SR hidden-state similarity against the matched-control
KC/SC similarity at each layer. It corresponds to the paper's check that the
observed KR-SR relation is not simply a shared-template artifact.
"""

import argparse
import json
import sys

import numpy as np

from config import MODELS, ANALYSIS_DIR, HIDDEN_STATES_DIR, N_BOOTSTRAP, SEED
from utils import bootstrap_ci, pair_cosine_similarities


def _require_controls(data: np.lib.npyio.NpzFile, model_name: str) -> None:
    required = ["k_control_hidden", "s_control_hidden"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"{model_name} hidden-state cache is missing {missing}. "
            "Run extraction on four-way KR/KC/SR/SC rows."
        )


def run_relative_similarity(
    data: np.lib.npyio.NpzFile,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    kr = data["knowledge_hidden"]
    sr = data["safety_hidden"]
    kc = data["k_control_hidden"]
    sc = data["s_control_hidden"]
    n_pairs, n_layers, _ = kr.shape

    layers = []
    for layer in range(n_layers):
        kr_l, sr_l = kr[:, layer, :], sr[:, layer, :]
        kc_l, sc_l = kc[:, layer, :], sc[:, layer, :]

        refusal_pairwise = pair_cosine_similarities(kr_l, sr_l)
        control_pairwise = pair_cosine_similarities(kc_l, sc_l)
        gap = refusal_pairwise - control_pairwise
        gap_lo, gap_hi = bootstrap_ci(gap, n_resamples=n_bootstrap, seed=SEED)

        layers.append({
            "layer": layer,
            "relative_layer": layer / max(n_layers - 1, 1),
            "pairwise_cosine_kr_sr": float(refusal_pairwise.mean()),
            "pairwise_cosine_kr_sr_std": float(refusal_pairwise.std()),
            "pairwise_cosine_kc_sc": float(control_pairwise.mean()),
            "pairwise_cosine_kc_sc_std": float(control_pairwise.std()),
            "pairwise_cosine_gap_kr_sr_minus_kc_sc": float(gap.mean()),
            "pairwise_cosine_gap_std": float(gap.std()),
            "pairwise_cosine_gap_ci95": [gap_lo, gap_hi],
        })

    return {
        "analysis": "template_control_relative_similarity",
        "n_pairs": int(n_pairs),
        "n_layers": int(n_layers),
        "layers": layers,
    }


def run_model(model_name: str, n_bootstrap: int = N_BOOTSTRAP) -> None:
    npz_path = HIDDEN_STATES_DIR / f"{model_name}.npz"
    if not npz_path.exists():
        print(f"  WARNING: {npz_path} not found; run extract_hidden_states.py first")
        return

    data = np.load(npz_path, allow_pickle=True)
    _require_controls(data, model_name)
    result = run_relative_similarity(data, n_bootstrap=n_bootstrap)
    result["model"] = model_name

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"{model_name}_relative_similarity.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="RQ1: template-control relative similarity")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all-models", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Use fewer bootstrap resamples")
    args = parser.parse_args()

    n_bootstrap = 50 if args.fast else N_BOOTSTRAP
    model_names = list(MODELS) if args.all_models else [
        k for k, v in MODELS.items() if v["path"] == args.model or k == args.model
    ]
    if not model_names:
        print(f"ERROR: model '{args.model}' not in config.MODELS", file=sys.stderr)
        sys.exit(1)

    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        run_model(model_name, n_bootstrap=n_bootstrap)


if __name__ == "__main__":
    main()
