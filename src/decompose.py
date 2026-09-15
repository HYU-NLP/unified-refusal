"""RQ1 direction geometry and RQ3 type-specific specialization.

RQ1 computes difference-in-means KR/KC and SR/SC shifts, their cosine
alignment, and their common/type-specific decomposition. The original
analysis's coordinate-permutation diagnostic is reported at the observed peak
layer.

RQ3 projects raw per-layer shifts onto the unit KR-specific residual at the
final layer. The same calculation across every reference layer provides the
reference-layer robustness grid.
"""

import argparse
import json
import sys

import numpy as np

from config import ANALYSIS_DIR, HIDDEN_STATES_DIR, MODELS, N_BOOTSTRAP, SEED
from utils import unit_vector


def _orthogonal_residual(vector: np.ndarray, basis_unit: np.ndarray) -> np.ndarray:
    return vector - float(np.dot(vector, basis_unit)) * basis_unit


def compute_layer_geometry(
    kr: np.ndarray,
    kc: np.ndarray,
    sr: np.ndarray,
    sc: np.ndarray,
) -> dict:
    """Compute shift alignment and decomposition for one model layer."""
    d_k_raw = (kr - kc).mean(axis=0)
    d_s_raw = (sr - sc).mean(axis=0)
    d_k = unit_vector(d_k_raw)
    d_s = unit_vector(d_s_raw)
    d_common = unit_vector(d_k + d_s)

    k_specific_raw = _orthogonal_residual(d_k, d_common)
    s_specific_raw = _orthogonal_residual(d_s, d_common)
    d_k_specific = unit_vector(k_specific_raw)
    d_s_specific = unit_vector(s_specific_raw)

    return {
        "d_k_raw": d_k_raw,
        "d_s_raw": d_s_raw,
        "d_k": d_k,
        "d_s": d_s,
        "d_common": d_common,
        "d_k_specific": d_k_specific,
        "d_s_specific": d_s_specific,
        "d_ks_contrast": unit_vector(d_k - d_s),
        "cos_dk_ds": float(np.dot(d_k, d_s)),
        "common_projection_k": float(np.dot(d_k, d_common)),
        "common_projection_s": float(np.dot(d_s, d_common)),
        "specific_residual_norm_k": float(np.linalg.norm(k_specific_raw)),
        "specific_residual_norm_s": float(np.linalg.norm(s_specific_raw)),
        "raw_shift_norm_k": float(np.linalg.norm(d_k_raw)),
        "raw_shift_norm_s": float(np.linalg.norm(d_s_raw)),
    }


def alignment_coordinate_permutation_pvalue(
    direction_k: np.ndarray,
    direction_s: np.ndarray,
    observed_alignment: float,
    n_permutations: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> float:
    """One-sided coordinate-permutation diagnostic from the analysis code."""
    if direction_k.shape != direction_s.shape or direction_k.ndim != 1:
        raise ValueError("direction_k and direction_s must be equal-shape vectors")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")

    rng = np.random.default_rng(seed)
    denominator = np.linalg.norm(direction_k) * np.linalg.norm(direction_s) + 1e-8
    exceedances = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(direction_k)
        alignment = float(np.dot(permuted, direction_s) / denominator)
        exceedances += int(alignment >= observed_alignment)
    return float((1 + exceedances) / (1 + n_permutations))


def _reference_layer_robustness(arrays: dict[str, np.ndarray]) -> dict:
    """Compute RQ3 divergence for every reference-layer/target-layer pair."""
    d_k_raw = arrays["d_k_raw"]
    d_s_raw = arrays["d_s_raw"]
    references = arrays["d_k_specific"]
    projection_k = references @ d_k_raw.T
    projection_s = references @ d_s_raw.T
    return {
        "reference_layers": list(range(len(references))),
        "target_layers": list(range(len(references))),
        "projection_k": projection_k.tolist(),
        "projection_s": projection_s.tolist(),
        "divergence_k_minus_s": (projection_k - projection_s).tolist(),
    }


def compute_analysis(
    model_name: str,
    data: np.lib.npyio.NpzFile,
    n_permutations: int = N_BOOTSTRAP,
) -> dict:
    required = [
        "knowledge_hidden",
        "safety_hidden",
        "k_control_hidden",
        "s_control_hidden",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{model_name} hidden-state cache is missing {missing}")

    kr = data["knowledge_hidden"]
    sr = data["safety_hidden"]
    kc = data["k_control_hidden"]
    sc = data["s_control_hidden"]
    n_pairs, n_layers, hidden_dim = kr.shape

    array_names = [
        "d_k_raw",
        "d_s_raw",
        "d_k",
        "d_s",
        "d_common",
        "d_k_specific",
        "d_s_specific",
        "d_ks_contrast",
    ]
    arrays = {
        name: np.zeros((n_layers, hidden_dim), dtype=np.float32)
        for name in array_names
    }
    layers = []

    for layer in range(n_layers):
        geometry = compute_layer_geometry(
            kr[:, layer], kc[:, layer], sr[:, layer], sc[:, layer]
        )
        for name in array_names:
            arrays[name][layer] = geometry[name]
        layers.append({
            "layer": layer,
            "relative_layer": layer / max(n_layers - 1, 1),
            "cos_dk_ds": geometry["cos_dk_ds"],
            "common_projection_k": geometry["common_projection_k"],
            "common_projection_s": geometry["common_projection_s"],
            "specific_residual_norm_k": geometry["specific_residual_norm_k"],
            "specific_residual_norm_s": geometry["specific_residual_norm_s"],
            "raw_shift_norm_k": geometry["raw_shift_norm_k"],
            "raw_shift_norm_s": geometry["raw_shift_norm_s"],
        })

    # RQ3: use raw DIM shifts and a unit final-layer reference.
    final_reference = arrays["d_k_specific"][-1]
    for layer, row in enumerate(layers):
        row["projection_on_final_k_specific_k"] = float(
            np.dot(arrays["d_k_raw"][layer], final_reference)
        )
        row["projection_on_final_k_specific_s"] = float(
            np.dot(arrays["d_s_raw"][layer], final_reference)
        )

    peak_layer = int(np.argmax([row["cos_dk_ds"] for row in layers]))
    peak_alignment = layers[peak_layer]["cos_dk_ds"]
    peak_pvalue = alignment_coordinate_permutation_pvalue(
        arrays["d_k_raw"][peak_layer],
        arrays["d_s_raw"][peak_layer],
        peak_alignment,
        n_permutations=n_permutations,
    )

    return {
        "model": model_name,
        "analysis": "refusal_direction_geometry_and_specialization",
        "n_pairs": int(n_pairs),
        "n_layers": int(n_layers),
        "hidden_dim": int(hidden_dim),
        "peak_alignment_layer": peak_layer,
        "peak_alignment": peak_alignment,
        "peak_alignment_permutation_pvalue": peak_pvalue,
        "n_permutations": n_permutations,
        "layers": layers,
        "reference_layer_robustness": _reference_layer_robustness(arrays),
        "arrays": arrays,
    }


def save_result(result: dict) -> None:
    model_name = result["model"]
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key != "arrays"}
    summary_path = ANALYSIS_DIR / f"{model_name}_direction_decomposition.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    arrays_path = ANALYSIS_DIR / f"{model_name}_directions.npz"
    np.savez_compressed(arrays_path, **result["arrays"], model=np.array(model_name))
    print(f"  Saved {summary_path}")
    print(f"  Saved {arrays_path}")


def run_model(model_name: str, n_permutations: int) -> None:
    npz_path = HIDDEN_STATES_DIR / f"{model_name}.npz"
    if not npz_path.exists():
        print(f"  WARNING: {npz_path} not found; run extraction first")
        return
    with np.load(npz_path, allow_pickle=False) as data:
        result = compute_analysis(model_name, data, n_permutations=n_permutations)
    save_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1/RQ3 refusal-direction analysis")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all-models", action="store_true")
    parser.add_argument("--n-permutations", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--fast", action="store_true", help="Use 50 permutations")
    args = parser.parse_args()

    n_permutations = 50 if args.fast else args.n_permutations
    model_names = list(MODELS) if args.all_models else [
        key for key, value in MODELS.items()
        if value["path"] == args.model or key == args.model
    ]
    if not model_names:
        print(f"ERROR: '{args.model}' not in config.MODELS", file=sys.stderr)
        sys.exit(1)

    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        run_model(model_name, n_permutations=n_permutations)


if __name__ == "__main__":
    main()
