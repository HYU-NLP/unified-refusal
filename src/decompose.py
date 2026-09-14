"""RQ1/RQ3 control-to-refusal direction decomposition.

For each layer, compute d_K = mean(KR - KC) and d_S = mean(SR - SC), then
measure their alignment and decompose them into common and type-specific
components. Direction arrays are saved for downstream analyses such as logit lens.
"""

import argparse
import json
import sys

import numpy as np

from config import MODELS, ANALYSIS_DIR, HIDDEN_STATES_DIR


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / (np.linalg.norm(v) + eps)


def _orthogonal_residual(v: np.ndarray, basis_unit: np.ndarray) -> np.ndarray:
    return v - float(np.dot(v, basis_unit)) * basis_unit


def compute_layer_geometry(kr, kc, sr, sc) -> dict:
    d_k_raw = (kr - kc).mean(axis=0)
    d_s_raw = (sr - sc).mean(axis=0)
    d_k = _unit(d_k_raw)
    d_s = _unit(d_s_raw)

    cos = float(np.dot(d_k, d_s))
    d_common = _unit(d_k + d_s)
    k_spec_raw = _orthogonal_residual(d_k, d_common)
    s_spec_raw = _orthogonal_residual(d_s, d_common)
    d_k_spec = _unit(k_spec_raw)
    d_s_spec = _unit(s_spec_raw)
    d_ks_contrast = _unit(d_k - d_s)

    return {
        "d_k": d_k,
        "d_s": d_s,
        "d_common": d_common,
        "d_k_specific": d_k_spec,
        "d_s_specific": d_s_spec,
        "d_ks_contrast": d_ks_contrast,
        "cos_dk_ds": cos,
        "common_projection_k": float(np.dot(d_k, d_common)),
        "common_projection_s": float(np.dot(d_s, d_common)),
        "specific_residual_norm_k": float(np.linalg.norm(k_spec_raw)),
        "specific_residual_norm_s": float(np.linalg.norm(s_spec_raw)),
        "raw_shift_norm_k": float(np.linalg.norm(d_k_raw)),
        "raw_shift_norm_s": float(np.linalg.norm(d_s_raw)),
    }


def compute_model(model_name: str) -> dict | None:
    npz_path = HIDDEN_STATES_DIR / f"{model_name}.npz"
    if not npz_path.exists():
        print(f"  WARNING: {npz_path} not found")
        return None

    data = np.load(npz_path, allow_pickle=True)
    required = ["knowledge_hidden", "safety_hidden", "k_control_hidden", "s_control_hidden"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{model_name} hidden-state cache is missing {missing}")

    kr = data["knowledge_hidden"]
    sr = data["safety_hidden"]
    kc = data["k_control_hidden"]
    sc = data["s_control_hidden"]
    n_pairs, n_layers, hidden_dim = kr.shape

    arrays = {
        "d_k": np.zeros((n_layers, hidden_dim), dtype=np.float32),
        "d_s": np.zeros((n_layers, hidden_dim), dtype=np.float32),
        "d_common": np.zeros((n_layers, hidden_dim), dtype=np.float32),
        "d_k_specific": np.zeros((n_layers, hidden_dim), dtype=np.float32),
        "d_s_specific": np.zeros((n_layers, hidden_dim), dtype=np.float32),
        "d_ks_contrast": np.zeros((n_layers, hidden_dim), dtype=np.float32),
    }
    layers = []

    for layer in range(n_layers):
        geom = compute_layer_geometry(kr[:, layer, :], kc[:, layer, :], sr[:, layer, :], sc[:, layer, :])
        for key in arrays:
            arrays[key][layer] = geom[key]
        layers.append({
            "layer": layer,
            "relative_layer": layer / max(n_layers - 1, 1),
            "cos_dk_ds": geom["cos_dk_ds"],
            "common_projection_k": geom["common_projection_k"],
            "common_projection_s": geom["common_projection_s"],
            "specific_residual_norm_k": geom["specific_residual_norm_k"],
            "specific_residual_norm_s": geom["specific_residual_norm_s"],
            "raw_shift_norm_k": geom["raw_shift_norm_k"],
            "raw_shift_norm_s": geom["raw_shift_norm_s"],
        })

    # RQ3: project each normalized shift onto the final-layer KR-specific axis.
    d_ref = arrays["d_k_specific"][-1]
    for layer, row in enumerate(layers):
        row["projection_on_final_k_specific_k"] = float(np.dot(arrays["d_k"][layer], d_ref))
        row["projection_on_final_k_specific_s"] = float(np.dot(arrays["d_s"][layer], d_ref))

    peak_layer = int(np.argmax([row["cos_dk_ds"] for row in layers]))
    return {
        "model": model_name,
        "analysis": "control_to_refusal_direction_decomposition",
        "n_pairs": int(n_pairs),
        "n_layers": int(n_layers),
        "hidden_dim": int(hidden_dim),
        "peak_alignment_layer": peak_layer,
        "peak_alignment": layers[peak_layer]["cos_dk_ds"],
        "layers": layers,
        "arrays": arrays,
    }


def save_result(result: dict) -> None:
    model_name = result["model"]
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {k: v for k, v in result.items() if k != "arrays"}
    summary_path = ANALYSIS_DIR / f"{model_name}_direction_decomposition.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    arrays_path = ANALYSIS_DIR / f"{model_name}_directions.npz"
    np.savez_compressed(arrays_path, **result["arrays"], model=np.array(model_name))
    print(f"  Saved {summary_path}")
    print(f"  Saved {arrays_path}")


def main():
    parser = argparse.ArgumentParser(description="RQ1/RQ3: direction decomposition")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all-models", action="store_true")
    args = parser.parse_args()

    model_names = list(MODELS) if args.all_models else [
        k for k, v in MODELS.items() if v["path"] == args.model or k == args.model
    ]
    if not model_names:
        print(f"ERROR: '{args.model}' not in config.MODELS", file=sys.stderr)
        sys.exit(1)

    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        result = compute_model(model_name)
        if result is not None:
            save_result(result)


if __name__ == "__main__":
    main()
