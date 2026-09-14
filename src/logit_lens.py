"""Direction logit-lens analysis.

Decodes directions from `decompose.py` through the model unembedding. This matches
the paper's vocabulary-level interpretation of the common refusal direction and
the K-S contrast direction.
"""

import argparse
import gc
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODELS, ANALYSIS_DIR


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / (np.linalg.norm(v) + eps)


def load_unembedding(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    W_U = model.lm_head.weight.detach().float().numpy()
    norm = model.model.norm
    raw_weight = norm.weight.detach().float().numpy()
    norm_cls = type(norm).__name__
    norm_weight = (1.0 + raw_weight) if "Gemma" in norm_cls else raw_weight
    norm_eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))
    logit_softcap = float(getattr(model.config, "final_logit_softcapping", 0.0) or 0.0)
    del model
    gc.collect()
    return tokenizer, W_U, norm_weight, norm_eps, logit_softcap


def direction_logits(direction: np.ndarray, W_U: np.ndarray, norm_weight: np.ndarray,
                     norm_eps: float, logit_softcap: float) -> np.ndarray:
    direction = _unit(direction)
    rms = np.sqrt((direction ** 2).mean(keepdims=True) + norm_eps)
    h = direction / rms * norm_weight
    logits = W_U @ h
    if logit_softcap > 0.0:
        logits = np.tanh(logits / logit_softcap) * logit_softcap
    return logits


def top_tokens(tokenizer, logits: np.ndarray, topk: int) -> dict:
    pos_ids = np.argsort(logits)[-topk:][::-1]
    neg_ids = np.argsort(logits)[:topk]
    return {
        "positive": [
            {"token_id": int(i), "token": tokenizer.decode([int(i)]), "score": float(logits[int(i)])}
            for i in pos_ids
        ],
        "negative": [
            {"token_id": int(i), "token": tokenizer.decode([int(i)]), "score": float(logits[int(i)])}
            for i in neg_ids
        ],
    }


def run_model(model_name: str, model_config: dict, layer: int | None, topk: int) -> None:
    directions_path = ANALYSIS_DIR / f"{model_name}_directions.npz"
    if not directions_path.exists():
        print(f"  WARNING: {directions_path} not found; run decompose.py first")
        return

    dirs = np.load(directions_path, allow_pickle=True)
    n_layers = dirs["d_common"].shape[0]
    target_layer = n_layers - 1 if layer is None else layer
    if target_layer < 0 or target_layer >= n_layers:
        raise ValueError(f"layer must be in [0, {n_layers - 1}], got {target_layer}")

    print(f"  Loading unembedding for {model_name} ...")
    tokenizer, W_U, norm_weight, norm_eps, logit_softcap = load_unembedding(model_config["path"])

    outputs = {}
    for name in ["d_common", "d_ks_contrast"]:
        logits = direction_logits(dirs[name][target_layer], W_U, norm_weight, norm_eps, logit_softcap)
        outputs[name] = top_tokens(tokenizer, logits, topk=topk)

    result = {
        "model": model_name,
        "analysis": "direction_logit_lens",
        "layer": int(target_layer),
        "topk": int(topk),
        "directions": outputs,
    }
    out_path = ANALYSIS_DIR / f"{model_name}_direction_logit_lens.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="RQ3: direction logit lens")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all-models", action="store_true")
    parser.add_argument("--layer", type=int, default=None, help="Layer to decode; default final layer")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    model_names = list(MODELS) if args.all_models else [
        k for k, v in MODELS.items() if v["path"] == args.model or k == args.model
    ]
    if not model_names:
        print(f"ERROR: '{args.model}' not in config.MODELS", file=sys.stderr)
        sys.exit(1)

    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        run_model(model_name, MODELS[model_name], layer=args.layer, topk=args.topk)


if __name__ == "__main__":
    main()
