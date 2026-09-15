"""RQ3: vocabulary logit lens.

For each condition, this module applies the model's final normalization and
unembedding to cached hidden states, averages across examples, and subtracts
the matched control. It then reports tokens associated with the common refusal
shift and the KR-minus-SR contrast. Reported tokens are restricted to readable,
word-initial ASCII words.
"""

import argparse
import gc
import json
import sys

import numpy as np

from config import ANALYSIS_DIR, HIDDEN_STATES_DIR, MODELS


def load_unembedding(model_path: str):
    """Load the tokenizer, output embedding, and final normalization on CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    output_embedding = model.get_output_embeddings()
    if output_embedding is None:
        raise ValueError(f"{model_path} does not expose an output embedding")
    unembedding = output_embedding.weight.detach().float().numpy()

    norm = model.model.norm
    raw_weight = norm.weight.detach().float().numpy()
    norm_weight = 1.0 + raw_weight if "Gemma" in type(norm).__name__ else raw_weight
    norm_eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))
    logit_softcap = float(
        getattr(model.config, "final_logit_softcapping", 0.0) or 0.0
    )
    del model
    gc.collect()
    return tokenizer, unembedding, norm_weight, norm_eps, logit_softcap


def condition_mean_logits(
    hidden: np.ndarray,
    unembedding: np.ndarray,
    norm_weight: np.ndarray,
    norm_eps: float,
    logit_softcap: float = 0.0,
    chunk_size: int = 8,
) -> np.ndarray:
    """Apply final norm and unembedding, then average logits over examples."""
    hidden = hidden.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(hidden ** 2, axis=-1, keepdims=True) + norm_eps)
    normalized = hidden / rms * norm_weight

    if logit_softcap <= 0.0:
        return unembedding @ normalized.mean(axis=0)

    # Gemma applies a nonlinear soft cap after unembedding, so average the
    # per-example capped logits rather than capping an already averaged vector.
    total = np.zeros(unembedding.shape[0], dtype=np.float64)
    for start in range(0, len(normalized), chunk_size):
        chunk = normalized[start:start + chunk_size] @ unembedding.T
        chunk = np.tanh(chunk / logit_softcap) * logit_softcap
        total += chunk.sum(axis=0)
    return (total / len(normalized)).astype(np.float32)


def _readable_word(tokenizer, token_id: int) -> str | None:
    raw = tokenizer.decode([token_id], skip_special_tokens=True)
    token_piece = tokenizer.convert_ids_to_tokens(token_id)
    word_initial = raw[:1].isspace() or str(token_piece).startswith(("Ġ", "▁"))
    word = raw.strip()
    if not word_initial or not 3 <= len(word) <= 18:
        return None
    if not word.isascii() or not word.isalpha():
        return None
    return word


def top_tokens(tokenizer, scores: np.ndarray, topk: int) -> dict:
    """Return top positive/negative unique word-initial ASCII tokens."""
    if topk < 1:
        raise ValueError("topk must be positive")

    def select(order: np.ndarray) -> list[dict]:
        selected = []
        seen = set()
        for index in order:
            token_id = int(index)
            word = _readable_word(tokenizer, token_id)
            if word is None or word.lower() in seen:
                continue
            selected.append({
                "token_id": token_id,
                "token": word,
                "score": float(scores[token_id]),
            })
            seen.add(word.lower())
            if len(selected) == topk:
                break
        return selected

    order = np.argsort(scores)
    return {
        "positive": select(order[::-1]),
        "negative": select(order),
    }


def compute_vocabulary_scores(
    kr: np.ndarray,
    kc: np.ndarray,
    sr: np.ndarray,
    sc: np.ndarray,
    unembedding: np.ndarray,
    norm_weight: np.ndarray,
    norm_eps: float,
    logit_softcap: float,
) -> dict[str, np.ndarray]:
    """Compute control-subtracted KR, SR, common, and K-S token scores."""
    kr_logits = condition_mean_logits(
        kr, unembedding, norm_weight, norm_eps, logit_softcap
    )
    kc_logits = condition_mean_logits(
        kc, unembedding, norm_weight, norm_eps, logit_softcap
    )
    sr_logits = condition_mean_logits(
        sr, unembedding, norm_weight, norm_eps, logit_softcap
    )
    sc_logits = condition_mean_logits(
        sc, unembedding, norm_weight, norm_eps, logit_softcap
    )
    k_shift = kr_logits - kc_logits
    s_shift = sr_logits - sc_logits
    return {
        "k_shift": k_shift,
        "s_shift": s_shift,
        "common_refusal": (k_shift + s_shift) / 2.0,
        "k_minus_s": k_shift - s_shift,
    }


def run_model(
    model_name: str,
    model_config: dict,
    layer: int | None,
    topk: int,
) -> None:
    hidden_path = HIDDEN_STATES_DIR / f"{model_name}.npz"
    if not hidden_path.exists():
        print(f"  WARNING: {hidden_path} not found; run extraction first")
        return

    with np.load(hidden_path, allow_pickle=False) as data:
        n_layers = data["knowledge_hidden"].shape[1]
        target_layer = n_layers - 1 if layer is None else layer
        if target_layer < 0 or target_layer >= n_layers:
            raise ValueError(
                f"layer must be in [0, {n_layers - 1}], got {target_layer}"
            )
        conditions = {
            "kr": data["knowledge_hidden"][:, target_layer],
            "kc": data["k_control_hidden"][:, target_layer],
            "sr": data["safety_hidden"][:, target_layer],
            "sc": data["s_control_hidden"][:, target_layer],
        }

    print(f"  Loading unembedding for {model_name} ...")
    tokenizer, unembedding, norm_weight, norm_eps, softcap = load_unembedding(
        model_config["path"]
    )
    scores = compute_vocabulary_scores(
        conditions["kr"],
        conditions["kc"],
        conditions["sr"],
        conditions["sc"],
        unembedding,
        norm_weight,
        norm_eps,
        softcap,
    )
    decoded = {name: top_tokens(tokenizer, values, topk) for name, values in scores.items()}

    result = {
        "model": model_name,
        "analysis": "condition_wise_vocabulary_logit_lens",
        "layer": int(target_layer),
        "topk": int(topk),
        "token_filter": "word-initial ASCII alphabetic tokens of length 3-18",
        "scores": decoded,
    }
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"{model_name}_logit_lens.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3: vocabulary logit lens")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str)
    group.add_argument("--all-models", action="store_true")
    parser.add_argument("--layer", type=int, default=None, help="Default: final layer")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    model_names = list(MODELS) if args.all_models else [
        key for key, value in MODELS.items()
        if value["path"] == args.model or key == args.model
    ]
    if not model_names:
        print(f"ERROR: '{args.model}' not in config.MODELS", file=sys.stderr)
        sys.exit(1)

    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        run_model(model_name, MODELS[model_name], layer=args.layer, topk=args.topk)


if __name__ == "__main__":
    main()
