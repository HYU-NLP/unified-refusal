# extract_hidden_states.py
import argparse
import gc
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODELS, DATASET_PATH, HIDDEN_STATES_DIR
from dataset import load_valid_pairs, has_control_prompts, RefusalPair


def format_prompt(question: str, tokenizer) -> torch.Tensor:
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt")["input_ids"]


def extract_last_token_hidden_states(
    model,
    tokenizer,
    questions: List[str],
) -> np.ndarray:
    """
    Collect last-token hidden state at every layer for each question.
    Returns float32 array [N, N_layers, hidden_dim].
    N_layers = embedding layer + transformer blocks.
    """
    device = next(model.parameters()).device
    all_hidden = []
    for q in questions:
        input_ids = format_prompt(q, tokenizer).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        last = torch.stack([h[0, -1, :] for h in out.hidden_states])  # [N_layers, d]
        all_hidden.append(last.float().cpu().numpy())
    return np.stack(all_hidden)  # [N, N_layers, d]


def save_extraction(
    model,
    tokenizer,
    pairs: List[RefusalPair],
    out_path: Path,
    model_path: str,
) -> None:
    k_questions = [p.knowledge_question for p in pairs]
    s_questions = [p.safety_question for p in pairs]

    print(f"    Extracting {len(pairs)} knowledge (refusal) questions ...")
    k_hidden = extract_last_token_hidden_states(model, tokenizer, k_questions)
    print(f"    Extracting {len(pairs)} safety (refusal) questions ...")
    s_hidden = extract_last_token_hidden_states(model, tokenizer, s_questions)

    n_layers = k_hidden.shape[1]
    hidden_dim = k_hidden.shape[2]

    save_kwargs = dict(
        knowledge_hidden=k_hidden,
        safety_hidden=s_hidden,
        pair_ids=np.array([p.pair_id for p in pairs]),
        categories=np.array([p.category for p in pairs]),
        layer_count=np.array(n_layers),
        hidden_dim=np.array(hidden_dim),
        model_path=np.array(model_path),
    )

    # Also extract control (normal-answer) prompts when available
    if has_control_prompts(pairs):
        k_ctrl = [p.k_control_question for p in pairs]
        s_ctrl = [p.s_control_question for p in pairs]
        print(f"    Extracting {len(pairs)} k_control (normal) questions ...")
        k_ctrl_hidden = extract_last_token_hidden_states(model, tokenizer, k_ctrl)
        print(f"    Extracting {len(pairs)} s_control (normal) questions ...")
        s_ctrl_hidden = extract_last_token_hidden_states(model, tokenizer, s_ctrl)
        save_kwargs["k_control_hidden"] = k_ctrl_hidden
        save_kwargs["s_control_hidden"] = s_ctrl_hidden

    np.savez_compressed(out_path, **save_kwargs)
    print(f"    Saved → {out_path}")


def extract_model(model_name: str, model_config: dict, pairs: List[RefusalPair]) -> None:
    out_path = HIDDEN_STATES_DIR / f"{model_name}.npz"
    if out_path.exists():
        print(f"  {model_name}: already extracted, skipping.")
        return

    print(f"  Loading {model_config['path']} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_config["path"])
    model = AutoModelForCausalLM.from_pretrained(
        model_config["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    save_extraction(model, tokenizer, pairs, out_path, model_config["path"])

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Stage 1: extract hidden states")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Model key from config.MODELS")
    group.add_argument("--all-models", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    HIDDEN_STATES_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_valid_pairs(DATASET_PATH, max_pairs=args.max_pairs)
    print(f"Loaded {len(pairs)} valid pairs from dataset.")
    if not pairs:
        raise ValueError("No valid rows were loaded from UNIFIED_REFUSAL_DATASET.")
    if not has_control_prompts(pairs):
        raise ValueError(
            "The main analyses require four-way KR/KC/SR/SC rows with "
            "k_control_prompt and s_control_prompt fields."
        )
    print("  control prompts detected: will extract KR/KC/SR/SC hidden states")

    if args.all_models:
        model_names = list(MODELS.keys())
    else:
        model_names = [
            k for k, v in MODELS.items()
            if v["path"] == args.model or k == args.model
        ]
        if not model_names:
            print(f"ERROR: model '{args.model}' not found in config.MODELS", file=sys.stderr)
            sys.exit(1)

    for name in model_names:
        print(f"\n=== {name} ===")
        extract_model(name, MODELS[name], pairs)


if __name__ == "__main__":
    main()
