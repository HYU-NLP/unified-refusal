"""Shared configuration for the refusal representation experiments.

Paths default to repo-local locations and can be overridden with environment
variables. This keeps the public release runnable without editing source files.
"""

import os
from pathlib import Path

SEED = 42
N_BOOTSTRAP = 1000
REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = os.environ.get(
    "UNIFIED_REFUSAL_DATASET",
    str(REPO_ROOT / "data" / "matched_quadruples.jsonl"),
)

MODELS = {
    "Meta-Llama-3-8B-Instruct": {
        "path": "meta-llama/Meta-Llama-3-8B-Instruct",
        "family": "llama",
    },
    "Qwen2.5-7B-Instruct": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "family": "qwen",
    },
    "gemma-2-9b-it": {
        "path": "google/gemma-2-9b-it",
        "family": "gemma",
    },
}


BASE_DIR = REPO_ROOT
OUTPUT_DIR = Path(os.environ.get("UNIFIED_REFUSAL_OUTPUT_DIR", str(BASE_DIR / "outputs")))
HIDDEN_STATES_DIR = OUTPUT_DIR / "hidden_states"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
