# A Unified Mechanistic Analysis of Knowledge- and Safety-Based Refusals

<p align="center">
  <a href="https://arxiv.org/abs/2609.00760"><img src="https://img.shields.io/badge/arXiv-2609.00760-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/HYU-NLP/unified-refusal/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB.svg" alt="Python 3.10 or later"></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.00760">📄 Paper</a> ·
  <a href="#installation">⚙️ Installation</a> ·
  <a href="#running-experiments">🚀 Experiments</a> ·
  <a href="data_construction/prompts.md">🧩 Construction Prompts</a>
</p>

**Official code and prompt materials for [A Unified Mechanistic Analysis of Knowledge- and Safety-Based Refusals](https://arxiv.org/abs/2609.00760).**

Yuri Son, Seunghee Kim, Hyuhng Joon Kim, and Taeuk Kim. *Accepted to EMNLP 2026 Main Conference.*

---

## Overview

This repository contains the core representation-analysis code for studying whether knowledge-based refusal (KR) and safety-based refusal (SR) share internal mechanisms in instruction-tuned language models.

The analysis is built around contrastive quadruples of the form `(KR, KC, SR, SC)`, where KR/SR are refusal prompts and KC/SC are matched answerable or benign controls. The code extracts layer-wise hidden states, estimates shared and type-specific refusal directions, measures cross-type probe transfer, and interprets late-layer specialization through a vocabulary logit lens.

## Resources

- [Paper (arXiv)](https://arxiv.org/abs/2609.00760)
- [Construction prompts](data_construction/prompts.md)
- [Source code](src)
- [MIT License](LICENSE)

## Repository Structure

| Path | Description |
| --- | --- |
| [`src/`](src) | Hidden-state extraction and the RQ1–RQ3 representation analyses |
| [`data_construction/prompts.md`](data_construction/prompts.md) | KR/SR and control-generation prompts plus validation and judging specifications |
| [`scripts/run_analysis.sh`](scripts/run_analysis.sh) | End-to-end script for the representation-analysis pipeline |
| [`requirements.txt`](requirements.txt) | Python dependencies |

## Implemented Analyses

The public pipeline implements the paper's central representation analyses for all three research questions:

| Research question | Implementation | Analysis |
| --- | --- | --- |
| RQ1: Shared Refusal Directions | [`analyze.py`](src/analyze.py), [`decompose.py`](src/decompose.py) | KR–SR versus KC–SC similarity, bootstrap intervals, per-layer refusal-direction alignment, common/type-specific decomposition, and the peak-layer alignment permutation diagnostic |
| RQ2: Asymmetric Cross-Type Transfer | [`cross_probe.py`](src/cross_probe.py) | Five-fold within-type probes, KR→SR and SR→KR transfer, and paired bootstrap intervals for the directional transfer gap |
| RQ3: Late Type-Specific Specialization | [`decompose.py`](src/decompose.py), [`logit_lens.py`](src/logit_lens.py) | Final-layer KR-specific projection, reference-layer robustness grid, and condition-wise vocabulary logit lens with readable-token filtering |

The release focuses on the reusable representation-analysis pipeline. The larger activation-steering and component-ablation experiments require model-specific intervention code and are not part of the default pipeline.

## Dataset Availability

The 213 analysis rows and the source workbook are not bundled with this repository. The prompt-based construction materials are available in [`data_construction/prompts.md`](data_construction/prompts.md), including:

- KR/SR matched-pair generation
- KC/SC control generation
- control validation
- LLM-as-judge quality evaluation

## Analysis Input Format

The analysis code reads a local, post-construction UTF-8 JSONL file. Each line must contain one contrastive quadruple with the following fields:

| Field | Required | Description |
| --- | --- | --- |
| `pair_id` | Yes | Unique identifier for the quadruple |
| `k_prompt` | Yes | Knowledge-based refusal prompt (KR) |
| `s_prompt` | Yes | Safety-based refusal prompt (SR) |
| `k_control_prompt` | Yes | Matched answerable knowledge control (KC) |
| `s_control_prompt` | Yes | Matched benign safety control (SC) |
| `category` | No | Category label; defaults to `unknown` |
| `pair_validation.final_pass` | No | Set to `false` to exclude the row; defaults to `true` |

Example input row:

```json
{"pair_id":"example-001","k_prompt":"...","s_prompt":"...","k_control_prompt":"...","s_control_prompt":"...","category":"example","pair_validation":{"final_pass":true}}
```

Point the code to a JSONL file in this format with:

```bash
export UNIFIED_REFUSAL_DATASET=/path/to/local_quadruples.jsonl
```

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/HYU-NLP/unified-refusal.git
cd unified-refusal
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The convenience script is written for Bash. On Windows, run it from WSL or Git Bash. To activate the environment in PowerShell instead, use `.venv\Scripts\Activate.ps1` and run the Python stages individually.

For gated Hugging Face models, request access to the relevant model repositories and authenticate first:

```bash
hf auth login
```

Hidden-state extraction requires enough CPU/GPU memory to load the selected model. The logit-lens stage loads the model weights on CPU in float32 and may require substantial system RAM.

## Model Configuration

Model keys and base-model paths are defined in [`src/config.py`](src/config.py). The public configuration includes:

- `Meta-Llama-3-8B-Instruct`
- `Qwen2.5-7B-Instruct`
- `gemma-2-9b-it`

Additional compatible Hugging Face models or local checkpoints can be registered by adding a model key, path, and family to `MODELS`.

Outputs default to `outputs/`. Override the location with:

```bash
export UNIFIED_REFUSAL_OUTPUT_DIR=/path/to/output_dir
```

## Running Experiments

Before running an experiment, set `UNIFIED_REFUSAL_DATASET` to your local four-way JSONL file as shown above.

Run one configured model on the first eight quadruples:

```bash
bash scripts/run_analysis.sh Meta-Llama-3-8B-Instruct 8
```

Run all configured models:

```bash
bash scripts/run_analysis.sh --all
```

The first argument must be a model key registered in `src/config.py` (for example, `Meta-Llama-3-8B-Instruct`). To use another model, add a corresponding entry to `MODELS` in that file.

The script runs these stages in order:

1. Hidden-state extraction (`extract_hidden_states.py`)
2. RQ1 template-control similarity and bootstrap intervals (`analyze.py`)
3. RQ2 asymmetric cross-type probing with paired bootstrap intervals (`cross_probe.py`)
4. RQ1 direction geometry and RQ3 specialization/robustness (`decompose.py`)
5. RQ3 condition-wise vocabulary logit lens (`logit_lens.py`)

Machine-readable hidden-state caches and analysis results are written under `outputs/hidden_states/` and `outputs/analysis/`, respectively.

Hidden-state extraction skips a model when its `.npz` cache already exists. If you change the input dataset or `--max-pairs` value, use a new `UNIFIED_REFUSAL_OUTPUT_DIR` or remove that model's old cache before rerunning.

## Notes

- Large or local artifacts are ignored by default, including `data/*.jsonl`, `data/*.xlsx`, `outputs/`, model checkpoints, hidden-state arrays, and logs.
- The default model configuration contains the three base instruction-tuned models listed above.

## Citation

```bibtex
@article{son2026unified,
  title={A Unified Mechanistic Analysis of Knowledge- and Safety-Based Refusals},
  author={Son, Yuri and Kim, Seunghee and Kim, Hyuhng Joon and Kim, Taeuk},
  journal={arXiv preprint arXiv:2609.00760},
  year={2026},
  url={https://arxiv.org/abs/2609.00760}
}
```

## License

This project is released under the [MIT License](LICENSE).
