# A Unified Mechanistic Analysis of Knowledge- and Safety-Based Refusals

---

#### Code and prompt materials for **[A Unified Mechanistic Analysis of Knowledge- and Safety-Based Refusals](https://arxiv.org/abs/2609.00760)**

##### Yuri Son, Seunghee Kim, Hyuhng Joon Kim, and Taeuk Kim. Accepted to EMNLP 2026 Main Conference.

---

This repository contains the core representation-analysis code for studying whether knowledge-based refusal (KR) and safety-based refusal (SR) share internal mechanisms in instruction-tuned language models.

The paper studies 213 contrastive quadruples of the form `(KR, KC, SR, SC)`, where KR/SR are refusal prompts and KC/SC are matched answerable or benign controls. The actual dataset rows used in the paper are **not released** in this repository. We release the core analysis code and dataset-construction prompts so that the construction protocol and main representation analyses can be inspected and reused.

## Paper

- arXiv: [https://arxiv.org/abs/2609.00760](https://arxiv.org/abs/2609.00760)
- Title: *A Unified Mechanistic Analysis of Knowledge- and Safety-Based Refusals*
- Venue: EMNLP 2026 Main Conference

## Repository Structure

| Path | Description |
| --- | --- |
| `src/` | Hidden-state extraction, representation analysis, cross-type probing, decomposition, and logit-lens analysis |
| `data_construction/prompts.md` | KR/SR generation, KC/SC control generation, validation, and LLM-as-judge prompts only |
| `scripts/` | Convenience scripts for running the main analysis pipeline |

## Dataset Status

The paper dataset is **not included** in this repository. This repository does not contain the 213 quadruple rows used in the paper, nor does it include generated JSONL/XLSX dataset files.

The original construction protocol used a spreadsheet template workbook to define categories, semantic frames, template skeletons, refusal bases, and slot families. That workbook and the generated dataset are not included.

The analysis code itself reads a local post-construction JSONL file. If you want to run the analysis on your own reconstructed data, set:

```bash
export UNIFIED_REFUSAL_DATASET=/path/to/local_quadruples.jsonl
```

That local JSONL is an analysis input format, not a released dataset schema.

## Dataset Construction Prompts Only

We release only the prompt-based construction protocol. The prompts are written around fields supplied from the spreadsheet template workbook. See [`data_construction/prompts.md`](data_construction/prompts.md) for:

- KR/SR matched pair generation prompt
- KC/SC control generation prompt
- control validation prompt
- LLM-as-judge quality prompt

No dataset construction helper scripts or generated dataset rows are included.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For gated Hugging Face models, authenticate first:

```bash
huggingface-cli login
```

## Model Configuration

Base model paths are defined in [`src/config.py`](src/config.py). Tuned checkpoints are placeholders in the public release and can be supplied with environment variables:

```bash
export UNIFIED_REFUSAL_LLAMA_TUNED_PATH=/path/to/llama-tuned-checkpoint
export UNIFIED_REFUSAL_QWEN_TUNED_PATH=/path/to/qwen-tuned-checkpoint
export UNIFIED_REFUSAL_GEMMA_TUNED_PATH=/path/to/gemma-tuned-checkpoint
```

Outputs default to `outputs/`. Override with:

```bash
export UNIFIED_REFUSAL_OUTPUT_DIR=/path/to/output_dir
```

## Running Experiments

Before running, set `UNIFIED_REFUSAL_DATASET` to your own local post-construction JSONL file.

Smoke run on one model and the first 8 local quadruples:

```bash
bash scripts/run_analysis.sh Meta-Llama-3-8B-Instruct 8
```

Run all configured models, after setting the tuned-checkpoint environment variables above:

```bash
bash scripts/run_analysis.sh --all
```

For a base-model-only run, pass the Hugging Face model key directly, as in the smoke-run example.

The main script runs hidden-state extraction, KR/SR representation analysis, cross-type probing, direction decomposition, and logit-lens analysis. It writes machine-readable outputs under `outputs/`.

## Notes

- No paper dataset rows are tracked in this repository.
- `data_construction/` contains prompts only, not generation outputs or dataset-construction scripts.
- Large/local artifacts are ignored by default: `data/*.jsonl`, `data/*.xlsx`, `outputs/`, model checkpoints, hidden-state `.npz` files, and logs.
- The code preserves original experiment/checkpoint keys for traceability.

## License

This code is released under the MIT License. See [LICENSE](LICENSE).

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
