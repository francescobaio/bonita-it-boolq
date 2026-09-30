<div align="center">

# 🐟 BONITA

Italian **Bo**olean-QA fi**ne**-tuning — an Italian instantiation of the [*Bonito*](https://arxiv.org/pdf/2402.18334) instruction-tuning pipeline.

Fine-tuning `sapienzanlp/Minerva-7B-instruct-v1.0` with QLoRA for Boolean Question Answering in Italian.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Base model](https://img.shields.io/badge/base%20model-Minerva--7B--instruct-orange)](https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0)
[![Dataset](https://img.shields.io/badge/dataset-boolq__italian-informational)](https://huggingface.co/datasets/sapienzanlp/boolq_italian)
[![Paper](https://img.shields.io/badge/paper-arXiv%3A2402.18334-b31b1b)](https://arxiv.org/pdf/2402.18334)

[Notebook](bonita.ipynb) · [Trained Checkpoint](#trained-checkpoint) · [Known Limitations](#known-limitations--honest-caveats)

</div>

## Overview

**BONITA** is our instantiation, for Italian, of the pipeline proposed in [*Bonito*](https://arxiv.org/pdf/2402.18334) — *"Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation"* (Nayak et al., 2024). In its general form, Bonito takes **raw, unannotated text** and a **task attribute**, and generates an **instruction-tuning dataset** for that task via a *meta-template*, without any human annotation:

<p align="center">
  <img src="assets/bonito-workflow.png" alt="Bonito workflow: unannotated text + task attribute → Bonito → instruction tuning data → specialized LLM" width="720">
  <br>
  <sub>The general Bonito pipeline (figure reproduced from Nayak et al., 2024). BONITA implements the right-hand side of this diagram — the fine-tuning of a specialized LLM — for Italian boolean QA.</sub>
</p>

Our brief for this project was to instantiate this idea for Italian on top of the [ita-bench](https://huggingface.co/collections/sapienzanlp/ita-bench-italian-benchmarks-for-llms-66337ca59e6df7d7d4933896) collection of Italian LLM benchmarks (developed by [Sapienza NLP](https://huggingface.co/sapienzanlp)), and to fine-tune an Italian instruction-tuned model on the result.

### Scope: what we actually built, and why

Given our timeline and compute budget (a single, shared 48GB GPU on the DISI Bologna HPC cluster), we deliberately scoped down to **one well-understood task** rather than attempting the full multi-task pipeline from day one: **boolean question answering**, on [`sapienzanlp/boolq_italian`](https://huggingface.co/datasets/sapienzanlp/boolq_italian). We picked it because it is one of the few ita-bench datasets where passage, question, and label are already cleanly separated, letting us focus our limited time on getting the meta-template, the QLoRA fine-tuning, and the evaluation methodology right — before trying to generalize to more task types.

Concretely: we take an **already-annotated** benchmark, reformat it with Bonito's meta-template, and **QLoRA fine-tune** `sapienzanlp/Minerva-7B-instruct-v1.0` to answer with `Risposta: Vero` / `Risposta: Falso`. We do **not** generate a new dataset from unannotated text, and we do not (yet) cover multiple task types — that generalization is documented as future work.

## How BONITA Works

Each `(passage, question, label)` triple from `boolq_italian` is turned into a `prompt` / `completion` pair using the same style of meta-template construction as Bonito's own **CTGA** (Conditional Task Generation with Attributes) format:

<p align="center">
  <img src="assets/ctga-construction.png" alt="CTGA construction: dataset examples + Jinja template → meta-template → applied to examples" width="820">
  <br>
  <sub>How a dataset example becomes a meta-template instance (figure reproduced from Nayak et al., 2024) — the same construction BONITA follows for boolq_italian.</sub>
</p>

Concretely, `bonita.prompting` renders:

```text
Prompt:
<|tasktype|>
boolean question answering
<|context|>
{{ passage }}
<|task|>

Completion:
{{ instruction }} {{ question }} Vero o Falso?
<|pipe|>
Risposta: Vero   (or: Risposta: Falso)
```

The instruction phrasing (`"Avendo letto il precedente passaggio, rispondi alla seguente domanda:"`, …) is sampled per example from a small set of Italian variants, to avoid overfitting to one exact wording.

The base model, `sapienzanlp/Minerva-7B-instruct-v1.0` — an Italian-centric instruction-tuned LLM built on the Mistral architecture — is loaded in **4-bit** (bitsandbytes NF4), with the special tokens above registered and the embedding matrix resized. Its weights are frozen and a **LoRA** adapter (`r=16`, `alpha=32`, dropout `0.05`, targeting `q_proj`/`v_proj`) is attached on top, trained with `trl.SFTTrainer`. This keeps the trainable footprint at **~0.17% of total parameters** (6.8M / 3.9B) — full fine-tuning of a 7B model was never on the table with our compute budget.

## Repository Structure

Our own contribution — the code, notebook, and writeup we authored:

```text
BONITA/
├── bonita.ipynb        # Full experiment, in one notebook: Part I — Training, Part II — Evaluation
├── src/bonita/          # Shared library, imported by both the notebook and train.py
│   ├── data.py            # load & format sapienzanlp/boolq_italian
│   ├── prompting.py        # meta-template rendering (prompt/completion construction)
│   ├── model.py             # base model + tokenizer loading, 4-bit quant, LoRA setup
│   ├── generation.py         # batched generation, Vero/Falso label extraction
│   ├── evaluation.py          # BLEU / BERTScore / METEOR / accuracy, error analysis
│   └── utils.py                # global seeding for reproducibility
├── train.py             # Standalone training script, actually submitted to the HPC cluster
├── report/               # Project report (LaTeX + PDF) — not pushed yet, see below
└── slides/                # Project presentation + figures — not pushed yet, see below
```

Everything else is supporting/config material: `bonita.sbatch` and `bonita.sh` (SLURM job + shell entrypoint), `pyproject.toml`, `uv.lock`, and `requirements.txt` (dependencies).

> `report/` and `slides/` exist locally but are currently excluded via `.gitignore` and not pushed to GitHub yet — see [Report & Slides](#report--slides). The two figures used above are duplicated into the tracked `assets/` folder so they keep rendering here regardless.

## Getting Started

**Requirements:** Python 3.11–3.12, and — for the actual QLoRA training/inference steps — a **CUDA-enabled Linux GPU** (`bitsandbytes` 4-bit quantization has no macOS/CPU backend). Everything else (data loading, prompt construction, unit-level testing of `src/bonita`) runs fine on any machine.

Clone the repo and install dependencies with [`uv`](https://docs.astral.sh/uv/) (the project's dependency manager — `pyproject.toml` + `uv.lock` are the source of truth):

```bash
git clone https://github.com/francescobaio/BONITA.git
cd BONITA
uv sync
```

Alternatively, with plain `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the Experiment

### Training

Training was run on the **DISI Bologna HPC cluster** (SLURM, one NVIDIA L40 48GB GPU) via [`train.py`](train.py), submitted with:

```bash
sbatch bonita.sbatch
```

`bonita.sbatch` calls [`bonita.sh`](bonita.sh), which activates the project's virtual environment and runs `train.py` (4 epochs, learning rate `2e-4`, automatic batch size search, `fp16`). The resulting LoRA adapter is the `checkpoint-4432` referenced throughout the notebook and code comments.

> We no longer have access to that cluster allocation, so this run is **not meant to be reproduced from this repo as-is** — it is documented for transparency and as a reference for anyone re-running it on equivalent hardware.

### Trained checkpoint

The LoRA adapter produced by the run above (`checkpoint-4432`, ~826 MB) is **not committed to this repository** — it's too large for a plain git push, and we don't use Git LFS here. It's hosted instead on a public OneDrive folder:

**Checkpoint download:** [OneDrive link — TODO]

Download it and point `load_finetuned_model` (used in Section 5 of the notebook) at the extracted folder:

```python
from bonita.model import load_finetuned_model

tokenizer, model, base_model = load_finetuned_model("path/to/checkpoint-4432")
```

### Notebook

The full, documented workflow — dataset loading, meta-template construction, model/LoRA setup, training, evaluation, qualitative error analysis, and base-vs-fine-tuned comparison — lives in a single notebook, split into two parts:

```bash
uv run jupyter lab bonita.ipynb
```

- **Part I — Training** (Sections 1–4): environment, dataset, model/LoRA setup, and documentation of the run that produced `checkpoint-4432`.
- **Part II — Evaluation** (Sections 5–9): loading the trained adapter, automatic metrics, qualitative error analysis, base-vs-fine-tuned comparison, and known limitations.

All logic shared across cells (and with `train.py`) lives in the importable [`src/bonita`](src/bonita) package rather than being duplicated — see its docstrings for details.

## Results

We compare the fine-tuned BONITA model against the un-adapted Minerva-7B-instruct base model on the same evaluation pipeline (BLEU, BERTScore F1, METEOR), as reported in our project report (see [Report & Slides](#report--slides)):

| Model                 |   BLEU | BERTScore F1 | METEOR |
| :--------------------- | -----: | -----------: | -----: |
| Base Minerva-7B        | 0.4772 |        0.8958 | 0.7295 |
| BONITA fine-tuned      | 0.4801 |        0.8963 | 0.7312 |

The fine-tuned model is consistently, but only marginally, better across all three metrics — evidence that Minerva-7B-instruct is already fairly competent on this task out of the box, and that QLoRA fine-tuning mainly sharpens formatting and response style rather than reasoning ability.

**Read this table together with the caveat right below** — we do not consider these deltas conclusive, and neither should you.

## Known Limitations & Honest Caveats ⚠️

We would rather document what is imperfect here than have it surface later. `checkpoint-4432` was trained once, under the conditions below, and we no longer have cluster access to retrain it — so these are documented rather than silently fixed:

- **Possible train/test overlap (most serious).** `train.py` splits the data with `train_test_split(test_size=0.3)` and **no fixed seed**; the exact held-out set from the real training run was never saved and cannot be reconstructed. The evaluation split reconstructed in the notebook (`seed=42`) is a *different* partition — some examples scored above may have been seen during training. **Treat the results table as indicative, not as a rigorous held-out evaluation.**
- **The reported deltas may just be decoding noise.** The original evaluation used Minerva's default sampling (`do_sample=True, temperature=0.4`) without overriding it, and generation was only ever run once. The table above (BLEU 0.4772→0.4801, BERTScore 0.8958→0.8963, METEOR 0.7295→0.7312) is not clearly distinguishable from run-to-run sampling variance. `bonita.generation.generate_predictions` now defaults to **deterministic (greedy) decoding** to remove this source of noise going forward, though it does not by itself resolve the overlap issue above.
- **Accuracy/format-adherence, not BLEU/METEOR, are the metrics that actually matter here.** BLEU/METEOR were designed for open-ended generation; for a two-class (`Vero`/`Falso`) task they are only weakly informative and are kept mainly for continuity with the project's original evaluation. The notebook additionally reports exact-match accuracy and format adherence (fraction of outputs with a parseable `Risposta: Vero/Falso`).
- **Dead special token.** `"{{ context }}"` is registered as a special token and its embedding was resized into the model, but the output template's `{{ context }}` placeholder is actually rendered by Jinja2 as an empty string (no `context` variable is passed at render time) — so that literal text never appears in any training example, and the embedding slot is untrained. See [`src/bonita/prompting.py`](src/bonita/prompting.py) for the full explanation; kept as-is rather than fixed, to stay consistent with the already-trained `checkpoint-4432`.
- **No model selection.** Checkpoints were saved every 500 steps (500 → 4432), but the final checkpoint was used without comparing validation loss across checkpoints.


## Report & Slides

The full write-up (`report/`, LaTeX) and the project presentation (`slides/`, Quarto/reveal.js) are part of this project but **not published in this repository yet** — they're still being finalized. Both are available locally under `BONITA/report/` and `BONITA/slides/` if you have the full working directory; links here will be added once they're public.

## References

- Nayak, N. V., Nan, Y., Trost, A., & Bach, S. H. (2024). *Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation.* [arXiv:2402.18334](https://arxiv.org/abs/2402.18334) — the Bonito paper this project instantiates for Italian.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Jiang, A. Q., et al. (2023). *Mistral 7B.* [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
- Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., & Toutanova, K. (2019). *BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions.* [arXiv:1905.10044](https://arxiv.org/abs/1905.10044)
- Sapienza NLP. *sapienzanlp/boolq_italian* [dataset card]. [huggingface.co/datasets/sapienzanlp/boolq_italian](https://huggingface.co/datasets/sapienzanlp/boolq_italian)
- Sapienza NLP. *sapienzanlp/Minerva-7B-instruct-v1.0* [model card]. [huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0](https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0)
- Sapienza NLP. [ita-bench](https://huggingface.co/collections/sapienzanlp/ita-bench-italian-benchmarks-for-llms-66337ca59e6df7d7d4933896) — Italian benchmarks for LLMs ([GitHub](https://github.com/SapienzaNLP/ita-bench)).

Full BibTeX entries are available in [`report/nlpreport.bib`](report/nlpreport.bib).

---

<div align="center">
<sub>Francesco Baiocchi & Leonardo Petrilli</sub>
</div>
