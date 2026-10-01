<div align="center">

# 🐟 BONITA

An Italian instantiation of [*Bonito*](https://arxiv.org/abs/2402.18334): a model that turns **unannotated Italian text** into **instruction-tuning data** for Boolean Question Answering.

[![Python](https://img.shields.io/badge/python-%E2%89%A53.11-blue?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-boolq__italian-informational)](https://huggingface.co/datasets/sapienzanlp/boolq_italian)

</div>

## Overview

[Bonito](https://arxiv.org/abs/2402.18334) (Nayak et al., 2024) is a model that, given a passage of raw text and a task type, generates a task for it: an instruction with its input and the expected answer. Running it over unannotated text yields an instruction-tuning dataset with no human annotation.

**BONITA** does the same for Italian, with boolean question answering as the task type: given a paragraph, it writes a yes/no question about it and its answer (`Vero` / `Falso`). As in Bonito, where the generator is trained on a pretrained (non-instruction-tuned) model, BONITA is a QLoRA adapter on [`sapienzanlp/Minerva-7B-base-v1.0`](https://huggingface.co/sapienzanlp/Minerva-7B-base-v1.0), trained on [`sapienzanlp/boolq_italian`](https://huggingface.co/datasets/sapienzanlp/boolq_italian) from the [ita-bench](https://huggingface.co/collections/sapienzanlp/ita-bench-italian-benchmarks-for-llms-66337ca59e6df7d7d4933896) collection.

Following the Bonito paper, BONITA is evaluated **extrinsically**: a student, [`sapienzanlp/Minerva-7B-instruct-v1.0`](https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0) with a new LoRA adapter, is trained only on the data BONITA generates from Italian Wikipedia, and compared with Minerva-7B-instruct without fine-tuning on held-out, human-labelled BoolQ examples.

## Pipeline

1. **Train BONITA**: a QLoRA adapter on Minerva-7B-base, trained on 80% of `boolq_italian` (`train` + `validation` merged, split by passage) in the Bonito meta-template.
2. **Collect unannotated text**: 10,000 paragraphs of prose from Italian Wikipedia (`wikimedia/wikipedia`, `20231101.it`), the same source as the BoolQ passages.
3. **Generate a synthetic dataset**: BONITA writes one question/answer pair per paragraph (greedy decoding); unparseable generations and duplicate questions are discarded.
4. **Train the student**: a new LoRA on Minerva-7B-instruct, trained **only** on the synthetic dataset.
5. **Evaluate** Minerva-7B-instruct without fine-tuning and the student on the remaining 20% (2,490 examples) by log-likelihood.

### Meta-template

Each BoolQ example becomes a prompt/completion pair; the four markers are added to the tokenizer as special tokens:

```text
<|tasktype|>
boolean question answering
<|context|>
{passage}
<|task|>                                  ← end of the prompt
{instruction} {question} Vero o Falso?
<|pipe|>
Risposta: Vero | Falso                    ← end of the completion
```

The instruction is drawn at random from three Italian paraphrases, so BONITA learns to generate both the question and its answer from a passage.

### Training

BONITA (on Minerva-7B-base) and the student (on Minerva-7B-instruct) share the same setup:

- **QLoRA**: model in 4-bit NF4 with `bfloat16` compute; LoRA on all linear projections, `r=16`, `alpha=32`, dropout `0.05`.
- **SFT** with `trl.SFTTrainer`, loss on the completion only: effective batch 16, learning rate `5e-5` with cosine decay and 3% warmup, at most 4 epochs.
- **Model selection**: dev loss every 100 steps on 5% of the training *passages*; the best checkpoint is kept, with early stopping after 3 evaluations without improvement.

### Evaluation

The student sees each test example as

```text
{passage}

Dopo aver letto il passaggio, rispondi alla seguente domanda: {question} Vero o Falso?
Risposta:
```

and the answer is whichever continuation between ` Vero` and ` Falso` has the higher log-likelihood, so no output parsing is involved. Minerva-7B-instruct without fine-tuning is scored on the same 4-bit model as the student, with the adapter disabled. Since the classes are unbalanced (61% `Vero`), we report **accuracy**, **macro-F1** and the always-`Vero` baseline.

To avoid leakage, the 80/20 split is done by passage, and test examples whose passage or question also appears (as a near-duplicate) in BONITA's training data are removed.

## Results

| Model                                 | Accuracy | Macro-F1 |
| :------------------------------------ | -------: | -------: |
| Majority baseline (`Vero`)            |      TBD |      TBD |
| Minerva-7B-instruct (no fine-tuning)  |      TBD |      TBD |
| Student (BONITA data only)            |      TBD |      TBD |

## Repository structure

```text
.
├── bonita.ipynb             # full pipeline and results (self-contained)
├── src/bonita/              # the notebook's code as a package, used by the scripts
│   ├── config.py            # model names, output paths, hyperparameters
│   ├── data.py              # BoolQ splits, meta-template, student/eval prompts
│   ├── wiki.py              # sampling of the Wikipedia paragraphs
│   ├── model.py             # 4-bit loading, LoRA, SFT training
│   ├── generation.py        # batched generation, parsing and filtering
│   └── evaluation.py        # log-likelihood scoring, accuracy and macro-F1
└── scripts/
    ├── train_bonita.py      # stage 1: train BONITA
    ├── generate.py          # stage 2: collect passages and generate the synthetic dataset
    └── train_student.py     # stage 3: train the student
```

## Getting started

Requires Python ≥ 3.11 and a CUDA GPU (`bitsandbytes` 4-bit quantization has no CPU/macOS backend); the pipeline was run on a single 24 GB GPU. Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/francescobaio/bonita-it-boolq.git
cd bonita-it-boolq
uv sync
```

## Resources

- **Slides**: [GitHub Pages — TODO](#) (source on the [`slides`](../../tree/slides) branch)
- **Report**: LaTeX source and PDF on the [`report`](../../tree/report) branch
- **Checkpoints**: [Google Drive — TODO](#), with the BONITA adapter and tokenizer (`bonita_full/`) and the student adapter (`student_bonita/`); place them in the repository root to skip the training stages

## References

- Nayak, N. V., Nan, Y., Trost, A., & Bach, S. H. (2024). *Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation.* [arXiv:2402.18334](https://arxiv.org/abs/2402.18334)
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Clark, C., et al. (2019). *BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions.* [arXiv:1905.10044](https://arxiv.org/abs/1905.10044)
- Sapienza NLP. [`sapienzanlp/boolq_italian`](https://huggingface.co/datasets/sapienzanlp/boolq_italian), [`sapienzanlp/Minerva-7B-base-v1.0`](https://huggingface.co/sapienzanlp/Minerva-7B-base-v1.0), [`sapienzanlp/Minerva-7B-instruct-v1.0`](https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0), [ita-bench](https://github.com/SapienzaNLP/ita-bench).

## Acknowledgements

This project builds on the work of [Sapienza NLP](https://nlp.uniroma1.it/) at Sapienza University of Rome. BONITA and the student are both based on [Minerva](https://nlp.uniroma1.it/minerva/), their family of LLMs trained from scratch on Italian ([Minerva-7B-base](https://huggingface.co/sapienzanlp/Minerva-7B-base-v1.0) and [Minerva-7B-instruct](https://huggingface.co/sapienzanlp/Minerva-7B-instruct-v1.0) respectively), and the training and evaluation data come from their Italian BoolQ and ita-bench. We thank them for making these resources openly available.

---

<div align="center">
<sub>Francesco Baiocchi & Leonardo Petrilli</sub>
</div>
