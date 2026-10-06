import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
from peft import PeftModel
from transformers import AutoTokenizer

from bonita.config import INSTRUCTED_MODEL, RESULTS_PATH, SEED, STUDENT_DIR, TEST_FRAC
from bonita.data import INSTRUCTION_VARIANTS, eval_prompts, load_boolq_splits
from bonita.evaluation import evaluate_model, summarize
from bonita.model import load_model

_, boolq_test = load_boolq_splits(TEST_FRAC, SEED)
gold_labels = np.array([int(ex["label"]) for ex in boolq_test])
print(f"{len(gold_labels)} test examples, {len(INSTRUCTION_VARIANTS)} instruction variants")

tokenizer = AutoTokenizer.from_pretrained(INSTRUCTED_MODEL)
model = PeftModel.from_pretrained(load_model(model_name=INSTRUCTED_MODEL), STUDENT_DIR)
model.eval()

# one test set per instruction variant: same questions and labels, different instruction
predictions: dict[str, list[np.ndarray]] = {"Baseline": [], "Student": []}
for instruction in INSTRUCTION_VARIANTS:
    prompts = eval_prompts(boolq_test, instruction)
    with model.disable_adapter():
        predictions["Baseline"].append(evaluate_model(model, tokenizer, prompts))
    predictions["Student"].append(evaluate_model(model, tokenizer, prompts))

results = {"always Vero (majority)": summarize(np.ones_like(gold_labels), gold_labels)}
for name, preds in predictions.items():
    per_variant = [summarize(pred, gold_labels) for pred in preds]
    for i, scores in enumerate(per_variant):
        results[f"{name}, instruction {i}"] = scores
    results[f"{name}, mean"] = {k: float(np.mean([s[k] for s in per_variant])) for k in per_variant[0]}

# the metrics go to the "results" key, keeping the generation statistics if already there
saved = {}
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        saved = json.load(f)
saved["results"] = results
with open(RESULTS_PATH, "w") as f:
    json.dump(saved, f, indent=1)
