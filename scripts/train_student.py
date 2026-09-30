import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import transformers
from transformers import AutoTokenizer

from bonita.config import BASE_MODEL, DEV_FRAC, GENERATED_PATH, SEED, STUDENT_DIR, STUDENT_EPOCHS
from bonita.data import read_jsonl, split_by_group, student_rows
from bonita.model import load_base_model, train_lora

transformers.set_seed(SEED)

student_train, student_dev = split_by_group(student_rows(read_jsonl(GENERATED_PATH)), "passage", frac=DEV_FRAC, seed=SEED)
print(f"student train: {len(student_train)}, dev: {len(student_dev)}")
print(student_train[0]["prompt"] + student_train[0]["completion"])

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
train_lora(load_base_model(), tokenizer, student_train, student_dev, STUDENT_DIR, STUDENT_EPOCHS)
