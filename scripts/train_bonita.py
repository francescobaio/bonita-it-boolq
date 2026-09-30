import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import transformers
from transformers import AutoTokenizer

from bonita.config import BASE_MODEL, BONITA_DIR, BONITA_EPOCHS, DEV_FRAC, SEED, TEST_FRAC
from bonita.data import SPECIAL_TOKENS, bonita_rows, load_boolq_splits, split_by_group
from bonita.model import load_base_model, train_lora

transformers.set_seed(SEED)

bonita_source, boolq_test = load_boolq_splits(TEST_FRAC, SEED)
print(f"BONITA source: {len(bonita_source)}, test: {len(boolq_test)}")
bonita_train, bonita_dev = split_by_group(bonita_rows(bonita_source, SEED), "passage", frac=DEV_FRAC, seed=SEED)
print(f"BONITA train: {len(bonita_train)}, dev: {len(bonita_dev)}")
print(bonita_train[0]["prompt"] + bonita_train[0]["completion"])

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)

train_lora(load_base_model(tokenizer), tokenizer, bonita_train, bonita_dev, BONITA_DIR, BONITA_EPOCHS)
tokenizer.save_pretrained(BONITA_DIR)  # needed to reload BONITA with its special tokens
