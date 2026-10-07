import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import transformers
from transformers import AutoTokenizer

from bonita.config import BASE_MODEL, BONITA_DIR, BONITA_EPOCHS, DEV_FRAC, SEED, TEST_FRAC
from bonita.data import SPECIAL_TOKENS, bonita_rows, load_boolq_splits, split_by_group
from bonita.model import load_model, train_lora

transformers.set_seed(SEED)

bonita_source, boolq_test = load_boolq_splits(TEST_FRAC, SEED)
print(f"BONITA source: {len(bonita_source)}, test: {len(boolq_test)}")
bonita_train, bonita_dev = split_by_group(bonita_rows(bonita_source, SEED), "passage", frac=DEV_FRAC, seed=SEED)
print(f"BONITA train: {len(bonita_train)}, dev: {len(bonita_dev)}")
print(bonita_train[0]["prompt"] + bonita_train[0]["completion"])

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)

model = load_model(tokenizer)
special_token_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
assert isinstance(special_token_ids, list)  # a list of tokens gives a list of ids
train_lora(model, tokenizer, bonita_train, bonita_dev, BONITA_DIR, BONITA_EPOCHS, trainable_token_ids=special_token_ids)
