import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import transformers

from bonita.config import BONITA_DIR, GENERATED_PATH, RESULTS_PATH, SEED, TEST_FRAC
from bonita.data import BONITA_INPUT, load_boolq_splits, write_jsonl
from bonita.generation import batch_generate, filter_generations
from bonita.model import load_bonita
from bonita.wiki import load_or_collect_passages

transformers.set_seed(SEED)

bonita_source, _ = load_boolq_splits(TEST_FRAC, SEED)
wiki_passages = load_or_collect_passages(bonita_source)
print(f"{len(wiki_passages)} passages")

model, tokenizer = load_bonita(BONITA_DIR)
generations = batch_generate(
    model,
    tokenizer,
    [BONITA_INPUT.format(passage=p["passage"]) for p in wiki_passages],
    max_new_tokens=256,
    stop_strings=["Risposta: Vero", "Risposta: Falso"],
    do_sample=False,
    use_cache=True,  # Minerva's config disables the KV cache
)

generated, counts = filter_generations(wiki_passages, generations)
write_jsonl(generated, GENERATED_PATH)
# the statistics go to the "generation" key of the results, keeping the metrics if already there
results = {}
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH) as f:
        results = json.load(f)
results["generation"] = counts
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=1)
