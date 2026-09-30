import os
import random

import numpy as np
from datasets import load_dataset

from .config import N_PASSAGES, SEED, WIKI_PATH, WIKI_SHARD
from .data import read_jsonl, write_jsonl


def min_paragraph_words(bonita_source: list[dict]) -> int:
    """5th percentile of the length (in words) of BONITA's training passages."""
    return int(np.percentile([len(ex["metadata"]["passage_translation"].split()) for ex in bonita_source], 5))


def candidate_paragraphs(text: str, min_words: int) -> list[str]:
    paragraphs = (p.strip() for p in text.split("\n"))
    return [p for p in paragraphs if p.endswith(".") and len(p.split()) >= min_words]


def collect_wiki_passages(min_words: int, n_passages: int = N_PASSAGES) -> list[dict]:
    """One random paragraph of prose from each of `n_passages` shuffled articles."""
    wiki = load_dataset("parquet", data_files=WIKI_SHARD, split="train").shuffle(seed=SEED)
    rng = random.Random(SEED)
    passages = []
    for article in wiki:
        candidates = candidate_paragraphs(article["text"], min_words)
        if candidates:
            passages.append({"passage": rng.choice(candidates)})
        if len(passages) >= n_passages:
            break
    return passages


def load_or_collect_passages(bonita_source: list[dict], path: str = WIKI_PATH) -> list[dict]:
    """The passages are collected once and saved to `path`; if the file already exists it is loaded instead."""
    if os.path.exists(path):
        print(f"loading the passages from {path}")
        return read_jsonl(path)
    min_words = min_paragraph_words(bonita_source)
    print(f"minimum paragraph length: {min_words} words")
    passages = collect_wiki_passages(min_words)
    write_jsonl(passages, path)
    return passages
