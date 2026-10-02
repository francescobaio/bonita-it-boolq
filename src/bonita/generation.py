import re

import torch
from tqdm.auto import tqdm

from .data import INSTRUCTION_VARIANTS, QUESTION_SUFFIX, normalize

LABEL_RE = re.compile(r"Risposta:\s*(Vero|Falso)\b", re.IGNORECASE)


@torch.no_grad()
def batch_generate(model, tokenizer, prompts: list[str], batch_size: int = 16, **generation_kwargs) -> list[str]:
    """Generates one completion per prompt (new tokens only, special tokens removed); needs left padding."""
    predictions = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        inputs = tokenizer(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(model.device)
        outputs = model.generate(
            **inputs, pad_token_id=tokenizer.pad_token_id, tokenizer=tokenizer, **generation_kwargs
        )
        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        predictions += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return predictions


def parse_generation(text: str) -> dict | None:
    """Splits a BONITA generation into task (instruction + question), question and label."""
    match = LABEL_RE.search(text)
    if match is None:
        return None
    task = text[: match.start()].strip()
    if not task.endswith(QUESTION_SUFFIX):
        return None
    question = task.removesuffix(QUESTION_SUFFIX)
    for variant in INSTRUCTION_VARIANTS:
        question = question.replace(variant, "")
    return {"task": task, "question": question.strip(), "label": match.group(1).capitalize() == "Vero"}


def filter_generations(passages: list[dict], generations: list[str]) -> tuple[list[dict], dict]:
    """Keeps the parseable, non-duplicate tasks; returns them with the generation statistics."""
    generated, seen = [], set()
    counts = {"generated": len(generations), "malformed": 0, "duplicate": 0}
    for passage, text in zip(passages, generations, strict=True):
        parsed = parse_generation(text)
        if parsed is None:
            counts["malformed"] += 1
            continue
        key = normalize(parsed["question"])
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        generated.append({"passage": passage["passage"], **parsed})
    counts["kept"] = len(generated)
    counts["vero_rate"] = sum(r["label"] for r in generated) / len(generated)
    return generated, counts
