import numpy as np
import torch
from tqdm.auto import tqdm

CHOICES = [" Falso", " Vero"]  # index = gold label (False = 0, True = 1)


def encode_request(tokenizer, context: str, continuation: str) -> tuple[list[int], int]:
    """Returns the token ids of context + continuation and the number of context tokens."""
    context_ids = tokenizer(context).input_ids
    return context_ids + tokenizer(continuation, add_special_tokens=False).input_ids, len(context_ids)


@torch.no_grad()
def continuation_logprobs(
    model, tokenizer, contexts: list[str], continuations: list[str], batch_size: int = 16
) -> np.ndarray:
    """Sum of the log-probabilities of each continuation given its context."""
    requests = [encode_request(tokenizer, c, x) for c, x in zip(contexts, continuations, strict=True)]
    scores = []
    for start in tqdm(range(0, len(requests), batch_size)):
        batch = requests[start : start + batch_size]
        # right padding: the positions of the real tokens are unchanged
        inputs = tokenizer.pad({"input_ids": [ids for ids, _ in batch]}, padding_side="right", return_tensors="pt")
        logits = model(**inputs.to(model.device)).logits
        for row, (ids, n_context) in enumerate(batch):
            # the token at position t is predicted by the logits at position t - 1
            logprobs = torch.log_softmax(logits[row, n_context - 1 : len(ids) - 1].float(), dim=-1)
            targets = torch.tensor(ids[n_context:], device=logprobs.device)
            scores.append(logprobs.gather(-1, targets[:, None]).sum().item())
    return np.array(scores)


def evaluate_model(model, tokenizer, prompts: list[str]) -> np.ndarray:
    """Scores both continuations for every prompt; the index of the best one is the predicted label."""
    contexts = [p for p in prompts for _ in CHOICES]
    continuations = [c for _ in prompts for c in CHOICES]
    return continuation_logprobs(model, tokenizer, contexts, continuations).reshape(-1, len(CHOICES)).argmax(axis=1)


def macro_f1(pred: np.ndarray, gold: np.ndarray) -> float:
    f1s = []
    for c in (0, 1):
        tp = np.sum((pred == c) & (gold == c))
        fp = np.sum((pred == c) & (gold != c))
        fn = np.sum((pred != c) & (gold == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(f1s))


def summarize(pred: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    return {"acc": float(np.mean(pred == gold)), "macro_f1": macro_f1(pred, gold)}
