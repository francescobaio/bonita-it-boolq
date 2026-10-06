import json
import random
import re

from datasets import concatenate_datasets, load_dataset

INSTRUCTION_VARIANTS = [
    "Dopo aver letto il passaggio, rispondi alla seguente domanda:",
    "In base al testo precedente, rispondi alla domanda:",
    "Secondo quanto riportato nel passaggio, rispondi alla domanda:",
]
QUESTION_SUFFIX = "Vero o Falso?"
SPECIAL_TOKENS = ["<|tasktype|>", "<|context|>", "<|task|>", "<|pipe|>"]

BONITA_INPUT = "<|tasktype|>\nboolean question answering\n<|context|>\n{passage}\n<|task|>"
BONITA_OUTPUT = "\n{instruction} {question}\n<|pipe|>\n{answer}"
PROMPT_TEMPLATE = "{passage}\n\n{task}\nRisposta:"  # student training and evaluation

WORD_RE = re.compile(r"\w+")


def normalize(question: str) -> str:
    return " ".join(WORD_RE.findall(question.lower()))


def answer_text(label: bool) -> str:
    return "Vero" if label else "Falso"


def split_by_group(rows: list[dict], key: str, frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Splits rows into (train, held-out) so that all rows sharing `key` end up on the same side."""
    groups = sorted({r[key] for r in rows})
    random.Random(seed).shuffle(groups)
    held_out = set(groups[: round(len(groups) * frac)])
    return [r for r in rows if r[key] not in held_out], [r for r in rows if r[key] in held_out]


def load_boolq_splits(test_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """BoolQ train + validation split 80/20 by original (English) passage: (BONITA source, test set)."""
    boolq = load_dataset("sapienzanlp/boolq_italian")
    boolq_all = [
        {**ex, "passage": ex["metadata"]["passage"]}
        for ex in concatenate_datasets([boolq["train"], boolq["validation"]]).to_list()
    ]
    bonita_source, boolq_test = split_by_group(boolq_all, "passage", frac=test_frac, seed=seed)

    # near-duplicate paragraphs: drop the test examples whose Italian passage or question is also in BONITA's data
    seen_passages = {ex["metadata"]["passage_translation"] for ex in bonita_source}
    seen_questions = {normalize(ex["input_translation"]) for ex in bonita_source}
    boolq_test = [
        ex
        for ex in boolq_test
        if ex["metadata"]["passage_translation"] not in seen_passages
        and normalize(ex["input_translation"]) not in seen_questions
    ]
    return bonita_source, boolq_test


def bonita_rows(bonita_source: list[dict], seed: int) -> list[dict]:
    """BoolQ examples in the BONITA meta-template (prompt = task type + passage, completion = task + answer)."""
    rng = random.Random(seed)
    return [
        {
            "prompt": BONITA_INPUT.format(passage=ex["metadata"]["passage_translation"]),
            "completion": BONITA_OUTPUT.format(
                instruction=rng.choice(INSTRUCTION_VARIANTS),
                question=f"{ex['input_translation']} {QUESTION_SUFFIX}",
                answer=f"Risposta: {answer_text(ex['label'])}",
            ),
            "passage": ex["passage"],
        }
        for ex in bonita_source
    ]


def student_rows(generated: list[dict]) -> list[dict]:
    """Generated tasks as student prompt/completion pairs (completion = " Vero" / " Falso")."""
    return [
        {
            "prompt": PROMPT_TEMPLATE.format(passage=r["passage"], task=r["task"]),
            "completion": " " + answer_text(r["label"]),
            "passage": r["passage"],
        }
        for r in generated
    ]


def eval_prompts(boolq_test: list[dict], instruction: str) -> list[str]:
    """Test examples in the student prompt format, all with the same `instruction`."""
    return [
        PROMPT_TEMPLATE.format(
            passage=ex["metadata"]["passage_translation"],
            task=f"{instruction} {ex['input_translation']} {QUESTION_SUFFIX}",
        )
        for ex in boolq_test
    ]


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
