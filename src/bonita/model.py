import tempfile

import pandas as pd
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from .config import BASE_MODEL, SEED

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


def load_model(tokenizer=None, model_name: str = BASE_MODEL) -> transformers.PreTrainedModel:
    """4-bit model; the embeddings are resized only if `tokenizer` has no free rows left for its added tokens."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    if tokenizer is not None and len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    return model


def load_bonita(bonita_dir: str):
    """4-bit model with the BONITA adapter, and its tokenizer (with the special tokens)."""
    tokenizer = AutoTokenizer.from_pretrained(bonita_dir)
    model = PeftModel.from_pretrained(load_model(tokenizer), bonita_dir)
    model.eval()
    return model, tokenizer


def init_special_tokens(model, tokenizer, tokens: list[str]) -> list[int]:
    """Initializes the input and output embeddings of `tokens` as the mean of those of their name.

    E.g. `<|pipe|>` starts from the mean of the rows of ` pipe`.

    New rows added by `resize_token_embeddings` all start from the same small mean vector: the tokens would be
    indistinguishable as input and too weak as output to ever be generated. Returns the ids of `tokens`.
    """
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    with torch.no_grad():
        for token, token_id in zip(tokens, token_ids, strict=True):
            name_ids = tokenizer(" " + token.strip("<|>"), add_special_tokens=False).input_ids
            for layer in (model.get_input_embeddings(), model.get_output_embeddings()):
                layer.weight[token_id] = layer.weight[name_ids].mean(dim=0)
    return token_ids


def lora_config(trainable_token_ids: list[int] | None = None) -> LoraConfig:
    """LoRA on all linear projections, plus the rows of `trainable_token_ids` in the (untied) embeddings and LM head."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        trainable_token_indices={"embed_tokens": trainable_token_ids, "lm_head": trainable_token_ids}
        if trainable_token_ids
        else None,
    )


def sft_config(output_dir: str, num_epochs: int) -> SFTConfig:
    """Training arguments shared by BONITA and the student; the checkpoint with the lowest dev loss is kept."""
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # float in [0, 1) = ratio of total steps
        bf16=True,
        completion_only_loss=True,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        seed=SEED,
    )


def train_lora(
    model,
    tokenizer,
    train_rows: list[dict],
    dev_rows: list[dict],
    output_dir: str,
    num_epochs: int,
    trainable_token_ids: list[int] | None = None,
) -> None:
    """Adds a fresh LoRA to `model`, trains it, saves the best adapter to `output_dir` and prints the dev losses."""
    transformers.set_seed(SEED)  # the LoRA weights are initialized before the Trainer sets its own seed
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config(trainable_token_ids))
    assert isinstance(model, PeftModel)  # not a PeftMixedModel (mixed=False)
    model.print_trainable_parameters()
    with tempfile.TemporaryDirectory() as ckpt_dir:  # intermediate checkpoints, deleted after training
        trainer = SFTTrainer(
            model=model,
            args=sft_config(ckpt_dir, num_epochs),
            train_dataset=Dataset.from_list(train_rows).select_columns(["prompt", "completion"]),
            eval_dataset=Dataset.from_list(dev_rows).select_columns(["prompt", "completion"]),
            processing_class=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()
    # the best checkpoint is loaded into `model`; only the LoRA and the trained token rows are saved
    # (PEFT would otherwise save the whole resized embeddings and LM head, ~1.7 GB)
    model.save_pretrained(output_dir, save_embedding_layers=False)
    tokenizer.save_pretrained(output_dir)
    print(pd.DataFrame(trainer.state.log_history).dropna(subset=["eval_loss"])[["step", "epoch", "eval_loss"]])
