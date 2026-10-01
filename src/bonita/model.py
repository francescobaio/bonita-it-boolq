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
    """4-bit model with the BONITA adapter, and its tokenizer (with the special tokens) padded on the left."""
    tokenizer = AutoTokenizer.from_pretrained(bonita_dir)
    tokenizer.padding_side = "left"  # batched generation: every prompt must end right before the new tokens
    model = PeftModel.from_pretrained(load_model(tokenizer), bonita_dir)
    model.eval()
    return model, tokenizer


def lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def sft_config(output_dir: str, num_epochs: int) -> SFTConfig:
    """Training arguments shared by BONITA and the student; the checkpoint with the lowest dev loss is kept."""
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # float in [0, 1) = ratio of total steps
        bf16=True,
        completion_only_loss=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",  # must match eval_strategy for load_best_model_at_end
        save_steps=100,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=25,
        report_to="tensorboard",
        seed=SEED,
    )


def train_lora(model, tokenizer, train_rows: list[dict], dev_rows: list[dict], output_dir: str, num_epochs: int) -> None:
    """Adds a fresh LoRA to `model`, trains it, saves the best adapter to `output_dir` and prints the dev losses."""
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config())
    model.print_trainable_parameters()
    trainer = SFTTrainer(
        model=model,  # type: ignore
        args=sft_config(f"outputs_{output_dir}", num_epochs),
        train_dataset=Dataset.from_list(train_rows).select_columns(["prompt", "completion"]),
        eval_dataset=Dataset.from_list(dev_rows).select_columns(["prompt", "completion"]),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(pd.DataFrame(trainer.state.log_history).dropna(subset=["eval_loss"])[["step", "epoch", "eval_loss"]])
