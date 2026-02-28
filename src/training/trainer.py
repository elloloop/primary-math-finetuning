from datasets import Dataset
from transformers import Trainer, TrainingArguments
from config.training_config import TRAINING_ARGS
from src.training.utils import build_data_collator
from src.training.callbacks import SimpleMetricsCallback


def tokenize_dataset(dataset: Dataset, tokenizer, max_seq_length: int):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_seq_length)
    tokenized = dataset.map(tok, batched=True, remove_columns=["text"])
    return tokenized


def run_training(model, tokenizer, train_ds: Dataset, eval_ds: Dataset, output_dir: str, overrides: dict | None = None):
    args = dict(TRAINING_ARGS)
    args["output_dir"] = output_dir
    if overrides:
        args.update(overrides)

    training_args = TrainingArguments(**args)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=build_data_collator(tokenizer),
        callbacks=[SimpleMetricsCallback()],
    )
    trainer.train(resume_from_checkpoint=overrides.get("resume_from_checkpoint") if overrides else None)
    trainer.save_model(f"{output_dir}/final")
    return trainer
