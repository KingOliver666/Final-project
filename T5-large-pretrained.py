#!/usr/bin/env python3
"""
Two‑stage fine‑tuning of T5‑large on your CSV dataset.
"""

import torch
import evaluate
import wandb

from datasets import load_dataset
from bert_score import BERTScorer
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# ─── Configuration & Device ────────────────────────────────────────────────────
MODEL_NAME      = "t5-large"
device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Hyperparameters ────────────────────────────────────────────────────────────
CHUNK_SIZE      = 512   # max input tokens
BATCH_SIZE      = 16     # per‑device batch size
STAGE1_MAX_SUM  = 250   # summary length for stage 1
STAGE2_MAX_SUM  = 350   # summary length for stage 2
EPOCHS_STAGE1   = 20
EPOCHS_STAGE2   = 20

# ─── Preprocessing Function ────────────────────────────────────────────────────
def preprocess_function(examples, tokenizer, max_target_length):
    inputs = tokenizer(
        [str(x) for x in examples.get("text", [])],
        max_length=CHUNK_SIZE,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        [str(x) for x in examples.get("summary", [])],
        max_length=max_target_length,
        padding="max_length",
        truncation=True,
    )["input_ids"]
    inputs["labels"] = labels
    return inputs

# ─── Metrics Function ──────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        use_stemmer=True
    )
    P, R, F1 = bertscore.score(decoded_preds, decoded_labels)
    result.update({
        "bertscore_precision": P.mean().item(),
        "bertscore_recall":    R.mean().item(),
        "bertscore_f1":        F1.mean().item(),
    })
    return {k: round(v, 4) for k, v in result.items()}

# ─── Stage Runner ──────────────────────────────────────────────────────────────
def run_stage(stage, init_model, epochs, train_ds, val_ds):
    wandb.init(project="t5_large_game_news", name=f"t5_stage{stage}", reinit=True)
    model = T5ForConditionalGeneration.from_pretrained(init_model).to(device)

    steps_per_epoch = max(1, len(train_ds) // BATCH_SIZE)
    save_steps      = steps_per_epoch * epochs

    args = Seq2SeqTrainingArguments(
        output_dir=f"t5_stage{stage}_output",
        do_train=True,
        do_eval=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_steps=steps_per_epoch,
        save_steps=save_steps,
        save_total_limit=1,
        logging_steps=steps_per_epoch,
        predict_with_generate=True,
        report_to="wandb",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nStage {stage} validation metrics:\n", metrics)
    ckpt_dir = f"t5_stage{stage}"
    trainer.save_model(ckpt_dir)
    return ckpt_dir

# ─── Main Pipeline ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Load & prepare dataset
    ds = load_dataset(
        "csv", 
        data_files={"train": "train.csv", "validation": "val.csv"}
    )
    ds = ds.remove_columns("word_count")
    ds = ds.rename_column("full_text",   "text")
    ds = ds.rename_column("brief_summary","summary")

    # 2) Initialize tokenizer, metrics, and collator
    tokenizer     = T5Tokenizer.from_pretrained(MODEL_NAME)
    rouge         = evaluate.load("rouge")
    bertscore     = BERTScorer(lang="en", rescale_with_baseline=False)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=tokenizer.pad_token_id
    )

    # 3) Record original column names so we can drop them after tokenization
    original_columns = ds["train"].column_names  # should be ["text","summary"]

    # 4) Tokenize for Stage 1
    train_ds1 = ds["train"].map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_target_length": STAGE1_MAX_SUM},
        remove_columns=original_columns,
    )
    val_ds1 = ds["validation"].map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_target_length": STAGE1_MAX_SUM},
        remove_columns=original_columns,
    )

    # 5) Run Stage 1
    ckpt1 = run_stage(
        stage=1,
        init_model=MODEL_NAME,
        epochs=EPOCHS_STAGE1,
        train_ds=train_ds1,
        val_ds=val_ds1,
    )

    # 6) Tokenize for Stage 2
    train_ds2 = ds["train"].map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_target_length": STAGE2_MAX_SUM},
        remove_columns=original_columns,
    )
    val_ds2 = ds["validation"].map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_target_length": STAGE2_MAX_SUM},
        remove_columns=original_columns,
    )

    # 7) Run Stage 2
    run_stage(
        stage=2,
        init_model=ckpt1,
        epochs=EPOCHS_STAGE2,
        train_ds=train_ds2,
        val_ds=val_ds2,
    )
