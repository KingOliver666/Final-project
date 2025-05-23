#!/usr/bin/env python3
"""
Two-stage fine‑tuning of google/pegasus-large with minimal checkpoints.
Adapted to avoid FP16 unscale errors by disabling mixed precision.
"""
import torch
from datasets import load_dataset
from transformers import (
    PegasusTokenizer,
    PegasusForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
from bert_score import BERTScorer
import wandb

# ─── Configuration & Device ─────────────────────────────────────────────────
MODEL_NAME      = "google/pegasus-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Hyperparameters ────────────────────────────────────────────────────────
CHUNK_SIZE      = 512   # max input tokens for Pegasus
BATCH_SIZE      = 1      # per-device batch size
STAGE1_MAX_SUM  = 250    # summary length for stage 1
STAGE2_MAX_SUM  = 350    # summary length for stage 2
EPOCHS_STAGE1   = 20     # epochs in stage 1
EPOCHS_STAGE2   = 20     # epochs in stage 2

# ─── Preprocessing Function ─────────────────────────────────────────────────
def preprocess_function(examples, tokenizer, max_target_length):
    texts = [str(x) for x in examples.get("text", [])]
    summaries = [str(x) for x in examples.get("summary", [])]
    inputs = tokenizer(
        texts,
        max_length=CHUNK_SIZE,
        padding="max_length",
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            summaries,
            max_length=max_target_length,
            padding="max_length",
            truncation=True
        )
    inputs["labels"] = labels["input_ids"]
    return inputs

# ─── Metrics Function ────────────────────────────────────────────────────────
def compute_metrics(eval_pred, tokenizer, rouge, bertscore):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    P, R, F1 = bertscore.score(decoded_preds, decoded_labels)
    result.update({
        "bertscore_precision": P.mean().item(),
        "bertscore_recall":    R.mean().item(),
        "bertscore_f1":        F1.mean().item()
    })
    return {k: round(v,4) for k,v in result.items()}

# ─── Stage Runner ──────────────────────────────────────────────────────────
def run_stage(stage, init_model, max_sum_len, epochs,
              train_ds, val_ds,
              tokenizer, rouge, bertscore, data_collator):
    wandb.init(project="pegasus_game_news", name=f"pegasus_stage{stage}", reinit=True)
    # load full precision
    model = PegasusForConditionalGeneration.from_pretrained(init_model).to(device)

    steps_per_epoch = max(1, len(train_ds) // BATCH_SIZE)
    save_steps = steps_per_epoch * epochs

    args = Seq2SeqTrainingArguments(
        output_dir=f"pegasus_stage{stage}_output",
        do_train=True,
        do_eval=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_steps=steps_per_epoch,
        save_steps=save_steps,
        save_total_limit=1,
        logging_steps=steps_per_epoch,
        report_to="wandb",
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, rouge, bertscore),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Stage {stage} validation metrics:\n{metrics}")
    ckpt_dir = f"pegasus_stage{stage}"
    trainer.save_model(ckpt_dir)
    return ckpt_dir

# ─── Main Pipeline ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load & prepare dataset
    ds = load_dataset("csv", data_files={"train":"train.csv","validation":"val.csv"})
    ds = ds.remove_columns("word_count")
    ds = ds.rename_column("full_text","text")
    ds = ds.rename_column("brief_summary","summary")

    # Initialize tokenizer, metrics, collator
    tokenizer  = PegasusTokenizer.from_pretrained(MODEL_NAME)
    rouge      = evaluate.load("rouge")
    bertscore  = BERTScorer(lang="en", rescale_with_baseline=False)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=tokenizer.pad_token_id
    )

    # Stage 1 tokenization
    train_ds1 = ds["train"].map(
        lambda x: preprocess_function(x, tokenizer, STAGE1_MAX_SUM),
        batched=True,
        remove_columns=["text","summary"]
    )
    val_ds1 = ds["validation"].map(
        lambda x: preprocess_function(x, tokenizer, STAGE1_MAX_SUM),
        batched=True,
        remove_columns=["text","summary"]
    )

    # Run Stage 1
    ckpt1 = run_stage(
        stage=1,
        init_model=MODEL_NAME,
        max_sum_len=STAGE1_MAX_SUM,
        epochs=EPOCHS_STAGE1,
        train_ds=train_ds1,
        val_ds=val_ds1,
        tokenizer=tokenizer,
        rouge=rouge,
        bertscore=bertscore,
        data_collator=data_collator
    )

    # Stage 2 tokenization
    train_ds2 = ds["train"].map(
        lambda x: preprocess_function(x, tokenizer, STAGE2_MAX_SUM),
        batched=True,
        remove_columns=["text","summary"]
    )
    val_ds2 = ds["validation"].map(
        lambda x: preprocess_function(x, tokenizer, STAGE2_MAX_SUM),
        batched=True,
        remove_columns=["text","summary"]
    )

    # Run Stage 2
    run_stage(
        stage=2,
        init_model=ckpt1,
        max_sum_len=STAGE2_MAX_SUM,
        epochs=EPOCHS_STAGE2,
        train_ds=train_ds2,
        val_ds=val_ds2,
        tokenizer=tokenizer,
        rouge=rouge,
        bertscore=bertscore,
        data_collator=data_collator
    )
