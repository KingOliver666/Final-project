#!/usr/bin/env python3
import torch
from datasets import load_dataset
from transformers import (
    T5TokenizerFast as T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
from bert_score import BERTScorer
import wandb

# ─── Hyperparameters ───────────────────────────────────────────────────────
CHUNK_SIZE      = 512               # tokens per chunk
BATCH_SIZE      = 8                 # examples per device per step
STAGE1_MAX_SUM  = 250               # max summary length for stage1
STAGE2_MAX_SUM  = 350               # max summary length for stage2
EPOCHS_STAGE1   = 20                # epochs in stage1
EPOCHS_STAGE2   = 20                # epochs in stage2
MODEL_NAME      = "t5-small"       # pretrained checkpoint

# ─── Preprocessing ─────────────────────────────────────────────────────────
def preprocess_function(examples, tokenizer, max_target_length):
    # prefix for T5 summarization
    inputs = ["summarize: " + str(x) for x in examples.get("text", [])]
    targets = [str(x) for x in examples.get("summary", [])]
    model_inputs = tokenizer(
        inputs,
        max_length=CHUNK_SIZE,
        padding="max_length",
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ─── Metrics ────────────────────────────────────────────────────────────────
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

# ─── Stage Runner ───────────────────────────────────────────────────────────
def run_stage(stage, init_model, max_sum_len, epochs,
              train_dataset, val_dataset,
              tokenizer, rouge, bertscore, data_collator):
    wandb.init(project="t5_game_news", name=f"t5_stage{stage}", reinit=True)
    model = T5ForConditionalGeneration.from_pretrained(init_model)

    steps_per_epoch = max(1, len(train_dataset) // BATCH_SIZE)
    save_steps = steps_per_epoch * epochs

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
        report_to="wandb",
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer, rouge, bertscore),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Stage {stage} validation metrics:", metrics)
    checkpoint_dir = f"t5_stage{stage}"
    trainer.save_model(checkpoint_dir)
    return checkpoint_dir

# ─── Main Pipeline ─────────────────────────────────────────────────────────
def main():
    # Load and clean dataset
    ds = load_dataset("csv", data_files={"train": "train.csv", "validation": "val.csv"})
    ds = ds.remove_columns("word_count")
    ds = ds.rename_column("full_text", "text")
    ds = ds.rename_column("brief_summary", "summary")

    # Initialize tokenizer, metrics, and data collator
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    rouge     = evaluate.load("rouge")
    bertscore = BERTScorer(lang="en", rescale_with_baseline=False)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=T5ForConditionalGeneration.from_pretrained(MODEL_NAME),
        label_pad_token_id=tokenizer.pad_token_id
    )

    # Stage 1 tokenization
    train_ds1 = ds["train"].map(
        lambda x: preprocess_function(x, tokenizer, STAGE1_MAX_SUM),
        batched=True,
        remove_columns=["text", "summary"]
    )
    val_ds1 = ds["validation"].map(
        lambda x: preprocess_function(x, tokenizer, STAGE1_MAX_SUM),
        batched=True,
        remove_columns=["text", "summary"]
    )

    # Run Stage 1
    ckpt1 = run_stage(
        stage=1,
        init_model=MODEL_NAME,
        max_sum_len=STAGE1_MAX_SUM,
        epochs=EPOCHS_STAGE1,
        train_dataset=train_ds1,
        val_dataset=val_ds1,
        tokenizer=tokenizer,
        rouge=rouge,
        bertscore=bertscore,
        data_collator=data_collator
    )

    # Stage 2 tokenization
    train_ds2 = ds["train"].map(
        lambda x: preprocess_function(x, tokenizer, STAGE2_MAX_SUM),
        batched=True,
        remove_columns=["text", "summary"]
    )
    val_ds2 = ds["validation"].map(
        lambda x: preprocess_function(x, tokenizer, STAGE2_MAX_SUM),
        batched=True,
        remove_columns=["text", "summary"]
    )

    # Run Stage 2
    run_stage(
        stage=2,
        init_model=ckpt1,
        max_sum_len=STAGE2_MAX_SUM,
        epochs=EPOCHS_STAGE2,
        train_dataset=train_ds2,
        val_dataset=val_ds2,
        tokenizer=tokenizer,
        rouge=rouge,
        bertscore=bertscore,
        data_collator=data_collator
    )

if __name__ == "__main__":
    main()
