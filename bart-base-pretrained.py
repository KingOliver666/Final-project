#!/usr/bin/env python3

import torch
from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
from bert_score import BERTScorer
import wandb

# ─── Hyperparameters ───────────────────────────────────────────────────────
CHUNK_SIZE      = 512
STRIDE          = 128
BATCH_SIZE      = 8
STAGE1_MAX_SUM  = 250
STAGE2_MAX_SUM  = 350
EPOCHS_STAGE1   = 20
EPOCHS_STAGE2   = 20
MODEL_NAME      = "facebook/bart-base"

# ─── Preprocess ────────────────────────────────────────────────────────────
def preprocess_fn(examples, tokenizer, max_len):
    inputs  = [str(x) for x in examples["text"]]
    targets = [str(x) for x in examples["summary"]]
    model_inputs = tokenizer(
        inputs,
        max_length=CHUNK_SIZE,
        padding="max_length",
        truncation=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_len,
            padding="max_length",
            truncation=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ─── Compute & Log Metrics ─────────────────────────────────────────────────
def compute_and_log(eval_pred, tokenizer, rouge, bertscore):
    preds, labels = eval_pred
    dec_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # compute
    result = rouge.compute(predictions=dec_preds, references=dec_labels)
    P, R, F1 = bertscore.score(dec_preds, dec_labels)
    result.update({
        "bertscore_precision": P.mean().item(),
        "bertscore_recall":    R.mean().item(),
        "bertscore_f1":        F1.mean().item()
    })
    # log to W&B
    wandb.log(result)
    return {k: round(v,4) for k,v in result.items()}

# ─── Stage Runner ──────────────────────────────────────────────────────────
def run_stage(stage, init_model, max_sum, epochs, train_ds, val_ds,
              tokenizer, rouge, bertscore, collator):
    wandb.init(project="bart_game_news", name=f"bart_base_stage{stage}", reinit=True)
    model = BartForConditionalGeneration.from_pretrained(init_model)

    steps_per_epoch = max(1, len(train_ds) // BATCH_SIZE)
    save_steps = steps_per_epoch * epochs

    args = Seq2SeqTrainingArguments(
        output_dir=f"bart_base_stage{stage}_output",
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
        data_collator=collator,
        compute_metrics=lambda p: compute_and_log(p, tokenizer, rouge, bertscore),
    )

    trainer.train()
    # final eval & log
    final_metrics = trainer.evaluate()
    wandb.log(final_metrics)
    print(f"Stage {stage} final metrics:", final_metrics)

    ckpt = f"bart_stage{stage}"
    trainer.save_model(ckpt)
    return ckpt

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    # load & clean
    ds = load_dataset("csv", data_files={"train":"train.csv","validation":"val.csv"})
    ds = ds.remove_columns("word_count")
    ds = ds.rename_column("full_text","text")
    ds = ds.rename_column("brief_summary","summary")

    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    rouge     = evaluate.load("rouge")
    bertscore = BERTScorer(lang="en", rescale_with_baseline=False)
    collator  = DataCollatorForSeq2Seq(
        tokenizer,
        model=BartForConditionalGeneration.from_pretrained(MODEL_NAME),
        label_pad_token_id=tokenizer.pad_token_id
    )

    # Stage 1 tokenization
    train1 = ds["train"].map(
        lambda x: preprocess_fn(x, tokenizer, STAGE1_MAX_SUM),
        batched=True, remove_columns=["text","summary"]
    )
    val1   = ds["validation"].map(
        lambda x: preprocess_fn(x, tokenizer, STAGE1_MAX_SUM),
        batched=True, remove_columns=["text","summary"]
    )

    m1 = run_stage(
        stage=1,
        init_model=MODEL_NAME,
        max_sum=STAGE1_MAX_SUM,
        epochs=EPOCHS_STAGE1,
        train_ds=train1,
        val_ds=val1,
        tokenizer=tokenizer,
        rouge=rouge,
        bertscore=bertscore,
        collator=collator
    )

    # Stage 2 tokenization
    train2 = ds["train"].map(
        lambda x: preprocess_fn(x, tokenizer, STAGE2_MAX_SUM),
        batched=True, remove_columns=["text","summary"]
    )
    val2   = ds["validation"].map(
        lambda x: preprocess_fn(x, tokenizer, STAGE2_MAX_SUM),
        batched=True, remove_columns=["text","summary"]
    )

    run_stage(
        stage=2,
        init_model=m1,
        max_sum=STAGE2_MAX_SUM,
        epochs=EPOCHS_STAGE2,
        train_ds=train2,
        val_ds=val2,
        tokenizer=tokenizer,
        rouge=rouge,
        bertscore=bertscore,
        collator=collator
    )

if __name__ == "__main__":
    main()
    wandb.finish()