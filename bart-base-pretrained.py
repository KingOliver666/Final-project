#!/usr/bin/env python3
import os
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
CHUNK_SIZE      = 512                  # tokens per chunk
STRIDE          = 128                  # overlap tokens between chunks
BATCH_SIZE      = 8                    # examples per device per step
STAGE1_MAX_SUM  = 250                  # max target length for stage 1
STAGE2_MAX_SUM  = 350                  # max target length for stage 2
EPOCHS_STAGE1   = 20                   # epochs in stage 1
EPOCHS_STAGE2   = 20                   # epochs in stage 2
MODEL_NAME      = "facebook/bart-base" # pretrained checkpoint

# ─── Preprocessing ─────────────────────────────────────────────────────────
def preprocess_function(examples, tokenizer, max_target_length):
    # ensure strings
    inputs  = [str(x) for x in examples["text"]]
    targets = [str(x) for x in examples["summary"]]

    # tokenize + fixed padding/truncation
    model_inputs = tokenizer(
        inputs,
        max_length=CHUNK_SIZE,
        padding="max_length",
        truncation=True,
        stride=STRIDE
    )
    # tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            stride=STRIDE
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# ─── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred, tokenizer, rouge, bertscore):
    preds, labels = eval_pred
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    # BERTScore
    P, R, F1 = bertscore.score(decoded_preds, decoded_labels)
    result.update({
        "bertscore_precision": P.mean().item(),
        "bertscore_recall":    R.mean().item(),
        "bertscore_f1":        F1.mean().item()
    })
    return {k: round(v,4) for k,v in result.items()}

# ─── Single‐Stage Runner ────────────────────────────────────────────────────
def run_stage(stage,
              model_init_path,
              max_summary_len,
              epochs,
              train_ds,
              val_ds,
              tokenizer,
              rouge,
              bertscore,
              data_collator):
    # initialize wandb
    wandb.init(
        project="bart_game_news",
        name=f"stage{stage}",
        reinit=True
    )
    # load or reuse model
    model = BartForConditionalGeneration.from_pretrained(model_init_path)

    # args
    args = Seq2SeqTrainingArguments(
        output_dir             = f"stage{stage}_output",
        evaluation_strategy    = "epoch",
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        num_train_epochs       = epochs,
        predict_with_generate  = True,
        logging_dir            = f"logs_stage{stage}",
        logging_steps          = 100,
        report_to              = "wandb",
    )

    # trainer
    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        data_collator   = data_collator,
        compute_metrics = lambda ep: compute_metrics(ep, tokenizer, rouge, bertscore),
    )

    # train & eval
    trainer.train()
    metrics = trainer.evaluate()
    print(f"Stage {stage} Validation Metrics:", metrics)

    # save for next stage
    outdir = f"bart_stage{stage}"
    trainer.save_model(outdir)
    return outdir

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    # 1) Load & clean data
    ds = load_dataset(
        "csv",
        data_files={
            "train": "train.csv",
            "validation": "val.csv"
        }
    )
    ds = ds.remove_columns("word_count")
    ds = ds.rename_column("full_text",    "text")
    ds = ds.rename_column("brief_summary","summary")

    # 2) Prep tokenizer, metrics, collator
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    rouge     = evaluate.load("rouge")
    bertscore = BERTScorer(lang="en", rescale_with_baseline=True)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=BartForConditionalGeneration.from_pretrained(MODEL_NAME),
        label_pad_token_id=tokenizer.pad_token_id
    )

    # 3) Tokenize for stage 1
    train_ds_1 = ds["train"].map(
        lambda x: preprocess_function(x, tokenizer, STAGE1_MAX_SUM),
        batched=True,
        remove_columns=["text","summary"]
    )
    val_ds_1   = ds["validation"].map(
        lambda x: preprocess_function(x, tokenizer, STAGE1_MAX_SUM),
        batched=True,
        remove_columns=["text","summary"]
    )

    # 4) Run stage 1
    stage1_model = run_stage(
        stage=1,
        model_init_path=MODEL_NAME,
        max_summary_len=STAGE1_MAX_SUM,
        epochs=EPOCHS_STAGE1,
        train_ds=train_ds_1,
        val_ds=val_ds_1,
        tokenizer=tokenizer,
        rouge=rouge,
        bertscore=bertscore,
        data_collator=data_collator
    )

    # 5) Tokenize for stage 2
    train_ds_2 = ds["train"].map(
        lambda x: preprocess_function(x, tokenizer, STAGE2_MAX_SUM),
        batched=True,
        remove_columns=["text","summary"]
    )
    val_ds_2   = ds["validation"].map(
        lambda x: preprocess_function(x, tokenizer, STAGE2_MAX_SUM),
        batched=True,
        remove_columns=["text","summary"]
    )

    # 6) Run stage 2
    run_stage(
        stage=2,
        model_init_path=stage1_model,
        max_summary_len=STAGE2_MAX_SUM,
        epochs=EPOCHS_STAGE2,
        train_ds=train_ds_2,
        val_ds=val_ds_2,
        tokenizer=tokenizer,
        rouge=rouge,
        bertscore=bertscore,
        data_collator=data_collator
    )

if __name__ == "__main__":
    main()
    wandb.finish()