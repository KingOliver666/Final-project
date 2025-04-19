#!/usr/bin/env python3  # <- Tells the OS to run this script using Python 3 from the environment’s PATH
"""
train_bart_summarizer.py  # <- Filename and brief description

Pretrain a BART-base model on CNN/DailyMail (two-stage summary lengths)
then fine-tune on game news dataset, tracking ROUGE & BERTScore in Weights & Biases.
"""

import os
import wandb  # <- Weights & Biases for experiment tracking
from datasets import load_dataset
import evaluate  # <- Hugging Face Evaluate for metrics
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ─── Hyperparameters ────────────────────────────────────────────────────────────
CHUNK_SIZE      = 512     # <- Number of tokens per input chunk
STRIDE          = 128     # <- Overlap tokens between consecutive chunks
BATCH_SIZE      = 8       # <- Number of chunks per batch during training
STAGE1_MAX_SUM  = 250     # <- Maximum target summary length in stage 1
STAGE2_MAX_SUM  = 350     # <- Maximum target summary length in stage 2
EPOCHS_STAGE1   = 20       # <- Number of epochs for stage 1
EPOCHS_STAGE2   = 20       # <- Number of epochs for stage 2

# Initialize W&B
wandb.init(project="bart_summarization_base_PC", name="bart_two_stage_run")

# Load metrics
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")


def main():
    # ─── 1) Initialize tokenizer and model from scratch ──────────────────────────
    tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")  # <- Loads vocabulary and encoding rules
    config    = BartConfig.from_pretrained("facebook/bart-base")         # <- Loads default BART-base config
    model     = BartForConditionalGeneration(config)  # <- Instantiates model with random weights (scratch)

    # ─── 2) Load CNN/DailyMail dataset ────────────────────────────────────────────
    cnn_dataset = load_dataset("cnn_dailymail", "3.0.0")  # <- Downloads & prepares train/val/test splits

    # ─── 3) Tokenization + chunking function ─────────────────────────────────────
    def tokenize_and_chunk(batch, max_sum_len):
        inputs = tokenizer(
            batch["article"],
            max_length=CHUNK_SIZE,
            truncation=True,
            return_overflowing_tokens=True,
            stride=STRIDE,
            return_attention_mask=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["highlights"],
                max_length=max_sum_len,
                truncation=True,
            )
        n_chunks = len(inputs["input_ids"])
        labels_ids = [labels["input_ids"]] * n_chunks
        return {
            "input_ids":      inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels":         labels_ids,
        }

    tokenized_s1 = cnn_dataset.map(
        lambda b: tokenize_and_chunk(b, STAGE1_MAX_SUM),
        batched=True,
        remove_columns=cnn_dataset["train"].column_names,
    )
    tokenized_s2 = cnn_dataset.map(
        lambda b: tokenize_and_chunk(b, STAGE2_MAX_SUM),
        batched=True,
        remove_columns=cnn_dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Metric computation function
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in labels as pad_token_id
        labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Compute ROUGE
        rouge_out = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Compute BERTScore
        bertscore_out = bertscore_metric.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        # Log to W&B
        wandb.log({
            "rouge1": rouge_out["rouge1"].mid.fmeasure,
            "rouge2": rouge_out["rouge2"].mid.fmeasure,
            "rougeL": rouge_out["rougeL"].mid.fmeasure,
            "bertscore_precision": sum(bertscore_out["precision"]) / len(bertscore_out["precision"]),
            "bertscore_recall": sum(bertscore_out["recall"]) / len(bertscore_out["recall"]),
            "bertscore_f1": sum(bertscore_out["f1"]) / len(bertscore_out["f1"]),
        })
        # Return combined metrics
        return {
            "rouge1": rouge_out["rouge1"].mid.fmeasure,
            "rouge2": rouge_out["rouge2"].mid.fmeasure,
            "rougeL": rouge_out["rougeL"].mid.fmeasure,
            "bertscore_f1": sum(bertscore_out["f1"]) / len(bertscore_out["f1"]),
        }

    # Stage training function to avoid duplication
    def stage_train(tokenized_data, output_dir, epochs):
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            evaluation_strategy="steps",
            eval_steps=5000,
            save_steps=5000,
            num_train_epochs=epochs,
            fp16=True,
            gradient_accumulation_steps=2,
            logging_dir=f"logs_{output_dir}",
            report_to="wandb",  # <- Enable W&B reporting
            logging_steps=1000,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()

    # Run Stage 1 & 2
    stage_train(tokenized_s1, "bart_cnn_stage1", EPOCHS_STAGE1)
    stage_train(tokenized_s2, "bart_cnn_stage2", EPOCHS_STAGE2)

    # Save pretrained checkpoint
    model.save_pretrained("bart_cnn_pretrained")
    tokenizer.save_pretrained("bart_cnn_pretrained")

    # Fine-tune on game news data
    game_news = load_dataset("csv", data_files={"train": "train.csv", "validation": "val.csv"})

    def preprocess_game(batch):
        inputs = tokenizer(batch["text"], max_length=CHUNK_SIZE, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["summary"], max_length=STAGE2_MAX_SUM, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_game = game_news.map(
        preprocess_game,
        batched=True,
        remove_columns=game_news["train"].column_names,
    )

    args_game = TrainingArguments(
        output_dir="bart_game_news_finetuned",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        num_train_epochs=3,
        fp16=True,
        gradient_accumulation_steps=2,
        logging_dir="logs_game_news",
        report_to="wandb",  # <- Enable W&B reporting
        logging_steps=500,
    )
    trainer_game = Trainer(
        model=model,
        args=args_game,
        train_dataset=tokenized_game["train"],
        eval_dataset=tokenized_game["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer_game.train()
    trainer_game.save_model("bart_game_news_final")


if __name__ == "__main__":
    main()  # <- Entry point: run all steps when the script is executed
