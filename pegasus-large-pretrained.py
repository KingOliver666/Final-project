#!/usr/bin/env python3
import torch
from datasets import load_dataset
from transformers import (
    PegasusTokenizerFast as PegasusTokenizer,
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
CHUNK_SIZE      = 512   # tokens per input chunk (max for PEGASUS)
BATCH_SIZE      = 1      # examples per device
STAGE1_MAX_SUM  = 250    # max summary length for stage1
STAGE2_MAX_SUM  = 350    # max summary length for stage2
EPOCHS_STAGE1   = 20     # epochs in stage1
EPOCHS_STAGE2   = 20     # epochs in stage2

# ─── Preprocessing Function ─────────────────────────────────────────────────
def preprocess_function(examples, tokenizer, max_target_length):
    inputs = [str(x) for x in examples.get("text", [])]
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
    return {k: round(v,4) for k, v in result.items()}

# ─── Stage Runner ──────────────────────────────────────────────────────────
def run_stage(stage, init_model, max_sum_len, epochs,
              train_ds, val_ds,
              tokenizer, rouge, bertscore, data_collator):
    wandb.init(project="pegasus_game_news", name=f"pegasus_stage{stage}", reinit=True)
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
    print(f"Stage {stage} validation metrics:\n", metrics)
    checkpoint = f"pegasus_stage{stage}"
    trainer.save_model(checkpoint)
    return checkpoint

# ─── Main Pipeline ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Load & preprocess dataset
    ds = load_dataset("csv", data_files={"train": "train.csv", "validation": "val.csv"})
    ds = ds.remove_columns("word_count")
    ds = ds.rename_column("full_text", "text")
    ds = ds.rename_column("brief_summary", "summary")

    # 2) Initialize tokenizer, metrics, collator
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    rouge     = evaluate.load("rouge")
    bertscore = BERTScorer(lang="en", rescale_with_baseline=False)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=PegasusForConditionalGeneration.from_pretrained(MODEL_NAME).to(device),
        label_pad_token_id=tokenizer.pad_token_id
    )

    # 3) Tokenize Stage 1
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

    # 4) Run Stage 1
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

    # 5) Tokenize Stage 2
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

    # 6) Run Stage 2
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
    wandb.finish()