#!/usr/bin/env python3
# summarize_full_text_bart_batched.py

import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─── Configuration ─────────────────────────────────────────────────────────────

INPUT_CSV       = "filtered_combined_news_text.csv"
OUTPUT_CSV      = "game_news_full_summaries_bart_batched.csv"
MODEL_NAME      = "facebook/bart-large-cnn"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE      = 1024     # tokens per chunk
STRIDE          = 128      # overlap
BATCH_SIZE      = 4        # number of chunks per batch
STAGE1_MAX_SUM  = 150
STAGE1_MIN_SUM  = 40
STAGE2_MAX_SUM  = 200
STAGE2_MIN_SUM  = 60

# ─── Load Model & Tokenizer ────────────────────────────────────────────────────

print(f"Loading model {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

# ─── Helper Functions ──────────────────────────────────────────────────────────

def chunk_text(text, max_tokens=CHUNK_SIZE, stride=STRIDE):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        yield tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        if end == len(tokens):
            break
        start += max_tokens - stride

def summarize_chunks_batched(chunks, max_sum, min_sum):
    """
    Summarize a list of chunks in batches of BATCH_SIZE.
    Returns a list of summaries in the same order.
    """
    summaries = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        # Tokenize batch
        inputs = tokenizer(
            batch,
            max_length=CHUNK_SIZE,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(DEVICE)
        # Generate summaries
        summary_ids = model.generate(
            **inputs,
            max_length=max_sum,
            min_length=min_sum,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        # Decode batch
        batch_summaries = [
            tokenizer.decode(g, skip_special_tokens=True)
            for g in summary_ids
        ]
        summaries.extend(batch_summaries)
    return summaries

def stage1_summaries(full_text):
    chunks = list(chunk_text(full_text))
    return summarize_chunks_batched(chunks, STAGE1_MAX_SUM, STAGE1_MIN_SUM)

def stage2_summary(chunk_summaries):
    combined = " ".join(chunk_summaries)
    # We can treat the entire combined text as one chunk for final pass
    final = summarize_chunks_batched([combined], STAGE2_MAX_SUM, STAGE2_MIN_SUM)
    return final[0]

# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    # only try to make a directory if OUTPUT_CSV contains one
    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    texts = df["full_text"].fillna("").tolist()

    final_summaries = []
    for text in tqdm(texts, desc="Full-text summarization (batched)"):
        # Stage 1: chunk-level batched
        chunk_summ = stage1_summaries(text)
        # Stage 2: aggregate
        final = stage2_summary(chunk_summ)
        final_summaries.append(final)

    df["brief_summary"] = final_summaries
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved summaries to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
