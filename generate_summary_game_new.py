#!/usr/bin/env python3
# summarize_full_text.py

import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─── Configuration ─────────────────────────────────────────────────────────────

INPUT_CSV = "filtered_combined_news_text.csv"
OUTPUT_CSV = "game_news_full_summaries.csv"
MODEL_NAME = "t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load Model & Tokenizer ────────────────────────────────────────────────────

print(f"Loading model {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

# ─── Helper Functions ──────────────────────────────────────────────────────────

def chunk_text(text, max_tokens=512, stride=50):
    """
    Yield overlapping text chunks that each encode to <= max_tokens tokens.
    Uses a sliding window with `stride` overlap.
    """
    # Tokenize without truncation to get full token list
    tokens = tokenizer.encode(text, add_special_tokens=False)
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_ids = tokens[start:end]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        yield chunk
        if end == len(tokens):
            break
        start += max_tokens - stride

def summarize_chunk(chunk, max_summary_length=60, min_summary_length=20):
    """
    Summarize a single chunk of text.
    """
    input_text = "summarize: " + chunk.replace("\n", " ")
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    summary_ids = model.generate(
        **inputs,
        max_length=max_summary_length,
        min_length=min_summary_length,
        do_sample=False
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def stage1_summaries(full_text):
    """
    Perform chunk-level summarization on the full text.
    Returns a list of summaries, one per chunk.
    """
    return [summarize_chunk(c) for c in chunk_text(full_text)]

def stage2_summary(chunk_summaries, max_summary_length=80, min_summary_length=30):
    """
    Aggregate chunk-level summaries into a final summary.
    """
    combined = " ".join(chunk_summaries)
    input_text = "summarize: " + combined
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    summary_ids = model.generate(
        **inputs,
        max_length=max_summary_length,
        min_length=min_summary_length,
        do_sample=False
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Load data
    print(f"Loading input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    texts = df["full_text"].fillna("").tolist()

    # Summarize each article
    final_summaries = []
    for text in tqdm(texts, desc="Full-text summarization"):
        # Stage 1: chunk-level
        chunk_summ = stage1_summaries(text)
        # Stage 2: aggregate
        final = stage2_summary(chunk_summ)
        final_summaries.append(final)

    # Attach and save
    df["brief_summary"] = final_summaries
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved summaries to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
