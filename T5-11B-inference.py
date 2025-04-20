#!/usr/bin/env python3
# summarize_full_text_t5_11b_batched.py

import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─── Configuration ─────────────────────────────────────────────────────────────

INPUT_CSV       = "filtered_combined_news_text.csv"
OUTPUT_CSV      = "game_news_full_summaries_t5_11b_batched.csv"
MODEL_NAME      = "google/t5-11b"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE      = 512     # tokens per chunk
STRIDE          = 128     # overlap
BATCH_SIZE      = 4       # due to large model, keep batch size at 1
STAGE1_MAX_SUM  = 250
STAGE1_MIN_SUM  = 80
STAGE2_MAX_SUM  = 350
STAGE2_MIN_SUM  = 120
PREFIX          = "summarize: "

# ─── Load Model & Tokenizer ────────────────────────────────────────────────────

print(f"Loading model {MODEL_NAME} on {DEVICE} with 8-bit quantization...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

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
    summaries = []
    for chunk in chunks:
        # Prepare single-item batch with prefix
        inp = PREFIX + chunk
        inputs = tokenizer(
            inp,
            max_length=CHUNK_SIZE + len(PREFIX.split()),
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        # Generate summary
        summary_ids = model.generate(
            **inputs,
            max_length=max_sum,
            min_length=min_sum,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    return summaries


def stage1_summaries(full_text):
    chunks = list(chunk_text(full_text))
    return summarize_chunks_batched(chunks, STAGE1_MAX_SUM, STAGE1_MIN_SUM)


def stage2_summary(chunk_summaries):
    combined = " ".join(chunk_summaries)
    return summarize_chunks_batched([combined], STAGE2_MAX_SUM, STAGE2_MIN_SUM)[0]

# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Loading input CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    texts = df.get("full_text", pd.Series()).fillna("").tolist()

    final_summaries = []
    for text in tqdm(texts, desc="Full-text summarization (batched)"):
        chunk_summ = stage1_summaries(text)
        final = stage2_summary(chunk_summ)
        final_summaries.append(final)

    df["brief_summary"] = final_summaries
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved summaries to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
