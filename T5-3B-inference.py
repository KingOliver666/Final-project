#!/usr/bin/env python3
import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV       = "filtered_combined_news_text.csv"
OUTPUT_CSV      = "game_news_full_summaries_t5_3b_batched.csv"
MODEL_NAME      = "t5-3b"  # HF hub id for T5‑3B
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# chunking
CHUNK_SIZE      = 512
STRIDE          = 128
BATCH_SIZE      = 8       # keep this at 1 for minimal VRAM

# two‑stage lengths
STAGE1_MAX_SUM  = 250
STAGE1_MIN_SUM  = 80
STAGE2_MAX_SUM  = 350
STAGE2_MIN_SUM  = 120

PREFIX          = "summarize: "

# ─── Quantization / Model Loading ────────────────────────────────────────────
print(f"Loading {MODEL_NAME} in 8‑bit on {DEVICE}…")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# ─── Helper Functions ─────────────────────────────────────────────────────────
def chunk_text(text):
    toks = tokenizer.encode(text, add_special_tokens=False)
    start = 0
    while start < len(toks):
        end = min(start + CHUNK_SIZE, len(toks))
        yield tokenizer.decode(toks[start:end], skip_special_tokens=True)
        if end == len(toks):
            break
        start += CHUNK_SIZE - STRIDE

def summarize_chunks(chunks, max_len, min_len):
    outs = []
    for chunk in chunks:
        inp = PREFIX + chunk
        inputs = tokenizer(
            inp,
            max_length=CHUNK_SIZE + len(PREFIX),
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)
        ids = model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        outs.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    return outs

def stage1_summaries(text):
    return summarize_chunks(chunk_text(text), STAGE1_MAX_SUM, STAGE1_MIN_SUM)

def stage2_summary(chunks_out):
    combined = " ".join(chunks_out)
    return summarize_chunks([combined], STAGE2_MAX_SUM, STAGE2_MIN_SUM)[0]

# ─── Main Pipeline ────────────────────────────────────────────────────────────
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    print(f"Reading {INPUT_CSV}…")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    texts = df.get("full_text", pd.Series()).fillna("").tolist()

    summaries = []
    for txt in tqdm(texts, desc="Summarizing"):
        s1 = stage1_summaries(txt)
        s2 = stage2_summary(s1)
        summaries.append(s2)

    df["brief_summary"] = summaries
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved summaries to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
