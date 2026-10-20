import unicodedata

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CSV_PATH = r"C:\\Users\\HP\\Desktop\\RAG\\data\\data\\punjabi_words.csv"
TOP_K = 3
MIN_SIMILARITY = 0.45


def normalise(text):
    return unicodedata.normalize("NFC", str(text)).strip()


def show_entry(row, heading):
    print(f"\n--- {heading} ---")
    print(f"Punjabi word: {row['word']}")
    print(f"Simple Punjabi: {row['simple_punjabi']}")
    print(f"Hindi Meaning: {row['hindi_meaning']}")
    print(f"English Meaning: {row['english_meaning']}")
    print(f"Example Sentence: {row['example_sentence']}")


df = pd.read_csv(CSV_PATH, encoding="utf-8-sig").fillna("")
df["word_clean"] = df["word"].map(normalise)
query = normalise(input("Enter your Punjabi word: "))

if not query:
    raise SystemExit("Please enter a Punjabi word.")

# Exact dictionary match: stop here, so FAISS cannot overwrite/confuse the answer.
matched = df[df["word_clean"] == query]
if not matched.empty:
    show_entry(matched.iloc[0], "Exact match found")
    raise SystemExit

# FAISS is only a fallback for words absent from the CSV.
texts = [
    f"Punjabi word: {row.word}\nSimple Punjabi: {row.simple_punjabi}\n"
    f"Hindi meaning: {row.hindi_meaning}\nEnglish meaning: {row.english_meaning}\n"
    f"Example: {row.example_sentence}"
    for row in df.itertuples(index=False)
]
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity
index.add(np.asarray(embeddings, dtype="float32"))
query_embedding = model.encode([query], normalize_embeddings=True, show_progress_bar=False)

k = min(TOP_K, len(df))
scores, indices = index.search(np.asarray(query_embedding, dtype="float32"), k=k)
if scores[0][0] < MIN_SIMILARITY:
    print("\nThis word was not found and no reliable similar match exists.")
    print("Check the spelling or add this word to punjabi_words.csv.")
else:
    print("\nExact word not found. Closest dictionary matches:")
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        show_entry(df.iloc[idx], f"Match {rank} (similarity: {score:.2f})")
