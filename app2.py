"""Punjabi dictionary RAG chatbot.

Run with: streamlit run app2.py
"""
import re
import unicodedata
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import io
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "data" / "punjabi_words.csv"
MIN_SIMILARITY = 0.75
TOP_K = 3


def normalise(text):
    # Keyboard/input tools can insert zero-width Unicode characters. They look
    # invisible but prevent an otherwise identical Punjabi word from matching.
    text = unicodedata.normalize("NFKC", str(text))
    return "".join(char for char in text if unicodedata.category(char) != "Cf").strip()


def punjabi_tokens(text):
    """Return Punjabi words from a mixed English/Hindi/Punjabi question."""
    return [normalise(token) for token in re.findall(r"[\u0A00-\u0A7F]+", text)]


@st.cache_resource(show_spinner="Dictionary model is loading...")
def load_retriever():
    df = pd.read_csv(r"C:\Users\HP\Desktop\RAG\data\data\punjabi_words.csv", encoding="utf-8-sig").fillna("")
    df["word_clean"] = df["word"].map(normalise)
    documents = [
        f"Punjabi word: {row.word}\nSimple Punjabi: {row.simple_punjabi}\n"
        f"Hindi meaning: {row.hindi_meaning}\nEnglish meaning: {row.english_meaning}\n"
        f"Example: {row.example_sentence}"
        for row in df.itertuples(index=False)
    ]
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype="float32"))
    return df, model, index


def entry_answer(row, confidence=None):
    prefix = ""
    if confidence is not None:
        prefix = f"Closest match (confidence: {confidence:.0%})\n\n"
    return (
        f"{prefix}**Punjabi word:** {row['word']}  \n"
        f"**Simple Punjabi:** {row['simple_punjabi']}  \n"
        f"**Hindi meaning:** {row['hindi_meaning']}  \n"
        f"**English meaning:** {row['english_meaning']}  \n"
        f"**Example:** {row['example_sentence']}"
    )


def answer(question, df, model, index):

    query = normalise(question)
    if not query:
        return "Please type a Punjabi word or a question.", []

    # Find exact Punjabi tokens before semantic search. This makes mixed prompts
    # such as "explain meaning of ਨਿਰੰਤਰ?" return the dictionary entry itself.
    query_words = set(punjabi_tokens(query))

    exact = df[
    df["word_clean"].apply(
        lambda dictionary_word: any(
            normalise(dictionary_word) == normalise(token)
            for token in query_words
        )
    )
]

    if exact.empty:
        exact = df[
        df["word_clean"].apply(
            lambda word: word and normalise(word) in query
        )
    ]
    if not exact.empty:
        return entry_answer(exact.iloc[0]), []

    # Also allow searches using a Hindi or English meaning.
    lower_query = query.casefold()
    meaning_match = df[
        df["hindi_meaning"].astype(str).str.casefold().str.contains(lower_query, regex=False)
        | df["english_meaning"].astype(str).str.casefold().str.contains(lower_query, regex=False)
    ]
    if not meaning_match.empty:
        return entry_answer(meaning_match.iloc[0]), []

    embedding = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    scores, indices = index.search(
        np.asarray(embedding, dtype="float32"), k=min(TOP_K, len(df))
    )
    best_score = float(scores[0][0])
    if best_score < MIN_SIMILARITY:
        return (
            "Mujhe is dictionary mein reliable answer nahi mila. "
            "Spelling check karein ya word ko CSV mein add karein.",
            [],
        )

    suggestions = [
        f"{df.iloc[idx]['word']} ({score:.0%})"
        for score, idx in zip(scores[0], indices[0])
    ]
    return entry_answer(df.iloc[indices[0][0]], best_score), suggestions


def create_audio(text):
    """Create Punjabi MP3 audio only when the user presses Listen."""
    speech_text = re.sub(r"\*\*", "", text).replace("  \n", ". ")
    audio_buffer = io.BytesIO()
    gTTS(text=speech_text, lang="pa", slow=False).write_to_fp(audio_buffer)
    return audio_buffer.getvalue()


def add_question(question, df, model, index):
    """Store one user question and its CSV-backed answer in chat history."""
    question = normalise(question)
    if not question:
        return
    response, suggestions = answer(question, df, model, index)
    st.session_state.messages.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": response, "suggestions": suggestions},
    ])


st.set_page_config(page_title="Punjabi RAG Chatbot", page_icon="💬")
st.title("ਪੰਜਾਬੀ RAG Chatbot")
st.caption("Type or speak a Punjabi word. Answers come only from your CSV dictionary.")

df, model, index = load_retriever()
if "messages" not in st.session_state:
    st.session_state.messages = []

for number, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("suggestions"):
            st.caption("Other possible matches: " + " • ".join(message["suggestions"]))

        # Only chatbot answers get the speaker control.
        if message["role"] == "assistant":
            if st.button("🔊 Listen", key=f"listen_{number}"):
                try:
                    message["audio"] = create_audio(message["content"])
                except Exception:
                    st.warning("Voice output could not be generated. Check your internet connection and try again.")
            if message.get("audio"):
                st.audio(message["audio"], format="audio/mp3")

# One search bar: write text, speak with mic, or press Send.
text_column, mic_column, send_column = st.columns([7, 2, 1])
with text_column:
    typed_text = st.text_input(
        "Punjabi word or question",
        placeholder="ਪੰਜਾਬੀ ਸ਼ਬਦ ਜਾਂ ਸਵਾਲ ਲਿਖੋ...",
        label_visibility="collapsed",
    )
with mic_column:
    spoken_text = speech_to_text(
        language="pa-IN",
        start_prompt="🎤 Speak",
        stop_prompt="⏹ Stop",
        just_once=True,
        use_container_width=True,
        key="punjabi_voice_input",
    )
with send_column:
    send_clicked = st.button("Send", use_container_width=True)

if spoken_text:
    add_question(spoken_text, df, model, index)
    st.rerun()
if send_clicked and typed_text:
    add_question(typed_text, df, model, index)
    st.rerun()

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
