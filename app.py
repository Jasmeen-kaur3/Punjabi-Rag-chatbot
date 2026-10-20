import streamlit as st
import pandas as pd
st.title("ਪੰਜਾਬੀ ਸ਼ਬਦਕੋਸ਼")
st.subheader("Punjabi word dictionary")
@st.cache_data
def load_data():
    df=pd.read_csv("punjabi_words.csv")
    df.columns=df.columns.str.strip()  # Strip whitespace from column names
    return df

df=load_data()
# st.title("ਪੰਜਾਬੀ ਸ਼ਬਦਕੋਸ਼")
# st.subheader("Punjabi word dictionary")

word=st.text_input("Enter your punjabi word: ")
if st.button ("Search"):
    user_word=word.strip()  # Strip whitespace from user input

    result=df[df["word"].astype(str).str.strip()==user_word]
    if not result.empty:
        st.success("Word found!")
        st.write("\nWord: ", result.iloc[0]["word"])
        st.write("Simple Punjabi: ", result.iloc[0]["simple_punjabi"])
        st.write("Hindi Meaning: ", result.iloc[0]["hindi_meaning"])
        st.write("English Meaning: ", result.iloc[0]["english_meaning"])
        st.write("Example: ", result.iloc[0]["example_sentence"])
else:
    st.error("Word not found in dataset")    