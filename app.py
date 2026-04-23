import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load model & vectorizer (we'll save them first)
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# UI
st.title("🎬 Sentiment Analysis App")

user_input = st.text_area("Enter a review:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)

    if result[0] == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😡")