import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
@st.cache_resource
def load_lstm():
    model = load_model("lstm_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_lstm()

# Streamlit UI
st.title("📰 Nepali News Sentiment Classifier (LSTM)")
st.write("Enter a Nepali news headline and get a Positive/Negative classification.")

headline = st.text_area("✍️ News Headline in Nepali", placeholder="Type your headline here...")

MAX_LEN = 100  # Change to the same maxlen you used during training

if st.button("Analyze"):
    if headline.strip() == "":
        st.warning("Please enter a valid headline.")
    else:
        # Tokenize and pad
        seq = tokenizer.texts_to_sequences([headline])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

        # Predict
        prediction = model.predict(padded)[0][0]

        # Interpret
        sentiment = "🌞 Positive" if prediction >= 0.5 else "🌧️ Negative"
        st.subheader(f"Sentiment: {sentiment}")
        st.caption(f"Confidence: {prediction:.2f}")


