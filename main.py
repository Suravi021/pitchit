# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import textstat
import uvicorn
import joblib


stop_words = set(stopwords.words("english"))

# -------------------------------
# Feature Extraction Function
# -------------------------------
def extract_features(text):
    text = text.strip()
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    
    num_words = len(words)
    num_sentences = len(sentences)

    avg_sentence_len = num_words / num_sentences if num_sentences > 0 else 0
    avg_word_len = np.mean([len(w) for w in words]) if words else 0

    stopword_ratio = sum(1 for w in words if w.lower() in stop_words) / num_words if num_words else 0
    uppercase_ratio = sum(1 for w in words if w.isupper()) / num_words if num_words else 0

    punctuation_count = sum(text.count(p) for p in ".,;:!?")
    punctuation_ratio = punctuation_count / len(text) if len(text) else 0

    unique_words = set(words)
    unique_word_ratio = len(unique_words) / num_words if num_words else 0
    type_token_ratio = unique_word_ratio

    freq = {}
    for w in words:
        w = w.lower()
        freq[w] = freq.get(w, 0) + 1
    hapax_ratio = sum(1 for w, c in freq.items() if c == 1) / num_words if num_words else 0

    char_lengths = [len(w) for w in words]
    burstiness = np.var(char_lengths) if char_lengths else 0

    digit_ratio = sum(c.isdigit() for c in text) / len(text) if len(text) > 0 else 0
    symbol_ratio = sum(c in "@#$%^&*(){}[]" for c in text) / len(text) if len(text) > 0 else 0

    flesch = textstat.flesch_reading_ease(text)
    coleman = textstat.coleman_liau_index(text)
    smog = textstat.smog_index(text)

    llm_patterns = [
        "as an ai", "as a language model", 
        "let me break it down", "in conclusion", 
        "overall", "step-by-step",
        "here’s a detailed explanation", "to summarize"
    ]
    llm_phrase_present = 1 if any(p in text.lower() for p in llm_patterns) else 0

    formal_words = ["therefore", "furthermore", "moreover", "additionally"]
    formal_ratio = sum(1 for w in words if w.lower() in formal_words) / num_words if num_words else 0

    repetition_score = len(words) / len(set(words)) if len(set(words)) else 0

    return [
        num_words,
        num_sentences,
        avg_sentence_len,
        avg_word_len,
        stopword_ratio,
        uppercase_ratio,
        punctuation_ratio,
        unique_word_ratio,
        type_token_ratio,
        hapax_ratio,
        burstiness,
        digit_ratio,
        symbol_ratio,
        flesch,
        coleman,
        smog,
        llm_phrase_present,
        formal_ratio,
        repetition_score
    ]


# -------------------------------
# Load Model
# -------------------------------
with open("rf_llm_detector.pkl", "rb") as f:
    model = joblib.load("rf_llm_detector.pkl")


# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI()

class TextRequest(BaseModel):
    text: str


@app.post("/detect")
def detect_text(req: TextRequest):
    features = extract_features(req.text)
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][prediction]

    # Print to your terminal
    print("\n--- INPUT TEXT ---")
    print(req.text)
    print("\n--- FEATURES ---")
    print(features)
    print("\n--- MODEL OUTPUT ---")
    print("Prediction:", "LLM" if prediction == 1 else "Human")
    print("Confidence:", float(prob))
    print("-------------------\n")

    return {
        "prediction": "LLM" if prediction == 1 else "Human",
        "confidence": float(prob)
    }


# Run the API
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
