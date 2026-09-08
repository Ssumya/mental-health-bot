from pathlib import Path
import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

# Load TF-IDF vectorizer
tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

# Load lightweight Logistic Regression parameters
model_data = np.load(BASE_DIR / "emotion_classifier.npz")

coef = model_data["coef"]
intercept = model_data["intercept"]

# Load label names
label_names = joblib.load(BASE_DIR / "label_names.pkl")


def predict_emotion(text):
    vector = tfidf.transform([text])

    # Logistic Regression decision scores
    scores = vector @ coef.T
    scores = np.asarray(scores).ravel() + intercept

    emotion_id = int(np.argmax(scores))

    return label_names[emotion_id]
