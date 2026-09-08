from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
emotion_model = joblib.load(BASE_DIR / "emotion_classifier.pkl")
label_names = joblib.load(BASE_DIR / "label_names.pkl")


def predict_emotion(text):
    vector = tfidf.transform([text])
    emotion_id = emotion_model.predict(vector)[0]
    return label_names[emotion_id]
